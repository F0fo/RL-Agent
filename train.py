"""Phased adversarial training:

  Phase 1  shooter PPO vs random placement                  (warm start)
  Phase 2+ alternating updates:
              placer PPO vs frozen shooter snapshot
              shooter PPO vs mixture(random | placer snapshots)

Snapshots make this fictitious self-play, which damps strategy cycling.
"""
from __future__ import annotations
import argparse
import os
import csv
import random
import time
import dataclasses
import numpy as np
import torch

from config import (DEVICE, BOARD_SIZE, SHIP_SIZES, SEED, SONAR_CHARGES,
                    PPOConfig, TrainConfig)
from battleship.game import BattleshipGame
from battleship.networks import ShooterNet, PlacerNet
from battleship.ppo import (PPOTrainer, collect_shooter_rollout, collect_placer_rollout,
                            random_placer, make_model_placer, make_model_evaluator, snapshot)
from battleship.baselines import make_heuristic_evaluator


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_mixed_placer(model_snapshots, random_prob: float, device):
    """Returns a placer_fn that with prob `random_prob` places randomly, otherwise
    samples one of the past placer snapshots uniformly and uses it."""
    def placer(game: BattleshipGame) -> None:
        if not model_snapshots or random.random() < random_prob:
            game.random_placement()
            return
        m = random.choice(model_snapshots)
        make_model_placer(m, device, deterministic=False)(game)
    return placer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shooter-pretrain", type=int, default=None,
                    help="Override TrainConfig.shooter_pretrain_updates")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--episodes", type=int, default=None,
                    help="Override PPOConfig.rollout_episodes")
    ap.add_argument("--ckpt-dir", type=str, default=None)
    ap.add_argument("--sonar", action="store_true",
                    help="Train the sonar-variant agent (200-action space, 7 obs channels)")
    ap.add_argument("--sonar-charges", type=int, default=SONAR_CHARGES,
                    help="Sonar charges per game when --sonar is set")
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny run to verify everything wires up")
    args = ap.parse_args()

    set_seed(SEED)
    ppo_cfg = PPOConfig()
    tr_cfg = TrainConfig()
    if args.episodes is not None:
        ppo_cfg.rollout_episodes = args.episodes
    if args.shooter_pretrain is not None:
        tr_cfg.shooter_pretrain_updates = args.shooter_pretrain
    if args.rounds is not None:
        tr_cfg.selfplay_rounds = args.rounds
    if args.ckpt_dir is not None:
        tr_cfg.ckpt_dir = args.ckpt_dir
    if args.smoke:
        tr_cfg.shooter_pretrain_updates = 2
        tr_cfg.selfplay_rounds = 1
        tr_cfg.placer_updates_per_round = 2
        tr_cfg.shooter_updates_per_round = 2
        tr_cfg.placer_eval_runs = 1
        ppo_cfg.rollout_episodes = 4
        ppo_cfg.batch_size = 64

    os.makedirs(tr_cfg.ckpt_dir, exist_ok=True)
    metrics_path = os.path.join(tr_cfg.ckpt_dir, tr_cfg.metrics_path)
    metrics_f = open(metrics_path, "w", newline="")
    writer = csv.writer(metrics_f)
    writer.writerow(["phase", "step", "agent", "avg_episode_len", "avg_return",
                     "policy_loss", "value_loss", "entropy", "elapsed_s"])
    t0 = time.time()

    sonar_charges = args.sonar_charges if args.sonar else 0
    in_channels = 7 if sonar_charges > 0 else 4
    num_action_channels = 2 if sonar_charges > 0 else 1
    print(f"[setup] device={DEVICE}  board={BOARD_SIZE}x{BOARD_SIZE}  ships={SHIP_SIZES}  "
          f"sonar={sonar_charges if sonar_charges > 0 else 'off'}  "
          f"d4_aug={tr_cfg.augment_symmetry}  "
          f"placer_ent={tr_cfg.placer_entropy_coef}  "
          f"placer_eval_runs={tr_cfg.placer_eval_runs}")

    shooter = ShooterNet(BOARD_SIZE, in_channels=in_channels,
                         num_action_channels=num_action_channels,
                         hidden=ppo_cfg.hidden).to(DEVICE)
    placer = PlacerNet(BOARD_SIZE, in_channels=2, hidden=ppo_cfg.hidden).to(DEVICE)
    shooter_trainer = PPOTrainer(shooter, ppo_cfg, DEVICE)
    # Placer needs more entropy to avoid mode collapse on its near-deterministic
    # action sequence (only 5 placement decisions per "episode").
    placer_ppo_cfg = dataclasses.replace(ppo_cfg, entropy_coef=tr_cfg.placer_entropy_coef)
    placer_trainer = PPOTrainer(placer, placer_ppo_cfg, DEVICE)
    heuristic_evaluator = make_heuristic_evaluator(seed=SEED) if tr_cfg.include_heuristic_in_pool else None

    placer_snapshots = []   # list of frozen PlacerNet
    shooter_snapshots = []  # list of frozen ShooterNet

    # ------------------------------------------------------------------
    # Phase 1: shooter vs random placement
    # ------------------------------------------------------------------
    print(f"[phase 1] shooter vs random placement, {tr_cfg.shooter_pretrain_updates} updates")
    for upd in range(tr_cfg.shooter_pretrain_updates):
        rollout, lengths, returns = collect_shooter_rollout(
            shooter, random_placer, ppo_cfg.rollout_episodes, ppo_cfg, DEVICE,
            sonar_charges=sonar_charges,
            augment_symmetry=tr_cfg.augment_symmetry)
        stats = shooter_trainer.update(rollout)
        if (upd + 1) % tr_cfg.log_every == 0 or upd == 0:
            avg_len = float(np.mean(lengths)); avg_ret = float(np.mean(returns))
            print(f"  upd {upd+1:4d}/{tr_cfg.shooter_pretrain_updates}  "
                  f"len={avg_len:5.1f}  ret={avg_ret:6.2f}  "
                  f"pl={stats.get('policy_loss', 0):+.3f}  "
                  f"vl={stats.get('value_loss', 0):.3f}  "
                  f"ent={stats.get('entropy', 0):.3f}")
            writer.writerow(["1", upd + 1, "shooter", avg_len, avg_ret,
                             stats.get("policy_loss", 0), stats.get("value_loss", 0),
                             stats.get("entropy", 0), time.time() - t0])
            metrics_f.flush()
    shooter_snapshots.append(snapshot(shooter))

    # ------------------------------------------------------------------
    # Phase 2+: alternating self-play
    # ------------------------------------------------------------------
    for rnd in range(tr_cfg.selfplay_rounds):
        # Build the placer's evaluation pool: last N shooter snapshots + heuristic.
        # Mixing opponents is what turns this into proper fictitious self-play —
        # a placement is only "good" if it's hard for many shooters, not just one.
        recent_snaps = shooter_snapshots[-tr_cfg.shooter_pool_size:]
        shooter_pool = [make_model_evaluator(s, DEVICE, deterministic=False)
                        for s in recent_snaps]
        if heuristic_evaluator is not None:
            shooter_pool.append(heuristic_evaluator)
        print(f"[phase 2 round {rnd+1}/{tr_cfg.selfplay_rounds}] placer vs pool of "
              f"{len(shooter_pool)} shooters ({len(recent_snaps)} snapshots"
              f"{' + heuristic' if heuristic_evaluator else ''})")
        for upd in range(tr_cfg.placer_updates_per_round):
            rollout, p_returns = collect_placer_rollout(
                placer, shooter_pool, ppo_cfg.rollout_episodes, placer_ppo_cfg, DEVICE,
                eval_runs=tr_cfg.placer_eval_runs, sonar_charges=sonar_charges)
            stats = placer_trainer.update(rollout)
            if (upd + 1) % tr_cfg.log_every == 0 or upd == 0:
                avg_shots = float(np.mean(p_returns))
                print(f"  placer upd {upd+1:4d}/{tr_cfg.placer_updates_per_round}  "
                      f"avg_shots_to_lose={avg_shots:5.2f}  "
                      f"pl={stats.get('policy_loss', 0):+.3f}  "
                      f"ent={stats.get('entropy', 0):.3f}")
                writer.writerow(["2p", rnd * 10000 + upd + 1, "placer",
                                 len(SHIP_SIZES), avg_shots,
                                 stats.get("policy_loss", 0), stats.get("value_loss", 0),
                                 stats.get("entropy", 0), time.time() - t0])
                metrics_f.flush()
        placer_snapshots.append(snapshot(placer))

        print(f"[phase 2 round {rnd+1}] shooter vs mixture(random | placer snapshots)")
        mixed = make_mixed_placer(placer_snapshots, tr_cfg.random_opponent_prob, DEVICE)
        for upd in range(tr_cfg.shooter_updates_per_round):
            rollout, lengths, returns = collect_shooter_rollout(
                shooter, mixed, ppo_cfg.rollout_episodes, ppo_cfg, DEVICE,
                sonar_charges=sonar_charges,
                augment_symmetry=tr_cfg.augment_symmetry)
            stats = shooter_trainer.update(rollout)
            if (upd + 1) % tr_cfg.log_every == 0 or upd == 0:
                avg_len = float(np.mean(lengths)); avg_ret = float(np.mean(returns))
                print(f"  shooter upd {upd+1:4d}/{tr_cfg.shooter_updates_per_round}  "
                      f"len={avg_len:5.1f}  ret={avg_ret:6.2f}  "
                      f"pl={stats.get('policy_loss', 0):+.3f}  "
                      f"ent={stats.get('entropy', 0):.3f}")
                writer.writerow(["2s", rnd * 10000 + upd + 1, "shooter", avg_len, avg_ret,
                                 stats.get("policy_loss", 0), stats.get("value_loss", 0),
                                 stats.get("entropy", 0), time.time() - t0])
                metrics_f.flush()
        shooter_snapshots.append(snapshot(shooter))

        torch.save(shooter.state_dict(), os.path.join(tr_cfg.ckpt_dir, f"shooter_r{rnd+1}.pt"))
        torch.save(placer.state_dict(), os.path.join(tr_cfg.ckpt_dir, f"placer_r{rnd+1}.pt"))

    torch.save(shooter.state_dict(), os.path.join(tr_cfg.ckpt_dir, "shooter_final.pt"))
    torch.save(placer.state_dict(), os.path.join(tr_cfg.ckpt_dir, "placer_final.pt"))
    # Persist setup so eval.py can reconstruct the right architecture.
    with open(os.path.join(tr_cfg.ckpt_dir, "setup.txt"), "w") as f:
        f.write(f"sonar_charges={sonar_charges}\n")
        f.write(f"in_channels={in_channels}\n")
        f.write(f"num_action_channels={num_action_channels}\n")
        f.write(f"hidden={ppo_cfg.hidden}\n")
    metrics_f.close()
    print(f"[done] checkpoints in {tr_cfg.ckpt_dir}/  metrics in {metrics_path}")


if __name__ == "__main__":
    main()
