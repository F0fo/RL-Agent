"""Evaluate trained shooter and placer against baselines.

Reports average shots-to-win (lower = better shooter, higher = better placer).
"""
from __future__ import annotations
import argparse
import numpy as np
import torch

from config import DEVICE, BOARD_SIZE, SONAR_CHARGES
from battleship.game import BattleshipGame
from battleship.networks import ShooterNet, PlacerNet
from battleship.ppo import _play_shooter_to_end, make_model_placer, random_placer
from battleship.baselines import random_shooter_play, hunt_target_shooter_play


def eval_shooter(shooter, placer_fn, n: int, device, label: str,
                 deterministic=False, sonar_charges: int = 0) -> float:
    """Trained-shooter rollout (sonar charges respected)."""
    shots = []
    extra = []
    for _ in range(n):
        g = BattleshipGame(sonar_charges=sonar_charges)
        placer_fn(g)
        shots.append(_play_shooter_to_end(shooter, g, device, deterministic=deterministic))
        extra.append(g.sonars_fired)
    arr = np.array(shots)
    sonar_avg = float(np.mean(extra)) if sonar_charges > 0 else 0.0
    extra_str = f"  sonar_used={sonar_avg:.2f}/{sonar_charges}" if sonar_charges > 0 else ""
    print(f"  {label:40s}  mean={arr.mean():5.2f}  std={arr.std():5.2f}  "
          f"min={arr.min()}  max={arr.max()}{extra_str}")
    return float(arr.mean())


def eval_baseline(play_fn, placer_fn, n: int, label: str, sonar_charges: int = 0) -> float:
    """Heuristic baselines never use sonar; we still pass sonar_charges to set up the
    same game shape so it's a fair comparison."""
    rng = np.random.default_rng(0)
    shots = []
    for _ in range(n):
        g = BattleshipGame(sonar_charges=sonar_charges)
        placer_fn(g)
        shots.append(play_fn(g, rng))
    arr = np.array(shots)
    print(f"  {label:40s}  mean={arr.mean():5.2f}  std={arr.std():5.2f}  "
          f"min={arr.min()}  max={arr.max()}")
    return float(arr.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shooter", type=str, default="checkpoints/shooter_final.pt")
    ap.add_argument("--placer", type=str, default="checkpoints/placer_final.pt")
    ap.add_argument("--n", type=int, default=500, help="games per matchup")
    ap.add_argument("--sonar", action="store_true",
                    help="Evaluate the sonar-variant agent (must match the trained model)")
    ap.add_argument("--sonar-charges", type=int, default=SONAR_CHARGES)
    args = ap.parse_args()

    sonar_charges = args.sonar_charges if args.sonar else 0
    in_channels = 7 if sonar_charges > 0 else 4
    num_action_channels = 2 if sonar_charges > 0 else 1
    print(f"[eval] device={DEVICE}  games per matchup={args.n}  "
          f"sonar={sonar_charges if sonar_charges > 0 else 'off'}")

    shooter = ShooterNet(BOARD_SIZE, in_channels=in_channels,
                         num_action_channels=num_action_channels).to(DEVICE)
    placer = PlacerNet(BOARD_SIZE, in_channels=2).to(DEVICE)
    shooter.load_state_dict(torch.load(args.shooter, map_location=DEVICE))
    placer.load_state_dict(torch.load(args.placer, map_location=DEVICE))
    shooter.eval(); placer.eval()

    model_placer = make_model_placer(placer, DEVICE, deterministic=False)

    print("\n=== Shooter benchmarks (vs random placement) ===")
    eval_baseline(random_shooter_play, random_placer, args.n, "random shooter",
                  sonar_charges=sonar_charges)
    eval_baseline(hunt_target_shooter_play, random_placer, args.n, "hunt+target heuristic",
                  sonar_charges=sonar_charges)
    eval_shooter(shooter, random_placer, args.n, DEVICE, "trained shooter (sample)",
                 sonar_charges=sonar_charges)
    eval_shooter(shooter, random_placer, args.n, DEVICE, "trained shooter (greedy)",
                 deterministic=True, sonar_charges=sonar_charges)

    print("\n=== Placer benchmarks (placer vs trained shooter) ===")
    print("  Higher avg = harder placement for the trained shooter.")
    eval_shooter(shooter, random_placer, args.n, DEVICE,
                 "trained shooter vs random placement",
                 sonar_charges=sonar_charges)
    eval_shooter(shooter, model_placer, args.n, DEVICE,
                 "trained shooter vs trained placer",
                 sonar_charges=sonar_charges)

    print("\n=== Cross-check (heuristic shooter vs trained placer) ===")
    eval_baseline(hunt_target_shooter_play, model_placer, args.n,
                  "hunt+target vs trained placer", sonar_charges=sonar_charges)


if __name__ == "__main__":
    main()
