"""Masked PPO trainer plus rollout collectors for shooter and placer."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Tuple
import copy
import random as _random
import numpy as np
import torch
import torch.nn.functional as F

from config import (PPOConfig, BOARD_SIZE, SHIP_SIZES, HIT_REWARD, MISS_REWARD,
                    SUNK_BONUS, WIN_BONUS, SONAR_COST)
from .game import BattleshipGame, Ship, UNKNOWN


# ----------------------------------------------------------------------
# Buffers and GAE
# ----------------------------------------------------------------------
@dataclass
class Rollout:
    obs: List[np.ndarray] = field(default_factory=list)
    masks: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[float] = field(default_factory=list)
    advantages: np.ndarray = None
    returns: np.ndarray = None

    def __len__(self) -> int:
        return len(self.actions)


def compute_gae(rewards, values_with_bootstrap, dones, gamma, lam):
    """rewards: T, values_with_bootstrap: T+1, dones: T (1 if step terminal)."""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in reversed(range(T)):
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * values_with_bootstrap[t + 1] * nonterm - values_with_bootstrap[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    returns = adv + np.array(values_with_bootstrap[:T], dtype=np.float32)
    return adv, returns


# ----------------------------------------------------------------------
# Masked Categorical helpers
# ----------------------------------------------------------------------
def _masked_dist(logits: torch.Tensor, mask: torch.Tensor) -> torch.distributions.Categorical:
    masked = logits.masked_fill(~mask, -1e9)
    return torch.distributions.Categorical(logits=masked)


def sample_masked(logits: torch.Tensor, mask: torch.Tensor):
    dist = _masked_dist(logits, mask)
    a = dist.sample()
    return a, dist.log_prob(a), dist.entropy()


def evaluate_masked(logits: torch.Tensor, mask: torch.Tensor, action: torch.Tensor):
    dist = _masked_dist(logits, mask)
    return dist.log_prob(action), dist.entropy()


# ----------------------------------------------------------------------
# PPO update
# ----------------------------------------------------------------------
class PPOTrainer:
    def __init__(self, model: torch.nn.Module, cfg: PPOConfig, device: torch.device):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    def update(self, rollout: Rollout) -> dict:
        N = len(rollout)
        if N == 0:
            return {}
        obs = torch.as_tensor(np.stack(rollout.obs), dtype=torch.float32, device=self.device)
        masks = torch.as_tensor(np.stack(rollout.masks), dtype=torch.bool, device=self.device)
        actions = torch.as_tensor(np.asarray(rollout.actions), dtype=torch.long, device=self.device)
        old_lp = torch.as_tensor(np.asarray(rollout.log_probs), dtype=torch.float32, device=self.device)
        adv = torch.as_tensor(rollout.advantages, dtype=torch.float32, device=self.device)
        ret = torch.as_tensor(rollout.returns, dtype=torch.float32, device=self.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        last = {}
        for _ in range(self.cfg.epochs_per_update):
            idx = np.random.permutation(N)
            for s in range(0, N, self.cfg.batch_size):
                b = idx[s:s + self.cfg.batch_size]
                logits, values = self.model(obs[b])
                lp, ent = evaluate_masked(logits, masks[b], actions[b])
                ratio = torch.exp(lp - old_lp[b])
                s1 = ratio * adv[b]
                s2 = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * adv[b]
                policy_loss = -torch.min(s1, s2).mean()
                value_loss = F.mse_loss(values, ret[b])
                ent_loss = -ent.mean()
                loss = policy_loss + self.cfg.value_coef * value_loss + self.cfg.entropy_coef * ent_loss
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.opt.step()
                last = {"policy_loss": float(policy_loss.item()),
                        "value_loss": float(value_loss.item()),
                        "entropy": float(-ent_loss.item())}
        return last


# ----------------------------------------------------------------------
# Helpers to copy a placement into a fresh game (so we can replay it)
# ----------------------------------------------------------------------
def clone_placement(source: BattleshipGame) -> BattleshipGame:
    """Clone a placed game (copying ships and board) but with shot/sonar state reset.
    Preserves sonar settings so replays use the same action space as training."""
    g = BattleshipGame(
        board_size=source.board_size, ship_sizes=source.ship_sizes,
        sonar_charges=source.sonar_max, sonar_range=source.sonar_range,
    )
    g.true_board = source.true_board.copy()
    g.ships = [Ship(size=s.size, cells=list(s.cells), hits=set()) for s in source.ships]
    return g


# ----------------------------------------------------------------------
# Placer interface: a callable that places ships into a game
# ----------------------------------------------------------------------
def random_placer(game: BattleshipGame) -> None:
    game.random_placement()


def make_model_placer(model: torch.nn.Module, device: torch.device, deterministic: bool = False):
    """Returns a placer_fn that uses the policy network to place all ships in `game`."""
    def placer(game: BattleshipGame) -> None:
        for ship_idx, size in enumerate(game.ship_sizes):
            obs = game.placer_observation(ship_idx)
            mask = game.placer_action_mask(size)
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, _ = model(obs_t)
                if deterministic:
                    masked = logits.masked_fill(~mask_t, -1e9)
                    a = int(masked.argmax(dim=-1).item())
                else:
                    a, _, _ = sample_masked(logits, mask_t)
                    a = int(a.item())
            W = game.board_size
            r = a // (2 * W)
            c = (a // 2) % W
            o = a % 2
            ok = game.place_ship(size, r, c, o)
            assert ok, f"model_placer chose invalid action {a}"
    return placer


# ----------------------------------------------------------------------
# Shooter rollout: each game = one episode
# ----------------------------------------------------------------------
def collect_shooter_rollout(
    shooter: torch.nn.Module,
    placer_fn: Callable[[BattleshipGame], None],
    num_episodes: int,
    cfg: PPOConfig,
    device: torch.device,
    sonar_charges: int = 0,
    augment_symmetry: bool = False,
) -> Tuple[Rollout, List[int], List[float]]:
    rollout = Rollout()
    adv_all, ret_all = [], []
    ep_lengths, ep_returns = [], []
    W = BOARD_SIZE
    # Generous step cap: with sonar, an episode can have more total actions.
    max_steps = W * W + (sonar_charges * 4 if sonar_charges > 0 else 0)

    for _ in range(num_episodes):
        g = BattleshipGame(sonar_charges=sonar_charges)
        placer_fn(g)
        if augment_symmetry:
            # Random D4 transform — shooter sees each placement under all 8
            # orientations on average, removing any orientation-specific prior.
            g.apply_d4_symmetry(int(_random.randint(0, 7)))
        ep_obs, ep_masks, ep_actions, ep_lp, ep_v, ep_r, ep_d = [], [], [], [], [], [], []
        done = False
        ret = 0.0
        while not done and len(ep_actions) < max_steps:
            obs = g.shooter_observation()
            mask = g.shooter_action_mask()
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, value = shooter(obs_t)
                a, lp, _ = sample_masked(logits, mask_t)
            a_int = int(a.item())
            kind, r, c = g.decode_action(a_int)
            if kind == "fire":
                res = g.fire(r, c)
                reward = (HIT_REWARD if res["hit"] else MISS_REWARD)
                if res["sunk"]:
                    reward += SUNK_BONUS
                if res["done"]:
                    reward += WIN_BONUS
                    done = True
            else:  # sonar — no terminal effect; reward defaults to opportunity cost
                g.sonar(r, c)
                reward = SONAR_COST
            ep_obs.append(obs)
            ep_masks.append(mask)
            ep_actions.append(a_int)
            ep_lp.append(float(lp.item()))
            ep_v.append(float(value.item()))
            ep_r.append(float(reward))
            ep_d.append(1.0 if done else 0.0)
            ret += reward
        ep_v_boot = ep_v + [0.0]
        adv, retn = compute_gae(ep_r, ep_v_boot, ep_d, cfg.gamma, cfg.gae_lambda)
        rollout.obs.extend(ep_obs)
        rollout.masks.extend(ep_masks)
        rollout.actions.extend(ep_actions)
        rollout.log_probs.extend(ep_lp)
        rollout.values.extend(ep_v)
        rollout.rewards.extend(ep_r)
        rollout.dones.extend(ep_d)
        adv_all.append(adv)
        ret_all.append(retn)
        ep_lengths.append(len(ep_actions))
        ep_returns.append(ret)

    rollout.advantages = np.concatenate(adv_all) if adv_all else np.zeros(0, dtype=np.float32)
    rollout.returns = np.concatenate(ret_all) if ret_all else np.zeros(0, dtype=np.float32)
    return rollout, ep_lengths, ep_returns


# ----------------------------------------------------------------------
# Placer rollout: each "episode" = len(SHIP_SIZES) decisions, terminal reward
# from how long the shooter takes to win on the resulting placement.
# ----------------------------------------------------------------------
# A "shooter evaluator" is any callable that, given a game with ships already placed,
# plays it to completion and returns shots-to-win. This lets us mix heterogeneous
# opponents (trained model snapshots + heuristic) into the placer's reward signal.
ShooterEvaluator = Callable[[BattleshipGame], int]


def make_model_evaluator(model: torch.nn.Module, device: torch.device, deterministic: bool = False) -> ShooterEvaluator:
    def evaluate(game: BattleshipGame) -> int:
        return _play_shooter_to_end(model, game, device, deterministic=deterministic)
    return evaluate


def _play_shooter_to_end(shooter, game: BattleshipGame, device: torch.device, deterministic: bool = False) -> int:
    """Play `game` to completion. Returns shots-to-win (sonar queries don't count
    against the score — only fired cells do, since shots-to-win is the metric)."""
    max_steps = game.board_size * game.board_size + game.sonar_max * 4 + 5
    for _ in range(max_steps):
        obs = game.shooter_observation()
        mask = game.shooter_action_mask()
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = shooter(obs_t)
            if deterministic:
                masked = logits.masked_fill(~mask_t, -1e9)
                a = int(masked.argmax(dim=-1).item())
            else:
                a_t, _, _ = sample_masked(logits, mask_t)
                a = int(a_t.item())
        kind, r, c = game.decode_action(a)
        if kind == "fire":
            res = game.fire(r, c)
            if res["done"]:
                return game.shots_fired
        else:
            game.sonar(r, c)
    return game.shots_fired  # safety fallback (shouldn't trigger)


def collect_placer_rollout(
    placer: torch.nn.Module,
    shooter_pool: List[ShooterEvaluator],
    num_episodes: int,
    cfg: PPOConfig,
    device: torch.device,
    eval_runs: int = 6,
    sonar_charges: int = 0,
) -> Tuple[Rollout, List[float]]:
    rollout = Rollout()
    adv_all, ret_all = [], []
    placer_returns = []
    W = BOARD_SIZE
    # Centering so that the placer's terminal reward is zero-mean-ish; ~50 shots on a
    # 10x10 board from a competent shooter is a reasonable anchor.
    SHOTS_BASELINE = 50.0
    SHOTS_SCALE = 20.0

    for _ in range(num_episodes):
        g = BattleshipGame(sonar_charges=sonar_charges)
        ep_obs, ep_masks, ep_actions, ep_lp, ep_v = [], [], [], [], []
        for ship_idx, size in enumerate(g.ship_sizes):
            obs = g.placer_observation(ship_idx)
            mask = g.placer_action_mask(size)
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, value = placer(obs_t)
                a, lp, _ = sample_masked(logits, mask_t)
            a_int = int(a.item())
            r = a_int // (2 * W)
            c = (a_int // 2) % W
            o = a_int % 2
            ok = g.place_ship(size, r, c, o)
            assert ok, "Placer chose an invalid action"
            ep_obs.append(obs)
            ep_masks.append(mask)
            ep_actions.append(a_int)
            ep_lp.append(float(lp.item()))
            ep_v.append(float(value.item()))

        # Score the placement by averaging shots-to-win across opponents drawn from
        # the shooter pool. Mixing opponents prevents the placer from overfitting to
        # a single shooter's quirks (which previously caused mode collapse).
        total_shots = 0
        for _ in range(eval_runs):
            replay = clone_placement(g)
            evaluator = _random.choice(shooter_pool)
            total_shots += evaluator(replay)
        avg_shots = total_shots / float(eval_runs)
        placer_returns.append(avg_shots)
        terminal = (avg_shots - SHOTS_BASELINE) / SHOTS_SCALE

        rewards = [0.0] * (len(g.ship_sizes) - 1) + [terminal]
        dones = [0.0] * (len(g.ship_sizes) - 1) + [1.0]
        ep_v_boot = ep_v + [0.0]
        adv, retn = compute_gae(rewards, ep_v_boot, dones, cfg.gamma, cfg.gae_lambda)

        rollout.obs.extend(ep_obs)
        rollout.masks.extend(ep_masks)
        rollout.actions.extend(ep_actions)
        rollout.log_probs.extend(ep_lp)
        rollout.values.extend(ep_v)
        rollout.rewards.extend(rewards)
        rollout.dones.extend(dones)
        adv_all.append(adv)
        ret_all.append(retn)

    rollout.advantages = np.concatenate(adv_all) if adv_all else np.zeros(0, dtype=np.float32)
    rollout.returns = np.concatenate(ret_all) if ret_all else np.zeros(0, dtype=np.float32)
    return rollout, placer_returns


# ----------------------------------------------------------------------
# Snapshots for fictitious self-play — keep frozen past versions to mix into
# the opponent distribution. Reduces strategy cycling.
# ----------------------------------------------------------------------
def snapshot(model: torch.nn.Module) -> torch.nn.Module:
    snap = copy.deepcopy(model)
    snap.eval()
    for p in snap.parameters():
        p.requires_grad_(False)
    return snap
