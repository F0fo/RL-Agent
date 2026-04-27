from dataclasses import dataclass
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BOARD_SIZE = 10
SHIP_SIZES = (5, 4, 3, 3, 2)
TOTAL_SHIP_CELLS = sum(SHIP_SIZES)

# Shooter rewards
HIT_REWARD = 1.0
MISS_REWARD = -0.05
SUNK_BONUS = 1.0
WIN_BONUS = 5.0

SEED = 42

# Sonar variant: submarine-style 3x3 yes/no query.
# Charges are limited per game so the agent must learn WHEN to spend them.
# A query with all-empty result clears 9 cells; a "yes" result narrows the belief.
SONAR_CHARGES = 3
SONAR_RANGE = 1     # half-width: 1 means a 3x3 area centered on the chosen cell
SONAR_COST = 0.0    # per-use reward (0 = signal is purely opportunity cost)


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    epochs_per_update: int = 4
    batch_size: int = 256
    grad_clip: float = 0.5
    rollout_episodes: int = 32
    hidden: int = 64


@dataclass
class TrainConfig:
    # Phase 1: shooter vs random placement
    shooter_pretrain_updates: int = 300
    # Phase 2+: alternating self-play rounds
    selfplay_rounds: int = 6
    placer_updates_per_round: int = 80
    shooter_updates_per_round: int = 80
    # Variance reduction: simulate K shooter playouts per placement to score the placer
    placer_eval_runs: int = 10
    # Mix opponent: probability of facing a random opponent (vs the trained one)
    random_opponent_prob: float = 0.5
    # Placer-only entropy coefficient (overrides PPOConfig.entropy_coef for the placer).
    # Cranked up to fight mode collapse — at 0.05 the placer locked into all-horizontal.
    placer_entropy_coef: float = 0.15
    # How many recent shooter snapshots to keep in the placer's evaluation pool
    shooter_pool_size: int = 3
    # Whether to include the hunt+target heuristic shooter in the placer's eval pool
    include_heuristic_in_pool: bool = True
    # D4 symmetry augmentation: apply a random rotation/flip to the placement
    # before the shooter plays. Breaks the shooter's orientation prior so the
    # placer can't exploit it by always choosing one orientation.
    augment_symmetry: bool = True
    # Logging / checkpointing
    log_every: int = 10
    ckpt_dir: str = "checkpoints"
    metrics_path: str = "metrics.csv"
