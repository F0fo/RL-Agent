"""Baseline shooters for benchmarking the trained agent."""
from __future__ import annotations
from typing import Callable, List, Tuple
import numpy as np

from .game import BattleshipGame, UNKNOWN, HIT, MISS


def make_heuristic_evaluator(seed: int = 0) -> Callable[[BattleshipGame], int]:
    """ShooterEvaluator that uses the hunt-and-target heuristic. Suitable for
    inclusion in the placer's reward pool."""
    rng = np.random.default_rng(seed)
    def evaluate(game: BattleshipGame) -> int:
        return hunt_target_shooter_play(game, rng)
    return evaluate


def random_shooter_play(game: BattleshipGame, rng: np.random.Generator) -> int:
    """Random shooter that never shoots the same cell twice. Returns shots-to-win."""
    W = game.board_size
    while True:
        unknown = np.argwhere(game.shot_board == UNKNOWN)
        idx = rng.integers(0, len(unknown))
        r, c = int(unknown[idx, 0]), int(unknown[idx, 1])
        res = game.fire(r, c)
        if res["done"]:
            return game.shots_fired


def hunt_target_shooter_play(game: BattleshipGame, rng: np.random.Generator) -> int:
    """Hunt-and-target heuristic: in HUNT mode, shoot a parity-2 checkerboard pattern;
    on a HIT, switch to TARGET mode and shoot adjacent unknown cells; on a SUNK,
    return to HUNT. Strong baseline for Battleship."""
    W = game.board_size
    targets: List[Tuple[int, int]] = []      # candidate cells from target mode
    hits_pending: List[Tuple[int, int]] = [] # current streak of hits not yet sunk

    def hunt_cell() -> Tuple[int, int]:
        # Cells with (r+c) even are the parity-2 checkerboard. Among those still
        # unknown, pick uniformly at random.
        cand = [(r, c) for r in range(W) for c in range(W)
                if (r + c) % 2 == 0 and game.shot_board[r, c] == UNKNOWN]
        if not cand:
            cand = [(r, c) for r in range(W) for c in range(W)
                    if game.shot_board[r, c] == UNKNOWN]
        return cand[int(rng.integers(0, len(cand)))]

    def add_neighbors(r: int, c: int) -> None:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < W and 0 <= nc < W and game.shot_board[nr, nc] == UNKNOWN:
                if (nr, nc) not in targets:
                    targets.append((nr, nc))

    while True:
        # If we have multiple aligned pending hits, prefer the two ends of the line
        if len(hits_pending) >= 2:
            rs = [p[0] for p in hits_pending]; cs = [p[1] for p in hits_pending]
            if len(set(rs)) == 1:  # horizontal
                r = rs[0]; lo, hi = min(cs) - 1, max(cs) + 1
                ends = [(r, lo), (r, hi)]
            elif len(set(cs)) == 1:  # vertical
                c = cs[0]; lo, hi = min(rs) - 1, max(rs) + 1
                ends = [(lo, c), (hi, c)]
            else:
                ends = []
            ends = [(r, c) for (r, c) in ends
                    if 0 <= r < W and 0 <= c < W and game.shot_board[r, c] == UNKNOWN]
            if ends:
                r, c = ends[int(rng.integers(0, len(ends)))]
            elif targets:
                r, c = targets.pop(int(rng.integers(0, len(targets))))
            else:
                r, c = hunt_cell()
        elif targets:
            # filter stale
            targets = [t for t in targets if game.shot_board[t[0], t[1]] == UNKNOWN]
            if targets:
                r, c = targets.pop(int(rng.integers(0, len(targets))))
            else:
                r, c = hunt_cell()
        else:
            r, c = hunt_cell()

        res = game.fire(r, c)
        if res["hit"]:
            hits_pending.append((r, c))
            add_neighbors(r, c)
            if res["sunk"]:
                # Clear pending state on sink
                hits_pending = []
                targets = [t for t in targets if game.shot_board[t[0], t[1]] == UNKNOWN]
        if res["done"]:
            return game.shots_fired
