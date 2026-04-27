"""Battleship game core: board state, ship placement, shot resolution."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

from config import BOARD_SIZE, SHIP_SIZES, TOTAL_SHIP_CELLS, SONAR_RANGE

# Observation values on the shooter's view of the board
UNKNOWN = 0
MISS = 1
HIT = 2
SUNK = 3


@dataclass
class Ship:
    size: int
    cells: List[Tuple[int, int]] = field(default_factory=list)
    hits: set = field(default_factory=set)

    @property
    def sunk(self) -> bool:
        return len(self.hits) == self.size


class BattleshipGame:
    """Single-board game state. The shooter fires at this board; ships are hidden."""

    def __init__(
        self,
        board_size: int = BOARD_SIZE,
        ship_sizes: Tuple[int, ...] = SHIP_SIZES,
        rng: Optional[np.random.Generator] = None,
        sonar_charges: int = 0,
        sonar_range: int = SONAR_RANGE,
    ):
        self.board_size = board_size
        self.ship_sizes = tuple(ship_sizes)
        self.total_ship_cells = sum(self.ship_sizes)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.sonar_max = int(sonar_charges)
        self.sonar_range = int(sonar_range)
        self.reset()

    @property
    def sonar_enabled(self) -> bool:
        return self.sonar_max > 0

    @property
    def num_obs_channels(self) -> int:
        # 4 shot states + (sonar_cleared, sonar_yes_overlap, charges_remaining)
        return 7 if self.sonar_enabled else 4

    @property
    def num_actions(self) -> int:
        # In sonar mode: first H*W are fire targets, next H*W are sonar centers.
        return self.board_size * self.board_size * (2 if self.sonar_enabled else 1)

    def reset(self) -> None:
        self.true_board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.shot_board = np.full((self.board_size, self.board_size), UNKNOWN, dtype=np.int8)
        self.ships: List[Ship] = []
        self.shots_fired = 0
        self.hits_scored = 0
        # Sonar bookkeeping
        self.sonar_charges_remaining = self.sonar_max
        self.sonar_cleared = np.zeros((self.board_size, self.board_size), dtype=bool)
        self.sonar_yes_overlap = np.zeros((self.board_size, self.board_size), dtype=np.int32)
        self.sonars_fired = 0

    # ------------------------------------------------------------------
    # Placement
    # ------------------------------------------------------------------
    def valid_placements(self, ship_size: int) -> np.ndarray:
        """Boolean mask of shape (H, W, 2). Index [r, c, o] is True if a ship
        of `ship_size` can start at (r, c) with orientation o (0=horizontal/right,
        1=vertical/down)."""
        H = W = self.board_size
        mask = np.zeros((H, W, 2), dtype=bool)
        empty = self.true_board == 0
        for r in range(H):
            for c in range(W):
                if c + ship_size <= W and empty[r, c:c + ship_size].all():
                    mask[r, c, 0] = True
                if r + ship_size <= H and empty[r:r + ship_size, c].all():
                    mask[r, c, 1] = True
        return mask

    def place_ship(self, ship_size: int, r: int, c: int, orient: int) -> bool:
        if orient == 0:
            if c + ship_size > self.board_size:
                return False
            cells = [(r, c + i) for i in range(ship_size)]
        else:
            if r + ship_size > self.board_size:
                return False
            cells = [(r + i, c) for i in range(ship_size)]
        for (rr, cc) in cells:
            if self.true_board[rr, cc] != 0:
                return False
        ship_id = len(self.ships) + 1
        for (rr, cc) in cells:
            self.true_board[rr, cc] = ship_id
        self.ships.append(Ship(size=ship_size, cells=cells))
        return True

    def random_placement(self) -> None:
        for size in self.ship_sizes:
            for _ in range(1000):
                r = int(self.rng.integers(0, self.board_size))
                c = int(self.rng.integers(0, self.board_size))
                o = int(self.rng.integers(0, 2))
                if self.place_ship(size, r, c, o):
                    break
            else:
                raise RuntimeError("Failed to place ship randomly")

    # ------------------------------------------------------------------
    # Firing
    # ------------------------------------------------------------------
    def fire(self, r: int, c: int) -> dict:
        result = {"hit": False, "sunk": False, "sunk_ship_size": 0,
                  "done": False, "already_shot": False}
        if self.shot_board[r, c] != UNKNOWN:
            result["already_shot"] = True
            return result
        self.shots_fired += 1
        ship_id = int(self.true_board[r, c])
        if ship_id == 0:
            self.shot_board[r, c] = MISS
        else:
            self.shot_board[r, c] = HIT
            self.hits_scored += 1
            ship = self.ships[ship_id - 1]
            ship.hits.add((r, c))
            result["hit"] = True
            if ship.sunk:
                result["sunk"] = True
                result["sunk_ship_size"] = ship.size
                for (rr, cc) in ship.cells:
                    self.shot_board[rr, cc] = SUNK
        if self.hits_scored == self.total_ship_cells:
            result["done"] = True
        return result

    # ------------------------------------------------------------------
    # Sonar
    # ------------------------------------------------------------------
    def sonar(self, r: int, c: int) -> dict:
        """Sonar query centered on (r, c). Returns whether ANY ship cell lies in
        the (2*range+1)^2 area (clipped to the board). Decrements charges."""
        result = {"sonar_hit": False, "charges_left": self.sonar_charges_remaining,
                  "invalid": False}
        if self.sonar_charges_remaining <= 0:
            result["invalid"] = True
            return result
        rr_lo = max(0, r - self.sonar_range)
        rr_hi = min(self.board_size, r + self.sonar_range + 1)
        cc_lo = max(0, c - self.sonar_range)
        cc_hi = min(self.board_size, c + self.sonar_range + 1)
        zone = self.true_board[rr_lo:rr_hi, cc_lo:cc_hi]
        has_ship = bool((zone > 0).any())
        self.sonar_charges_remaining -= 1
        self.sonars_fired += 1
        if has_ship:
            result["sonar_hit"] = True
            self.sonar_yes_overlap[rr_lo:rr_hi, cc_lo:cc_hi] += 1
        else:
            self.sonar_cleared[rr_lo:rr_hi, cc_lo:cc_hi] = True
        result["charges_left"] = self.sonar_charges_remaining
        return result

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Symmetry augmentation
    # ------------------------------------------------------------------
    def apply_d4_symmetry(self, k: int) -> None:
        """Apply a D4 (dihedral) symmetry to the placement in-place.

        k ∈ {0..7}: k % 4 = number of 90° CW rotations; k >= 4 = horizontal
        flip first. Used during shooter training to prevent the policy from
        learning orientation-specific priors (the "all ships are horizontal"
        shortcut)."""
        if k <= 0 or len(self.ships) == 0:
            return
        H = self.board_size
        flip = k >= 4
        rot = k % 4

        def transform(r: int, c: int) -> Tuple[int, int]:
            if flip:
                c = H - 1 - c
            for _ in range(rot):
                r, c = c, H - 1 - r
            return r, c

        new_board = np.zeros_like(self.true_board)
        new_ships: List[Ship] = []
        for ship in self.ships:
            new_cells = [transform(r, c) for (r, c) in ship.cells]
            sid = len(new_ships) + 1
            for (r, c) in new_cells:
                new_board[r, c] = sid
            new_ships.append(Ship(size=ship.size, cells=new_cells, hits=set()))
        self.true_board = new_board
        self.ships = new_ships
        # Caller is responsible for not having fired any shots yet — sonar/shot
        # state is not transformed because rollouts apply this right after
        # placement, before play begins.

    def shooter_observation(self) -> np.ndarray:
        """Without sonar: 4 channels (shot-state one-hot).
        With sonar: 7 channels — adds (sonar_cleared, sonar_yes_overlap_norm,
        charges_remaining_norm) so the policy knows what sonar told it and how
        much sonar is left."""
        H = W = self.board_size
        if not self.sonar_enabled:
            obs = np.zeros((4, H, W), dtype=np.float32)
            for v in (UNKNOWN, MISS, HIT, SUNK):
                obs[v] = (self.shot_board == v).astype(np.float32)
            return obs
        obs = np.zeros((7, H, W), dtype=np.float32)
        for v in (UNKNOWN, MISS, HIT, SUNK):
            obs[v] = (self.shot_board == v).astype(np.float32)
        obs[4] = self.sonar_cleared.astype(np.float32)
        obs[5] = np.clip(self.sonar_yes_overlap.astype(np.float32) / float(self.sonar_max), 0.0, 1.0)
        obs[6] = float(self.sonar_charges_remaining) / float(self.sonar_max)
        return obs

    def shooter_action_mask(self) -> np.ndarray:
        """Without sonar: H*W mask (True for cells not yet fired at).
        With sonar: 2*H*W mask. Indices [0, H*W) = fire (mask = unknown cells).
        Indices [H*W, 2*H*W) = sonar (mask = all True iff charges_remaining > 0)."""
        fire_mask = (self.shot_board == UNKNOWN).flatten()
        if not self.sonar_enabled:
            return fire_mask
        if self.sonar_charges_remaining > 0:
            sonar_mask = np.ones(self.board_size * self.board_size, dtype=bool)
        else:
            sonar_mask = np.zeros(self.board_size * self.board_size, dtype=bool)
        return np.concatenate([fire_mask, sonar_mask])

    def decode_action(self, action: int) -> Tuple[str, int, int]:
        """Map a flat action index to ('fire'|'sonar', row, col)."""
        HW = self.board_size * self.board_size
        if action < HW:
            return "fire", action // self.board_size, action % self.board_size
        s = action - HW
        return "sonar", s // self.board_size, s % self.board_size

    def placer_observation(self, ship_idx: int) -> np.ndarray:
        """For sequential placement: channel 0 = occupied cells, channel 1 = current
        ship's normalized size broadcast across the grid. Shape (2, H, W)."""
        obs = np.zeros((2, self.board_size, self.board_size), dtype=np.float32)
        obs[0] = (self.true_board > 0).astype(np.float32)
        if ship_idx < len(self.ship_sizes):
            obs[1] = self.ship_sizes[ship_idx] / float(max(self.ship_sizes))
        return obs

    def placer_action_mask(self, ship_size: int) -> np.ndarray:
        """Flat mask over actions indexed as r*W*2 + c*2 + orient."""
        valid = self.valid_placements(ship_size)
        return valid.reshape(-1)
