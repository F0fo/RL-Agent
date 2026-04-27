"""FastAPI server for the Battleship-vs-AI live demo.

Two boards per session: `human_board` (the AI fires here, has the human's ships)
and `ai_board` (the human fires here, has the trained placer's ships). Sessions
live in memory keyed by a cookie.

Run locally:  uvicorn web.server:app --reload --port 8080
Production:   uvicorn web.server:app --host 0.0.0.0 --port $PORT
"""
from __future__ import annotations
import os
import sys
import time
import uuid
import threading
from typing import Dict, List, Optional

print(f"[boot] python={sys.version.split()[0]} cwd={os.getcwd()} files={sorted(os.listdir('.'))}", flush=True)

import numpy as np
import torch
print(f"[boot] torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
from fastapi import Cookie, FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import BOARD_SIZE, SHIP_SIZES, SONAR_CHARGES
from battleship.game import BattleshipGame
from battleship.networks import PlacerNet, ShooterNet
from battleship.ppo import make_model_placer, sample_masked


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "checkpoints_sonar_v2")
SESSION_TTL_SECONDS = int(os.environ.get("SESSION_TTL_SECONDS", 1800))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[startup] device={DEVICE}  checkpoints={CHECKPOINT_DIR}  ttl={SESSION_TTL_SECONDS}s")


# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------
shooter_path = os.path.join(CHECKPOINT_DIR, "shooter_final.pt")
placer_path = os.path.join(CHECKPOINT_DIR, "placer_final.pt")
print(f"[boot] looking for checkpoints at {os.path.abspath(CHECKPOINT_DIR)}", flush=True)
if not os.path.isfile(shooter_path) or not os.path.isfile(placer_path):
    print(f"[boot] CHECKPOINT_DIR contents: {os.listdir(CHECKPOINT_DIR) if os.path.isdir(CHECKPOINT_DIR) else 'NOT A DIR'}", flush=True)
    raise FileNotFoundError(
        f"Could not find trained checkpoints in {CHECKPOINT_DIR}. "
        "Set CHECKPOINT_DIR or train first."
    )
print(f"[boot] loading shooter from {shooter_path}", flush=True)

shooter_model = ShooterNet(BOARD_SIZE, in_channels=7, num_action_channels=2).to(DEVICE)
shooter_model.load_state_dict(torch.load(shooter_path, map_location=DEVICE))
shooter_model.eval()

print(f"[boot] loading placer from {placer_path}", flush=True)
placer_model = PlacerNet(BOARD_SIZE, in_channels=2).to(DEVICE)
placer_model.load_state_dict(torch.load(placer_path, map_location=DEVICE))
placer_model.eval()

placer_fn = make_model_placer(placer_model, DEVICE, deterministic=False)
print("[boot] models loaded successfully", flush=True)


# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
class Session:
    """One game session = two BattleshipGames + turn bookkeeping."""

    def __init__(self):
        self.id = str(uuid.uuid4())
        self.created_at = time.time()
        self.last_active = self.created_at
        self.history: List[dict] = []
        self.winner: Optional[str] = None
        self.turn = "human"
        # AI shoots at this:
        self.human_board = BattleshipGame(sonar_charges=SONAR_CHARGES)
        self.human_board.random_placement()
        # Human shoots at this:
        self.ai_board = BattleshipGame(sonar_charges=SONAR_CHARGES)
        placer_fn(self.ai_board)

    @property
    def started(self) -> bool:
        return self.human_board.shots_fired + self.human_board.sonars_fired \
             + self.ai_board.shots_fired + self.ai_board.sonars_fired > 0

    def shuffle_human(self) -> None:
        if self.started:
            raise HTTPException(400, "Cannot shuffle after the game has started")
        self.human_board = BattleshipGame(sonar_charges=SONAR_CHARGES)
        self.human_board.random_placement()

    # --- Turn application ---------------------------------------------
    def human_act(self, kind: str, row: int, col: int) -> dict:
        if self.winner:
            raise HTTPException(400, "Game over")
        if self.turn != "human":
            raise HTTPException(400, "Not your turn")
        if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            raise HTTPException(400, "Out of bounds")
        if kind == "fire":
            res = self.ai_board.fire(row, col)
            if res["already_shot"]:
                raise HTTPException(400, "Already fired at that cell")
        elif kind == "sonar":
            if self.ai_board.sonar_charges_remaining <= 0:
                raise HTTPException(400, "No sonar charges remaining")
            res = self.ai_board.sonar(row, col)
        else:
            raise HTTPException(400, "Unknown action kind")

        action = {"side": "human", "kind": kind, "row": row, "col": col, "result": res}
        self.history.append(action)
        if self.ai_board.hits_scored == self.ai_board.total_ship_cells:
            self.winner = "human"
        else:
            self.turn = "ai"
        return action

    def ai_act(self) -> dict:
        if self.winner:
            return None
        # AI's view = human_board's shooter observation
        obs = self.human_board.shooter_observation()
        mask = self.human_board.shooter_action_mask()
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            logits, _ = shooter_model(obs_t)
            a_t, _, _ = sample_masked(logits, mask_t)
            a = int(a_t.item())
        kind, r, c = self.human_board.decode_action(a)
        if kind == "fire":
            res = self.human_board.fire(r, c)
        else:
            res = self.human_board.sonar(r, c)
        action = {"side": "ai", "kind": kind, "row": r, "col": c, "result": res}
        self.history.append(action)
        if self.human_board.hits_scored == self.human_board.total_ship_cells:
            self.winner = "ai"
        else:
            self.turn = "human"
        return action

    # --- Serialization ------------------------------------------------
    def to_state(self) -> dict:
        # Reveal AI ships only after the game ends.
        reveal_ai = self.winner is not None
        return {
            "session_id": self.id,
            "turn": self.turn,
            "winner": self.winner,
            "started": self.started,
            "ship_sizes": list(SHIP_SIZES),
            "human_board": board_to_dict(self.human_board, reveal_ships=True),
            "ai_board": board_to_dict(self.ai_board, reveal_ships=reveal_ai),
            "history": self.history[-30:],
        }


def board_to_dict(g: BattleshipGame, reveal_ships: bool) -> dict:
    return {
        "shot_board": g.shot_board.tolist(),
        "sonar_cleared": g.sonar_cleared.astype(int).tolist(),
        "sonar_yes_overlap": g.sonar_yes_overlap.tolist(),
        "sonar_charges_remaining": int(g.sonar_charges_remaining),
        "sonar_max": int(g.sonar_max),
        "shots_fired": int(g.shots_fired),
        "sonars_fired": int(g.sonars_fired),
        "hits_scored": int(g.hits_scored),
        "total_ship_cells": int(g.total_ship_cells),
        "true_board": g.true_board.tolist() if reveal_ships else None,
    }


sessions: Dict[str, Session] = {}
sessions_lock = threading.Lock()


def get_or_create_session(session_id: Optional[str], response: Response) -> Session:
    with sessions_lock:
        if session_id and session_id in sessions:
            s = sessions[session_id]
            s.last_active = time.time()
            return s
        s = Session()
        sessions[s.id] = s
        response.set_cookie(
            "session_id", s.id, max_age=SESSION_TTL_SECONDS,
            httponly=True, samesite="lax",
        )
        return s


def cleanup_loop():
    while True:
        time.sleep(60)
        now = time.time()
        with sessions_lock:
            stale = [sid for sid, s in sessions.items()
                     if now - s.last_active > SESSION_TTL_SECONDS]
            for sid in stale:
                sessions.pop(sid, None)
        if stale:
            print(f"[cleanup] removed {len(stale)} idle sessions, {len(sessions)} active")


threading.Thread(target=cleanup_loop, daemon=True).start()


# ----------------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------------
app = FastAPI(title="Battleship vs AI")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/healthz")
def healthz():
    return {"ok": True, "active_sessions": len(sessions)}


@app.get("/api/state")
def get_state(response: Response, session_id: Optional[str] = Cookie(None)):
    s = get_or_create_session(session_id, response)
    return {"state": s.to_state()}


@app.post("/api/new_game")
def new_game(response: Response):
    s = Session()
    with sessions_lock:
        sessions[s.id] = s
    response.set_cookie(
        "session_id", s.id, max_age=SESSION_TTL_SECONDS,
        httponly=True, samesite="lax",
    )
    return {"ok": True, "state": s.to_state()}


@app.post("/api/shuffle")
def shuffle(response: Response, session_id: Optional[str] = Cookie(None)):
    s = get_or_create_session(session_id, response)
    s.shuffle_human()
    return {"ok": True, "state": s.to_state()}


class ActionBody(BaseModel):
    kind: str
    row: int
    col: int


@app.post("/api/action")
def action(body: ActionBody, response: Response,
           session_id: Optional[str] = Cookie(None)):
    s = get_or_create_session(session_id, response)
    human_action = s.human_act(body.kind, body.row, body.col)
    ai_action = None
    if not s.winner and s.turn == "ai":
        ai_action = s.ai_act()
    return {
        "ok": True,
        "human_action": human_action,
        "ai_action": ai_action,
        "state": s.to_state(),
    }
