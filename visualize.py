"""Record a single game as a time-lapse video.

Plays one episode with the trained shooter (and optionally trained placer) and
renders one frame per action:

  - Left  panel: agent's view (shot history + sonar overlays)
  - Right panel: ground truth (ships visible)
  - Title:       step, action, hits, charges remaining

Saves PNGs to a directory and optionally stitches them into a GIF and/or MP4.

Examples:
  python visualize.py --shooter checkpoints/shooter_final.pt
  python visualize.py --shooter checkpoints_sonar/shooter_final.pt --sonar \\
      --placer checkpoints_sonar/placer_final.pt --gif --mp4 --fps 4
"""
from __future__ import annotations
import argparse
import glob
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

from config import DEVICE, BOARD_SIZE, SONAR_CHARGES
from battleship.game import BattleshipGame, UNKNOWN, MISS, HIT, SUNK
from battleship.networks import ShooterNet, PlacerNet
from battleship.ppo import sample_masked, make_model_placer, random_placer


# ----------------------------------------------------------------------
# Color palette
# ----------------------------------------------------------------------
C_UNKNOWN     = "#cfd8dc"   # fog
C_MISS        = "#90caf9"   # light blue
C_HIT         = "#ef5350"   # red
C_SUNK        = "#5e2128"   # dark red
C_SHIP        = "#37474f"   # dark slate
C_WATER       = "#e3f2fd"   # pale blue
C_SONAR_NO    = "#a5d6a7"   # light green tint for cleared cells
C_SONAR_YES   = "#fbc02d"   # gold border for "yes" zones
C_LAST_ACTION = "#212121"   # near-black, thick border
C_GRID        = "#90a4ae"


def agent_cell_color(game: BattleshipGame, r: int, c: int) -> str:
    v = int(game.shot_board[r, c])
    if v == HIT:    return C_HIT
    if v == SUNK:   return C_SUNK
    if v == MISS:   return C_MISS
    # UNKNOWN: prefer the sonar-cleared shade if applicable
    if game.sonar_enabled and game.sonar_cleared[r, c]:
        return C_SONAR_NO
    return C_UNKNOWN


def truth_cell_color(game: BattleshipGame, r: int, c: int) -> str:
    v = int(game.shot_board[r, c])
    if v == HIT:    return C_HIT
    if v == SUNK:   return C_SUNK
    if v == MISS:   return C_MISS
    return C_SHIP if game.true_board[r, c] > 0 else C_WATER


def draw_grid(ax, game: BattleshipGame, kind: str, last_action=None):
    """kind in {'agent', 'truth'}. last_action = (kind, r, c) | None."""
    H = W = game.board_size
    color_fn = agent_cell_color if kind == "agent" else truth_cell_color
    for r in range(H):
        for c in range(W):
            ax.add_patch(Rectangle(
                (c, H - 1 - r), 1, 1,
                facecolor=color_fn(game, r, c),
                edgecolor=C_GRID, linewidth=0.6,
            ))

    if kind == "agent" and game.sonar_enabled:
        # Persistent gold borders on every cell ever inside a "yes" sonar zone.
        for r in range(H):
            for c in range(W):
                if game.sonar_yes_overlap[r, c] > 0:
                    ax.add_patch(Rectangle(
                        (c + 0.04, H - 1 - r + 0.04), 0.92, 0.92,
                        fill=False, edgecolor=C_SONAR_YES, linewidth=1.6,
                    ))

    # Highlight the most recent action on both panels (so audience can follow).
    if last_action is not None:
        akind, ar, ac = last_action
        ax.add_patch(Rectangle(
            (ac + 0.02, H - 1 - ar + 0.02), 0.96, 0.96,
            fill=False, edgecolor=C_LAST_ACTION, linewidth=3,
        ))
        # If sonar action, ring the surrounding 3x3 area.
        if akind == "sonar":
            rr_lo = max(0, ar - game.sonar_range)
            rr_hi = min(H, ar + game.sonar_range + 1)
            cc_lo = max(0, ac - game.sonar_range)
            cc_hi = min(W, ac + game.sonar_range + 1)
            ax.add_patch(Rectangle(
                (cc_lo, H - rr_hi),
                cc_hi - cc_lo, rr_hi - rr_lo,
                fill=False, edgecolor=C_LAST_ACTION, linewidth=2, linestyle="--",
            ))

    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    # Axis labels: column letters on top, row numbers on left
    for c in range(W):
        ax.text(c + 0.5, H + 0.15, chr(ord('A') + c), ha="center", va="bottom",
                fontsize=8, color="#546e7a")
    for r in range(H):
        ax.text(-0.2, H - 1 - r + 0.5, str(r + 1), ha="right", va="center",
                fontsize=8, color="#546e7a")


def render_frame(game: BattleshipGame, frame_idx: int, last_action, action_str: str,
                 output_dir: str, banner: str = ""):
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.5, 5.5))
    draw_grid(ax_l, game, "agent", last_action=last_action)
    draw_grid(ax_r, game, "truth", last_action=last_action)
    ax_l.set_title("Agent's view", fontsize=12, pad=18)
    ax_r.set_title("Ground truth (ships shown)", fontsize=12, pad=18)

    sonar_str = (f"  |  sonar {game.sonar_charges_remaining}/{game.sonar_max}"
                 if game.sonar_enabled else "")
    title = (f"Step {frame_idx}    {action_str}    "
             f"hits {game.hits_scored}/{game.total_ship_cells}    "
             f"shots fired {game.shots_fired}{sonar_str}")
    if banner:
        title = banner + "\n" + title
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
# Episode runner that records frames as it plays
# ----------------------------------------------------------------------
def play_and_record(shooter, placer_fn, sonar_charges: int, device,
                    output_dir: str, deterministic: bool, hold_frames: int = 8):
    g = BattleshipGame(sonar_charges=sonar_charges)
    placer_fn(g)

    frame_idx = 0
    render_frame(g, frame_idx, last_action=None, action_str="initial state",
                 output_dir=output_dir)
    frame_idx += 1

    max_steps = g.board_size * g.board_size + g.sonar_max * 4 + 5
    for _ in range(max_steps):
        obs = g.shooter_observation()
        mask = g.shooter_action_mask()
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = shooter(obs_t)
            if deterministic:
                a = int(logits.masked_fill(~mask_t, -1e9).argmax(dim=-1).item())
            else:
                a_t, _, _ = sample_masked(logits, mask_t)
                a = int(a_t.item())

        kind, r, c = g.decode_action(a)
        cell = f"{chr(ord('A') + c)}{r + 1}"
        if kind == "fire":
            res = g.fire(r, c)
            label = f"FIRE {cell} → "
            if res["sunk"]:        label += f"SUNK ({res['sunk_ship_size']})"
            elif res["hit"]:       label += "HIT"
            else:                  label += "miss"
            done = res["done"]
        else:
            res = g.sonar(r, c)
            label = f"SONAR {cell} → " + ("ship in zone" if res["sonar_hit"] else "all clear")
            done = False

        render_frame(g, frame_idx, last_action=(kind, r, c), action_str=label,
                     output_dir=output_dir)
        frame_idx += 1
        if done:
            break

    # Hold the final frame so the video doesn't end abruptly.
    for _ in range(hold_frames):
        render_frame(g, frame_idx, last_action=None,
                     action_str="game over",
                     banner=f"Solved in {g.shots_fired} shots",
                     output_dir=output_dir)
        frame_idx += 1
    return frame_idx, g.shots_fired


# ----------------------------------------------------------------------
# Stitching frames into GIF/MP4 via imageio
# ----------------------------------------------------------------------
def compile_video(frame_dir: str, gif_path: str = None, mp4_path: str = None, fps: int = 3):
    try:
        import imageio.v2 as imageio
    except ImportError:
        print("[warn] imageio not installed — cannot stitch video. "
              "Install with: pip install imageio imageio-ffmpeg")
        return
    frames = sorted(glob.glob(os.path.join(frame_dir, "frame_*.png")))
    if not frames:
        print("[warn] no frames found")
        return
    images = [imageio.imread(f) for f in frames]
    # Frames can differ by a few pixels because of matplotlib's tight bbox.
    # Pad them all to the largest size so video encoders are happy.
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)
    # MP4 (H.264) wants even dimensions
    max_h = max_h + (max_h % 2)
    max_w = max_w + (max_w % 2)
    padded = []
    for img in images:
        if img.shape[:2] != (max_h, max_w):
            pad = np.full((max_h, max_w, img.shape[2]), 255, dtype=img.dtype)
            pad[:img.shape[0], :img.shape[1]] = img
            padded.append(pad)
        else:
            padded.append(img)

    if gif_path:
        imageio.mimsave(gif_path, padded, duration=1.0 / fps, loop=0)
        print(f"[saved] {gif_path}  ({len(padded)} frames, {fps} fps)")

    if mp4_path:
        try:
            imageio.mimsave(mp4_path, padded, fps=fps, codec="libx264",
                            quality=8, macro_block_size=1)
            print(f"[saved] {mp4_path}  ({len(padded)} frames, {fps} fps)")
        except Exception as e:
            print(f"[warn] MP4 export failed: {e}\n"
                  "Install imageio-ffmpeg: pip install imageio-ffmpeg")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shooter", type=str, required=True)
    ap.add_argument("--placer", type=str, default=None,
                    help="Optional placer checkpoint; if omitted, ships are placed randomly")
    ap.add_argument("--sonar", action="store_true",
                    help="Use the sonar-variant agent (must match the trained model)")
    ap.add_argument("--sonar-charges", type=int, default=SONAR_CHARGES)
    ap.add_argument("--output", type=str, default="frames")
    ap.add_argument("--deterministic", action="store_true",
                    help="Greedy action selection (more reproducible video)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gif", action="store_true", help="Compile a GIF from the frames")
    ap.add_argument("--mp4", action="store_true", help="Compile an MP4 from the frames")
    ap.add_argument("--fps", type=int, default=3)
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)

    sonar_charges = args.sonar_charges if args.sonar else 0
    in_channels = 7 if sonar_charges > 0 else 4
    num_action_channels = 2 if sonar_charges > 0 else 1

    shooter = ShooterNet(BOARD_SIZE, in_channels=in_channels,
                         num_action_channels=num_action_channels).to(DEVICE)
    shooter.load_state_dict(torch.load(args.shooter, map_location=DEVICE))
    shooter.eval()

    if args.placer:
        placer = PlacerNet(BOARD_SIZE, in_channels=2).to(DEVICE)
        placer.load_state_dict(torch.load(args.placer, map_location=DEVICE))
        placer.eval()
        placer_fn = make_model_placer(placer, DEVICE, deterministic=False)
    else:
        placer_fn = random_placer

    os.makedirs(args.output, exist_ok=True)
    # Clear previous frames so old episodes don't leak into the video.
    for f in glob.glob(os.path.join(args.output, "frame_*.png")):
        os.remove(f)

    print(f"[setup] device={DEVICE}  sonar={sonar_charges if sonar_charges > 0 else 'off'}  "
          f"deterministic={args.deterministic}")
    n_frames, shots = play_and_record(
        shooter, placer_fn, sonar_charges, DEVICE,
        args.output, args.deterministic)
    print(f"[done] {n_frames} frames in {args.output}/  game solved in {shots} shots")

    if args.gif or args.mp4:
        gif_path = os.path.join(args.output, "timelapse.gif") if args.gif else None
        mp4_path = os.path.join(args.output, "timelapse.mp4") if args.mp4 else None
        compile_video(args.output, gif_path=gif_path, mp4_path=mp4_path, fps=args.fps)


if __name__ == "__main__":
    main()
