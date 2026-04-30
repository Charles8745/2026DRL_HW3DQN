"""Dashboard GIF animation: agent in Gridworld (left) + loss curve (right)."""

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import numpy as np
import torch
from matplotlib.patches import Rectangle

from src.gridworld_env import Gridworld
from src.model import build_model
from src.utils import ACTION_SET, encode_state, set_seed


# Colour map for the four pieces (RGB 0–1).
PIECE_COLORS = {
    'P': (0.20, 0.45, 0.90),   # Player — blue
    '+': (0.20, 0.75, 0.30),   # Goal — green
    '-': (0.85, 0.25, 0.25),   # Pit — red
    'W': (0.45, 0.45, 0.45),   # Wall — gray
}


def _list_snapshots(snapshots_dir: Path) -> list[tuple[int, Path]]:
    """Return [(epoch_int, path), ...] sorted by epoch."""
    out = []
    for p in snapshots_dir.glob('epoch_*.pth'):
        m = re.match(r'epoch_(\d+)\.pth', p.name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return out


def _draw_grid(ax, game: Gridworld) -> None:
    ax.clear()
    size = game.board.size
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)  # row 0 at top
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_aspect('equal')
    ax.grid(True, color='black', linewidth=0.5)
    for name, piece in game.board.components.items():
        r, c = piece.pos
        colour = PIECE_COLORS.get(piece.code, (0.7, 0.7, 0.7))
        ax.add_patch(Rectangle((c - 0.45, r - 0.45), 0.9, 0.9,
                                facecolor=colour, edgecolor='black'))
        ax.text(c, r, piece.code, ha='center', va='center',
                color='white', fontsize=20, fontweight='bold')


def _draw_loss(ax, losses: np.ndarray, current_step: int,
               yscale: str = 'log') -> None:
    ax.clear()
    if len(losses) == 0:
        ax.text(0.5, 0.5, '(no losses recorded yet)',
                ha='center', va='center', transform=ax.transAxes)
        return
    ax.plot(losses, color='lightgray', linewidth=0.5)
    if current_step > 0:
        ax.plot(np.arange(current_step), losses[:current_step],
                color='C3', linewidth=1.0)
        ax.axvline(current_step, color='C3', linestyle='--', linewidth=1.0)
    ax.set_xlabel('Training step')
    ax.set_ylabel('Loss')
    if yscale == 'log':
        # avoid zeros / negatives crashing log scale
        positive = losses[losses > 0]
        if len(positive) > 0:
            ax.set_yscale('log')
            ax.set_ylim(positive.min() * 0.5, positive.max() * 2.0)
    ax.set_title('Training loss')


def _step_to_epoch_index(epoch: int, total_epochs: int,
                         total_steps: int) -> int:
    """Approximate which loss-step corresponds to a given epoch."""
    if total_epochs <= 0:
        return 0
    return min(int(total_steps * (epoch / total_epochs)), total_steps)


def _frame(figsize, snapshot_epoch, max_epoch, move_idx, action,
           game, losses, current_step, outcome, yscale):
    fig, (ax_grid, ax_loss) = plt.subplots(1, 2, figsize=figsize)
    _draw_grid(ax_grid, game)
    ax_grid.set_title(f'Epoch {snapshot_epoch} | Move {move_idx} | Action: {action}')
    _draw_loss(ax_loss, losses, current_step, yscale=yscale)
    fig.suptitle(f'Status: epoch {snapshot_epoch}/{max_epoch} | Outcome: {outcome}',
                 fontsize=12)
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return img


def make_dashboard_gif(
    *,
    exp_dir: str,
    test_mode: str | None = None,
    fps: int = 5,
    figsize: tuple = (12, 5),
    loss_yscale: str = 'log',
    max_test_steps: int = 15,
    out_filename: str = 'dashboard.gif',
) -> str:
    """Build a dashboard GIF from snapshots of one experiment. Returns gif path."""
    exp_path = Path(exp_dir)
    snapshots_dir = exp_path / 'snapshots'
    losses_path = exp_path / 'losses.npy'
    metrics_path = exp_path / 'metrics.json'

    losses = np.load(losses_path) if losses_path.exists() else np.array([])
    with open(metrics_path) as f:
        metrics = json.load(f)
    mode = test_mode or metrics['mode']
    max_epoch = metrics['hyperparams']['epochs']
    total_steps = len(losses)

    snaps = _list_snapshots(snapshots_dir)
    frames: list[np.ndarray] = []

    for snap_epoch, snap_path in snaps:
        # Deterministic test board per snapshot:
        # - static mode: always same board (mode handles it)
        # - random/player: seed shifts per snapshot, reproducible
        set_seed(snap_epoch)
        game = Gridworld(size=4, mode=mode)
        sd = torch.load(snap_path, weights_only=True)
        model = build_model()
        model.load_state_dict(sd)
        model.eval()

        current_step = _step_to_epoch_index(snap_epoch, max_epoch, total_steps)

        # Render initial frame (move 0, no action yet)
        frames.append(_frame(
            figsize, snap_epoch, max_epoch, move_idx=0, action='—',
            game=game, losses=losses, current_step=current_step,
            outcome='playing', yscale=loss_yscale,
        ))

        state = encode_state(game)
        outcome = 'playing'
        for step in range(1, max_test_steps + 1):
            with torch.no_grad():
                qval = model(state)
            action_idx = int(torch.argmax(qval).item())
            action = ACTION_SET[action_idx]
            game.makeMove(action)
            state = encode_state(game)
            reward = game.reward()
            if reward > 0:
                outcome = 'WIN'
            elif reward == -10:
                outcome = 'LOST (pit)'
            elif step == max_test_steps:
                outcome = 'LOST (timeout)'
            else:
                outcome = 'playing'
            frames.append(_frame(
                figsize, snap_epoch, max_epoch, move_idx=step, action=action,
                game=game, losses=losses, current_step=current_step,
                outcome=outcome, yscale=loss_yscale,
            ))
            if reward != -1:
                # Hold final frame for ~0.8s
                hold = max(1, int(round(fps * 0.8)))
                for _ in range(hold):
                    frames.append(frames[-1])
                break

    out_path = exp_path / out_filename
    duration_ms = int(1000 / fps)
    imageio.mimsave(str(out_path), frames, duration=duration_ms, loop=0)
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description='Dashboard GIF generator (HW3-1).')
    parser.add_argument('--exp', required=True,
                        choices=['naive_static', 'replay_static', 'replay_random'])
    parser.add_argument('--fps', type=int, default=5)
    parser.add_argument('--max-steps', type=int, default=15)
    args = parser.parse_args()
    yscale = 'log' if 'naive' in args.exp else 'linear'
    out = make_dashboard_gif(
        exp_dir=f'results/HW3-1/{args.exp}',
        fps=args.fps, loss_yscale=yscale, max_test_steps=args.max_steps,
    )
    print(f'GIF written: {out}')


if __name__ == '__main__':
    main()
