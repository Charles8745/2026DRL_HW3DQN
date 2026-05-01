import json
import os
import numpy as np
import torch
from src.dqn_dueling import train_dueling


def test_smoke_run(tmp_path):
    """Tiny-budget run on player mode to keep it fast (<5s)."""
    out_dir = tmp_path / "dueling_player"
    metrics = train_dueling(
        epochs=20,
        mem_size=50,
        batch_size=10,
        max_moves=20,
        snapshot_every=10,
        mode='player',
        seed=0,
        out_dir=str(out_dir),
        eval_n_games=10,
    )

    assert metrics['stage'] == 'HW3-2: Enhanced DQN Variants for player mode'
    assert metrics['experiment'] == 'dueling_player'
    assert metrics['method'] == 'dueling'
    assert metrics['mode'] == 'player'
    assert 'win_rate' in metrics
    # Dueling DQN does NOT use a target network — sync_freq must not appear
    assert 'sync_freq' not in metrics['hyperparams']

    assert (out_dir / 'checkpoint.pth').exists()
    assert (out_dir / 'losses.npy').exists()
    assert (out_dir / 'loss.png').exists()
    assert (out_dir / 'metrics.json').exists()
    assert (out_dir / 'snapshots').is_dir()

    losses = np.load(out_dir / 'losses.npy')
    assert losses.ndim == 1
    assert len(losses) >= 1

    snaps = sorted(os.listdir(out_dir / 'snapshots'))
    assert len(snaps) >= 2
    # Snapshots must load into a DuelingMLP (NOT the Sequential build_model).
    from src.model import build_dueling_model
    for s in snaps:
        sd = torch.load(out_dir / 'snapshots' / s, weights_only=True)
        m = build_dueling_model()
        m.load_state_dict(sd)
