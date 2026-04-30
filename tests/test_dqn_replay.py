import json
import os
import numpy as np
import torch
from src.dqn_replay import train_replay


def test_smoke_run(tmp_path):
    """Short run on static mode with tiny replay buffer to keep it fast."""
    out_dir = tmp_path / "replay_static"
    metrics = train_replay(
        epochs=20,
        mem_size=50,
        batch_size=10,
        max_moves=20,
        snapshot_every=10,
        mode='static',
        seed=0,
        out_dir=str(out_dir),
        eval_n_games=10,
    )

    assert metrics['stage'] == 'HW3-1: Naive DQN for static mode'
    assert metrics['experiment'] == 'replay_static'
    assert metrics['method'] == 'replay'

    assert (out_dir / 'checkpoint.pth').exists()
    assert (out_dir / 'losses.npy').exists()
    assert (out_dir / 'loss.png').exists()
    assert (out_dir / 'metrics.json').exists()
    assert (out_dir / 'snapshots').is_dir()

    losses = np.load(out_dir / 'losses.npy')
    assert losses.ndim == 1

    with open(out_dir / 'metrics.json') as f:
        data = json.load(f)
    assert data['hyperparams']['mem_size'] == 50
    assert data['hyperparams']['batch_size'] == 10

    snaps = sorted(os.listdir(out_dir / 'snapshots'))
    assert len(snaps) >= 2
