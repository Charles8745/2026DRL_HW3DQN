import json
import os
import numpy as np
import torch
from src.dqn_naive import train_naive


def test_smoke_run(tmp_path):
    """5-epoch run end-to-end. Verifies all artifacts exist and are well-formed."""
    out_dir = tmp_path / "naive_static"
    metrics = train_naive(
        epochs=5,
        snapshot_every=2,
        mode='static',
        seed=0,
        out_dir=str(out_dir),
    )

    # Returned dict
    assert metrics['stage'] == 'HW3-1: Naive DQN for static mode'
    assert metrics['experiment'] == 'naive_static'
    assert metrics['method'] == 'naive'
    assert metrics['mode'] == 'static'
    assert 'win_rate_1000' not in metrics  # smoke test uses smaller eval
    assert 'win_rate' in metrics
    assert 'training_wall_time_sec' in metrics

    # Files
    assert (out_dir / 'checkpoint.pth').exists()
    assert (out_dir / 'losses.npy').exists()
    assert (out_dir / 'loss.png').exists()
    assert (out_dir / 'metrics.json').exists()
    assert (out_dir / 'snapshots').is_dir()

    # losses.npy non-empty
    losses = np.load(out_dir / 'losses.npy')
    assert losses.ndim == 1
    assert len(losses) >= 1

    # metrics.json valid
    with open(out_dir / 'metrics.json') as f:
        data = json.load(f)
    assert data['stage'] == 'HW3-1: Naive DQN for static mode'
    assert 'hyperparams' in data
    assert data['hyperparams']['epochs'] == 5

    # Snapshots: at epochs 0, 2, 4 → exactly 3 files
    snaps = sorted(os.listdir(out_dir / 'snapshots'))
    assert len(snaps) == 3, f"expected 3 snapshots, got {snaps}"
    for s in snaps:
        assert s.startswith('epoch_') and s.endswith('.pth')
        # state dict loadable
        sd = torch.load(out_dir / 'snapshots' / s, weights_only=True)
        assert isinstance(sd, dict)
