import json
import os
import numpy as np
import torch
from src.dqn_double import train_double


def test_smoke_run(tmp_path):
    """Tiny-budget run on player mode to keep it fast (<5s)."""
    out_dir = tmp_path / "double_player"
    metrics = train_double(
        epochs=20,
        mem_size=50,
        batch_size=10,
        max_moves=20,
        sync_freq=5,
        snapshot_every=10,
        mode='player',
        seed=0,
        out_dir=str(out_dir),
        eval_n_games=10,
    )

    assert metrics['stage'] == 'HW3-2: Enhanced DQN Variants for player mode'
    assert metrics['experiment'] == 'double_player'
    assert metrics['method'] == 'double'
    assert metrics['mode'] == 'player'
    assert 'win_rate' in metrics
    assert 'training_wall_time_sec' in metrics
    assert metrics['hyperparams']['sync_freq'] == 5

    assert (out_dir / 'checkpoint.pth').exists()
    assert (out_dir / 'losses.npy').exists()
    assert (out_dir / 'loss.png').exists()
    assert (out_dir / 'metrics.json').exists()
    assert (out_dir / 'snapshots').is_dir()

    losses = np.load(out_dir / 'losses.npy')
    assert losses.ndim == 1
    assert len(losses) >= 1

    with open(out_dir / 'metrics.json') as f:
        data = json.load(f)
    assert data['hyperparams']['mem_size'] == 50
    assert data['hyperparams']['batch_size'] == 10

    snaps = sorted(os.listdir(out_dir / 'snapshots'))
    assert len(snaps) >= 2
    for s in snaps:
        assert s.startswith('epoch_') and s.endswith('.pth')
        sd = torch.load(out_dir / 'snapshots' / s, weights_only=True)
        assert isinstance(sd, dict)
        # Saved snapshot must be the ONLINE model state dict —
        # so it should be loadable by build_model().
        from src.model import build_model
        m = build_model()
        m.load_state_dict(sd)
