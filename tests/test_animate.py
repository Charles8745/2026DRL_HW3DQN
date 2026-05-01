import os
from src.dqn_naive import train_naive
from src.animate import make_dashboard_gif


def test_smoke_gif(tmp_path):
    """Train a tiny naive run, then build a GIF from its snapshots."""
    exp_dir = tmp_path / "naive_static"
    train_naive(
        epochs=4, snapshot_every=2, mode='static', seed=0,
        out_dir=str(exp_dir), eval_n_games=5,
    )
    out_gif = make_dashboard_gif(
        exp_dir=str(exp_dir),
        fps=4,
        loss_yscale='linear',
        max_test_steps=6,
        out_filename='dashboard.gif',
    )
    assert os.path.exists(out_gif)
    assert os.path.getsize(out_gif) > 0
    # Should be a binary file (GIF magic bytes 'GIF8')
    with open(out_gif, 'rb') as f:
        assert f.read(4) in (b'GIF8',)


def test_make_dashboard_gif_dueling(tmp_path):
    """Smoke test for HW3-2: animate against a DuelingMLP snapshot via the
    new model_factory parameter."""
    import json
    import numpy as np
    import torch
    from src.animate import make_dashboard_gif
    from src.model import build_dueling_model

    exp_dir = tmp_path / "dueling_player"
    snaps = exp_dir / "snapshots"
    snaps.mkdir(parents=True)

    # Two tiny snapshots so the GIF has at least 2 segments
    for ep in (0, 10):
        m = build_dueling_model()
        torch.save(m.state_dict(), snaps / f"epoch_{ep:04d}.pth")

    np.save(exp_dir / "losses.npy", np.linspace(1.0, 0.1, 50).astype(np.float32))
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump({
            "stage": "HW3-2: Enhanced DQN Variants for player mode",
            "experiment": "dueling_player",
            "mode": "player",
            "method": "dueling",
            "hyperparams": {"epochs": 10},
        }, f)

    gif_path = make_dashboard_gif(
        exp_dir=str(exp_dir),
        fps=2,
        loss_yscale='linear',
        max_test_steps=4,
        model_factory=build_dueling_model,
    )
    assert (exp_dir / "dashboard.gif").exists()
    assert gif_path.endswith("dashboard.gif")
    assert os.path.getsize(gif_path) > 0
    # Should be a binary file (GIF magic bytes 'GIF8')
    with open(gif_path, 'rb') as f:
        assert f.read(4) in (b'GIF8',)
