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
