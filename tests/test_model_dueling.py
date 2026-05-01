import torch
from src.model import DuelingMLP, build_dueling_model


def test_dueling_forward_shape_default():
    """DuelingMLP with defaults: input (B, 64) -> output (B, 4)."""
    model = build_dueling_model()
    x = torch.randn(5, 64)
    y = model(x)
    assert y.shape == (5, 4)


def test_dueling_forward_shape_single_sample():
    """Works for batch size 1 (typical action-selection call)."""
    model = build_dueling_model()
    x = torch.randn(1, 64)
    y = model(x)
    assert y.shape == (1, 4)


def test_dueling_advantage_zero_mean_aggregation():
    """The mean-baseline aggregation forces (Q - V).mean(dim=1) == 0
    (i.e. advantages are centered) for every sample.
    """
    model = build_dueling_model()
    x = torch.randn(8, 64)
    q = model(x)                              # (B, 4)
    # Recompute V independently to subtract out
    h = model.trunk(x)
    v = model.value_head(h)                   # (B, 1)
    centered_advantage = q - v                # (B, 4)
    means = centered_advantage.mean(dim=1)    # (B,)
    assert torch.allclose(means, torch.zeros_like(means), atol=1e-6)


def test_dueling_factory_custom_dimensions():
    """Custom dims propagate through trunk + heads."""
    model = build_dueling_model(in_dim=16, hidden1=24, hidden2=12, n_actions=3)
    x = torch.randn(2, 16)
    assert model(x).shape == (2, 3)
    # Spot-check head shapes
    assert model.value_head.out_features == 1
    assert model.advantage_head.out_features == 3


def test_dueling_existing_build_model_unchanged():
    """Adding DuelingMLP must not break the existing Sequential factory."""
    from src.model import build_model
    m = build_model()
    assert isinstance(m, torch.nn.Sequential)
    assert m(torch.randn(3, 64)).shape == (3, 4)
