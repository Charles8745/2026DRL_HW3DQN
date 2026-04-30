import torch
from src.model import build_model


def test_default_output_shape():
    """build_model() with defaults: input (B, 64) → output (B, 4)."""
    model = build_model()
    x = torch.randn(7, 64)  # batch of 7
    y = model(x)
    assert y.shape == (7, 4)


def test_default_layer_sizes():
    """Default architecture: 64 → 150 → 100 → 4 (Listing 3.2)."""
    model = build_model()
    linear_layers = [m for m in model if isinstance(m, torch.nn.Linear)]
    assert len(linear_layers) == 3
    assert linear_layers[0].in_features == 64 and linear_layers[0].out_features == 150
    assert linear_layers[1].in_features == 150 and linear_layers[1].out_features == 100
    assert linear_layers[2].in_features == 100 and linear_layers[2].out_features == 4


def test_custom_dimensions():
    """build_model accepts custom dims."""
    model = build_model(in_dim=16, hidden1=32, hidden2=24, out_dim=8)
    x = torch.randn(3, 16)
    assert model(x).shape == (3, 8)
