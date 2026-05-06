"""HW3-4 Rainbow DQN tests (NoisyLinear, network, buffers, projection, smoke)."""

import pytest
import torch


def test_module_imports():
    """The rainbow module must import without error."""
    import src.rainbow  # noqa: F401


# ============================================================
# NoisyLinear tests
# ============================================================

def test_noisy_linear_forward_shape_and_param_layout():
    """NoisyLinear should output shape (B, out_features) and own
    {weight,bias}_{mu,sigma} as nn.Parameter, plus {weight,bias}_epsilon as buffer."""
    from src.rainbow import NoisyLinear

    layer = NoisyLinear(64, 32, sigma_init=0.5)
    x = torch.randn(8, 64)
    y = layer(x)
    assert y.shape == (8, 32)
    # Learnable params
    assert isinstance(layer.weight_mu, torch.nn.Parameter)
    assert isinstance(layer.weight_sigma, torch.nn.Parameter)
    assert isinstance(layer.bias_mu, torch.nn.Parameter)
    assert isinstance(layer.bias_sigma, torch.nn.Parameter)
    # Noise buffers (not learnable)
    assert layer.weight_epsilon.requires_grad is False
    assert layer.bias_epsilon.requires_grad is False


def test_noisy_linear_reset_noise_changes_epsilon():
    """reset_noise() must resample weight_epsilon and bias_epsilon."""
    from src.rainbow import NoisyLinear

    torch.manual_seed(0)
    layer = NoisyLinear(64, 32)
    eps_before = layer.weight_epsilon.clone()
    layer.reset_noise()
    eps_after = layer.weight_epsilon
    assert not torch.equal(eps_before, eps_after)


def test_noisy_linear_eval_mode_is_deterministic():
    """In eval mode, two forwards on same input must give identical output
    (no noise applied; uses mu only)."""
    from src.rainbow import NoisyLinear

    layer = NoisyLinear(64, 32)
    layer.eval()
    x = torch.randn(2, 64)
    y1 = layer(x)
    layer.reset_noise()  # should have no effect in eval mode
    y2 = layer(x)
    assert torch.equal(y1, y2)


def test_noisy_linear_train_mode_uses_noise():
    """In train mode, output should differ between resampled noise calls."""
    from src.rainbow import NoisyLinear

    torch.manual_seed(0)
    layer = NoisyLinear(64, 32, sigma_init=0.5)
    layer.train()
    x = torch.ones(1, 64)  # fixed deterministic input
    y1 = layer(x).clone()
    layer.reset_noise()
    y2 = layer(x).clone()
    # With fresh noise + non-zero sigma, outputs should differ.
    assert not torch.equal(y1, y2)


# ============================================================
# DistributionalDuelingMLP tests
# ============================================================

def test_distributional_model_forward_shapes():
    """forward(state) -> (B, n_actions); forward_dist(state) -> (B, n_actions, n_atoms)."""
    from src.rainbow import build_rainbow_model

    model = build_rainbow_model()
    x = torch.randn(8, 64)
    q = model(x)
    dist = model.forward_dist(x)
    assert q.shape == (8, 4)
    assert dist.shape == (8, 4, 51)


def test_distributional_model_dist_is_valid_probability():
    """Each (B, action) row of forward_dist must sum to ~1 (softmax over atoms)
    and contain only non-negative entries."""
    from src.rainbow import build_rainbow_model

    model = build_rainbow_model()
    x = torch.randn(4, 64)
    dist = model.forward_dist(x)
    assert (dist >= 0).all()
    sums = dist.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_distributional_model_expected_q_matches_dist_dot_support():
    """forward(s) should equal sum_i z_i * dist(s, ., i), where z is the
    fixed support buffer."""
    from src.rainbow import build_rainbow_model

    model = build_rainbow_model()
    x = torch.randn(2, 64)
    dist = model.forward_dist(x)
    q_from_dist = (dist * model.support).sum(dim=-1)   # (B, 4)
    q_direct = model(x)
    assert torch.allclose(q_from_dist, q_direct, atol=1e-6)


def test_distributional_model_reset_noise_propagates():
    """model.reset_noise() must call reset_noise on every NoisyLinear inside it.
    Detect by checking that at least one NoisyLinear's weight_epsilon changes."""
    from src.rainbow import build_rainbow_model, NoisyLinear

    torch.manual_seed(0)
    model = build_rainbow_model()
    noisy_layers = [m for m in model.modules() if isinstance(m, NoisyLinear)]
    assert len(noisy_layers) >= 4   # 2 V-head layers + 2 A-head layers
    eps_before = [m.weight_epsilon.clone() for m in noisy_layers]
    model.reset_noise()
    eps_after = [m.weight_epsilon for m in noisy_layers]
    differences = [not torch.equal(b, a) for b, a in zip(eps_before, eps_after)]
    assert all(differences)
