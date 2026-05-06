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


# ============================================================
# SumTree tests
# ============================================================

def test_sum_tree_total_is_sum_of_priorities():
    """After 5 adds, .total must equal the sum of priorities."""
    from src.rainbow import SumTree

    tree = SumTree(capacity=8)
    priorities = [1.0, 2.0, 3.0, 4.0, 5.0]
    for i, p in enumerate(priorities):
        tree.add(p, data=f'item-{i}')
    assert abs(tree.total - sum(priorities)) < 1e-6


def test_sum_tree_sample_finds_correct_leaf():
    """Given priorities [1, 2, 3] (cum: 1, 3, 6), sample(0.5) -> idx 0;
    sample(2) -> idx 1; sample(5) -> idx 2."""
    from src.rainbow import SumTree

    tree = SumTree(capacity=4)
    tree.add(1.0, 'a')
    tree.add(2.0, 'b')
    tree.add(3.0, 'c')
    _, _, data_a = tree.sample(0.5)
    _, _, data_b = tree.sample(2.0)
    _, _, data_c = tree.sample(5.0)
    assert data_a == 'a'
    assert data_b == 'b'
    assert data_c == 'c'


def test_sum_tree_update_changes_total_and_redirects_sampling():
    """Updating a leaf's priority must update the running total and the
    sampling distribution."""
    from src.rainbow import SumTree

    tree = SumTree(capacity=4)
    tree.add(1.0, 'a')
    tree.add(1.0, 'b')
    idx_b, _, _ = tree.sample(1.5)   # b is at cumulative range [1, 2)
    assert tree.total == 2.0
    tree.update(idx_b, 9.0)
    assert abs(tree.total - 10.0) < 1e-6
    # Now b dominates the distribution; sample(5.0) should fall on b.
    _, _, data = tree.sample(5.0)
    assert data == 'b'


def test_sum_tree_overwrites_oldest_when_full():
    """Adding past capacity must overwrite the oldest leaf circularly."""
    from src.rainbow import SumTree

    tree = SumTree(capacity=2)
    tree.add(1.0, 'a')
    tree.add(2.0, 'b')
    tree.add(3.0, 'c')   # overwrites 'a'
    # total now = 2 + 3 = 5
    assert abs(tree.total - 5.0) < 1e-6
    # Confirm 'a' is gone: sample full range, must hit only 'b' or 'c'.
    seen = set()
    for s in [0.5, 1.5, 2.5, 3.5, 4.5]:
        _, _, data = tree.sample(s)
        seen.add(data)
    assert 'a' not in seen


# ============================================================
# PrioritizedReplayBuffer tests
# ============================================================

def _dummy_transition(reward=-1.0, done=False):
    s = torch.zeros(1, 64)
    return (s, 0, reward, s, done)


def test_per_buffer_push_then_sample_shapes():
    """push 50 transitions, sample 8: must return (transitions, indices, weights)
    with len(transitions) == 8 and weights in (0, 1]."""
    from src.rainbow import PrioritizedReplayBuffer

    buf = PrioritizedReplayBuffer(capacity=64, alpha=0.5,
                                   beta_start=0.4, beta_end=1.0)
    for _ in range(50):
        buf.push(_dummy_transition())
    transitions, idxs, w = buf.sample(8, frac=0.0)
    assert len(transitions) == 8
    assert len(idxs) == 8
    assert w.shape == (8,)
    assert (w > 0).all() and (w <= 1.0 + 1e-6).all()


def test_per_buffer_new_transition_gets_max_priority():
    """A freshly pushed transition should be sampled at least once if we
    ask for n samples >= n_existing — because new items inherit max priority,
    not zero. Concretely: push 10 items, set first 9 to tiny priority,
    push the 10th — the 10th must dominate sampling."""
    from src.rainbow import PrioritizedReplayBuffer
    import random
    import numpy as np

    random.seed(0)
    np.random.seed(0)
    buf = PrioritizedReplayBuffer(capacity=16, alpha=1.0,
                                   beta_start=0.4, beta_end=1.0)
    for i in range(9):
        buf.push((f'old-{i}',))
    # Drop priorities of the 9 existing leaves to a tiny value so the next
    # push (which inherits current max) clearly dominates.
    transitions_before, idxs_before, _ = buf.sample(9, frac=0.0)
    for idx in idxs_before:
        buf.tree.update(idx, 1e-6)
    buf.push(('new',))
    transitions_after, _, _ = buf.sample(50, frac=0.0)
    payloads = [t[0] for t in transitions_after]
    assert payloads.count('new') >= 25  # dominates roughly half of samples


def test_per_buffer_update_priorities_shifts_distribution():
    """Update one leaf's priority way up; subsequent sampling skews to it."""
    from src.rainbow import PrioritizedReplayBuffer

    buf = PrioritizedReplayBuffer(capacity=8, alpha=1.0,
                                   beta_start=0.4, beta_end=1.0)
    for i in range(4):
        buf.push((f't-{i}',))
    _, idxs, _ = buf.sample(4, frac=0.0)
    chosen = idxs[0]
    buf.update_priorities([chosen], [100.0])
    transitions, _, _ = buf.sample(50, frac=0.0)
    # Most samples should now correspond to the high-priority leaf's payload.
    target_payload_index = chosen - (buf.tree.capacity - 1)
    target_payload = buf.tree.data[target_payload_index]
    matches = sum(1 for t in transitions if t == target_payload)
    assert matches >= 25


# ============================================================
# NStepBuffer tests
# ============================================================

def test_n_step_buffer_returns_none_until_n_filled():
    """With n=3, the first 2 appends return None; the 3rd returns the
    first n-step transition."""
    from src.rainbow import NStepBuffer

    buf = NStepBuffer(n=3, gamma=0.9)
    s = torch.zeros(1, 64)
    out0 = buf.append(s, 0, -1.0, s, False)
    out1 = buf.append(s, 1, -1.0, s, False)
    out2 = buf.append(s, 2, +10.0, s, True)
    assert out0 is None
    assert out1 is None
    assert out2 is not None
    s1, a, R_n, s_next, d = out2
    # n-step return: -1 + 0.9*(-1) + 0.9^2*10 = -1.9 + 8.1 = 6.2
    assert abs(R_n - 6.2) < 1e-6
    assert a == 0     # action of the OLDEST transition in the window
    assert d is True  # tail done -> downstream target uses 0
    assert torch.equal(s_next, s)


def test_n_step_buffer_truncates_on_done():
    """If done at step 1 (before filling n), append should not return until
    flush; flush returns the truncated n-step starting at step 0."""
    from src.rainbow import NStepBuffer

    buf = NStepBuffer(n=3, gamma=0.9)
    s = torch.zeros(1, 64)
    out0 = buf.append(s, 0, -1.0, s, False)
    out1 = buf.append(s, 7, +10.0, s, True)
    assert out0 is None
    assert out1 is None
    flushed = list(buf.flush())
    assert len(flushed) == 2          # one for action 0, one for action 7
    s1, a0, R0, _, d0 = flushed[0]
    assert a0 == 0
    # Truncated 2-step: -1 + 0.9*10 = 8.0
    assert abs(R0 - 8.0) < 1e-6
    assert d0 is True
    s1b, a1, R1, _, d1 = flushed[1]
    assert a1 == 7
    assert abs(R1 - 10.0) < 1e-6
    assert d1 is True


def test_n_step_buffer_continues_after_full_window():
    """After 3 fills + 1 more append, every subsequent append yields a new
    n-step transition (sliding window)."""
    from src.rainbow import NStepBuffer

    buf = NStepBuffer(n=3, gamma=0.9)
    s = torch.zeros(1, 64)
    outs = []
    for i in range(5):
        out = buf.append(s, i, -1.0, s, False)
        outs.append(out)
    # outs[0], outs[1] are None; outs[2..4] are valid n-step transitions.
    assert outs[0] is None
    assert outs[1] is None
    for o in outs[2:]:
        assert o is not None
    # action field tracks the OLDEST transition in the window:
    assert outs[2][1] == 0
    assert outs[3][1] == 1
    assert outs[4][1] == 2


# ============================================================
# project_distribution tests
# ============================================================

def test_project_distribution_preserves_total_mass():
    """Projection of a valid distribution must yield a valid distribution
    (rows sum to ~1, all non-negative)."""
    from src.rainbow import project_distribution

    n_atoms = 51
    v_min, v_max = -10.0, 10.0
    support = torch.linspace(v_min, v_max, n_atoms)
    B = 16
    next_dist = torch.softmax(torch.randn(B, n_atoms), dim=-1)
    rewards = torch.tensor([-1.0] * B)
    dones = torch.tensor([0.0] * B)
    m = project_distribution(next_dist, rewards, dones,
                              gamma_n=0.9 ** 3,
                              support=support, v_min=v_min, v_max=v_max,
                              n_atoms=n_atoms)
    assert m.shape == (B, n_atoms)
    sums = m.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)
    assert (m >= 0).all()


def test_project_distribution_done_is_point_mass_at_clipped_reward():
    """For done=True, the projection of any next_dist must collapse to a
    distribution concentrated near clip(R, v_min, v_max)."""
    from src.rainbow import project_distribution

    n_atoms = 51
    v_min, v_max = -10.0, 10.0
    support = torch.linspace(v_min, v_max, n_atoms)
    B = 1
    next_dist = torch.softmax(torch.randn(B, n_atoms), dim=-1)
    rewards = torch.tensor([10.0])     # in-range reward
    dones = torch.tensor([1.0])
    m = project_distribution(next_dist, rewards, dones,
                              gamma_n=0.9 ** 3,
                              support=support, v_min=v_min, v_max=v_max,
                              n_atoms=n_atoms)
    expected_value = (m * support).sum(dim=-1).item()
    # Point mass at +10 -> expected value should be ~+10.
    assert abs(expected_value - 10.0) < 1e-3
    # Distribution sum still ~1.
    assert abs(m.sum().item() - 1.0) < 1e-4


# ============================================================
# train_rainbow end-to-end smoke
# ============================================================

def test_train_rainbow_smoke_writes_all_artifacts(tmp_path):
    """Tiny-budget end-to-end run; verifies every HW3-1/2/3-style artifact
    is produced and metrics record component flags + key hyperparams."""
    from src.rainbow import train_rainbow, STAGE_LABEL, build_rainbow_model

    out_dir = tmp_path / 'rainbow_smoke'
    metrics = train_rainbow(
        epochs=4,
        mem_size=64,
        batch_size=8,
        max_moves=6,
        sync_freq=2,
        n_step=3,
        n_atoms=11,        # smaller for speed
        v_min=-10.0,
        v_max=10.0,
        snapshot_every=2,
        mode='static',
        seed=0,
        out_dir=str(out_dir),
        eval_n_games=2,
    )

    assert metrics['stage'] == STAGE_LABEL
    assert metrics['method'] == 'rainbow'
    assert metrics['mode'] == 'static'
    assert metrics['components'] == {
        'double': True, 'dueling': True, 'per': True,
        'n_step': True, 'distributional': True, 'noisy': True,
    }
    assert metrics['hyperparams']['n_step'] == 3
    assert metrics['hyperparams']['n_atoms'] == 11
    assert metrics['hyperparams']['v_min'] == -10.0
    assert metrics['hyperparams']['v_max'] == 10.0

    # Artifact set matches HW3-1/2/3.
    assert (out_dir / 'checkpoint.pth').exists()
    assert (out_dir / 'losses.npy').exists()
    assert (out_dir / 'loss.png').exists()
    assert (out_dir / 'metrics.json').exists()
    assert (out_dir / 'snapshots').is_dir()

    # Snapshots round-trip through build_rainbow_model with matching kwargs.
    sd = torch.load(out_dir / 'checkpoint.pth', weights_only=True)
    fresh = build_rainbow_model(n_atoms=11, v_min=-10.0, v_max=10.0)
    fresh.load_state_dict(sd)        # must not raise
