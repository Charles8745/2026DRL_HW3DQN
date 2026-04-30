import json
import random
import numpy as np
import torch
from src.utils import (
    ACTION_SET, set_seed, encode_state, epsilon_greedy,
    running_mean, save_metrics,
)
from src.gridworld_env import Gridworld


def test_action_set():
    assert ACTION_SET == {0: 'u', 1: 'd', 2: 'l', 3: 'r'}


def test_set_seed_makes_torch_deterministic():
    set_seed(123)
    a = torch.randn(5)
    set_seed(123)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_set_seed_makes_numpy_deterministic():
    set_seed(123)
    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_encode_state_shape_and_dtype():
    g = Gridworld(size=4, mode='static')
    s = encode_state(g)
    assert s.shape == (1, 64)
    assert s.dtype == torch.float32


def test_encode_state_adds_noise():
    """Two calls on the same state should differ (noise injected)."""
    g = Gridworld(size=4, mode='static')
    s1 = encode_state(g)
    s2 = encode_state(g)
    assert not torch.equal(s1, s2)


def test_epsilon_greedy_pure_exploration():
    """epsilon=1.0 → always random (call many times, should not always be argmax)."""
    set_seed(42)
    qval = torch.tensor([[10.0, 0.0, 0.0, 0.0]])  # argmax is 0
    chosen = [epsilon_greedy(qval, epsilon=1.0) for _ in range(100)]
    # with eps=1, all picks are uniform random over {0,1,2,3}, not always 0
    assert set(chosen) == {0, 1, 2, 3} or len(set(chosen)) >= 3


def test_epsilon_greedy_pure_exploitation():
    """epsilon=0 → always argmax."""
    qval = torch.tensor([[0.0, 5.0, 0.0, 0.0]])  # argmax is 1
    chosen = [epsilon_greedy(qval, epsilon=0.0) for _ in range(20)]
    assert all(c == 1 for c in chosen)


def test_running_mean_basic():
    x = np.ones(100)
    y = running_mean(x, N=10)
    assert y.shape == (90,)
    assert np.allclose(y, 1.0)


def test_save_metrics_writes_valid_json(tmp_path):
    out = tmp_path / "m.json"
    save_metrics(str(out), stage="HW3-1", win_rate=0.89, hyperparams={"lr": 0.001})
    with open(out) as f:
        data = json.load(f)
    assert data["stage"] == "HW3-1"
    assert data["win_rate"] == 0.89
    assert data["hyperparams"]["lr"] == 0.001
