"""HW3-3 Lightning-converted DQN tests."""

import os
import random
from collections import deque

import numpy as np
import pytest
import torch

from src.model import build_dueling_model


# -------- RolloutDataset --------


def test_rollout_dataset_plays_one_game_per_iter():
    """Each __iter__ call should play exactly one game and stop."""
    from src.dqn_lightning import RolloutDataset

    online = build_dueling_model()
    ds = RolloutDataset(online_model=online, mode='static', mem_size=100,
                        batch_size=10, max_moves=15, epsilon=1.0)
    # First call: plays one game, may yield 0 minibatches (buffer not full).
    batches_1 = list(iter(ds))
    n_buf_after_game_1 = len(ds.replay)
    # Second call: plays a second game; buffer must have grown (or stayed at
    # mem_size cap), proving the deque persists across iter calls.
    batches_2 = list(iter(ds))
    assert len(ds.replay) >= n_buf_after_game_1   # persisted, did not reset
    # static mode + ε=1.0 + max_moves=15: one game ≤ 16 transitions
    # (`mov > max_moves` lets one extra step through, matching HW3-1/2 semantics).
    assert n_buf_after_game_1 <= 16


def test_rollout_dataset_yields_minibatches_when_buffer_full():
    """Once buffer > batch_size, every move yields a (s1, a, r, s2, d) tuple."""
    from src.dqn_lightning import RolloutDataset

    online = build_dueling_model()
    ds = RolloutDataset(online_model=online, mode='static', mem_size=100,
                        batch_size=2, max_moves=20, epsilon=1.0)
    # Pre-fill the buffer manually so the very first move yields.
    s = torch.zeros(1, 64)
    for _ in range(5):
        ds.replay.append((s, 0, -1.0, s, False))
    batches = list(iter(ds))
    assert len(batches) >= 1
    s1, a, r, s2, d = batches[0]
    # Shapes must be (B, 64) / (B,) / (B,) / (B, 64) / (B,)  with B = batch_size = 2.
    assert s1.shape == (2, 64)
    assert s2.shape == (2, 64)
    assert a.shape == (2,)
    assert r.shape == (2,)
    assert d.shape == (2,)
    assert r.dtype == torch.float32
    assert d.dtype == torch.float32


def test_rollout_dataset_uses_online_model_for_action_selection():
    """Action selection should call online_model in torch.no_grad context.
    Detect this by counting forward calls when ε=0 (purely greedy)."""
    from src.dqn_lightning import RolloutDataset

    calls = {'n': 0}

    class CountingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(64, 4)
        def forward(self, x):
            calls['n'] += 1
            return self.lin(x)

    online = CountingModel()
    ds = RolloutDataset(online_model=online, mode='static', mem_size=100,
                        batch_size=10, max_moves=8, epsilon=0.0)
    list(iter(ds))
    # Each move calls online once for action selection. One game ≤ 9 moves
    # (`mov > max_moves` lets one extra step through, matching HW3-1/2 semantics).
    assert 1 <= calls['n'] <= 9
