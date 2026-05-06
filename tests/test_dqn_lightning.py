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


# -------- DQNLightningModule --------


def test_lightning_module_initial_target_equals_online():
    """At construction the target network must mirror the online network."""
    from src.dqn_lightning import DQNLightningModule

    m = DQNLightningModule(lr=1e-3, gamma=0.9, sync_freq=500,
                           epochs=10, sched=False, huber=False)
    for p_o, p_t in zip(m.online.parameters(), m.target.parameters()):
        assert torch.equal(p_o.data, p_t.data)


def test_lightning_module_loss_class_switches_with_huber():
    """huber=True should select SmoothL1Loss; huber=False -> MSELoss."""
    from src.dqn_lightning import DQNLightningModule

    m_mse = DQNLightningModule(lr=1e-3, gamma=0.9, sync_freq=500,
                               epochs=10, sched=False, huber=False)
    m_huber = DQNLightningModule(lr=1e-3, gamma=0.9, sync_freq=500,
                                 epochs=10, sched=False, huber=True)
    assert isinstance(m_mse.loss_fn, torch.nn.MSELoss)
    assert isinstance(m_huber.loss_fn, torch.nn.SmoothL1Loss)


def test_lightning_module_training_step_returns_scalar_loss():
    """Feed a known minibatch, verify loss is a 0-d differentiable tensor."""
    from src.dqn_lightning import DQNLightningModule

    m = DQNLightningModule(lr=1e-3, gamma=0.9, sync_freq=500,
                           epochs=10, sched=False, huber=False)
    B = 4
    s1 = torch.randn(B, 64)
    a = torch.tensor([0, 1, 2, 3])
    r = torch.tensor([-1.0, -1.0, 10.0, -1.0])
    s2 = torch.randn(B, 64)
    d = torch.tensor([0.0, 0.0, 1.0, 0.0])
    loss = m.training_step((s1, a, r, s2, d), batch_idx=0)
    assert loss.ndim == 0
    assert loss.requires_grad


def test_lightning_module_target_sync_fires_on_correct_step():
    """on_train_batch_end should copy online → target every sync_freq calls."""
    from src.dqn_lightning import DQNLightningModule

    m = DQNLightningModule(lr=1e-3, gamma=0.9, sync_freq=3,
                           epochs=10, sched=False, huber=False)
    # Mutate online to differ from target.
    with torch.no_grad():
        for p in m.online.parameters():
            p.add_(1.0)
    # First two calls: no sync yet.
    m.on_train_batch_end(None, None, None, 0)
    m.on_train_batch_end(None, None, None, 0)
    not_synced = any(
        not torch.equal(po.data, pt.data)
        for po, pt in zip(m.online.parameters(), m.target.parameters())
    )
    assert not_synced
    # Third call: sync fires.
    m.on_train_batch_end(None, None, None, 0)
    for p_o, p_t in zip(m.online.parameters(), m.target.parameters()):
        assert torch.equal(p_o.data, p_t.data)


def test_lightning_module_configure_optimizers_no_sched():
    """sched=False -> bare optimizer."""
    from src.dqn_lightning import DQNLightningModule

    m = DQNLightningModule(lr=1e-3, gamma=0.9, sync_freq=500,
                           epochs=10, sched=False, huber=False)
    opt = m.configure_optimizers()
    assert isinstance(opt, torch.optim.Adam)


def test_lightning_module_configure_optimizers_with_sched():
    """sched=True -> dict with optimizer + CosineAnnealingLR."""
    from src.dqn_lightning import DQNLightningModule

    m = DQNLightningModule(lr=1e-3, gamma=0.9, sync_freq=500,
                           epochs=100, sched=True, huber=False)
    out = m.configure_optimizers()
    assert isinstance(out, dict)
    assert isinstance(out['optimizer'], torch.optim.Adam)
    sched_cfg = out['lr_scheduler']
    assert isinstance(sched_cfg['scheduler'],
                      torch.optim.lr_scheduler.CosineAnnealingLR)
    assert sched_cfg['interval'] == 'epoch'
    # T_max should equal epochs.
    assert sched_cfg['scheduler'].T_max == 100
    # eta_min should be 1e-5 per spec.
    assert abs(sched_cfg['scheduler'].eta_min - 1e-5) < 1e-12


# -------- SnapshotCallback --------


def test_snapshot_callback_writes_every_n_games(tmp_path):
    """on_train_epoch_end called 6 times with every=2 -> writes at game 2, 4, 6."""
    from src.dqn_lightning import DQNLightningModule, SnapshotCallback

    snaps_dir = tmp_path / 'snapshots'
    snaps_dir.mkdir()
    cb = SnapshotCallback(snaps_dir, every=2)
    module = DQNLightningModule(lr=1e-3, gamma=0.9, sync_freq=500,
                                epochs=10, sched=False, huber=False)
    for _ in range(6):
        cb.on_train_epoch_end(trainer=None, pl_module=module)
    assert (snaps_dir / 'epoch_0002.pth').exists()
    assert (snaps_dir / 'epoch_0004.pth').exists()
    assert (snaps_dir / 'epoch_0006.pth').exists()
    assert not (snaps_dir / 'epoch_0001.pth').exists()
    assert not (snaps_dir / 'epoch_0003.pth').exists()
    assert not (snaps_dir / 'epoch_0005.pth').exists()


def test_snapshot_callback_state_dict_loads_into_dueling_model(tmp_path):
    """Saved snapshot must round-trip through build_dueling_model."""
    from src.dqn_lightning import DQNLightningModule, SnapshotCallback

    snaps_dir = tmp_path / 'snapshots'
    snaps_dir.mkdir()
    cb = SnapshotCallback(snaps_dir, every=1)
    module = DQNLightningModule(lr=1e-3, gamma=0.9, sync_freq=500,
                                epochs=10, sched=False, huber=False)
    cb.on_train_epoch_end(trainer=None, pl_module=module)
    sd = torch.load(snaps_dir / 'epoch_0001.pth', weights_only=True)
    fresh = build_dueling_model()
    fresh.load_state_dict(sd)        # must not raise
