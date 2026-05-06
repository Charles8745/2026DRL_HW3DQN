# HW3-3 Lightning + Training Tricks — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the HW3-2 Combined (Double + Dueling) DQN to PyTorch Lightning, with three orthogonal training tricks (gradient clipping / cosine lr schedule / Huber loss) toggleable via CLI flags. Produce a 5-cell ablation on `random` mode (baseline + each trick alone + all_tricks), each with the same artifact set as HW3-1/2 (loss.png, dashboard.gif, metrics.json, checkpoint.pth, losses.npy, snapshots/).

**Architecture:** A single `src/dqn_lightning.py` containing `RolloutDataset(IterableDataset)` (episode rollout + replay buffer + minibatch sampler), `DQNLightningModule(LightningModule)` (Combined Double+Dueling with optional Huber loss + optional cosine scheduler), `SnapshotCallback` (saves vanilla `state_dict` per N games for `animate.py` compatibility), and `train_lightning(...)` orchestrator. Three tricks are gated by CLI flags `--clip / --sched / --huber`; `gradient_clip_val` is a Lightning Trainer flag, `CosineAnnealingLR` is wired in `configure_optimizers`, and Huber switches the loss class at module construction. `animate.py` adds 5 new `--exp` choices and routes them to `results/HW3-3/`.

**Tech Stack:** PyTorch 2.11, PyTorch Lightning ≥2.5,<3, NumPy, matplotlib, imageio, pytest 9.

**Spec:** [`docs/superpowers/specs/2026-05-06-hw3-dqn-stage3-design.md`](../specs/2026-05-06-hw3-dqn-stage3-design.md)

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `requirements.txt` | MODIFY | Add `pytorch-lightning>=2.5,<3` (1 line) |
| `src/dqn_lightning.py` | CREATE | All Lightning training code (RolloutDataset, DQNLightningModule, SnapshotCallback, train_lightning, CLI) |
| `src/animate.py` | MODIFY | Add 5 HW3-3 `--exp` choices, route to `results/HW3-3/`, `build_dueling_model` for all 5 |
| `tests/test_dqn_lightning.py` | CREATE | RolloutDataset unit test, LightningModule init test, SnapshotCallback unit test, parametrized 5-combo smoke test |
| `results/HW3-3/{baseline,clip,sched,huber,full}_random/*` | CREATE | 5 sets of training artifacts (loss.png, dashboard.gif, metrics.json, checkpoint.pth, losses.npy, snapshots/) |
| `HW3_3_report.md` | CREATE | Chinese 6–8 page report mirroring HW3-1/2 structure |
| `chatlog3.md` | CREATE | HW3-3 conversation log |
| `README.md` | MODIFY | Mark HW3-3 ✅; add HW3-3 analysis section with 5 loss.png + 5 dashboard.gif + metrics table |

`src/model.py`, `src/utils.py`, `src/gridworld_env.py`, `src/gridboard.py`, all `dqn_*` (naive/replay/double/dueling/double_dueling) **are not modified**.

---

## Task 1 — Add `pytorch-lightning` dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1.1: Install pytorch-lightning into venv**

The venv is uv-managed (no `pip` shim). Install via uv:

```bash
cd "/Users/charles88/Downloads/HW3_ DQN"
uv pip install "pytorch-lightning>=2.5,<3"
```

Expected: install completes; new `pytorch_lightning` package visible in `.venv/lib/python3.12/site-packages/`.

- [ ] **Step 1.2: Capture exact installed version**

```bash
.venv/bin/python -c "import pytorch_lightning as pl; print(pl.__version__)"
```

Record the printed version (e.g. `2.5.1`). Use it to pin in next step.

- [ ] **Step 1.3: Add pin to `requirements.txt`**

Append exactly one line at the bottom (alphabetical order in this file is not maintained — append is fine and matches existing convention):

```
pytorch-lightning==<exact_version_from_1.2>
```

Then re-freeze the rest by running:

```bash
.venv/bin/python -m pip freeze 2>/dev/null || uv pip freeze | grep -E "lightning|torchmetrics|fsspec" | head -20
```

If `pytorch-lightning` install pulled in `torchmetrics`, `lightning-utilities`, or updated `fsspec`, add those exact pins too (one per line, like the other entries).

- [ ] **Step 1.4: Verify import works**

```bash
.venv/bin/python -c "from pytorch_lightning import LightningModule, Trainer; from pytorch_lightning.callbacks import Callback; from torch.utils.data import IterableDataset, DataLoader; print('OK')"
```

Expected: `OK`. No deprecation warnings about `pytorch_lightning` vs `lightning` namespace (Lightning ≥2.0 supports both; we use `pytorch_lightning` for consistency with existing PyTorch-2.x docs).

- [ ] **Step 1.5: Commit**

```bash
git add requirements.txt
git commit -m "$(cat <<'EOF'
deps: add pytorch-lightning for HW3-3

Adds pytorch-lightning>=2.5,<3 to requirements.txt for the HW3-3
Lightning-converted DQN. Pinned to exact version installed alongside
its transitive deps (torchmetrics, lightning-utilities) where the
install resolver pulled them in.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2 — `RolloutDataset` (TDD)

`RolloutDataset` is an `IterableDataset` that plays exactly one Gridworld game per `__iter__` call, owns the replay buffer (persists across calls), and yields one minibatch tuple `(s1, a, r, s2, done)` after every move where the buffer has at least `batch_size + 1` transitions. This is the only non-trivial new abstraction; we test it directly.

**Files:**
- Create: `src/dqn_lightning.py` (initial skeleton with just `RolloutDataset` + imports)
- Create: `tests/test_dqn_lightning.py` (initial skeleton with just `RolloutDataset` tests)

- [ ] **Step 2.1: Write failing tests for `RolloutDataset`**

Create `tests/test_dqn_lightning.py` with exactly this content:

```python
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
    # Each move calls online once for action selection. One game ≤ 9 calls
    # (`mov > max_moves` lets one extra step through, matching HW3-1/2 semantics).
    assert 1 <= calls['n'] <= 9
```

- [ ] **Step 2.2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_dqn_lightning.py -v
```

Expected: 3 ERRORs at collection, all `ImportError: cannot import name 'RolloutDataset' from 'src.dqn_lightning'` (module doesn't exist yet).

- [ ] **Step 2.3: Create `src/dqn_lightning.py` with `RolloutDataset` only**

Create the file with exactly this content (we'll grow it task by task):

```python
"""Lightning-wrapped Combined DQN with optional training tricks (HW3-3)."""

import argparse
import random
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback

from src.dqn_naive import _plot_loss
from src.gridworld_env import Gridworld
from src.model import build_dueling_model
from src.utils import (
    ACTION_SET, encode_state, epsilon_greedy, evaluate,
    save_metrics, set_seed,
)


STAGE_LABEL = 'HW3-3: Lightning-converted DQN with Training Tricks for random mode'


class RolloutDataset(IterableDataset):
    """Episode-driven rollout + replay buffer + minibatch sampling.

    One ``__iter__`` call plays exactly one Gridworld game; Lightning
    re-creates the iterator each epoch via ``iter(loader)``, so total games
    played equals ``Trainer(max_epochs=...)``. The replay buffer persists
    across calls (instance attribute).
    """

    def __init__(self, *, online_model: nn.Module, mode: str, mem_size: int,
                 batch_size: int, max_moves: int, epsilon: float):
        super().__init__()
        self.online = online_model
        self.mode = mode
        self.batch_size = batch_size
        self.max_moves = max_moves
        self.epsilon = epsilon
        self.replay: deque = deque(maxlen=mem_size)

    def __iter__(self):
        game = Gridworld(size=4, mode=self.mode)
        state1 = encode_state(game)
        mov = 0
        while True:
            mov += 1
            with torch.no_grad():
                qval = self.online(state1)
            action_idx = epsilon_greedy(qval, self.epsilon)
            action = ACTION_SET[action_idx]
            game.makeMove(action)
            state2 = encode_state(game)
            reward = game.reward()
            done = reward > 0
            self.replay.append((state1, action_idx, reward, state2, done))
            state1 = state2

            if len(self.replay) > self.batch_size:
                minibatch = random.sample(list(self.replay), self.batch_size)
                yield self._collate(minibatch)

            if reward != -1 or mov > self.max_moves:
                break

    @staticmethod
    def _collate(minibatch):
        s1 = torch.cat([m[0] for m in minibatch])
        a = torch.tensor([m[1] for m in minibatch])
        r = torch.tensor([m[2] for m in minibatch], dtype=torch.float32)
        s2 = torch.cat([m[3] for m in minibatch])
        d = torch.tensor([m[4] for m in minibatch], dtype=torch.float32)
        return s1, a, r, s2, d
```

- [ ] **Step 2.4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_dqn_lightning.py -v
```

Expected: 3 PASS.

- [ ] **Step 2.5: Commit**

```bash
git add src/dqn_lightning.py tests/test_dqn_lightning.py
git commit -m "$(cat <<'EOF'
feat(hw3-3): add RolloutDataset for episode-driven Lightning training

RolloutDataset is an IterableDataset that plays one Gridworld game per
__iter__ call and yields minibatch tuples once the deque-based replay
buffer is full. Buffer persists across iter calls (instance attr), so
Lightning's per-epoch iterator re-creation gives the right semantics:
one Lightning epoch = one game.

Includes 3 unit tests (one game per call, minibatch shapes/dtypes when
buffer full, online-model used for action selection).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3 — `DQNLightningModule` (TDD)

The Lightning module wraps two `DuelingMLP`s (online + target), implements the Double DQN target formula in `training_step`, syncs the target network every `sync_freq` minibatch updates via `on_train_batch_end`, and conditionally registers a `CosineAnnealingLR` scheduler in `configure_optimizers`. Loss class (`MSELoss` vs `SmoothL1Loss`) is selected at construction.

**Files:**
- Modify: `src/dqn_lightning.py` (append)
- Modify: `tests/test_dqn_lightning.py` (append)

- [ ] **Step 3.1: Write failing tests for `DQNLightningModule`**

Append to `tests/test_dqn_lightning.py`:

```python
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
```

- [ ] **Step 3.2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_dqn_lightning.py -v
```

Expected: 3 PASS (RolloutDataset) + 6 ERRORs/FAILs (`DQNLightningModule` does not exist).

- [ ] **Step 3.3: Append `DQNLightningModule` to `src/dqn_lightning.py`**

Append to the end of `src/dqn_lightning.py` (after the `RolloutDataset` class):

```python
class DQNLightningModule(LightningModule):
    """Combined Double + Dueling DQN, Lightning-wrapped, with optional tricks.

    online_model & target_model are both ``DuelingMLP`` instances. Target is a
    hard copy synced every ``sync_freq`` minibatch updates inside
    ``on_train_batch_end``. Loss is ``MSELoss`` (default) or ``SmoothL1Loss``
    when ``huber=True``. With ``sched=True``, ``configure_optimizers`` returns
    Adam + ``CosineAnnealingLR(T_max=epochs, eta_min=1e-5)`` keyed to
    ``interval='epoch'`` (one game = one epoch).
    """

    def __init__(self, *, lr: float, gamma: float, sync_freq: int,
                 epochs: int, sched: bool, huber: bool):
        super().__init__()
        self.save_hyperparameters()
        self.online = build_dueling_model()
        self.target = build_dueling_model()
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.loss_fn = nn.SmoothL1Loss() if huber else nn.MSELoss()
        self._global_update = 0
        self.training_losses: list[float] = []

    def training_step(self, batch, batch_idx):
        s1, a, r, s2, d = batch
        Q1 = self.online(s1)
        with torch.no_grad():
            online_next = self.online(s2)
            next_actions = online_next.argmax(dim=1, keepdim=True)
            target_next = self.target(s2)
            next_q = target_next.gather(1, next_actions).squeeze(1)
        Y = r + self.hparams.gamma * (1 - d) * next_q
        X = Q1.gather(1, a.long().unsqueeze(1)).squeeze(1)
        loss = self.loss_fn(X, Y.detach())
        self.training_losses.append(float(loss.item()))
        return loss

    def on_train_batch_end(self, *args, **kwargs):
        self._global_update += 1
        if self._global_update % self.hparams.sync_freq == 0:
            self.target.load_state_dict(self.online.state_dict())

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.online.parameters(), lr=self.hparams.lr)
        if not self.hparams.sched:
            return opt
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.hparams.epochs, eta_min=1e-5)
        return {
            'optimizer': opt,
            'lr_scheduler': {'scheduler': sched, 'interval': 'epoch'},
        }
```

- [ ] **Step 3.4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_dqn_lightning.py -v
```

Expected: 9 PASS.

- [ ] **Step 3.5: Commit**

```bash
git add src/dqn_lightning.py tests/test_dqn_lightning.py
git commit -m "$(cat <<'EOF'
feat(hw3-3): add DQNLightningModule with Double + Dueling target + optional tricks

DQNLightningModule wraps build_dueling_model() x 2 (online + target),
implements the Double DQN target formula in training_step, syncs target
every sync_freq updates in on_train_batch_end, and conditionally returns
CosineAnnealingLR(T_max=epochs, eta_min=1e-5) at interval='epoch' from
configure_optimizers when sched=True. Loss class switches to SmoothL1Loss
when huber=True.

6 unit tests cover initial target=online equality, loss class switch,
training_step shape/grad, target sync timing, and both optimizer config
branches.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4 — `SnapshotCallback` (TDD)

A small Lightning callback that writes `online.state_dict()` to `snapshots/epoch_NNNN.pth` every N games (N = `snapshot_every`). Tracks game count internally so it doesn't depend on `trainer.current_epoch` indexing nuances.

**Files:**
- Modify: `src/dqn_lightning.py` (append)
- Modify: `tests/test_dqn_lightning.py` (append)

- [ ] **Step 4.1: Write failing tests**

Append to `tests/test_dqn_lightning.py`:

```python
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
```

- [ ] **Step 4.2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_dqn_lightning.py -v
```

Expected: 9 PASS + 2 ERRORs (`SnapshotCallback` not defined).

- [ ] **Step 4.3: Append `SnapshotCallback` to `src/dqn_lightning.py`**

Append (after `DQNLightningModule`):

```python
class SnapshotCallback(Callback):
    """Save online model state_dict every N games as ``epoch_NNNN.pth``.

    Tracks game count internally rather than relying on
    ``trainer.current_epoch`` to avoid Lightning version off-by-one nuances.
    Naming matches HW3-1/2 (``epoch_<NNNN>.pth``) so ``animate.py`` works
    unchanged.
    """

    def __init__(self, snapshots_dir: Path, every: int):
        self.snapshots_dir = snapshots_dir
        self.every = every
        self._game = 0

    def on_train_epoch_end(self, trainer, pl_module):
        self._game += 1
        if self._game % self.every == 0:
            torch.save(
                pl_module.online.state_dict(),
                self.snapshots_dir / f'epoch_{self._game:04d}.pth',
            )
```

- [ ] **Step 4.4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_dqn_lightning.py -v
```

Expected: 11 PASS.

- [ ] **Step 4.5: Commit**

```bash
git add src/dqn_lightning.py tests/test_dqn_lightning.py
git commit -m "$(cat <<'EOF'
feat(hw3-3): add SnapshotCallback writing vanilla state_dict every N games

SnapshotCallback tracks game count internally and writes
online.state_dict() to snapshots/epoch_NNNN.pth every snapshot_every
games. The vanilla-state_dict format is what animate.py expects, so
HW3-3 dashboards work without touching animate's load logic.

2 unit tests: cadence (writes at games 2/4/6 with every=2) and
round-trip into build_dueling_model().

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5 — `train_lightning` orchestrator + parametrized smoke tests

`train_lightning(...)` ties everything together: `set_seed` + `pl.seed_everything`, build module + dataset + DataLoader, configure `Trainer` with `gradient_clip_val` and `SnapshotCallback`, call `trainer.fit`, then save final `checkpoint.pth`, `losses.npy`, `loss.png`, run `evaluate(...)`, and write `metrics.json`. The smoke test parametrizes over all 5 trick combos used by the ablation.

**Files:**
- Modify: `src/dqn_lightning.py` (append `train_lightning`)
- Modify: `tests/test_dqn_lightning.py` (append parametrized smoke test)

- [ ] **Step 5.1: Write failing parametrized smoke test**

Append to `tests/test_dqn_lightning.py`:

```python
# -------- train_lightning end-to-end smoke --------


@pytest.mark.parametrize("clip,sched,huber,tag", [
    (False, False, False, 'baseline'),
    (True,  False, False, 'clip'),
    (False, True,  False, 'sched'),
    (False, False, True,  'huber'),
    (True,  True,  True,  'full'),
])
def test_train_lightning_smoke(tmp_path, clip, sched, huber, tag):
    """Tiny-budget end-to-end run for each ablation cell. Verifies all
    artifacts are produced and metrics record the trick state correctly.
    """
    from src.dqn_lightning import train_lightning, STAGE_LABEL

    out_dir = tmp_path / f'{tag}_static'
    metrics = train_lightning(
        epochs=4,
        mem_size=20,
        batch_size=4,
        max_moves=8,
        sync_freq=2,
        snapshot_every=2,
        mode='static',
        seed=0,
        out_dir=str(out_dir),
        eval_n_games=2,
        clip=clip, sched=sched, huber=huber,
    )

    assert metrics['stage'] == STAGE_LABEL
    assert metrics['method'] == 'lightning_combined'
    assert metrics['mode'] == 'static'
    assert metrics['tricks'] == {'clip': clip, 'sched': sched, 'huber': huber}
    assert metrics['hyperparams']['gradient_clip_val'] == (10.0 if clip else None)
    assert metrics['hyperparams']['loss_fn'] == ('SmoothL1Loss' if huber else 'MSELoss')
    if sched:
        assert 'CosineAnnealingLR' in metrics['hyperparams']['lr_scheduler']
    else:
        assert metrics['hyperparams']['lr_scheduler'] is None

    # Artifact set matches HW3-1/2.
    assert (out_dir / 'checkpoint.pth').exists()
    assert (out_dir / 'losses.npy').exists()
    assert (out_dir / 'loss.png').exists()
    assert (out_dir / 'metrics.json').exists()
    assert (out_dir / 'snapshots').is_dir()

    # Snapshots load round-trip into a DuelingMLP.
    snaps = sorted(os.listdir(out_dir / 'snapshots'))
    assert len(snaps) >= 1
    for s in snaps:
        sd = torch.load(out_dir / 'snapshots' / s, weights_only=True)
        build_dueling_model().load_state_dict(sd)
```

- [ ] **Step 5.2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_dqn_lightning.py -v
```

Expected: 11 PASS + 5 ERRORs (`train_lightning` not defined).

- [ ] **Step 5.3: Append `train_lightning` and `main` to `src/dqn_lightning.py`**

Append (after `SnapshotCallback`):

```python
def train_lightning(
    *,
    epochs: int = 5000,
    gamma: float = 0.9,
    epsilon: float = 0.3,
    lr: float = 1e-3,
    mem_size: int = 1000,
    batch_size: int = 200,
    max_moves: int = 50,
    sync_freq: int = 500,
    mode: str = 'random',
    seed: int = 42,
    snapshot_every: int = 250,
    out_dir: str = 'results/HW3-3/baseline_random',
    eval_n_games: int = 1000,
    clip: bool = False,
    sched: bool = False,
    huber: bool = False,
) -> dict:
    """Train Lightning-wrapped Combined DQN with optional tricks. Saves the
    same artifact set as HW3-2 variants. Returns metrics dict.
    """
    set_seed(seed)
    pl.seed_everything(seed, workers=True)
    out_path = Path(out_dir)
    snapshots_dir = out_path / 'snapshots'
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    module = DQNLightningModule(
        lr=lr, gamma=gamma, sync_freq=sync_freq, epochs=epochs,
        sched=sched, huber=huber,
    )
    torch.save(module.online.state_dict(), snapshots_dir / 'epoch_0000.pth')

    dataset = RolloutDataset(
        online_model=module.online, mode=mode, mem_size=mem_size,
        batch_size=batch_size, max_moves=max_moves, epsilon=epsilon,
    )
    loader = DataLoader(dataset, batch_size=None, num_workers=0)

    trainer = Trainer(
        max_epochs=epochs,
        gradient_clip_val=10.0 if clip else 0.0,
        gradient_clip_algorithm='norm' if clip else None,
        callbacks=[SnapshotCallback(snapshots_dir, snapshot_every)],
        enable_progress_bar=True,
        enable_checkpointing=False,
        logger=False,
        accelerator='cpu',
        devices=1,
    )
    t0 = time.time()
    trainer.fit(module, loader)
    wall_time = time.time() - t0

    torch.save(module.online.state_dict(), out_path / 'checkpoint.pth')
    losses_arr = np.array(module.training_losses, dtype=np.float32)
    np.save(out_path / 'losses.npy', losses_arr)
    title_bits = []
    if clip: title_bits.append('clip')
    if sched: title_bits.append('sched')
    if huber: title_bits.append('huber')
    title_tag = '+'.join(title_bits) if title_bits else 'baseline'
    _plot_loss(losses_arr, out_path / 'loss.png',
               title=f'Lightning Combined ({title_tag}, {mode}) — training loss')

    eval_result = evaluate(module.online, mode=mode, n_games=eval_n_games)
    tail = losses_arr[-100:] if len(losses_arr) >= 100 else losses_arr
    metrics = {
        'stage': STAGE_LABEL,
        'experiment': out_path.name,
        'mode': mode,
        'method': 'lightning_combined',
        'tricks': {'clip': clip, 'sched': sched, 'huber': huber},
        'hyperparams': {
            'epochs': epochs, 'gamma': gamma, 'epsilon': epsilon, 'lr': lr,
            'mem_size': mem_size, 'batch_size': batch_size,
            'max_moves': max_moves, 'sync_freq': sync_freq, 'seed': seed,
            'snapshot_every': snapshot_every,
            'gradient_clip_val': 10.0 if clip else None,
            'lr_scheduler': 'CosineAnnealingLR(eta_min=1e-5)' if sched else None,
            'loss_fn': 'SmoothL1Loss' if huber else 'MSELoss',
        },
        'final_loss_mean_last_100': float(tail.mean()) if len(tail) else 0.0,
        'final_loss_std_last_100': float(tail.std()) if len(tail) else 0.0,
        'win_rate': eval_result['win_rate'],
        'avg_steps_per_win': eval_result['avg_steps_per_win'],
        'n_eval_games': eval_result['n_games'],
        'training_wall_time_sec': float(wall_time),
    }
    save_metrics(str(out_path / 'metrics.json'), **metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Lightning Combined DQN with optional training tricks (HW3-3).')
    parser.add_argument('--mode', default='random',
                        choices=['static', 'player', 'random'])
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--epsilon', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--mem-size', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--max-moves', type=int, default=50)
    parser.add_argument('--sync-freq', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--snapshot-every', type=int, default=250)
    parser.add_argument('--clip', action='store_true',
                        help='Enable gradient norm clipping (max_norm=10.0).')
    parser.add_argument('--sched', action='store_true',
                        help='Enable CosineAnnealingLR (eta_min=1e-5).')
    parser.add_argument('--huber', action='store_true',
                        help='Use Huber loss (SmoothL1Loss) instead of MSE.')
    parser.add_argument('--out-dir', default=None,
                        help='Default: auto-named from trick combo.')
    args = parser.parse_args()

    if args.out_dir:
        out_dir = args.out_dir
    else:
        active = [n for n, on in (('clip', args.clip), ('sched', args.sched),
                                  ('huber', args.huber)) if on]
        if not active:
            tag = 'baseline'
        elif len(active) == 3:
            tag = 'full'
        else:
            tag = '_'.join(active)
        out_dir = f'results/HW3-3/{tag}_{args.mode}'

    train_lightning(
        epochs=args.epochs, gamma=args.gamma, epsilon=args.epsilon, lr=args.lr,
        mem_size=args.mem_size, batch_size=args.batch_size,
        max_moves=args.max_moves, sync_freq=args.sync_freq,
        mode=args.mode, seed=args.seed,
        snapshot_every=args.snapshot_every, out_dir=out_dir,
        clip=args.clip, sched=args.sched, huber=args.huber,
    )


if __name__ == '__main__':
    main()
```

- [ ] **Step 5.4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_dqn_lightning.py -v
```

Expected: 16 PASS (3 RolloutDataset + 6 LightningModule + 2 SnapshotCallback + 5 parametrized smoke).

- [ ] **Step 5.5: Run full test suite to verify no regression**

```bash
.venv/bin/python -m pytest -q
```

Expected: 49 PASS (33 pre-existing HW3-1/2 + 16 new HW3-3). If any pre-existing test fails, stop and investigate — HW3-3 must not break HW3-1/2.

- [ ] **Step 5.6: Commit**

```bash
git add src/dqn_lightning.py tests/test_dqn_lightning.py
git commit -m "$(cat <<'EOF'
feat(hw3-3): add train_lightning orchestrator + CLI + parametrized smoke

train_lightning(...) wires up DQNLightningModule + RolloutDataset +
DataLoader + Trainer (with gradient_clip_val if --clip) +
SnapshotCallback, runs trainer.fit, then saves vanilla checkpoint.pth,
losses.npy, loss.png, and metrics.json (with HW3-1/2-compatible schema
plus a tricks bool dict and trick-specific hyperparam fields).

CLI exposes three boolean flags --clip / --sched / --huber and
auto-names out_dir as results/HW3-3/<tag>_<mode>/ from active tricks.

5 parametrized smoke tests (one per ablation cell) verify artifact
production, metrics schema, and snapshot round-trip into
build_dueling_model.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6 — Update `src/animate.py` for HW3-3 dispatch

Add 5 new `--exp` choices, route them to `results/HW3-3/`, and use `build_dueling_model` for all 5 (the Combined backbone).

**Files:**
- Modify: `src/animate.py:193-220` (the `main()` function only)

- [ ] **Step 6.1: Read the current `main()` to confirm starting state**

```bash
sed -n '193,224p' src/animate.py
```

Confirm `main()` matches what the spec §4.2 expects (HW3-1 + HW3-2 dispatch only, 7 choices total).

- [ ] **Step 6.2: Replace `main()` with HW3-3-aware version**

Use Edit on `src/animate.py`. Find this exact block (lines 193–223 currently):

```python
def main():
    parser = argparse.ArgumentParser(description='Dashboard GIF generator.')
    parser.add_argument('--exp', required=True, choices=[
        # HW3-1
        'naive_static', 'replay_static', 'replay_random',
        # HW3-2
        'replay_player', 'double_player', 'dueling_player', 'combined_player',
    ])
    parser.add_argument('--fps', type=int, default=5)
    parser.add_argument('--max-steps', type=int, default=15)
    args = parser.parse_args()

    # Stage-1 dir for HW3-1 exps; stage-2 dir for player-mode HW3-2 exps.
    stage_dir = 'HW3-2' if args.exp.endswith('_player') else 'HW3-1'

    # Dueling-architecture exps need build_dueling_model for snapshot loading.
    factory = (build_dueling_model
               if args.exp in ('dueling_player', 'combined_player')
               else build_model)

    yscale = 'log' if 'naive' in args.exp else 'linear'
    out = make_dashboard_gif(
        exp_dir=f'results/{stage_dir}/{args.exp}',
        fps=args.fps, loss_yscale=yscale, max_test_steps=args.max_steps,
        model_factory=factory,
    )
    print(f'GIF written: {out}')
```

Replace it with:

```python
def main():
    parser = argparse.ArgumentParser(description='Dashboard GIF generator.')
    parser.add_argument('--exp', required=True, choices=[
        # HW3-1
        'naive_static', 'replay_static', 'replay_random',
        # HW3-2
        'replay_player', 'double_player', 'dueling_player', 'combined_player',
        # HW3-3
        'baseline_random', 'clip_random', 'sched_random',
        'huber_random', 'full_random',
    ])
    parser.add_argument('--fps', type=int, default=5)
    parser.add_argument('--max-steps', type=int, default=15)
    args = parser.parse_args()

    # Stage dispatch.
    hw3_3 = {'baseline_random', 'clip_random', 'sched_random',
             'huber_random', 'full_random'}
    if args.exp in hw3_3:
        stage_dir = 'HW3-3'
    elif args.exp.endswith('_player'):
        stage_dir = 'HW3-2'
    else:
        stage_dir = 'HW3-1'

    # Model factory: Dueling for HW3-2 dueling/combined and all HW3-3 cells.
    dueling_exps = {'dueling_player', 'combined_player'} | hw3_3
    factory = build_dueling_model if args.exp in dueling_exps else build_model

    yscale = 'log' if 'naive' in args.exp else 'linear'
    out = make_dashboard_gif(
        exp_dir=f'results/{stage_dir}/{args.exp}',
        fps=args.fps, loss_yscale=yscale, max_test_steps=args.max_steps,
        model_factory=factory,
    )
    print(f'GIF written: {out}')
```

- [ ] **Step 6.3: Verify animate's argparse accepts new choices**

```bash
.venv/bin/python -m src.animate --exp full_random --help
```

Expected: shows help with `full_random` listed in choices. (It will not actually generate a GIF here since `--help` prints and exits.)

- [ ] **Step 6.4: Verify HW3-1/2 animate tests still pass**

```bash
.venv/bin/python -m pytest tests/test_animate.py -v
```

Expected: all existing animate tests PASS (we only changed `main()`'s dispatch table; `make_dashboard_gif` is untouched).

- [ ] **Step 6.5: Commit**

```bash
git add src/animate.py
git commit -m "$(cat <<'EOF'
feat(hw3-3): teach animate.py the 5 HW3-3 random-mode experiments

Adds the 5 HW3-3 ablation cells to animate.py's --exp choices,
routes them to results/HW3-3/, and uses build_dueling_model for
all 5 (Combined backbone). make_dashboard_gif itself is unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7 — Run the 5 ablation experiments

Each run is ~25–35s on CPU; total ~3 minutes. Each writes its full artifact set to `results/HW3-3/<tag>_random/`.

**Files:**
- Create (via training): `results/HW3-3/baseline_random/`, `results/HW3-3/clip_random/`, `results/HW3-3/sched_random/`, `results/HW3-3/huber_random/`, `results/HW3-3/full_random/`

- [ ] **Step 7.1: Run baseline (no tricks)**

```bash
.venv/bin/python -m src.dqn_lightning
```

Expected: tqdm progress bar from Lightning; training completes; `results/HW3-3/baseline_random/{loss.png, losses.npy, metrics.json, checkpoint.pth, snapshots/}` produced.

- [ ] **Step 7.2: Run +clip**

```bash
.venv/bin/python -m src.dqn_lightning --clip
```

Expected: produces `results/HW3-3/clip_random/`.

- [ ] **Step 7.3: Run +sched**

```bash
.venv/bin/python -m src.dqn_lightning --sched
```

Expected: produces `results/HW3-3/sched_random/`.

- [ ] **Step 7.4: Run +huber**

```bash
.venv/bin/python -m src.dqn_lightning --huber
```

Expected: produces `results/HW3-3/huber_random/`.

- [ ] **Step 7.5: Run +all**

```bash
.venv/bin/python -m src.dqn_lightning --clip --sched --huber
```

Expected: produces `results/HW3-3/full_random/`.

- [ ] **Step 7.6: Sanity-check the 5 metrics.json files**

```bash
for d in baseline clip sched huber full; do
  echo "=== $d ==="
  .venv/bin/python -c "import json; m=json.load(open('results/HW3-3/${d}_random/metrics.json')); \
    print(f\"  win_rate={m['win_rate']}\"); \
    print(f\"  loss mean±std={m['final_loss_mean_last_100']:.5f}±{m['final_loss_std_last_100']:.5f}\"); \
    print(f\"  wall_time={m['training_wall_time_sec']:.2f}s\")"
done
```

Expected (from spec §5.3, ±20% slack):
- `baseline.win_rate >= 0.85` (≥ HW3-1 replay_random's 85.5% — sanity check that Lightning conversion didn't break training)
- `full.win_rate > baseline.win_rate` by at least 0.03 (3 percentage points)
- `full.loss_std` < `baseline.loss_std` × 0.7 (at least 30% reduction)
- All 5 wall_times in 20–60s range

If sanity checks fail:
- baseline win_rate < 0.85 → check `pl.seed_everything` ran, check `module.online` reference is the same network being optimized (RolloutDataset shouldn't deep-copy), check target sync timing.
- full not better than baseline → check `gradient_clip_val=10.0` is actually passed (print `trainer.gradient_clip_val` after `Trainer(...)`), check scheduler stepping (log `optimizer.param_groups[0]['lr']` at start vs end of training).
- Re-run only the failing cell after fixing.

- [ ] **Step 7.7: Commit (training artifacts)**

```bash
git add results/HW3-3/
git commit -m "$(cat <<'EOF'
experiment(hw3-3): run 5-group ablation on random mode

Five training runs of the Lightning Combined DQN on random mode,
seed=42, epochs=5000. Cells: baseline (no tricks), clip only,
sched only, huber only, and all-tricks. Each cell writes loss.png,
losses.npy, metrics.json, checkpoint.pth, and snapshots/ under
results/HW3-3/<tag>_random/.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8 — Generate 5 dashboard GIFs

**Files:**
- Create (via animate): `results/HW3-3/<tag>_random/dashboard.gif` for each of the 5 cells

- [ ] **Step 8.1: Generate all 5 GIFs in one loop**

```bash
for exp in baseline_random clip_random sched_random huber_random full_random; do
  .venv/bin/python -m src.animate --exp $exp
done
```

Expected: 5 lines `GIF written: results/HW3-3/<exp>/dashboard.gif`. Each ~5–15s to render.

- [ ] **Step 8.2: Spot-check the 5 GIFs exist and are non-trivial**

```bash
ls -la results/HW3-3/*/dashboard.gif
```

Expected: 5 files, each > 100 KB (small GIFs would mean rendering failed).

- [ ] **Step 8.3: Commit**

```bash
git add results/HW3-3/
git commit -m "$(cat <<'EOF'
experiment(hw3-3): add dashboard GIFs for 5 random-mode runs

Runs animate --exp on each of the 5 HW3-3 cells to produce
dashboard.gif side-by-side with each loss.png. animate's
DuelingMLP dispatch is reused (Combined backbone for all 5 cells).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9 — Write `HW3_3_report.md`

A 6–8 page Chinese report mirroring the structure of `HW3_2_report.md`. Lengths and section weights are fixed below; numbers come from `results/HW3-3/<tag>_random/metrics.json` (read them after Task 7).

**Files:**
- Create: `HW3_3_report.md`

- [ ] **Step 9.1: Read the 5 metrics.json files into one summary**

```bash
.venv/bin/python <<'EOF'
import json
rows = []
for tag in ['baseline', 'clip', 'sched', 'huber', 'full']:
    m = json.load(open(f'results/HW3-3/{tag}_random/metrics.json'))
    rows.append((
        tag,
        m['final_loss_mean_last_100'],
        m['final_loss_std_last_100'],
        m['win_rate'],
        m['avg_steps_per_win'],
        m['training_wall_time_sec'],
    ))
print('| 變體 | Loss mean | Loss std | Win rate | Avg steps | Wall (s) |')
print('|---|---|---|---|---|---|')
for tag, lm, ls, wr, av, wt in rows:
    print(f'| {tag} | {lm:.5f} | {ls:.5f} | {wr*100:.1f}% | {av:.2f} | {wt:.2f} |')
EOF
```

Save the printed Markdown table to use in §7 of the report.

- [ ] **Step 9.2: Create `HW3_3_report.md`**

Create the file with this exact structure (fill in the bracketed `[from metrics]` placeholders using values from Step 9.1; do NOT leave them as placeholders in the committed file):

```markdown
# HW3-3: Lightning-converted DQN with Training Tricks for random mode
## PyTorch → Lightning 框架轉換 + 三個訓練技巧的消融研究

> 作者：charles88　|　課程：深度強化學習　|　日期：2026-05-06
> Repo：https://github.com/Charles8745/2026DRL_HW3DQN

---

## 1. 作業目標

本階段（HW3-3）目的有二：
1. 把 HW3-2 完成的 Combined DQN（Double + Dueling）從 vanilla PyTorch 移植到
   PyTorch Lightning，驗證框架轉換不破壞既有結果並讓訓練 tricks 變得宣告式。
2. 加入三個正交的 training tricks（gradient norm clipping、CosineAnnealingLR、
   Huber loss），在 HW3-1 留下的痛點環境（random mode）上做 5 組消融比較。

## 2. random mode 重訪：HW3-1 留下的痛點

| Mode | Player 位置 | 其他物件 | HW3-1 win rate | HW3-1 loss mean ± std |
|---|---|---|---|---|
| static | 固定 | 固定 | 100% | 0.0014 ± 0.0003 |
| player | 隨機 | 固定 | 100%（HW3-2 4 變體） | 0.0003–0.005（4 組） |
| random | 隨機 | **全隨機** | **85.5%** | **0.0613 ± 0.0573** |

random mode 的 loss std 與 mean 同量級 — 訓練全程都伴隨大幅波動，明顯有
「壞 batch 把網路踢偏」的徵兆。HW3-2 player mode 太簡單 → 4 變體都 100%、
看不出 tricks 效果；HW3-3 把舞台移回 random，才看得到 tricks 的價值。

## 3. 從 HW3-2 PyTorch Combined 到 PyTorch Lightning

### 3.1 自動 vs 手動 optimization

選 `automatic_optimization=True`：`training_step` 只回傳 loss，Lightning 自動
做 backward → grad clip → optimizer.step → scheduler.step。代價是 episode
rollout 不能放在 `training_step` 內、必須包進 `IterableDataset`；好處是三個
tricks 都能用「一行 Lightning API」啟用，呼應作業 #1 的精神。

### 3.2 RolloutDataset：把 episode 包成 IterableDataset

關鍵設計：**一次 `__iter__` 呼叫 = 一局 game = 一個 Lightning epoch**。
Lightning 為每個 epoch 重新 `iter(loader)`，所以 `Trainer(max_epochs=5000)`
等於玩 5000 局；replay buffer 是 dataset 的 instance attribute，跨 epoch
持續累積。

```python
class RolloutDataset(IterableDataset):
    def __iter__(self):
        game = Gridworld(size=4, mode=self.mode)
        state1 = encode_state(game)
        while True:
            with torch.no_grad():
                qval = self.online(state1)
            action_idx = epsilon_greedy(qval, self.epsilon)
            game.makeMove(ACTION_SET[action_idx])
            state2 = encode_state(game)
            self.replay.append((state1, action_idx, game.reward(), state2, ...))
            state1 = state2
            if len(self.replay) > self.batch_size:
                yield self._collate(random.sample(list(self.replay), self.batch_size))
            if reward != -1 or mov > self.max_moves:
                break
```

### 3.3 DQNLightningModule + Trainer + SnapshotCallback

模組三大元件：
- `training_step`：Double DQN target 計算 + loss → 回傳；Lightning 自動 backward。
- `on_train_batch_end`：global_step += 1；每 500 個 step 同步 target。
- `configure_optimizers`：Adam（+ CosineAnnealingLR if `--sched`）。

外掛 `SnapshotCallback` 在 `on_train_epoch_end` 存 `online.state_dict()` 為
`epoch_NNNN.pth`，與 HW3-1/2 的 animate.py 完全相容（vanilla state_dict 載入）。

### 3.4 等價性驗證

baseline_random（無 tricks，純 Lightning Combined）win rate = [from metrics]，
高於 HW3-1 replay_random 的 85.5%（Combined > replay 的預期效果）；
loss mean = [from metrics]，比 HW3-1 replay_random 的 0.0613 [明顯下降/相近]。
證明 Lightning 框架轉換沒有破壞訓練。

## 4. Trick A — Gradient Norm Clipping

### 4.1 原理

DQN target $Y$ 在 target net sync 後會跳變（target 用了新權重），對應的
gradient 也偶爾變大，破壞 Adam 的 momentum。Gradient norm clipping 把
所有 grad 的整體 L2 norm 上限切到 `c=10.0`：

$$g \leftarrow g \cdot \min(1, c / \|g\|_2)$$

### 4.2 程式碼（一行 Lightning API）

```python
Trainer(gradient_clip_val=10.0, gradient_clip_algorithm='norm', ...)
```

### 4.3 訓練結果

![clip loss](results/HW3-3/clip_random/loss.png)

![clip dashboard](results/HW3-3/clip_random/dashboard.gif)

| 指標 | 數值 |
|---|---|
| Final loss (mean ± std) | [from metrics] |
| Win rate | [from metrics] |
| 平均勝場步數 | [from metrics] |
| 訓練時間 | [from metrics] |

[一段 80–120 字的解讀：clip 對 std 的影響、與 baseline 比的差異]

## 5. Trick B — CosineAnnealingLR

### 5.1 原理

固定 lr=1e-3 在訓練後期太大，loss 已收斂仍被推來推去。Cosine annealing
讓 lr 從 1e-3 平滑降到 1e-5：

$$\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(t\pi/T))$$

### 5.2 程式碼

```python
def configure_optimizers(self):
    opt = torch.optim.Adam(self.online.parameters(), lr=self.hparams.lr)
    if not self.hparams.sched:
        return opt
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=self.hparams.epochs, eta_min=1e-5)
    return {'optimizer': opt,
            'lr_scheduler': {'scheduler': sched, 'interval': 'epoch'}}
```

### 5.3 訓練結果

![sched loss](results/HW3-3/sched_random/loss.png)

![sched dashboard](results/HW3-3/sched_random/dashboard.gif)

| 指標 | 數值 |
|---|---|
| Final loss (mean ± std) | [from metrics] |
| Win rate | [from metrics] |
| 平均勝場步數 | [from metrics] |
| 訓練時間 | [from metrics] |

[一段 80–120 字的解讀：sched 對後期 mean 的影響、與 baseline 比]

## 6. Trick C — Huber Loss (Smooth L1)

### 6.1 原理

random mode 的 reward 分布有重尾，MSE 對 outlier 過度敏感（誤差平方放大）。
Huber 在小誤差時用 MSE（保持 smooth gradient），大誤差時退為 L1（gradient
上限固定）：

$$L_\delta(x) = \begin{cases} \tfrac{1}{2}x^2 & |x| \le \delta \\ \delta(|x| - \tfrac{1}{2}\delta) & |x| > \delta \end{cases}$$

DQN paper（Mnih 2015）描述為 "clip TD error to [-1, 1]"，等價於 $\delta=1$ 的 Huber。

### 6.2 程式碼（一行差）

```python
self.loss_fn = nn.SmoothL1Loss() if huber else nn.MSELoss()
```

### 6.3 訓練結果

![huber loss](results/HW3-3/huber_random/loss.png)

![huber dashboard](results/HW3-3/huber_random/dashboard.gif)

| 指標 | 數值 |
|---|---|
| Final loss (mean ± std) | [from metrics] |
| Win rate | [from metrics] |
| 平均勝場步數 | [from metrics] |
| 訓練時間 | [from metrics] |

[一段 80–120 字的解讀：huber 同時影響 mean 與 std；與 clip 的差異（loss 端 vs grad 端）]

## 7. 5 組對比

![baseline loss](results/HW3-3/baseline_random/loss.png)

![baseline dashboard](results/HW3-3/baseline_random/dashboard.gif)

![full loss](results/HW3-3/full_random/loss.png)

![full dashboard](results/HW3-3/full_random/dashboard.gif)

[from Step 9.1 的 metrics 對比表]

**討論**：
- 哪個單獨 trick 對 win_rate 提升最多？[依數據]
- 哪個單獨 trick 對 loss std 削減最多？[依數據]
- full（三 tricks 合用）vs 各自加總：是否大於 marginal sum？
- 訓練時間代價：[依 wall_time 數據]

## 8. 結論

完成 HW3 第三階段（Lightning + Training Tricks）的全部要求：把 HW3-2
Combined 移到 PyTorch Lightning（重用既有 `build_dueling_model`、保持
checkpoint 與 animate 相容）；加入三個正交 tricks 並在 random mode 做
5 組消融。最有價值的觀察：[從 §7 數據點出 1–2 個關鍵 takeaway，例如
「Huber 對 random mode 的 loss std 改善最大」或「三 tricks 合用 win_rate
比 baseline 高 X 個百分點」]。

回顧 HW3 三階段的學習軌跡：HW3-1 用 Replay 對抗 random mode 的非平穩；
HW3-2 用 Double + Dueling 控制 Q 高估與 sample efficiency；HW3-3 用框架
轉換 + tricks 把 random mode 從 85.5% 推到 [final win rate]%。每一階段
解決前一階段留下的痛點。
```

> **重要**：把所有 `[from metrics]` 與 `[依數據]` 都用 Step 9.1 算出來的實際數值取代；commit 時不能留任何 `[...]` 字樣（檢查方法：`grep -n '\[' HW3_3_report.md` 應該只匹配 markdown 的 image / link 語法 `![...]` 與 `[...](...)`）。

- [ ] **Step 9.3: Verify report has no unfilled placeholders**

```bash
grep -nE '\[from |\[依數據' HW3_3_report.md
```

Expected: no matches. If any, fill them in from the metrics or remove the section.

- [ ] **Step 9.4: Commit**

```bash
git add HW3_3_report.md
git commit -m "$(cat <<'EOF'
docs(hw3-3): add HW3-3 Chinese report

8-section report covering Lightning conversion of Combined DQN +
3-trick ablation on random mode (baseline / clip / sched / huber /
full). Mirrors HW3-1/2 report structure (theory → code → result
per trick + final 5-cell comparison table + conclusion connecting
all three HW3 stages).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10 — Write `chatlog3.md`

A faithful conversation log of the HW3-3 work, mirroring `chatlog.md` (HW3-1) and `chatlog2.md` (HW3-2).

**Files:**
- Create: `chatlog3.md`

- [ ] **Step 10.1: Inspect HW3-2's chatlog2.md as a format template**

```bash
head -40 chatlog2.md
```

Confirm the structure: H1 title + meta line + `## Turn N` for each conversation turn + `**User：**` / `**Claude：**` blocks + tool-use summaries (no full tool output).

- [ ] **Step 10.2: Create `chatlog3.md`**

Walk through the conversation history of THIS implementation session and emit the chatlog. Structure:

```markdown
# HW3-3 開發對話紀錄

> 對話日期：2026-05-06
> 模型：Claude Opus 4.7 (1M context)
> 範圍：HW3-3 (Lightning 框架轉換 + 三個 training tricks ablation on random mode)

## Turn 1
**User：** [the original /superpowers:brainstorm message verbatim]

**Claude：** [summary of brainstorming flow: explored project, asked Q1–Q5, walked through design §1, wrote spec, committed]

## Turn 2
**User：** Pytorch lightning

**Claude：** [confirmed framework choice, asked Q2]

## Turn 3
... [continue per turn]

## Turn N (最後一輪)
**User：** [whatever closes out HW3-3]

**Claude：** [final summary]
```

For each turn, include 1–3 sentences of "what the assistant did" rather than verbatim long outputs. Embed key decisions (e.g. "選 Lightning"、"Combined on random mode"、"三 tricks: clip+sched+huber"、"5-cell ablation") inline so the log is readable as a standalone document.

- [ ] **Step 10.3: Commit**

```bash
git add chatlog3.md
git commit -m "$(cat <<'EOF'
docs(hw3-3): add HW3-3 chatlog

Faithful turn-by-turn log of HW3-3 brainstorming + implementation
conversation. Mirrors chatlog.md (HW3-1) and chatlog2.md (HW3-2)
format: per-turn user / Claude blocks with tool-use summarised
rather than pasted in full.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11 — Update `README.md`

Mark HW3-3 as ✅, append a HW3-3 analysis section under existing HW3-1/2 sections, and update the project blurb at the top.

**Files:**
- Modify: `README.md`

- [ ] **Step 11.1: Read current README structure**

```bash
.venv/bin/python -c "
with open('README.md') as f:
    for i, l in enumerate(f, 1):
        if l.startswith('#') or l.startswith('##') or 'HW3-' in l[:60]:
            print(f'{i:4d}: {l.rstrip()}')
"
```

Use the printed line numbers to locate:
- The status line (currently `**階段**：HW3-1 ✅ + HW3-2 ✅（本次更新） + HW3-3 ⏳`)
- The HW3-2 analysis section header (so we know where to append HW3-3)

- [ ] **Step 11.2: Update status line**

Edit `README.md` line 4. Replace:

```markdown
> **階段**：HW3-1 ✅ + **HW3-2 ✅（本次更新）** + HW3-3 ⏳
```

with:

```markdown
> **階段**：HW3-1 ✅ + HW3-2 ✅ + **HW3-3 ✅（本次更新）**
```

- [ ] **Step 11.3: Update intro blurb**

Edit `README.md` lines 11–20 (the bullet list under "本專案在 4×4 Gridworld..."). Append after the existing HW3-2 bullet block:

```markdown
**HW3-3：Lightning-converted DQN + Training Tricks for `random` mode**（本次更新內容）
- **PyTorch → PyTorch Lightning 移植** — Combined backbone 重用既有 `build_dueling_model`，三個 tricks 全 declarative
- **Trick A：Gradient norm clipping**（max_norm=10.0，作用點：grad）
- **Trick B：CosineAnnealingLR**（eta_min=1e-5，作用點：lr）
- **Trick C：Huber loss / SmoothL1**（作用點：loss function）
- **5 組消融**：baseline / clip / sched / huber / all_tricks，全部跑 random mode

```

- [ ] **Step 11.4: Update "特色" / "專案結構" / 後續階段表 if present**

If `README.md` has a "特色" section listing test counts (HW3-2 said "33 個 pytest 測試"), update to reflect the new total (33 + 16 = **49 個 pytest 測試**).

If `README.md` has a "專案結構" or stage table marking `HW3-3：⏳`, change it to `HW3-3：✅` and link to `HW3_3_report.md`.

Use grep to find these:

```bash
grep -nE '33 個 pytest|HW3-3.*⏳|HW3-3：⏳' README.md
```

Apply the corresponding Edit calls inline.

- [ ] **Step 11.5: Append HW3-3 analysis section**

Append at the end of `README.md` (after the HW3-2 analysis section):

```markdown

## HW3-3 分析結果

### 1. 訓練 Loss 曲線

**Baseline（無 tricks）**
![Baseline Loss](results/HW3-3/baseline_random/loss.png)

**+Gradient Clipping**
![Clip Loss](results/HW3-3/clip_random/loss.png)

**+CosineAnnealingLR**
![Sched Loss](results/HW3-3/sched_random/loss.png)

**+Huber Loss**
![Huber Loss](results/HW3-3/huber_random/loss.png)

**+All Tricks**
![Full Loss](results/HW3-3/full_random/loss.png)

### 2. 量化指標（5 組對比）

[paste the markdown table from Task 9 Step 9.1 verbatim]

### 3. 策略動畫

**Baseline**
![Baseline Dashboard](results/HW3-3/baseline_random/dashboard.gif)

**+Gradient Clipping**
![Clip Dashboard](results/HW3-3/clip_random/dashboard.gif)

**+CosineAnnealingLR**
![Sched Dashboard](results/HW3-3/sched_random/dashboard.gif)

**+Huber Loss**
![Huber Dashboard](results/HW3-3/huber_random/dashboard.gif)

**+All Tricks**
![Full Dashboard](results/HW3-3/full_random/dashboard.gif)

> **詳細分析**：見 [HW3_3_report.md](HW3_3_report.md)。
```

- [ ] **Step 11.6: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
docs(readme): add HW3-3 section + mark stage 3 complete

- Status line: HW3-3 from ⏳ to ✅
- New intro bullet block summarising HW3-3 contents
- Bumped pytest count to 49 (was 33)
- New HW3-3 analysis section: 5 loss.png, metrics table, 5 dashboard.gif
- Pointer to HW3_3_report.md for full report

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12 — Final verification

- [ ] **Step 12.1: Full test suite green**

```bash
.venv/bin/python -m pytest -q
```

Expected: 49 PASS, 0 FAIL.

- [ ] **Step 12.2: All 5 result dirs have full artifact set**

```bash
for d in baseline clip sched huber full; do
  echo "=== ${d}_random ==="
  ls -1 "results/HW3-3/${d}_random/" | sort
  ls -1 "results/HW3-3/${d}_random/snapshots/" | head -3
done
```

Expected: each dir has `checkpoint.pth`, `dashboard.gif`, `loss.png`, `losses.npy`, `metrics.json`, `snapshots/`. Each `snapshots/` has `epoch_0000.pth` plus several `epoch_NNNN.pth` files.

- [ ] **Step 12.3: No accidental binary blobs in git history (sanity)**

```bash
git log --stat --since="1 day ago" -- '*.pth' '*.gif' '*.npy' '*.png' | head -60
```

Expected: only files inside `results/HW3-3/` modified in the last day. If you see anything outside that dir, investigate.

- [ ] **Step 12.4: README links resolve**

```bash
grep -oE 'results/HW3-3/[a-z_]+/[a-z._]+' README.md HW3_3_report.md | sort -u | while read p; do
  test -e "$p" && echo "OK   $p" || echo "MISS $p"
done
```

Expected: all `OK`, no `MISS`.

- [ ] **Step 12.5: git status clean**

```bash
git status
```

Expected: `nothing to commit, working tree clean`.

---

## Self-Review Notes

Post-write self-review against spec:

| Spec section | Plan task |
|---|---|
| §0 needs mapping | Tasks 1, 5, 7, 8, 9 (deps + impl + experiments + GIFs + report) |
| §1.1 random mode focus | Task 7 runs all 5 cells in `mode='random'` |
| §1.2 Combined as backbone | Task 3 builds DQNLightningModule on `build_dueling_model` × 2 |
| §1.3 three tricks via flags | Task 5's CLI exposes `--clip --sched --huber` |
| §1.4 declarative | Task 5 uses `gradient_clip_val` flag, `configure_optimizers` dict, loss class swap |
| §1.5 reuse base | Tasks 3, 5 import from `model.py` / `utils.py` / `dqn_naive.py` (`_plot_loss`); only `animate.py` modified |
| §1.6 single seed=42 | Task 7 step 7.1–7.5 uses default seed=42 |
| §2.1 Combined target formula | Task 3 step 3.3 implements Double DQN target |
| §2.2 clip max_norm=10.0 | Task 5 step 5.3 sets `gradient_clip_val=10.0` and Task 5 metric writes 10.0 |
| §2.3 CosineAnnealingLR T_max=epochs eta_min=1e-5 interval='epoch' | Task 3 step 3.3 |
| §2.4 SmoothL1Loss | Task 3 step 3.3 |
| §3 file structure | This plan's "File Structure" table |
| §4.1 dqn_lightning.py code | Tasks 2, 3, 4, 5 build it incrementally — full code is in those tasks |
| §4.2 animate.py modify | Task 6 |
| §4.3 metrics schema | Task 5 step 5.3's `metrics = {...}` literal |
| §4.4 requirements.txt | Task 1 |
| §5.1 5-cell ablation table | Task 7 |
| §5.2 commands | Task 7 step 7.1–7.5 |
| §5.3 sanity checks | Task 7 step 7.6 |
| §5.4 evaluate reuse | Task 5 calls `evaluate(module.online, ...)` |
| §6 report structure | Task 9's report scaffold matches |
| §7 tests | Tasks 2.1, 3.1, 4.1, 5.1 |
| §8 chatlog | Task 10 |
| §9 README updates | Task 11 |
| §10 commits | 8 commits across this plan match (deps + 3 feat impl + 1 feat animate + 1 experiment + 1 GIF + 1 docs report + 1 docs chatlog + 1 docs README ≈ 9; spec said 8 — slight oversplit on tests vs feat is fine and improves bisect-ability) |
| §11 out of scope | Plan does not include any out-of-scope item |

No gaps. No unfilled placeholders in plan steps.
