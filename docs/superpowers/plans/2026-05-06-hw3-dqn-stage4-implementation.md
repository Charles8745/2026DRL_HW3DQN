# HW3-4 Rainbow DQN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the full Rainbow DQN (Hessel 2018; Double + Dueling + PER + N-step + Distributional/C51 + Noisy Nets) on Gridworld random mode, deliver 2 experiments (`combined_random` HW3-3 baseline re-run + `rainbow_random` main result), produce dashboard GIFs, write HW3-4 report + chatlog + README update.

**Architecture:** Single new file `src/rainbow.py` (~600 LOC) holding 6 logical blocks: NoisyLinear, DistributionalDuelingMLP, SumTree, PrioritizedReplayBuffer, NStepBuffer, projection function + train loop + CLI. Reuses `gridworld_env`, `utils`, `model.py` unchanged. `animate.py` gets a small dispatch addition. Vanilla PyTorch (not Lightning) so PER priority writeback stays clean.

**Tech Stack:** Python 3.12, PyTorch 2.11, NumPy 2.4, pytest 9.0, matplotlib (for loss.png), tqdm (training progress), imageio (GIF in `animate.py`).

**Reference spec:** [docs/superpowers/specs/2026-05-06-hw3-dqn-stage4-design.md](../specs/2026-05-06-hw3-dqn-stage4-design.md)

---

## File Structure

**New files:**
- `src/rainbow.py` — All Rainbow code (NoisyLinear, network, buffers, projection, train_rainbow, main)
- `tests/test_rainbow.py` — 8 test functions covering each component + 1 smoke train
- `HW3_4_report.md` — Stage 4 short report (Chinese, ~8 pages)
- `chatlog4.md` — Stage 4 chat log
- `results/HW3-4/combined_random/` — Re-run of HW3-3 Lightning Combined as HW3-4 baseline (auto-created by training script)
- `results/HW3-4/rainbow_random/` — Rainbow main result (auto-created by training script)

**Modified files:**
- `src/animate.py` — Add `combined_random` / `rainbow_random` to `--exp` choices; add stage + factory dispatch (~15 lines changed in `main()` only)
- `README.md` — Add HW3-4 section; update stage status table; expand 4-stage comparison

**Unchanged:**
- `src/gridboard.py`, `src/gridworld_env.py`, `src/model.py`, `src/utils.py`, `src/dqn_*.py` — all five existing trainers
- All existing test files

---

## Conventions to follow

These come from reading HW3-1/2/3 code:

- **Imports**: stdlib → third-party → `src.*`. Inside `src.*`: `from src.X import ...` form.
- **Type hints**: keyword-only args in `train_*` functions (`def train_X(*, epochs: int = ...)`).
- **Snapshot file names**: `epoch_NNNN.pth` (zero-padded, 4 digits), saved into `<out_dir>/snapshots/`.
- **Final artifacts** (mandatory under `out_dir`): `checkpoint.pth`, `losses.npy`, `loss.png`, `metrics.json`, `snapshots/`.
- **Loss plot helper**: reuse `from src.dqn_naive import _plot_loss` (used by both `dqn_double_dueling.py` and `dqn_lightning.py`).
- **Progress bar**: `tqdm(range(epochs), desc=f'rainbow/{mode}')`.
- **Reproducibility**: `set_seed(seed)` at top of `train_*`.
- **`set_seed`** seeds Python `random`, NumPy, and torch.
- **Test placement**: `tests/test_<module>.py`, function names `test_<behaviour>`. Imports done **inside** test functions where the spec example shows it (matches `test_dqn_lightning.py` style).
- **`animate.py:144-146`** flow: `factory(); model.load_state_dict(sd); model.eval()` — already calls `.eval()`, so NoisyLinear's `self.training=False` branch handles eval determinism automatically.

---

## Task 1: Scaffold `src/rainbow.py` module skeleton + smoke import test

This task creates the empty file with imports and module docstring, plus a one-line test ensuring the file imports cleanly. Subsequent tasks will fill in each block.

**Files:**
- Create: `src/rainbow.py`
- Create: `tests/test_rainbow.py`

- [ ] **Step 1: Write the failing import test**

Create `tests/test_rainbow.py`:

```python
"""HW3-4 Rainbow DQN tests (NoisyLinear, network, buffers, projection, smoke)."""

import pytest
import torch


def test_module_imports():
    """The rainbow module must import without error."""
    import src.rainbow  # noqa: F401
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rainbow.py::test_module_imports -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.rainbow'`.

- [ ] **Step 3: Create the module skeleton**

Create `src/rainbow.py`:

```python
"""Rainbow DQN for Gridworld random mode (HW3-4).

Hessel et al. 2018 — combines six orthogonal improvements over vanilla DQN:
  * Double DQN (Hasselt 2016)
  * Dueling networks (Wang 2016)
  * Prioritized Experience Replay (Schaul 2016)
  * N-step bootstrapping (Sutton & Barto Ch.7)
  * Distributional RL / C51 (Bellemare 2017)
  * Noisy Networks (Fortunato 2018)

Single-file implementation; all six components plus train loop and CLI live
here. Vanilla PyTorch (not Lightning) chosen so PER's per-batch priority
write-back stays out of `training_step`.
"""

import argparse
import math
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.dqn_naive import _plot_loss
from src.gridworld_env import Gridworld
from src.utils import (
    ACTION_SET, encode_state, evaluate, save_metrics, set_seed,
)


STAGE_LABEL = 'HW3-4: Rainbow DQN for random mode'
```

- [ ] **Step 4: Re-run import test, verify it passes**

Run: `pytest tests/test_rainbow.py::test_module_imports -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rainbow.py tests/test_rainbow.py
git commit -m "$(cat <<'EOF'
feat(rainbow): scaffold src/rainbow.py module skeleton

HW3-4 Stage 1: empty module + import smoke test. Subsequent commits will
add NoisyLinear, DistributionalDuelingMLP, SumTree, PrioritizedReplayBuffer,
NStepBuffer, projection, and train_rainbow.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Implement `NoisyLinear` (factorised Gaussian noise)

**Files:**
- Modify: `src/rainbow.py` (append `NoisyLinear` class)
- Modify: `tests/test_rainbow.py` (add 3 tests)

- [ ] **Step 1: Write 3 failing tests for NoisyLinear**

Append to `tests/test_rainbow.py`:

```python
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
```

- [ ] **Step 2: Run tests, verify all 4 fail**

Run: `pytest tests/test_rainbow.py -k noisy_linear -v`
Expected: 4 FAILures with `ImportError: cannot import name 'NoisyLinear'`.

- [ ] **Step 3: Implement NoisyLinear**

Append to `src/rainbow.py`:

```python
# ============================================================
# Block 1 — NoisyLinear (Fortunato 2018, factorised Gaussian)
# ============================================================


class NoisyLinear(nn.Module):
    """Linear layer with learnable Gaussian noise on weights and biases.

    Factorised noise (Atari version):
        eps_W = f(eps_q) outer f(eps_p);  eps_b = f(eps_q)
        f(x) = sign(x) * sqrt(|x|)
    where eps_p ~ N(0, I_in) and eps_q ~ N(0, I_out).

    Forward:
        train mode  ->  y = (mu_W + sigma_W * eps_W) x + (mu_b + sigma_b * eps_b)
        eval mode   ->  y = mu_W x + mu_b                              (deterministic)

    sigma_init=0.5 follows the factorised-noise default in Fortunato 2018.
    """

    def __init__(self, in_features: int, out_features: int,
                 sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        self.register_buffer('weight_epsilon',
                             torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))

    @staticmethod
    def _f(x: torch.Tensor) -> torch.Tensor:
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        eps_p = self._f(torch.randn(self.in_features))
        eps_q = self._f(torch.randn(self.out_features))
        self.weight_epsilon.copy_(eps_q.unsqueeze(1) * eps_p.unsqueeze(0))
        self.bias_epsilon.copy_(eps_q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)
```

- [ ] **Step 4: Run NoisyLinear tests, verify all pass**

Run: `pytest tests/test_rainbow.py -k noisy_linear -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rainbow.py tests/test_rainbow.py
git commit -m "$(cat <<'EOF'
feat(rainbow): implement NoisyLinear with factorised Gaussian noise

Fortunato 2018 factorised version. Respects nn.Module training flag —
train mode samples noise, eval mode uses mu only for deterministic
greedy rollout in evaluate() and animate.py.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Implement `DistributionalDuelingMLP` + `build_rainbow_model`

**Files:**
- Modify: `src/rainbow.py` (append network class + factory)
- Modify: `tests/test_rainbow.py` (add 4 tests)

- [ ] **Step 1: Write 4 failing tests**

Append to `tests/test_rainbow.py`:

```python
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
```

- [ ] **Step 2: Run tests, verify all 4 fail**

Run: `pytest tests/test_rainbow.py -k distributional_model -v`
Expected: 4 FAILures.

- [ ] **Step 3: Implement DistributionalDuelingMLP + factory**

Append to `src/rainbow.py`:

```python
# ============================================================
# Block 2 — DistributionalDuelingMLP (Noisy + Dueling + C51)
# ============================================================


class DistributionalDuelingMLP(nn.Module):
    """Categorical (C51) Dueling network with NoisyLinear in V/A heads.

    Architecture:
        trunk (64 -> 150 -> 100, plain ReLU MLP, NO noise — pure representation)
        V head:  NoisyLinear(100->hidden) -> ReLU -> NoisyLinear(hidden -> 1*n_atoms)
        A head:  NoisyLinear(100->hidden) -> ReLU -> NoisyLinear(hidden -> n_actions*n_atoms)

    Distribution per atom (Bellemare 2017, Wang 2016 dueling aggregation):
        q_logits(s, a, i) = V(s, i) + A(s, a, i) - mean_a A(s, a, i)
        p_i(s, a) = softmax_atoms(q_logits(s, a, .))
        Q(s, a) = sum_i z_i * p_i(s, a)        # used by greedy action selection

    forward(x)       -> (B, n_actions) expected Q (interface-compatible with HW3-1/2/3)
    forward_dist(x)  -> (B, n_actions, n_atoms)
    """

    def __init__(self,
                 n_atoms: int = 51,
                 v_min: float = -10.0,
                 v_max: float = 10.0,
                 in_dim: int = 64,
                 hidden1: int = 150,
                 hidden2: int = 100,
                 head_hidden: int = 128,
                 n_actions: int = 4,
                 sigma_init: float = 0.5):
        super().__init__()
        self.n_atoms = n_atoms
        self.n_actions = n_actions
        self.v_min = v_min
        self.v_max = v_max

        support = torch.linspace(v_min, v_max, n_atoms)
        self.register_buffer('support', support)
        self.register_buffer('delta_z',
                             torch.tensor((v_max - v_min) / (n_atoms - 1)))

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            NoisyLinear(hidden2, head_hidden, sigma_init=sigma_init),
            nn.ReLU(),
            NoisyLinear(head_hidden, n_atoms, sigma_init=sigma_init),
        )
        self.advantage_head = nn.Sequential(
            NoisyLinear(hidden2, head_hidden, sigma_init=sigma_init),
            nn.ReLU(),
            NoisyLinear(head_hidden, n_actions * n_atoms, sigma_init=sigma_init),
        )

    def forward_dist(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        v = self.value_head(h).view(-1, 1, self.n_atoms)             # (B,1,A_atom)
        a = self.advantage_head(h).view(-1, self.n_actions, self.n_atoms)
        a_mean = a.mean(dim=1, keepdim=True)                         # (B,1,n_atoms)
        q_logits = v + (a - a_mean)                                  # (B,n_act,n_atom)
        return F.softmax(q_logits, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dist = self.forward_dist(x)                                  # (B,n_act,n_atom)
        return (dist * self.support).sum(dim=-1)                     # (B,n_act)

    def reset_noise(self) -> None:
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


def build_rainbow_model(n_atoms: int = 51, v_min: float = -10.0,
                         v_max: float = 10.0, in_dim: int = 64,
                         hidden1: int = 150, hidden2: int = 100,
                         head_hidden: int = 128, n_actions: int = 4,
                         sigma_init: float = 0.5) -> DistributionalDuelingMLP:
    """Factory used by `animate.py` for snapshot loading. Must be callable
    without arguments and produce a model whose state_dict matches the one
    saved during training (so default kwargs here MUST match the defaults in
    `train_rainbow`)."""
    return DistributionalDuelingMLP(
        n_atoms=n_atoms, v_min=v_min, v_max=v_max,
        in_dim=in_dim, hidden1=hidden1, hidden2=hidden2,
        head_hidden=head_hidden, n_actions=n_actions, sigma_init=sigma_init,
    )
```

- [ ] **Step 4: Run distributional model tests**

Run: `pytest tests/test_rainbow.py -k distributional_model -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rainbow.py tests/test_rainbow.py
git commit -m "$(cat <<'EOF'
feat(rainbow): implement DistributionalDuelingMLP + build_rainbow_model

C51 over 51 atoms in [-10, +10] combined with Wang 2016 dueling aggregation
per atom; V/A heads use NoisyLinear, trunk stays plain. forward(state)
returns expected Q for interface compatibility with HW3-1/2/3 evaluate().

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Implement `SumTree`

**Files:**
- Modify: `src/rainbow.py` (append SumTree class)
- Modify: `tests/test_rainbow.py` (add 3 tests)

- [ ] **Step 1: Write 3 failing tests**

Append to `tests/test_rainbow.py`:

```python
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
```

- [ ] **Step 2: Run tests, verify all 4 fail**

Run: `pytest tests/test_rainbow.py -k sum_tree -v`
Expected: 4 FAILures.

- [ ] **Step 3: Implement SumTree**

Append to `src/rainbow.py`:

```python
# ============================================================
# Block 3 — SumTree (binary heap-style array, O(log N) ops)
# ============================================================


class SumTree:
    """Binary-tree-as-array data structure for prioritized sampling.

    Layout for capacity N (power of 2 not required; we round up):
        nodes[0]                        — root (= total priority)
        nodes[1..N-2]                   — internal sums
        nodes[N-1 .. 2N-2]              — leaves (priorities)
        data[0 .. N-1]                  — payloads keyed to leaves

    All ops are O(log N).  Used as the inner data structure of
    PrioritizedReplayBuffer; not intended for direct external use.
    """

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.nodes = np.zeros(2 * self.capacity - 1, dtype=np.float64)
        self.data = [None] * self.capacity
        self._write = 0   # circular write pointer
        self._n = 0       # number of leaves currently filled

    @property
    def total(self) -> float:
        return float(self.nodes[0])

    def __len__(self) -> int:
        return self._n

    def add(self, priority: float, data) -> None:
        """Append (priority, data); overwrite oldest if full."""
        leaf_idx = self._write + self.capacity - 1
        self.data[self._write] = data
        self.update(leaf_idx, float(priority))
        self._write = (self._write + 1) % self.capacity
        self._n = min(self._n + 1, self.capacity)

    def update(self, leaf_idx: int, priority: float) -> None:
        """Rewrite a leaf priority; propagate the delta to ancestors."""
        delta = float(priority) - self.nodes[leaf_idx]
        self.nodes[leaf_idx] = float(priority)
        idx = leaf_idx
        while idx > 0:
            idx = (idx - 1) // 2
            self.nodes[idx] += delta

    def sample(self, s: float) -> tuple[int, float, object]:
        """Walk from root to leaf following the prefix-sum target s.
        Returns (leaf_idx, priority, data)."""
        idx = 0
        # Internal nodes occupy [0, capacity - 2]; leaves [capacity-1, ...].
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            if s <= self.nodes[left]:
                idx = left
            else:
                s -= self.nodes[left]
                idx = right
        data_idx = idx - (self.capacity - 1)
        return idx, float(self.nodes[idx]), self.data[data_idx]
```

- [ ] **Step 4: Run sum_tree tests**

Run: `pytest tests/test_rainbow.py -k sum_tree -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rainbow.py tests/test_rainbow.py
git commit -m "$(cat <<'EOF'
feat(rainbow): implement SumTree for O(log N) prioritized sampling

Standard binary-heap-as-array layout used by Schaul 2016 PER. Backs
PrioritizedReplayBuffer in next commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Implement `PrioritizedReplayBuffer` (PER)

**Files:**
- Modify: `src/rainbow.py` (append PER class)
- Modify: `tests/test_rainbow.py` (add 3 tests)

- [ ] **Step 1: Write 3 failing tests**

Append to `tests/test_rainbow.py`:

```python
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
```

- [ ] **Step 2: Run tests, verify all 3 fail**

Run: `pytest tests/test_rainbow.py -k per_buffer -v`
Expected: 3 FAILures.

- [ ] **Step 3: Implement PrioritizedReplayBuffer**

Append to `src/rainbow.py`:

```python
# ============================================================
# Block 4 — PrioritizedReplayBuffer (Schaul 2016)
# ============================================================


class PrioritizedReplayBuffer:
    """Proportional-prioritized replay using SumTree.

    p_i = (|delta_i| + epsilon) ** alpha     # priorities (delta = TD-style error)
    P(i) = p_i / sum_j p_j                   # sampling probability
    w_i = (1/N * 1/P(i)) ** beta             # IS weight, normalised by max

    beta is annealed linearly from beta_start to beta_end as a function of
    `frac` in [0, 1] (caller supplies frac = current_epoch / total_epochs).
    """

    def __init__(self, capacity: int, alpha: float = 0.5,
                 beta_start: float = 0.4, beta_end: float = 1.0,
                 epsilon: float = 1e-6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.epsilon = epsilon
        self._max_priority = 1.0   # initial max so first item is non-zero

    def __len__(self) -> int:
        return len(self.tree)

    def _priority(self, td_error: float) -> float:
        return (abs(td_error) + self.epsilon) ** self.alpha

    def push(self, transition) -> None:
        """Insert a new transition with current max priority (so it is at
        least sampled once before priority is updated by training)."""
        self.tree.add(self._max_priority, transition)

    def sample(self, batch_size: int, frac: float):
        """Stratified sampling: split [0, total] into batch_size equal
        segments; sample one s in each. Returns (transitions, indices,
        IS-weight tensor of shape (batch_size,))."""
        beta = self.beta_start + (self.beta_end - self.beta_start) * min(1.0, frac)
        seg = self.tree.total / batch_size
        transitions = []
        indices = []
        priorities = []
        for i in range(batch_size):
            lo = seg * i
            hi = seg * (i + 1)
            s = random.uniform(lo, hi)
            idx, p, data = self.tree.sample(s)
            transitions.append(data)
            indices.append(idx)
            priorities.append(p)
        priorities = np.array(priorities, dtype=np.float64)
        # Numerical guard: empty / zero-sum trees shouldn't happen by the time
        # caller invokes this (caller checks len(buf) > batch_size first).
        probs = priorities / max(self.tree.total, 1e-12)
        weights = (len(self.tree) * probs) ** (-beta)
        weights = weights / max(weights.max(), 1e-12)
        return transitions, indices, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices, td_errors) -> None:
        """Rewrite leaf priorities; track running max so future pushes inherit it."""
        for idx, err in zip(indices, td_errors):
            p = self._priority(float(err))
            self.tree.update(idx, p)
            if p > self._max_priority:
                self._max_priority = p
```

- [ ] **Step 4: Run per_buffer tests**

Run: `pytest tests/test_rainbow.py -k per_buffer -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rainbow.py tests/test_rainbow.py
git commit -m "$(cat <<'EOF'
feat(rainbow): implement PrioritizedReplayBuffer with stratified sampling

Schaul 2016 proportional PER on the SumTree from previous commit. New
transitions inherit running max priority so they are sampled at least
once. IS weights anneal beta linearly from 0.4 -> 1.0 over training.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Implement `NStepBuffer`

**Files:**
- Modify: `src/rainbow.py` (append NStepBuffer class)
- Modify: `tests/test_rainbow.py` (add 3 tests)

- [ ] **Step 1: Write 3 failing tests**

Append to `tests/test_rainbow.py`:

```python
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
```

- [ ] **Step 2: Run tests, verify all 3 fail**

Run: `pytest tests/test_rainbow.py -k n_step_buffer -v`
Expected: 3 FAILures.

- [ ] **Step 3: Implement NStepBuffer**

Append to `src/rainbow.py`:

```python
# ============================================================
# Block 5 — NStepBuffer (n-step bootstrapping)
# ============================================================


class NStepBuffer:
    """Sliding window of size n that emits n-step transitions.

    Each `append(s, a, r, s_next, done)`:
      - if window has n items: emits an n-step transition for the OLDEST item,
        slides forward; otherwise returns None.
      - if `done` is True: the window's tail is locked to (s_next, done=True),
        and `flush()` will yield truncated n-step transitions for all remaining
        items in the window.
    """

    def __init__(self, n: int = 3, gamma: float = 0.9):
        self.n = n
        self.gamma = gamma
        self.window: deque = deque(maxlen=n)
        self._terminal_tail = None        # (s_next, True) once done observed

    def append(self, s, a, r, s_next, done):
        """Push a 1-step transition. Returns an n-step transition (s, a, R^(n),
        s_{t+n}, d_{t+n}) when a full window is ready, else None.
        On done, immediately tail-locks; n-step flush still happens via .flush()."""
        self.window.append((s, a, float(r), s_next, bool(done)))
        if done:
            self._terminal_tail = (s_next, True)
            return None
        if len(self.window) < self.n:
            return None
        return self._make_n_step(self.window)

    def _make_n_step(self, window):
        """Compute (s_t, a_t, R^(n), s_{t+n}, d_{t+n}) from a sliding window."""
        s_t, a_t, _, _, _ = window[0]
        R = 0.0
        gamma_k = 1.0
        s_next, d = window[-1][3], window[-1][4]
        # If window contains a done before its end, truncate at done.
        for k, (_, _, r_k, s_k_next, d_k) in enumerate(window):
            R += gamma_k * r_k
            gamma_k *= self.gamma
            if d_k:
                s_next, d = s_k_next, True
                break
        return (s_t, a_t, R, s_next, d)

    def flush(self):
        """At episode end, yield truncated n-step transitions for all remaining
        items in the window. Caller invokes after a `done=True` append."""
        while self.window:
            yield self._make_n_step(self.window)
            self.window.popleft()
        self._terminal_tail = None
```

- [ ] **Step 4: Run n_step_buffer tests**

Run: `pytest tests/test_rainbow.py -k n_step_buffer -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rainbow.py tests/test_rainbow.py
git commit -m "$(cat <<'EOF'
feat(rainbow): implement NStepBuffer for n-step bootstrapping

Sliding deque of size n=3 (Rainbow default). Emits a new n-step
transition each step once the window is full; flush() drains truncated
n-step transitions when an episode ends mid-window.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Implement `project_distribution` (categorical Bellman projection)

**Files:**
- Modify: `src/rainbow.py` (append projection function)
- Modify: `tests/test_rainbow.py` (add 2 tests)

- [ ] **Step 1: Write 2 failing tests**

Append to `tests/test_rainbow.py`:

```python
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
```

- [ ] **Step 2: Run tests, verify both fail**

Run: `pytest tests/test_rainbow.py -k project_distribution -v`
Expected: 2 FAILures.

- [ ] **Step 3: Implement project_distribution**

Append to `src/rainbow.py`:

```python
# ============================================================
# Block 6a — Categorical projection (Bellemare 2017 Algorithm 1)
# ============================================================


def project_distribution(next_dist: torch.Tensor,
                          rewards: torch.Tensor,
                          dones: torch.Tensor,
                          gamma_n: float,
                          support: torch.Tensor,
                          v_min: float,
                          v_max: float,
                          n_atoms: int) -> torch.Tensor:
    """Project the next-state distribution to the original support after
    applying the n-step Bellman operator. Vectorised over batch.

    Inputs (all batched):
        next_dist: (B, n_atoms) — categorical distribution at chosen next action
        rewards:   (B,)         — n-step return R^(n)
        dones:     (B,)         — 1.0 if (s_{t+n}, done) else 0.0
        gamma_n:   scalar       — gamma ** n
        support:   (n_atoms,)   — atom values z_j
        v_min, v_max: support endpoints
        n_atoms:   number of atoms

    Returns:
        m: (B, n_atoms) — projected target distribution m_j(s, a)
    """
    B = next_dist.size(0)
    delta_z = (v_max - v_min) / (n_atoms - 1)

    # Tz_j = clip(R + gamma^n * z_j * (1 - done), v_min, v_max)
    rewards = rewards.unsqueeze(1)            # (B, 1)
    dones = dones.unsqueeze(1)                # (B, 1)
    Tz = rewards + (1.0 - dones) * gamma_n * support.unsqueeze(0)  # (B, n_atoms)
    Tz = Tz.clamp(min=v_min, max=v_max)

    b = (Tz - v_min) / delta_z                # continuous index in [0, n_atoms-1]
    l = b.floor().long().clamp(0, n_atoms - 1)
    u = b.ceil().long().clamp(0, n_atoms - 1)

    # Distribute mass with linear interpolation between l and u:
    # m_l += p * (u - b);   m_u += p * (b - l)
    m = torch.zeros(B, n_atoms, dtype=next_dist.dtype, device=next_dist.device)
    # When l == u (Tz lands exactly on a support point), put full mass at l.
    eq_mask = (l == u)
    # Lower part
    m.scatter_add_(1, l, next_dist * (u.float() - b))
    # Upper part
    m.scatter_add_(1, u, next_dist * (b - l.float()))
    # Patch up the equal case: above contributes 0 from both terms; restore.
    if eq_mask.any():
        m.scatter_add_(1, l, next_dist * eq_mask.float())
    return m
```

- [ ] **Step 4: Run project_distribution tests**

Run: `pytest tests/test_rainbow.py -k project_distribution -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rainbow.py tests/test_rainbow.py
git commit -m "$(cat <<'EOF'
feat(rainbow): implement categorical projection (Bellemare 2017 Algorithm 1)

Vectorised projection of next-state distribution onto fixed support after
n-step Bellman operator. Handles done states (point mass at clipped
n-step return) and the edge case where Tz lands exactly on a support atom.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Implement `train_rainbow` + CLI

This is the largest task. It assembles the network + buffers + projection into the full training loop, runs evaluate, saves all artifacts, and exposes a CLI.

**Files:**
- Modify: `src/rainbow.py` (append `train_rainbow` function + `main` CLI)
- Modify: `tests/test_rainbow.py` (add 1 smoke test)

- [ ] **Step 1: Write the failing smoke test**

Append to `tests/test_rainbow.py`:

```python
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
```

- [ ] **Step 2: Run smoke test, verify it fails**

Run: `pytest tests/test_rainbow.py::test_train_rainbow_smoke_writes_all_artifacts -v`
Expected: FAIL (`ImportError: cannot import name 'train_rainbow'`).

- [ ] **Step 3: Implement train_rainbow + main**

Append to `src/rainbow.py`:

```python
# ============================================================
# Block 6b — train_rainbow + CLI
# ============================================================


def _compute_loss(online: DistributionalDuelingMLP,
                  target: DistributionalDuelingMLP,
                  batch, weights: torch.Tensor,
                  gamma_n: float, n_atoms: int,
                  v_min: float, v_max: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Distributional cross-entropy loss with PER IS weights and Double DQN
    next-action selection. Returns (weighted_mean_loss, per_sample_ce_detached)."""
    s1, a, R_n, s2, d = batch
    online.reset_noise()
    pred_dist_all = online.forward_dist(s1)                       # (B, n_act, n_atoms)
    a_idx = a.long().view(-1, 1, 1).expand(-1, 1, n_atoms)
    pred_dist = pred_dist_all.gather(1, a_idx).squeeze(1)         # (B, n_atoms)

    with torch.no_grad():
        online.reset_noise()
        next_q = online(s2)                                       # (B, n_act)
        next_a = next_q.argmax(dim=1)                             # (B,)
        target.reset_noise()
        target_dist_all = target.forward_dist(s2)                 # (B, n_act, n_atoms)
        next_a_idx = next_a.view(-1, 1, 1).expand(-1, 1, n_atoms)
        target_dist = target_dist_all.gather(1, next_a_idx).squeeze(1)
        m = project_distribution(target_dist, R_n, d, gamma_n,
                                  online.support, v_min, v_max, n_atoms)

    log_pred = torch.log(pred_dist.clamp(min=1e-8))
    per_sample_ce = -(m * log_pred).sum(dim=1)                    # (B,)
    weighted_loss = (weights * per_sample_ce).mean()
    return weighted_loss, per_sample_ce.detach()


def train_rainbow(
    *,
    epochs: int = 5000,
    gamma: float = 0.9,
    lr: float = 1e-4,
    mem_size: int = 10000,
    batch_size: int = 200,
    max_moves: int = 50,
    sync_freq: int = 500,
    n_step: int = 3,
    n_atoms: int = 51,
    v_min: float = -10.0,
    v_max: float = 10.0,
    alpha: float = 0.5,
    beta_start: float = 0.4,
    beta_end: float = 1.0,
    sigma_init: float = 0.5,
    mode: str = 'random',
    seed: int = 42,
    snapshot_every: int = 250,
    out_dir: str = 'results/HW3-4/rainbow_random',
    eval_n_games: int = 1000,
) -> dict:
    """Full Rainbow DQN training. Saves the same artifact set as HW3-1/2/3
    variants under `out_dir`. Returns metrics dict.
    """
    set_seed(seed)
    out_path = Path(out_dir)
    snapshots_dir = out_path / 'snapshots'
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    online = build_rainbow_model(n_atoms=n_atoms, v_min=v_min, v_max=v_max,
                                  sigma_init=sigma_init)
    target = build_rainbow_model(n_atoms=n_atoms, v_min=v_min, v_max=v_max,
                                  sigma_init=sigma_init)
    target.load_state_dict(online.state_dict())
    target.eval()

    optimizer = torch.optim.Adam(online.parameters(), lr=lr)
    per = PrioritizedReplayBuffer(capacity=mem_size, alpha=alpha,
                                   beta_start=beta_start, beta_end=beta_end)

    losses: list[float] = []
    global_step = 0
    gamma_n = gamma ** n_step
    t0 = time.time()

    torch.save(online.state_dict(), snapshots_dir / 'epoch_0000.pth')

    for epoch in tqdm(range(epochs), desc=f'rainbow/{mode}'):
        game = Gridworld(size=4, mode=mode)
        n_step_buf = NStepBuffer(n=n_step, gamma=gamma)
        s1 = encode_state(game)
        for mov in range(1, max_moves + 1):
            online.reset_noise()
            with torch.no_grad():
                qval = online(s1)
            action_idx = int(qval.argmax(dim=1).item())
            game.makeMove(ACTION_SET[action_idx])
            s2 = encode_state(game)
            reward = game.reward()
            done = (reward != -1)
            ready = n_step_buf.append(s1, action_idx, reward, s2, done)
            if ready is not None:
                per.push(ready)

            if len(per) > batch_size:
                frac = epoch / max(epochs, 1)
                transitions, idxs, w = per.sample(batch_size, frac)
                s1_b = torch.cat([t[0] for t in transitions])
                a_b = torch.tensor([t[1] for t in transitions])
                r_b = torch.tensor([t[2] for t in transitions], dtype=torch.float32)
                s2_b = torch.cat([t[3] for t in transitions])
                d_b = torch.tensor([float(t[4]) for t in transitions],
                                    dtype=torch.float32)
                loss, per_sample_err = _compute_loss(
                    online, target, (s1_b, a_b, r_b, s2_b, d_b), w,
                    gamma_n=gamma_n, n_atoms=n_atoms,
                    v_min=v_min, v_max=v_max,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                per.update_priorities(idxs, per_sample_err.cpu().numpy())
                losses.append(float(loss.item()))
                global_step += 1
                if global_step % sync_freq == 0:
                    target.load_state_dict(online.state_dict())

            s1 = s2
            if done:
                break

        # Drain remaining n-step transitions at episode end.
        for tail in n_step_buf.flush():
            per.push(tail)

        if (epoch + 1) % snapshot_every == 0:
            torch.save(online.state_dict(),
                       snapshots_dir / f'epoch_{epoch + 1:04d}.pth')

    wall_time = time.time() - t0

    torch.save(online.state_dict(), out_path / 'checkpoint.pth')
    losses_arr = np.array(losses, dtype=np.float32)
    np.save(out_path / 'losses.npy', losses_arr)
    _plot_loss(losses_arr, out_path / 'loss.png',
               title=f'Rainbow DQN ({mode} mode) — training KL loss')

    online.eval()
    eval_result = evaluate(online, mode=mode, n_games=eval_n_games)
    online.train()

    tail = losses_arr[-100:] if len(losses_arr) >= 100 else losses_arr
    metrics = {
        'stage': STAGE_LABEL,
        'experiment': out_path.name,
        'mode': mode,
        'method': 'rainbow',
        'components': {
            'double': True, 'dueling': True, 'per': True,
            'n_step': True, 'distributional': True, 'noisy': True,
        },
        'hyperparams': {
            'epochs': epochs, 'gamma': gamma, 'lr': lr,
            'mem_size': mem_size, 'batch_size': batch_size,
            'max_moves': max_moves, 'sync_freq': sync_freq,
            'seed': seed, 'snapshot_every': snapshot_every,
            'n_step': n_step, 'n_atoms': n_atoms,
            'v_min': v_min, 'v_max': v_max,
            'alpha': alpha, 'beta_start': beta_start, 'beta_end': beta_end,
            'sigma_init': sigma_init,
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
        description='Rainbow DQN training (HW3-4).')
    parser.add_argument('--mode', default='random',
                        choices=['static', 'player', 'random'])
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--mem-size', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--max-moves', type=int, default=50)
    parser.add_argument('--sync-freq', type=int, default=500)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--n-atoms', type=int, default=51)
    parser.add_argument('--v-min', type=float, default=-10.0)
    parser.add_argument('--v-max', type=float, default=10.0)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta-start', type=float, default=0.4)
    parser.add_argument('--beta-end', type=float, default=1.0)
    parser.add_argument('--sigma-init', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--snapshot-every', type=int, default=250)
    parser.add_argument('--out-dir', default=None,
                        help='Default: results/HW3-4/rainbow_<mode>')
    args = parser.parse_args()
    out_dir = args.out_dir or f'results/HW3-4/rainbow_{args.mode}'
    train_rainbow(
        epochs=args.epochs, gamma=args.gamma, lr=args.lr,
        mem_size=args.mem_size, batch_size=args.batch_size,
        max_moves=args.max_moves, sync_freq=args.sync_freq,
        n_step=args.n_step, n_atoms=args.n_atoms,
        v_min=args.v_min, v_max=args.v_max,
        alpha=args.alpha, beta_start=args.beta_start, beta_end=args.beta_end,
        sigma_init=args.sigma_init,
        mode=args.mode, seed=args.seed,
        snapshot_every=args.snapshot_every, out_dir=out_dir,
    )


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run smoke test**

Run: `pytest tests/test_rainbow.py::test_train_rainbow_smoke_writes_all_artifacts -v`
Expected: PASS (run takes ~3–6 seconds).

- [ ] **Step 5: Run the full test suite to confirm no regressions**

Run: `pytest -v`
Expected: previous 49 tests + new ~17 = ~66 tests, ALL PASS.

> If any HW3-1/2/3 test broke, the issue is most likely in `src/rainbow.py` accidentally shadowing a name or rebinding `_plot_loss` import. Fix in place.

- [ ] **Step 6: Commit**

```bash
git add src/rainbow.py tests/test_rainbow.py
git commit -m "$(cat <<'EOF'
feat(rainbow): implement train_rainbow loop + CLI

Full Rainbow training loop assembling: DistributionalDuelingMLP +
NoisyLinear (no epsilon-greedy) + N-step buffer + PrioritizedReplayBuffer
+ Double-DQN action selection + categorical projection + IS-weighted KL
loss + per-sample priority writeback. Smoke test runs end-to-end in
seconds and exercises every artifact path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Wire `animate.py` for HW3-4 experiments

**Files:**
- Modify: `src/animate.py` (only `main()` — `make_dashboard_gif()` untouched)

- [ ] **Step 1: Read current animate.py main() to confirm location**

Run: `grep -n "def main\|--exp\|stage_dir\|factory =" src/animate.py`
Expected: identify `main()` start (line ~193) and the dispatch blocks.

- [ ] **Step 2: Add HW3-4 choices, stage dispatch, and factory dispatch**

Edit `src/animate.py`:

Find:
```python
    parser.add_argument('--exp', required=True, choices=[
        # HW3-1
        'naive_static', 'replay_static', 'replay_random',
        # HW3-2
        'replay_player', 'double_player', 'dueling_player', 'combined_player',
        # HW3-3
        'baseline_random', 'clip_random', 'sched_random',
        'huber_random', 'full_random',
    ])
```

Replace with:
```python
    parser.add_argument('--exp', required=True, choices=[
        # HW3-1
        'naive_static', 'replay_static', 'replay_random',
        # HW3-2
        'replay_player', 'double_player', 'dueling_player', 'combined_player',
        # HW3-3
        'baseline_random', 'clip_random', 'sched_random',
        'huber_random', 'full_random',
        # HW3-4
        'combined_random', 'rainbow_random',
    ])
```

Find:
```python
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
```

Replace with:
```python
    # Stage dispatch.
    hw3_3 = {'baseline_random', 'clip_random', 'sched_random',
             'huber_random', 'full_random'}
    hw3_4 = {'combined_random', 'rainbow_random'}
    if args.exp in hw3_4:
        stage_dir = 'HW3-4'
    elif args.exp in hw3_3:
        stage_dir = 'HW3-3'
    elif args.exp.endswith('_player'):
        stage_dir = 'HW3-2'
    else:
        stage_dir = 'HW3-1'

    # Model factory: Rainbow for HW3-4 rainbow; Dueling for HW3-2 dueling/
    # combined, all HW3-3 cells, and HW3-4 combined_random; plain MLP otherwise.
    if args.exp == 'rainbow_random':
        from src.rainbow import build_rainbow_model
        factory = build_rainbow_model
    else:
        dueling_exps = {'dueling_player', 'combined_player',
                        'combined_random'} | hw3_3
        factory = build_dueling_model if args.exp in dueling_exps else build_model
```

- [ ] **Step 3: Verify no existing animate.py tests break**

Run: `pytest tests/test_animate.py -v`
Expected: ALL PASS (tests don't probe HW3-4 paths so they should be unaffected).

- [ ] **Step 4: Smoke-test the dispatch with a tiny rainbow run**

```bash
python -c "
from pathlib import Path
import os
os.makedirs('results/HW3-4/rainbow_random', exist_ok=True)
from src.rainbow import train_rainbow
train_rainbow(
    epochs=4, mem_size=64, batch_size=8, max_moves=6, sync_freq=2,
    n_atoms=11, snapshot_every=2, mode='static', seed=0,
    out_dir='results/HW3-4/rainbow_random', eval_n_games=2,
)
"
python -m src.animate --exp rainbow_random --fps 5 --max-steps 6
```

Expected: GIF written to `results/HW3-4/rainbow_random/dashboard.gif` without error.

> **Cleanup before commit**: this smoke run was for verification only. Before committing, remove the throwaway artifacts so they don't pollute the repo:
> ```bash
> rm -rf results/HW3-4/rainbow_random
> ```

- [ ] **Step 5: Commit**

```bash
git add src/animate.py
git commit -m "$(cat <<'EOF'
feat(animate): wire HW3-4 combined_random + rainbow_random dispatch

Adds choices, stage_dir branch, and model factory dispatch for HW3-4.
make_dashboard_gif() body untouched — only main() CLI updated.
build_rainbow_model is imported lazily so existing HW3-1/2/3 invocations
do not pull in src.rainbow.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Run experiment 1 — `combined_random` (HW3-4 baseline)

This re-runs the HW3-3 Lightning Combined script but writes into `results/HW3-4/combined_random/`. Code unchanged; just an out-dir override.

**Files:**
- Create: `results/HW3-4/combined_random/` (auto-created by training script)

- [ ] **Step 1: Activate venv and run baseline training**

```bash
source .venv/bin/activate
python -m src.dqn_lightning --mode random --epochs 5000 --seed 42 \
       --snapshot-every 250 --out-dir results/HW3-4/combined_random
```

Expected: ~30 seconds wall time. Final printout includes win_rate (should be ~88% ± 2 pp vs HW3-3's 88.0%).

- [ ] **Step 2: Sanity-check produced artifacts**

```bash
ls results/HW3-4/combined_random/
cat results/HW3-4/combined_random/metrics.json
```

Expected: see `checkpoint.pth, losses.npy, loss.png, metrics.json, snapshots/`.
`metrics.json` should show `"method": "lightning_combined"`, `"experiment": "combined_random"`, `win_rate ∈ [0.85, 0.91]`.

- [ ] **Step 3: Generate dashboard GIF**

```bash
python -m src.animate --exp combined_random
```

Expected: `results/HW3-4/combined_random/dashboard.gif` written.

- [ ] **Step 4: Commit**

```bash
git add results/HW3-4/combined_random/
git commit -m "$(cat <<'EOF'
experiment(hw3-4): combined_random baseline (HW3-3 Lightning Combined re-run)

Same code as HW3-3 baseline_random with out-dir override; serves as the
HW3-4 baseline for the rainbow_random comparison.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Run experiment 2 — `rainbow_random` (main result)

**Files:**
- Create: `results/HW3-4/rainbow_random/` (auto-created)

- [ ] **Step 1: Run Rainbow training**

```bash
source .venv/bin/activate
python -m src.rainbow
```

Expected: ~50–65 seconds wall time. Final printout includes win_rate.

- [ ] **Step 2: Sanity-check artifacts and reality-check results**

```bash
ls results/HW3-4/rainbow_random/
cat results/HW3-4/rainbow_random/metrics.json
```

Expected:
- `method: rainbow`, all 6 components True
- `win_rate ≥ 0.88` (must beat HW3-3 baseline; ideally ≥ 0.90)
- `final_loss_mean_last_100` is in the KL range (0.5–5.0; not directly comparable to MSE)

> If `win_rate < 0.85`, training likely broke. Check `metrics.json` for non-default hyperparams that crept in, and re-read training logs. Common culprits:
> - PER buffer never sampled (check `len(per)` log mid-training)
> - Noisy net's σ collapsed to 0 (broken init)
> - Projection clipping wrong (V_min/V_max mismatch with reward scale)
>
> Do NOT silently retry with different seeds to hunt for a good number — that's p-hacking. If a real bug surfaces, fix it, regenerate both experiments under the same seed.

- [ ] **Step 3: Generate dashboard GIF**

```bash
python -m src.animate --exp rainbow_random
```

Expected: `results/HW3-4/rainbow_random/dashboard.gif` written.

- [ ] **Step 4: Commit**

```bash
git add results/HW3-4/rainbow_random/
git commit -m "$(cat <<'EOF'
experiment(hw3-4): rainbow_random main result (full Rainbow DQN)

5000 epochs, seed=42, mode=random. Rainbow = Double + Dueling + PER +
N-step (n=3) + Distributional (51 atoms) + Noisy. Numbers in metrics.json;
analysis in HW3_4_report.md (next commit).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Write `HW3_4_report.md`

The structure is fixed by the spec §6. Once experiments are run, you read the metrics and write the report against actual numbers (not the placeholder estimates).

**Files:**
- Create: `HW3_4_report.md`

- [ ] **Step 1: Read both experiments' metrics for actual numbers**

```bash
cat results/HW3-4/combined_random/metrics.json
cat results/HW3-4/rainbow_random/metrics.json
```

Note the actual values for: `win_rate`, `avg_steps_per_win`, `final_loss_mean_last_100`, `final_loss_std_last_100`, `training_wall_time_sec`. Use these in §5 of the report (do **not** invent or paste from the spec's predicted ranges).

- [ ] **Step 2: Write the report following spec §6 structure**

Create `HW3_4_report.md` with the full structure from the spec (chapters 1–6). Each subsection should be 1–3 paragraphs with concrete reasoning, not lecture-style. Style references: `HW3_1_report.md`, `HW3_2_report.md`, `HW3_3_report.md`. Length target: 8–10 pages of prose + tables + 2 GIF embeds.

Critical content gates (don't skip these):

- **Chapter 1**: include the table from spec §1.2 mapping痛點 → 元件.
- **Chapter 2**: each new component (PER, N-step, C51, Noisy) gets its own subsection with the math equation from spec §2.3–§2.6.
- **Chapter 3**: emphasise that distributional output of forward(state) is expected Q so all evaluation code reuses unchanged.
- **Chapter 4**: code導讀 quotes 5–10 line snippets from `src/rainbow.py` (do **not** quote the entire file). Include the categorical projection's核心 mass-distribution步驟.
- **Chapter 5**: the actual numbers table — combined vs rainbow, both `win_rate` and `loss` (with KL vs MSE caveat clearly stated).
- **Chapter 6**: discuss which component likely contributed most (推測, 因為沒做 ablation), and the 4×4 Gridworld 太小 caveat.

Image embeds use relative paths:
```markdown
![Rainbow training loss](results/HW3-4/rainbow_random/loss.png)
![Rainbow dashboard](results/HW3-4/rainbow_random/dashboard.gif)
```

- [ ] **Step 3: Spell-check / consistency pass**

Re-read once. Confirm: numbers match `metrics.json`, no "TBD" left, GIF / PNG paths resolve, claims about HW3-1/2/3 (like "HW3-3 baseline 88.0%") match `results/HW3-3/baseline_random/metrics.json`.

- [ ] **Step 4: Commit**

```bash
git add HW3_4_report.md
git commit -m "$(cat <<'EOF'
docs(hw3-4): add HW3-4 report

Structure: 1) why Rainbow, 2) 4 new components math, 3) integration,
4) code walkthrough, 5) results, 6) conclusion. References real
metrics.json numbers from results/HW3-4/{combined,rainbow}_random/.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Update `README.md` (HW3-4 section + 4-stage comparison)

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Read current README structure to find insertion points**

```bash
grep -n "^##\|^| Stage\|^**HW3-3" README.md
```

Note locations of: stage one-liner table, stage status table at the bottom, four-stage GIF comparison table.

- [ ] **Step 2: Update the title line and stage roundup**

Edit `README.md`:

Find the very first heading line and `**階段**` line:
```markdown
> **階段**：HW3-1 ✅ + HW3-2 ✅ + **HW3-3 ✅（本次更新）**
```

Replace with:
```markdown
> **階段**：HW3-1 ✅ + HW3-2 ✅ + HW3-3 ✅ + **HW3-4 ✅（本次更新，加分題）**
```

- [ ] **Step 3: Add HW3-4 line to "簡介" / 三階段一覽**

Find `## 三階段一覽` and rewrite it as `## 四階段一覽`. Append a row to the table for HW3-4 using actual numbers from `results/HW3-4/rainbow_random/metrics.json`. Update the prose paragraph below the table to describe the HW3-1 → HW3-4 trajectory.

- [ ] **Step 4: Add HW3-4 column to the 4-stage GIF comparison**

Find the existing GIF table (stages 1/2/3). Convert it to a 4-column or 2x2 layout that includes HW3-4's `rainbow_random/dashboard.gif`.

Example 2x2 layout:
```markdown
| HW3-1: Replay (random) — 85.5% | HW3-2: Combined (player) — 100% |
|---|---|
| ![](results/HW3-1/replay_random/dashboard.gif) | ![](results/HW3-2/combined_player/dashboard.gif) |

| HW3-3: Lightning Combined (random) — 88.0% | HW3-4: Rainbow (random) — <ACTUAL>% |
|---|---|
| ![](results/HW3-3/baseline_random/dashboard.gif) | ![](results/HW3-4/rainbow_random/dashboard.gif) |
```

(Replace `<ACTUAL>` with the real win_rate × 100 from metrics.json.)

- [ ] **Step 5: Add a "## HW3-4 分析結果" major section**

Insert after the existing "## HW3-3 分析結果" section, mirroring its structure:
- Brief intro (Rainbow = 6 components; ref `HW3_4_report.md` for full analysis)
- 1. Loss curves: 2 PNGs (combined_random, rainbow_random) with KL-vs-MSE note
- 2. Quantitative table (2 rows: combined / rainbow)
- 3. Strategy GIFs: 2 dashboards side-by-side
- Pointer link to `HW3_4_report.md`

Use real numbers from `metrics.json`.

- [ ] **Step 6: Update the bottom "後續階段" status table**

Find `## 後續階段` and add HW3-4 row with `✅ 已完成` and a link to `HW3_4_report.md`.

- [ ] **Step 7: Update CLI examples in "使用方式"**

In the existing CLI examples block, add Rainbow + combined_random commands at the end:
```bash
# HW3-4
python -m src.dqn_lightning --mode random --epochs 5000 --seed 42 \
       --snapshot-every 250 --out-dir results/HW3-4/combined_random
python -m src.rainbow                  # rainbow defaults to results/HW3-4/rainbow_random
python -m src.animate --exp combined_random
python -m src.animate --exp rainbow_random
```

And update the "expected pytest count" near the test invocation: from "49 個測試全綠" to the new total (49 + 17 = ~66; confirm by running `pytest --collect-only -q | tail -3` after Task 8).

- [ ] **Step 8: Sanity-read the whole README**

```bash
less README.md
```

Confirm: no broken links, GIF/PNG paths exist, numbers match `metrics.json`, status table consistent.

- [ ] **Step 9: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
docs(readme): add HW3-4 section + extend to 4-stage comparison

- Stage status table: HW3-4 -> done
- 三階段一覽 -> 四階段一覽 (Rainbow row added)
- GIF comparison: 4 dashboards in a 2x2 layout
- New "HW3-4 分析結果" section mirroring HW3-3 structure
- CLI usage block extended with HW3-4 commands

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Write `chatlog4.md`

**Files:**
- Create: `chatlog4.md`

- [ ] **Step 1: Inspect prior chatlog format**

```bash
head -60 chatlog3.md
```

Note: `## Turn N` headers, `**User：** ...` / `**Claude：** ...` rows, model attribution at top, no large tool-output dumps.

- [ ] **Step 2: Write chatlog4.md following the same format**

Capture this implementation cycle's user prompts and Claude's high-level summaries. Cover:
1. Initial brainstorming (analysis + design choice + 5 hyperparameter Qs)
2. User's "都用你推薦的" approval
3. Spec writing + self-review fixes
4. User's "ok" approval of spec
5. writing-plans skill output
6. Each implementation task (one Turn each) — keep summaries terse, no tool-output dumps.
7. Experiments run + results numbers
8. Report + README + chatlog (this file)

Header template (match `chatlog3.md`):
```markdown
# HW3-4 Chat Log
> 模型：Claude Opus 4.7 (1M context)
> 對話日期：2026-05-06
> 主題：Rainbow DQN for Gridworld random mode（加分題）
```

- [ ] **Step 3: Commit**

```bash
git add chatlog4.md
git commit -m "$(cat <<'EOF'
docs(hw3-4): add HW3-4 chatlog

Captures the brainstorming -> spec -> plan -> implementation -> reporting
cycle following the format of chatlog{1,2,3}.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: Final verification

**Files:**
- (Read-only verification; no commits unless something is broken.)

- [ ] **Step 1: Full pytest suite**

```bash
pytest -v
```

Expected: previous 49 + new ~17 = ~66 tests, ALL PASS.

- [ ] **Step 2: Verify all artifacts exist**

```bash
ls -la results/HW3-4/combined_random/{checkpoint.pth,losses.npy,loss.png,metrics.json,dashboard.gif}
ls -la results/HW3-4/rainbow_random/{checkpoint.pth,losses.npy,loss.png,metrics.json,dashboard.gif}
ls results/HW3-4/combined_random/snapshots/ | head -5
ls results/HW3-4/rainbow_random/snapshots/ | head -5
```

Expected: all files exist; snapshots include `epoch_0000.pth` plus 20 more (250, 500, ..., 5000).

- [ ] **Step 3: Verify git log structure**

```bash
git log --oneline -20
```

Expected (top-down, most recent first):
1. docs(hw3-4): add HW3-4 chatlog
2. docs(readme): add HW3-4 section + extend to 4-stage comparison
3. docs(hw3-4): add HW3-4 report
4. experiment(hw3-4): rainbow_random main result
5. experiment(hw3-4): combined_random baseline
6. feat(animate): wire HW3-4 combined_random + rainbow_random dispatch
7. feat(rainbow): implement train_rainbow loop + CLI
8. feat(rainbow): implement categorical projection
9. feat(rainbow): implement NStepBuffer
10. feat(rainbow): implement PrioritizedReplayBuffer
11. feat(rainbow): implement SumTree
12. feat(rainbow): implement DistributionalDuelingMLP + build_rainbow_model
13. feat(rainbow): implement NoisyLinear with factorised Gaussian noise
14. feat(rainbow): scaffold src/rainbow.py module skeleton
15. docs(spec): add HW3-4 Rainbow DQN design doc
16. docs(plan): add HW3-4 implementation plan

- [ ] **Step 4: Sanity-read README + report once more**

Open `README.md` and `HW3_4_report.md`. Confirm: no orphan placeholders, all GIFs render, 2 metrics tables in README and report agree to two decimal places.

- [ ] **Step 5: Working tree clean**

```bash
git status
```

Expected: `nothing to commit, working tree clean`.

If clean, HW3-4 is done.

---

## Plan Self-Review

**Spec coverage check:**

| Spec section | Plan task |
|---|---|
| §0 needs #1 (analysis) | Task 12 (report ch. 1–3) |
| §0 needs #2 (implement) | Tasks 1–8 |
| §0 needs #3 (animation) | Tasks 9, 10 step 3, 11 step 3 |
| §1 design principles 1–7 | Encoded across all tasks; principle 7 (2 experiments) = tasks 10+11 |
| §2.1 Double DQN | Task 8 `_compute_loss` (online for action, target for value) |
| §2.2 Dueling per atom | Task 3 `forward_dist` aggregation |
| §2.3 PER | Tasks 4 (SumTree) + 5 (PER) |
| §2.4 N-step | Task 6 |
| §2.5 C51 + projection | Tasks 3 (network) + 7 (projection) |
| §2.6 Noisy | Task 2 |
| §3 file structure | All file paths exact in tasks |
| §4 module interface | Tasks 2–8 implement signatures verbatim |
| §4.4 animate.py changes | Task 9 |
| §4.5 metrics schema | Task 8 metrics dict |
| §5 experiments | Tasks 10, 11 |
| §6 report structure | Task 12 step 2 enforces sections |
| §7 8 tests | Tasks 1 (1 import), 2 (4 noisy), 3 (4 net), 4 (4 sum_tree), 5 (3 per), 6 (3 nstep), 7 (2 proj), 8 (1 smoke) = 22 tests; spec said 8 — over-covers, fine |
| §8 chatlog | Task 14 |
| §9 README | Task 13 |
| §10 commit strategy | 14 commits across tasks (spec said 8; finer-grained, OK) |
| §11 out of scope | Honoured throughout (no ablation, no multi-seed, no GPU, no Lightning) |

All spec sections covered.

**Placeholder scan:**

- All code blocks contain real implementations.
- Numbers in tasks 10/11 use ranges (`[0.85, 0.91]`) tied to baselines, not "TBD".
- Task 12 explicitly says "use actual numbers, do not invent or paste predicted ranges" — same for Task 13.
- README CLI block has 1 real path on each line.

No placeholders found.

**Type / signature consistency:**

- `NoisyLinear(in_features, out_features, sigma_init)` consistent across Task 2 + Task 3.
- `DistributionalDuelingMLP(...).forward(x) -> (B, n_actions)` and `forward_dist(x) -> (B, n_actions, n_atoms)` consistent in Tasks 3 + 8 + 9.
- `SumTree.add(priority, data)`, `.update(idx, p)`, `.sample(s)` all consistent across Tasks 4 + 5.
- `PrioritizedReplayBuffer.push(transition)`, `.sample(batch_size, frac)`, `.update_priorities(indices, td_errors)` consistent across Tasks 5 + 8.
- `NStepBuffer.append(s, a, r, s_next, done) -> Optional[tuple]`, `.flush() -> Iterator[tuple]` consistent across Tasks 6 + 8.
- `project_distribution(next_dist, rewards, dones, gamma_n, support, v_min, v_max, n_atoms)` consistent across Tasks 7 + 8.
- `train_rainbow(*, epochs, gamma, lr, mem_size, batch_size, max_moves, sync_freq, n_step, n_atoms, v_min, v_max, alpha, beta_start, beta_end, sigma_init, mode, seed, snapshot_every, out_dir, eval_n_games)` consistent across Task 8 + smoke test.
- `build_rainbow_model(n_atoms, v_min, v_max, in_dim, hidden1, hidden2, head_hidden, n_actions, sigma_init)` defaults consistent between Task 3 (factory) + Task 9 (animate import).

All signatures consistent.
