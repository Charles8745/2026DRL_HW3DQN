# HW3-2: Enhanced DQN Variants for player mode — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a GitHub-ready HW3 Stage 2 — Double DQN, Dueling DQN, and the combined Double+Dueling variant trained on Gridworld `player` mode, with a four-way comparison (incl. baseline DQN+Replay), dashboard GIFs, a Chinese understanding report, an HW3-2 chatlog, and an updated README.

**Architecture:** Each variant lives in its own `src/dqn_<variant>.py` mirroring HW3-1 style (one variant per file, fully self-contained CLI). They reuse `gridworld_env`, `utils`, and the existing `dqn_replay.py` (which serves as the player-mode baseline). The shared `DuelingMLP` ships in `src/model.py` next to the existing `build_model()`. Animation reuses `src/animate.py` extended to dispatch the correct model factory per experiment. Smoke tests validate each new training entry point. After code is built, four real experiments are run and committed; finally `HW3_2_report.md`, `chatlog2.md`, and the README update land.

**Tech Stack:** Python 3.12 · PyTorch (CPU) · NumPy · Matplotlib · imageio · tqdm · pytest · uv (existing — no new deps)

---

## File Structure

**New source files (`src/`):**
- `src/dqn_double.py` — `train_double()` + CLI (Double DQN)
- `src/dqn_dueling.py` — `train_dueling()` + CLI (Dueling DQN)
- `src/dqn_double_dueling.py` — `train_double_dueling()` + CLI (combined)

**Modified source files:**
- `src/model.py` — add `DuelingMLP` class + `build_dueling_model()` factory; existing `build_model()` unchanged
- `src/animate.py` — extend `make_dashboard_gif()` with optional `model_factory` parameter; extend `main()` CLI to accept HW3-2 exp names and pick the right factory + stage directory

**New tests (`tests/`):**
- `tests/test_model_dueling.py` — DuelingMLP forward shape + advantage zero-mean property + factory parameterization
- `tests/test_dqn_double.py` — smoke test (small-budget run; verifies artifacts and Double-DQN-specific keys in metrics.json)
- `tests/test_dqn_dueling.py` — smoke test
- `tests/test_dqn_double_dueling.py` — smoke test

**Modified tests:**
- `tests/test_animate.py` — add a smoke test that runs `make_dashboard_gif` with `build_dueling_model` against a tiny DuelingMLP snapshot

**Generated artifacts (committed after experiment tasks):**
- `results/HW3-2/replay_player/{loss.png, losses.npy, metrics.json, checkpoint.pth, snapshots/, dashboard.gif}`
- `results/HW3-2/double_player/{...}`
- `results/HW3-2/dueling_player/{...}`
- `results/HW3-2/combined_player/{...}`

**Final deliverables:**
- `HW3_2_report.md` (repo root)
- `chatlog2.md` (repo root)
- `README.md` (modify — add HW3-2 sections, fix HW3-3 description in stage table)

---

## Task 1: Add `DuelingMLP` to `src/model.py`

**Files:**
- Modify: `src/model.py`
- Create: `tests/test_model_dueling.py`

- [ ] **Step 1: Write failing tests for DuelingMLP**

Create `/Users/charles88/Downloads/HW3_ DQN/tests/test_model_dueling.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd "/Users/charles88/Downloads/HW3_ DQN" && pytest tests/test_model_dueling.py -v
```
Expected: FAIL with `ImportError: cannot import name 'DuelingMLP' from 'src.model'`.

- [ ] **Step 3: Implement DuelingMLP + factory in `src/model.py`**

Append to `/Users/charles88/Downloads/HW3_ DQN/src/model.py` (after `build_model`):

```python
import torch


class DuelingMLP(torch.nn.Module):
    """Dueling network: shared trunk -> V(s) head + A(s,a) head, combined with
    mean-baseline aggregation (Wang et al. 2016, eq. 9).

        Q(s, a) = V(s) + ( A(s, a) - mean_a A(s, a) )
    """

    def __init__(self, in_dim: int = 64, hidden1: int = 150,
                 hidden2: int = 100, n_actions: int = 4):
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden1), torch.nn.ReLU(),
            torch.nn.Linear(hidden1, hidden2), torch.nn.ReLU(),
        )
        self.value_head = torch.nn.Linear(hidden2, 1)
        self.advantage_head = torch.nn.Linear(hidden2, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        v = self.value_head(h)                              # (B, 1)
        a = self.advantage_head(h)                          # (B, n_actions)
        return v + (a - a.mean(dim=1, keepdim=True))        # (B, n_actions)


def build_dueling_model(in_dim: int = 64, hidden1: int = 150,
                        hidden2: int = 100, n_actions: int = 4) -> DuelingMLP:
    """Factory mirroring `build_model` but returning a DuelingMLP."""
    return DuelingMLP(in_dim, hidden1, hidden2, n_actions)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd "/Users/charles88/Downloads/HW3_ DQN" && pytest tests/test_model_dueling.py -v
```
Expected: all 5 tests PASS.

- [ ] **Step 5: Run the full test suite to verify no regressions**

Run:
```bash
cd "/Users/charles88/Downloads/HW3_ DQN" && pytest -v
```
Expected: HW3-1's 24 tests + 5 new dueling-model tests = 29 tests, all PASS.

- [ ] **Step 6: Commit**

Run:
```bash
cd "/Users/charles88/Downloads/HW3_ DQN"
git add src/model.py tests/test_model_dueling.py
git commit -m "$(cat <<'EOF'
feat(model): add DuelingMLP with mean-baseline aggregation

Introduce DuelingMLP and build_dueling_model() factory next to the
existing build_model(): shared 64→150→100 trunk, V(s) head (→1),
A(s,a) head (→n_actions), combined as Q = V + (A - mean A) per
Wang et al. 2016 eq. 9. Tests cover output shape (batch + single
sample), the advantage zero-mean property, custom-dim factory, and
that the existing Sequential factory is untouched.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Implement Double DQN training (`src/dqn_double.py`)

**Files:**
- Create: `src/dqn_double.py`
- Create: `tests/test_dqn_double.py`

- [ ] **Step 1: Write the smoke test for `train_double`**

Create `/Users/charles88/Downloads/HW3_ DQN/tests/test_dqn_double.py`:

```python
import json
import os
import numpy as np
import torch
from src.dqn_double import train_double


def test_smoke_run(tmp_path):
    """Tiny-budget run on player mode to keep it fast (<5s)."""
    out_dir = tmp_path / "double_player"
    metrics = train_double(
        epochs=20,
        mem_size=50,
        batch_size=10,
        max_moves=20,
        sync_freq=5,
        snapshot_every=10,
        mode='player',
        seed=0,
        out_dir=str(out_dir),
        eval_n_games=10,
    )

    assert metrics['stage'] == 'HW3-2: Enhanced DQN Variants for player mode'
    assert metrics['experiment'] == 'double_player'
    assert metrics['method'] == 'double'
    assert metrics['mode'] == 'player'
    assert 'win_rate' in metrics
    assert 'training_wall_time_sec' in metrics
    assert metrics['hyperparams']['sync_freq'] == 5

    assert (out_dir / 'checkpoint.pth').exists()
    assert (out_dir / 'losses.npy').exists()
    assert (out_dir / 'loss.png').exists()
    assert (out_dir / 'metrics.json').exists()
    assert (out_dir / 'snapshots').is_dir()

    losses = np.load(out_dir / 'losses.npy')
    assert losses.ndim == 1
    assert len(losses) >= 1

    with open(out_dir / 'metrics.json') as f:
        data = json.load(f)
    assert data['hyperparams']['mem_size'] == 50
    assert data['hyperparams']['batch_size'] == 10

    snaps = sorted(os.listdir(out_dir / 'snapshots'))
    assert len(snaps) >= 2
    for s in snaps:
        assert s.startswith('epoch_') and s.endswith('.pth')
        sd = torch.load(out_dir / 'snapshots' / s, weights_only=True)
        assert isinstance(sd, dict)
        # Saved snapshot must be the ONLINE model state dict —
        # so it should be loadable by build_model().
        from src.model import build_model
        m = build_model()
        m.load_state_dict(sd)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd "/Users/charles88/Downloads/HW3_ DQN" && pytest tests/test_dqn_double.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'src.dqn_double'`.

- [ ] **Step 3: Implement `src/dqn_double.py`**

Create `/Users/charles88/Downloads/HW3_ DQN/src/dqn_double.py`:

```python
"""Double DQN training (HW3-2) — Hasselt et al. 2016.

Decouples action selection (online net) from value estimation (target net)
to mitigate the systematic Q-value over-estimation of vanilla DQN.

    Y = r + gamma * (1 - done) * Q_target(s', argmax_a' Q_online(s', a'))

Target network is hard-synced from online every `sync_freq` *training steps*
(global gradient updates, not epochs).
"""

import argparse
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.dqn_naive import _plot_loss
from src.gridworld_env import Gridworld
from src.model import build_model
from src.utils import (
    ACTION_SET, encode_state, epsilon_greedy, evaluate,
    save_metrics, set_seed,
)


STAGE_LABEL = 'HW3-2: Enhanced DQN Variants for player mode'


def train_double(
    *,
    epochs: int = 3000,
    gamma: float = 0.9,
    epsilon: float = 0.3,
    lr: float = 1e-3,
    mem_size: int = 1000,
    batch_size: int = 200,
    max_moves: int = 50,
    sync_freq: int = 500,
    mode: str = 'player',
    seed: int = 42,
    snapshot_every: int = 150,
    out_dir: str = 'results/HW3-2/double_player',
    eval_n_games: int = 1000,
) -> dict:
    """Train Double DQN. Saves checkpoint, snapshots/, losses.npy, loss.png,
    and metrics.json under `out_dir`. Returns metrics dict.

    For test/smoke runs, set `epochs`, `mem_size`, `batch_size`, `sync_freq`,
    and `eval_n_games` low.
    """
    set_seed(seed)
    out_path = Path(out_dir)
    snapshots_dir = out_path / 'snapshots'
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    online_model = build_model()
    target_model = build_model()
    target_model.load_state_dict(online_model.state_dict())
    target_model.eval()

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(online_model.parameters(), lr=lr)

    replay: deque = deque(maxlen=mem_size)
    losses: list[float] = []
    global_step = 0
    t0 = time.time()

    torch.save(online_model.state_dict(), snapshots_dir / 'epoch_0000.pth')

    for i in tqdm(range(epochs), desc=f'double/{mode}'):
        game = Gridworld(size=4, mode=mode)
        state1 = encode_state(game)
        status = 1
        mov = 0
        while status == 1:
            mov += 1
            qval = online_model(state1)
            action_idx = epsilon_greedy(qval, epsilon)
            action = ACTION_SET[action_idx]
            game.makeMove(action)
            state2 = encode_state(game)
            reward = game.reward()
            done = reward > 0
            replay.append((state1, action_idx, reward, state2, done))
            state1 = state2

            if len(replay) > batch_size:
                minibatch = random.sample(list(replay), batch_size)
                state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
                action_batch = torch.tensor([a for (s1, a, r, s2, d) in minibatch])
                reward_batch = torch.tensor(
                    [r for (s1, a, r, s2, d) in minibatch], dtype=torch.float32)
                state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
                done_batch = torch.tensor(
                    [d for (s1, a, r, s2, d) in minibatch], dtype=torch.float32)

                Q1 = online_model(state1_batch)
                with torch.no_grad():
                    online_next = online_model(state2_batch)
                    next_actions = online_next.argmax(dim=1, keepdim=True)
                    target_next = target_model(state2_batch)
                    next_q = target_next.gather(1, next_actions).squeeze(1)
                Y = reward_batch + gamma * (1 - done_batch) * next_q
                X = Q1.gather(
                    dim=1, index=action_batch.long().unsqueeze(dim=1)
                ).squeeze()
                loss = loss_fn(X, Y.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))

                global_step += 1
                if global_step % sync_freq == 0:
                    target_model.load_state_dict(online_model.state_dict())

            if reward != -1 or mov > max_moves:
                status = 0

        if (i + 1) % snapshot_every == 0:
            torch.save(online_model.state_dict(),
                       snapshots_dir / f'epoch_{i + 1:04d}.pth')

    wall_time = time.time() - t0
    torch.save(online_model.state_dict(), out_path / 'checkpoint.pth')
    losses_arr = np.array(losses, dtype=np.float32)
    np.save(out_path / 'losses.npy', losses_arr)
    _plot_loss(losses_arr, out_path / 'loss.png',
               title=f'Double DQN ({mode} mode) — training loss')

    eval_result = evaluate(online_model, mode=mode, n_games=eval_n_games)
    tail = losses_arr[-100:] if len(losses_arr) >= 100 else losses_arr
    metrics = {
        'stage': STAGE_LABEL,
        'experiment': f'double_{mode}',
        'mode': mode,
        'method': 'double',
        'hyperparams': {
            'epochs': epochs, 'gamma': gamma, 'epsilon': epsilon, 'lr': lr,
            'mem_size': mem_size, 'batch_size': batch_size,
            'max_moves': max_moves, 'sync_freq': sync_freq, 'seed': seed,
            'snapshot_every': snapshot_every,
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
    parser = argparse.ArgumentParser(description='Double DQN training (HW3-2).')
    parser.add_argument('--mode', default='player',
                        choices=['static', 'player', 'random'])
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--epsilon', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--mem-size', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--max-moves', type=int, default=50)
    parser.add_argument('--sync-freq', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--snapshot-every', type=int, default=150)
    parser.add_argument('--out-dir', default=None,
                        help='Default: results/HW3-2/double_<mode>')
    args = parser.parse_args()
    out_dir = args.out_dir or f'results/HW3-2/double_{args.mode}'
    train_double(
        epochs=args.epochs, gamma=args.gamma, epsilon=args.epsilon, lr=args.lr,
        mem_size=args.mem_size, batch_size=args.batch_size,
        max_moves=args.max_moves, sync_freq=args.sync_freq,
        mode=args.mode, seed=args.seed,
        snapshot_every=args.snapshot_every, out_dir=out_dir,
    )


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd "/Users/charles88/Downloads/HW3_ DQN" && pytest tests/test_dqn_double.py -v
```
Expected: 1 test PASS in under ~5 seconds.

- [ ] **Step 5: Run the full test suite to verify no regressions**

Run:
```bash
cd "/Users/charles88/Downloads/HW3_ DQN" && pytest -v
```
Expected: 30 tests PASS (29 previous + 1 new).

- [ ] **Step 6: Commit**

Run:
```bash
cd "/Users/charles88/Downloads/HW3_ DQN"
git add src/dqn_double.py tests/test_dqn_double.py
git commit -m "$(cat <<'EOF'
feat: implement Double DQN training (HW3-2)

train_double() decouples action selection (online net) from value
estimation (target net) per Hasselt 2016 to mitigate Q-value
over-estimation:
    Y = r + gamma * (1 - done) * Q_target(s', argmax Q_online(s'))
Target net hard-synced from online every sync_freq global training
steps (default 500). Reuses dqn_replay's buffer/minibatch/CLI
shape; adds --sync-freq flag and writes metrics.json with
method='double'. Smoke test verifies artifacts and that saved
snapshots load into the online build_model().

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Implement Dueling DQN training (`src/dqn_dueling.py`)

**Files:**
- Create: `src/dqn_dueling.py`
- Create: `tests/test_dqn_dueling.py`

- [ ] **Step 1: Write the smoke test**

Create `/Users/charles88/Downloads/HW3_ DQN/tests/test_dqn_dueling.py`:

```python
import json
import os
import numpy as np
import torch
from src.dqn_dueling import train_dueling


def test_smoke_run(tmp_path):
    """Tiny-budget run on player mode to keep it fast (<5s)."""
    out_dir = tmp_path / "dueling_player"
    metrics = train_dueling(
        epochs=20,
        mem_size=50,
        batch_size=10,
        max_moves=20,
        snapshot_every=10,
        mode='player',
        seed=0,
        out_dir=str(out_dir),
        eval_n_games=10,
    )

    assert metrics['stage'] == 'HW3-2: Enhanced DQN Variants for player mode'
    assert metrics['experiment'] == 'dueling_player'
    assert metrics['method'] == 'dueling'
    assert metrics['mode'] == 'player'
    assert 'win_rate' in metrics
    # Dueling DQN does NOT use a target network — sync_freq must not appear
    assert 'sync_freq' not in metrics['hyperparams']

    assert (out_dir / 'checkpoint.pth').exists()
    assert (out_dir / 'losses.npy').exists()
    assert (out_dir / 'loss.png').exists()
    assert (out_dir / 'metrics.json').exists()
    assert (out_dir / 'snapshots').is_dir()

    losses = np.load(out_dir / 'losses.npy')
    assert losses.ndim == 1
    assert len(losses) >= 1

    snaps = sorted(os.listdir(out_dir / 'snapshots'))
    assert len(snaps) >= 2
    # Snapshots must load into a DuelingMLP (NOT the Sequential build_model).
    from src.model import build_dueling_model
    for s in snaps:
        sd = torch.load(out_dir / 'snapshots' / s, weights_only=True)
        m = build_dueling_model()
        m.load_state_dict(sd)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd "/Users/charles88/Downloads/HW3_ DQN" && pytest tests/test_dqn_dueling.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'src.dqn_dueling'`.

- [ ] **Step 3: Implement `src/dqn_dueling.py`**

Create `/Users/charles88/Downloads/HW3_ DQN/src/dqn_dueling.py`:

```python
"""Dueling DQN training (HW3-2) — Wang et al. 2016.

Replaces the Q-network with V(s) + A(s,a) heads sharing a trunk:
    Q(s, a) = V(s) + ( A(s, a) - mean_a' A(s, a') )
Target is the same one-step bootstrap as DQN+Replay (no target network);
the change here is purely architectural.
"""

import argparse
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.dqn_naive import _plot_loss
from src.gridworld_env import Gridworld
from src.model import build_dueling_model
from src.utils import (
    ACTION_SET, encode_state, epsilon_greedy, evaluate,
    save_metrics, set_seed,
)


STAGE_LABEL = 'HW3-2: Enhanced DQN Variants for player mode'


def train_dueling(
    *,
    epochs: int = 3000,
    gamma: float = 0.9,
    epsilon: float = 0.3,
    lr: float = 1e-3,
    mem_size: int = 1000,
    batch_size: int = 200,
    max_moves: int = 50,
    mode: str = 'player',
    seed: int = 42,
    snapshot_every: int = 150,
    out_dir: str = 'results/HW3-2/dueling_player',
    eval_n_games: int = 1000,
) -> dict:
    """Train Dueling DQN. Same training loop as DQN+Replay but with
    `build_dueling_model` swapped in for the network factory.
    """
    set_seed(seed)
    out_path = Path(out_dir)
    snapshots_dir = out_path / 'snapshots'
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    model = build_dueling_model()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    replay: deque = deque(maxlen=mem_size)
    losses: list[float] = []
    t0 = time.time()

    torch.save(model.state_dict(), snapshots_dir / 'epoch_0000.pth')

    for i in tqdm(range(epochs), desc=f'dueling/{mode}'):
        game = Gridworld(size=4, mode=mode)
        state1 = encode_state(game)
        status = 1
        mov = 0
        while status == 1:
            mov += 1
            qval = model(state1)
            action_idx = epsilon_greedy(qval, epsilon)
            action = ACTION_SET[action_idx]
            game.makeMove(action)
            state2 = encode_state(game)
            reward = game.reward()
            done = reward > 0
            replay.append((state1, action_idx, reward, state2, done))
            state1 = state2

            if len(replay) > batch_size:
                minibatch = random.sample(list(replay), batch_size)
                state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
                action_batch = torch.tensor([a for (s1, a, r, s2, d) in minibatch])
                reward_batch = torch.tensor(
                    [r for (s1, a, r, s2, d) in minibatch], dtype=torch.float32)
                state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
                done_batch = torch.tensor(
                    [d for (s1, a, r, s2, d) in minibatch], dtype=torch.float32)

                Q1 = model(state1_batch)
                with torch.no_grad():
                    Q2 = model(state2_batch)
                Y = reward_batch + gamma * (
                    (1 - done_batch) * torch.max(Q2, dim=1)[0])
                X = Q1.gather(
                    dim=1, index=action_batch.long().unsqueeze(dim=1)
                ).squeeze()
                loss = loss_fn(X, Y.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))

            if reward != -1 or mov > max_moves:
                status = 0

        if (i + 1) % snapshot_every == 0:
            torch.save(model.state_dict(),
                       snapshots_dir / f'epoch_{i + 1:04d}.pth')

    wall_time = time.time() - t0
    torch.save(model.state_dict(), out_path / 'checkpoint.pth')
    losses_arr = np.array(losses, dtype=np.float32)
    np.save(out_path / 'losses.npy', losses_arr)
    _plot_loss(losses_arr, out_path / 'loss.png',
               title=f'Dueling DQN ({mode} mode) — training loss')

    eval_result = evaluate(model, mode=mode, n_games=eval_n_games)
    tail = losses_arr[-100:] if len(losses_arr) >= 100 else losses_arr
    metrics = {
        'stage': STAGE_LABEL,
        'experiment': f'dueling_{mode}',
        'mode': mode,
        'method': 'dueling',
        'hyperparams': {
            'epochs': epochs, 'gamma': gamma, 'epsilon': epsilon, 'lr': lr,
            'mem_size': mem_size, 'batch_size': batch_size,
            'max_moves': max_moves, 'seed': seed,
            'snapshot_every': snapshot_every,
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
    parser = argparse.ArgumentParser(description='Dueling DQN training (HW3-2).')
    parser.add_argument('--mode', default='player',
                        choices=['static', 'player', 'random'])
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--epsilon', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--mem-size', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--max-moves', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--snapshot-every', type=int, default=150)
    parser.add_argument('--out-dir', default=None,
                        help='Default: results/HW3-2/dueling_<mode>')
    args = parser.parse_args()
    out_dir = args.out_dir or f'results/HW3-2/dueling_{args.mode}'
    train_dueling(
        epochs=args.epochs, gamma=args.gamma, epsilon=args.epsilon, lr=args.lr,
        mem_size=args.mem_size, batch_size=args.batch_size,
        max_moves=args.max_moves, mode=args.mode, seed=args.seed,
        snapshot_every=args.snapshot_every, out_dir=out_dir,
    )


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd "/Users/charles88/Downloads/HW3_ DQN" && pytest tests/test_dqn_dueling.py -v
```
Expected: 1 test PASS in under ~5 seconds.

- [ ] **Step 5: Run the full test suite**

Run:
```bash
cd "/Users/charles88/Downloads/HW3_ DQN" && pytest -v
```
Expected: 31 tests PASS.

- [ ] **Step 6: Commit**

Run:
```bash
cd "/Users/charles88/Downloads/HW3_ DQN"
git add src/dqn_dueling.py tests/test_dqn_dueling.py
git commit -m "$(cat <<'EOF'
feat: implement Dueling DQN training (HW3-2)

train_dueling() reuses the DQN+Replay loop verbatim but swaps the
network factory to build_dueling_model (V/A heads + mean-baseline
aggregation). No target network — improvement is purely
architectural per Wang et al. 2016. Smoke test verifies snapshots
round-trip through DuelingMLP.load_state_dict() and that
metrics.json carries method='dueling' with no sync_freq key.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Implement Combined Double + Dueling training (`src/dqn_double_dueling.py`)

**Files:**
- Create: `src/dqn_double_dueling.py`
- Create: `tests/test_dqn_double_dueling.py`

- [ ] **Step 1: Write the smoke test**

Create `/Users/charles88/Downloads/HW3_ DQN/tests/test_dqn_double_dueling.py`:

```python
import json
import os
import numpy as np
import torch
from src.dqn_double_dueling import train_double_dueling


def test_smoke_run(tmp_path):
    """Tiny-budget run on player mode to keep it fast (<5s)."""
    out_dir = tmp_path / "combined_player"
    metrics = train_double_dueling(
        epochs=20,
        mem_size=50,
        batch_size=10,
        max_moves=20,
        sync_freq=5,
        snapshot_every=10,
        mode='player',
        seed=0,
        out_dir=str(out_dir),
        eval_n_games=10,
    )

    assert metrics['stage'] == 'HW3-2: Enhanced DQN Variants for player mode'
    assert metrics['experiment'] == 'combined_player'
    assert metrics['method'] == 'double_dueling'
    assert metrics['mode'] == 'player'
    assert metrics['hyperparams']['sync_freq'] == 5

    assert (out_dir / 'checkpoint.pth').exists()
    assert (out_dir / 'losses.npy').exists()
    assert (out_dir / 'loss.png').exists()
    assert (out_dir / 'metrics.json').exists()
    assert (out_dir / 'snapshots').is_dir()

    snaps = sorted(os.listdir(out_dir / 'snapshots'))
    assert len(snaps) >= 2
    # Snapshots must load into a DuelingMLP (combined uses dueling network).
    from src.model import build_dueling_model
    for s in snaps:
        sd = torch.load(out_dir / 'snapshots' / s, weights_only=True)
        m = build_dueling_model()
        m.load_state_dict(sd)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd "/Users/charles88/Downloads/HW3_ DQN" && pytest tests/test_dqn_double_dueling.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'src.dqn_double_dueling'`.

- [ ] **Step 3: Implement `src/dqn_double_dueling.py`**

Create `/Users/charles88/Downloads/HW3_ DQN/src/dqn_double_dueling.py`:

```python
"""Combined Double + Dueling DQN training (HW3-2).

Stacks two orthogonal improvements:
  * Double DQN — decoupled action selection (online) and value estimation (target)
  * Dueling DQN — V(s) + A(s,a) heads with mean-baseline aggregation
"""

import argparse
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.dqn_naive import _plot_loss
from src.gridworld_env import Gridworld
from src.model import build_dueling_model
from src.utils import (
    ACTION_SET, encode_state, epsilon_greedy, evaluate,
    save_metrics, set_seed,
)


STAGE_LABEL = 'HW3-2: Enhanced DQN Variants for player mode'


def train_double_dueling(
    *,
    epochs: int = 3000,
    gamma: float = 0.9,
    epsilon: float = 0.3,
    lr: float = 1e-3,
    mem_size: int = 1000,
    batch_size: int = 200,
    max_moves: int = 50,
    sync_freq: int = 500,
    mode: str = 'player',
    seed: int = 42,
    snapshot_every: int = 150,
    out_dir: str = 'results/HW3-2/combined_player',
    eval_n_games: int = 1000,
) -> dict:
    """Train Double + Dueling DQN. Saves the same artifact set as the other
    HW3-2 variants. Returns metrics dict.
    """
    set_seed(seed)
    out_path = Path(out_dir)
    snapshots_dir = out_path / 'snapshots'
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    online_model = build_dueling_model()
    target_model = build_dueling_model()
    target_model.load_state_dict(online_model.state_dict())
    target_model.eval()

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(online_model.parameters(), lr=lr)

    replay: deque = deque(maxlen=mem_size)
    losses: list[float] = []
    global_step = 0
    t0 = time.time()

    torch.save(online_model.state_dict(), snapshots_dir / 'epoch_0000.pth')

    for i in tqdm(range(epochs), desc=f'combined/{mode}'):
        game = Gridworld(size=4, mode=mode)
        state1 = encode_state(game)
        status = 1
        mov = 0
        while status == 1:
            mov += 1
            qval = online_model(state1)
            action_idx = epsilon_greedy(qval, epsilon)
            action = ACTION_SET[action_idx]
            game.makeMove(action)
            state2 = encode_state(game)
            reward = game.reward()
            done = reward > 0
            replay.append((state1, action_idx, reward, state2, done))
            state1 = state2

            if len(replay) > batch_size:
                minibatch = random.sample(list(replay), batch_size)
                state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
                action_batch = torch.tensor([a for (s1, a, r, s2, d) in minibatch])
                reward_batch = torch.tensor(
                    [r for (s1, a, r, s2, d) in minibatch], dtype=torch.float32)
                state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
                done_batch = torch.tensor(
                    [d for (s1, a, r, s2, d) in minibatch], dtype=torch.float32)

                Q1 = online_model(state1_batch)
                with torch.no_grad():
                    online_next = online_model(state2_batch)
                    next_actions = online_next.argmax(dim=1, keepdim=True)
                    target_next = target_model(state2_batch)
                    next_q = target_next.gather(1, next_actions).squeeze(1)
                Y = reward_batch + gamma * (1 - done_batch) * next_q
                X = Q1.gather(
                    dim=1, index=action_batch.long().unsqueeze(dim=1)
                ).squeeze()
                loss = loss_fn(X, Y.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))

                global_step += 1
                if global_step % sync_freq == 0:
                    target_model.load_state_dict(online_model.state_dict())

            if reward != -1 or mov > max_moves:
                status = 0

        if (i + 1) % snapshot_every == 0:
            torch.save(online_model.state_dict(),
                       snapshots_dir / f'epoch_{i + 1:04d}.pth')

    wall_time = time.time() - t0
    torch.save(online_model.state_dict(), out_path / 'checkpoint.pth')
    losses_arr = np.array(losses, dtype=np.float32)
    np.save(out_path / 'losses.npy', losses_arr)
    _plot_loss(losses_arr, out_path / 'loss.png',
               title=f'Double + Dueling DQN ({mode} mode) — training loss')

    eval_result = evaluate(online_model, mode=mode, n_games=eval_n_games)
    tail = losses_arr[-100:] if len(losses_arr) >= 100 else losses_arr
    metrics = {
        'stage': STAGE_LABEL,
        'experiment': f'combined_{mode}',
        'mode': mode,
        'method': 'double_dueling',
        'hyperparams': {
            'epochs': epochs, 'gamma': gamma, 'epsilon': epsilon, 'lr': lr,
            'mem_size': mem_size, 'batch_size': batch_size,
            'max_moves': max_moves, 'sync_freq': sync_freq, 'seed': seed,
            'snapshot_every': snapshot_every,
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
        description='Double + Dueling DQN training (HW3-2).')
    parser.add_argument('--mode', default='player',
                        choices=['static', 'player', 'random'])
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--epsilon', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--mem-size', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--max-moves', type=int, default=50)
    parser.add_argument('--sync-freq', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--snapshot-every', type=int, default=150)
    parser.add_argument('--out-dir', default=None,
                        help='Default: results/HW3-2/combined_<mode>')
    args = parser.parse_args()
    out_dir = args.out_dir or f'results/HW3-2/combined_{args.mode}'
    train_double_dueling(
        epochs=args.epochs, gamma=args.gamma, epsilon=args.epsilon, lr=args.lr,
        mem_size=args.mem_size, batch_size=args.batch_size,
        max_moves=args.max_moves, sync_freq=args.sync_freq,
        mode=args.mode, seed=args.seed,
        snapshot_every=args.snapshot_every, out_dir=out_dir,
    )


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd "/Users/charles88/Downloads/HW3_ DQN" && pytest tests/test_dqn_double_dueling.py -v
```
Expected: 1 test PASS in under ~5 seconds.

- [ ] **Step 5: Run the full test suite**

Run:
```bash
cd "/Users/charles88/Downloads/HW3_ DQN" && pytest -v
```
Expected: 32 tests PASS.

- [ ] **Step 6: Commit**

Run:
```bash
cd "/Users/charles88/Downloads/HW3_ DQN"
git add src/dqn_double_dueling.py tests/test_dqn_double_dueling.py
git commit -m "$(cat <<'EOF'
feat: implement Double + Dueling combined training (HW3-2)

train_double_dueling() stacks the two orthogonal improvements:
DuelingMLP for the network plus the Double DQN target formula
(online net selects action, target net evaluates value). Hard
target sync every sync_freq global steps. metrics.json carries
method='double_dueling' and experiment='combined_<mode>'.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Extend `src/animate.py` to support HW3-2 experiments

**Note:** The HW3-2 spec §4.5 says "function logic unchanged, only main() changes". That's not quite achievable: `make_dashboard_gif()` currently hard-codes `build_model()` for snapshot loading, but Dueling/Combined snapshots are `DuelingMLP` state dicts. We extend the function with an optional `model_factory` parameter (default = `build_model`, preserving HW3-1 behavior) and dispatch from `main()`. This is the minimal change consistent with the spec's intent.

**Files:**
- Modify: `src/animate.py`
- Modify: `tests/test_animate.py`

- [ ] **Step 1: Read the existing animate test to understand its shape**

Run:
```bash
cat "/Users/charles88/Downloads/HW3_ DQN/tests/test_animate.py"
```
Expected: see one HW3-1 smoke test using `build_model`. We'll add a parallel test for `build_dueling_model`.

- [ ] **Step 2: Add a failing test for the dueling-snapshot path**

Append to `/Users/charles88/Downloads/HW3_ DQN/tests/test_animate.py`:

```python
def test_make_dashboard_gif_dueling(tmp_path):
    """Smoke test for HW3-2: animate against a DuelingMLP snapshot via the
    new model_factory parameter."""
    import json
    import numpy as np
    import torch
    from src.animate import make_dashboard_gif
    from src.model import build_dueling_model

    exp_dir = tmp_path / "dueling_player"
    snaps = exp_dir / "snapshots"
    snaps.mkdir(parents=True)

    # Two tiny snapshots so the GIF has at least 2 segments
    for ep in (0, 10):
        m = build_dueling_model()
        torch.save(m.state_dict(), snaps / f"epoch_{ep:04d}.pth")

    np.save(exp_dir / "losses.npy", np.linspace(1.0, 0.1, 50).astype(np.float32))
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump({
            "stage": "HW3-2: Enhanced DQN Variants for player mode",
            "experiment": "dueling_player",
            "mode": "player",
            "method": "dueling",
            "hyperparams": {"epochs": 10},
        }, f)

    gif_path = make_dashboard_gif(
        exp_dir=str(exp_dir),
        fps=2,
        loss_yscale='linear',
        max_test_steps=4,
        model_factory=build_dueling_model,
    )
    assert (exp_dir / "dashboard.gif").exists()
    assert gif_path.endswith("dashboard.gif")
```

- [ ] **Step 3: Run the new test to verify it fails**

Run:
```bash
cd "/Users/charles88/Downloads/HW3_ DQN" && pytest tests/test_animate.py::test_make_dashboard_gif_dueling -v
```
Expected: FAIL with `TypeError: make_dashboard_gif() got an unexpected keyword argument 'model_factory'`.

- [ ] **Step 4: Add `model_factory` parameter to `make_dashboard_gif`**

Edit `/Users/charles88/Downloads/HW3_ DQN/src/animate.py`. Replace the function signature and the snapshot-loading lines:

Replace:
```python
def make_dashboard_gif(
    *,
    exp_dir: str,
    test_mode: str | None = None,
    fps: int = 5,
    figsize: tuple = (12, 5),
    loss_yscale: str = 'log',
    max_test_steps: int = 15,
    out_filename: str = 'dashboard.gif',
) -> str:
```

With:
```python
def make_dashboard_gif(
    *,
    exp_dir: str,
    test_mode: str | None = None,
    fps: int = 5,
    figsize: tuple = (12, 5),
    loss_yscale: str = 'log',
    max_test_steps: int = 15,
    out_filename: str = 'dashboard.gif',
    model_factory=None,
) -> str:
```

And replace these two lines inside the snapshot loop:
```python
        sd = torch.load(snap_path, weights_only=True)
        model = build_model()
        model.load_state_dict(sd)
```

With:
```python
        sd = torch.load(snap_path, weights_only=True)
        factory = model_factory or build_model
        model = factory()
        model.load_state_dict(sd)
```

- [ ] **Step 5: Update `main()` to dispatch HW3-2 exp names + correct factory + stage dir**

Replace the existing `main()` body in `/Users/charles88/Downloads/HW3_ DQN/src/animate.py` with:

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
    from src.model import build_dueling_model
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

- [ ] **Step 6: Run tests to verify both old and new tests pass**

Run:
```bash
cd "/Users/charles88/Downloads/HW3_ DQN" && pytest tests/test_animate.py -v
```
Expected: 2 tests PASS (existing HW3-1 + new HW3-2 dueling).

- [ ] **Step 7: Run the full test suite**

Run:
```bash
cd "/Users/charles88/Downloads/HW3_ DQN" && pytest -v
```
Expected: 33 tests PASS.

- [ ] **Step 8: Commit**

Run:
```bash
cd "/Users/charles88/Downloads/HW3_ DQN"
git add src/animate.py tests/test_animate.py
git commit -m "$(cat <<'EOF'
feat(animate): support HW3-2 experiments with model factory dispatch

Add optional model_factory parameter to make_dashboard_gif (default
build_model — HW3-1 behavior preserved). Extend main() with the
four HW3-2 exp names, choose results/HW3-{1,2} from the exp suffix,
and select build_dueling_model for dueling/combined snapshots so
DuelingMLP state dicts load correctly. New smoke test renders a
two-snapshot GIF from a DuelingMLP.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Run all 4 player-mode experiments

**Note:** These tasks run real training (not smoke tests). Estimated wall-clock on CPU: ~12s + ~18s + ~15s + ~22s ≈ 70 seconds total. Each experiment is committed separately so the history shows the order results landed.

**Files:**
- Create (training output):
  - `results/HW3-2/replay_player/{loss.png, losses.npy, metrics.json, checkpoint.pth, snapshots/}`
  - `results/HW3-2/double_player/{...}`
  - `results/HW3-2/dueling_player/{...}`
  - `results/HW3-2/combined_player/{...}`

- [ ] **Step 1: Activate venv**

Run:
```bash
cd "/Users/charles88/Downloads/HW3_ DQN" && source .venv/bin/activate
```

- [ ] **Step 2: Run baseline DQN+Replay on player mode**

Run:
```bash
python -m src.dqn_replay --mode player --epochs 3000 --seed 42 \
       --snapshot-every 150 \
       --out-dir results/HW3-2/replay_player
```
Expected: tqdm bar finishes; `results/HW3-2/replay_player/` contains `checkpoint.pth`, `losses.npy`, `loss.png`, `metrics.json`, and `snapshots/` with 21 `epoch_*.pth` files (epoch_0000 + 20 × 150-step snapshots).

Verify metrics.json:
```bash
cat results/HW3-2/replay_player/metrics.json
```
Expected: `"win_rate"` should be roughly **0.85–0.95** (player mode is more constrained than random; failure here means re-check seed and inspect a few snapshots before going further).

**Note on the `stage` field:** `dqn_replay.py` imports `STAGE_LABEL` from `dqn_naive`, so this baseline's `metrics.json` will read `"stage": "HW3-1: Naive DQN for static mode"`. That's expected — we're intentionally not modifying HW3-1 code. The directory (`results/HW3-2/replay_player/`) and `experiment` field (`"replay_player"`) make the stage assignment unambiguous; downstream code (the report and README tables) keys on those, not on `stage`.

- [ ] **Step 3: Commit baseline experiment**

Run:
```bash
cd "/Users/charles88/Downloads/HW3_ DQN"
git add results/HW3-2/replay_player
git commit -m "$(cat <<'EOF'
experiment: run DQN+Replay baseline on player mode (HW3-2)

3000 epochs, seed=42. Snapshots every 150 epochs (21 total incl.
epoch_0000). Serves as the no-improvement baseline for the four-way
HW3-2 comparison.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: Run Double DQN on player mode**

Run:
```bash
python -m src.dqn_double --mode player --epochs 3000 --seed 42
```
Expected: produces `results/HW3-2/double_player/` (default out-dir from CLI). Check `metrics.json["win_rate"]` is **≥ baseline** (target ~92–96%); inspect loss curve in `loss.png` for cleaner convergence than baseline.

- [ ] **Step 5: Commit Double DQN experiment**

Run:
```bash
cd "/Users/charles88/Downloads/HW3_ DQN"
git add results/HW3-2/double_player
git commit -m "$(cat <<'EOF'
experiment: run Double DQN on player mode (HW3-2)

3000 epochs, seed=42, sync_freq=500. Decoupled action selection
should reduce Q over-estimation vs baseline; expect comparable or
higher win rate with steadier loss tail.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 6: Run Dueling DQN on player mode**

Run:
```bash
python -m src.dqn_dueling --mode player --epochs 3000 --seed 42
```
Expected: produces `results/HW3-2/dueling_player/`. Snapshots are DuelingMLP state dicts (will be loaded with `build_dueling_model` in Task 7).

- [ ] **Step 7: Commit Dueling DQN experiment**

Run:
```bash
cd "/Users/charles88/Downloads/HW3_ DQN"
git add results/HW3-2/dueling_player
git commit -m "$(cat <<'EOF'
experiment: run Dueling DQN on player mode (HW3-2)

3000 epochs, seed=42. Pure architectural change (V/A heads with
mean-baseline aggregation, no target net) — improvement comes from
better sample efficiency on shared V(s) statistics.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 8: Run combined Double + Dueling on player mode**

Run:
```bash
python -m src.dqn_double_dueling --mode player --epochs 3000 --seed 42
```
Expected: produces `results/HW3-2/combined_player/`. This run carries both improvements; expect win rate ≥ each individual variant.

- [ ] **Step 9: Commit combined experiment**

Run:
```bash
cd "/Users/charles88/Downloads/HW3_ DQN"
git add results/HW3-2/combined_player
git commit -m "$(cat <<'EOF'
experiment: run Double + Dueling on player mode (HW3-2)

3000 epochs, seed=42, sync_freq=500. Stacks both orthogonal
improvements; expected to top the four-way comparison on win rate
and loss stability.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 10: Sanity-check all four metrics.json side-by-side**

Run:
```bash
cd "/Users/charles88/Downloads/HW3_ DQN"
for exp in replay_player double_player dueling_player combined_player; do
    echo "=== $exp ==="
    python -c "import json; d=json.load(open('results/HW3-2/$exp/metrics.json')); print(f\"win_rate={d['win_rate']:.3f}  final_loss={d['final_loss_mean_last_100']:.4f}±{d['final_loss_std_last_100']:.4f}  wall_time={d['training_wall_time_sec']:.1f}s\")"
done
```
Expected: a 4-line summary. The win-rate ordering should roughly be `replay ≤ double ≈ dueling ≤ combined`; if any variant is dramatically lower (e.g. <80%), inspect the loss curve and the corresponding snapshot before proceeding.

---

## Task 7: Generate dashboard GIFs for all 4 experiments

**Files:**
- Create:
  - `results/HW3-2/replay_player/dashboard.gif`
  - `results/HW3-2/double_player/dashboard.gif`
  - `results/HW3-2/dueling_player/dashboard.gif`
  - `results/HW3-2/combined_player/dashboard.gif`

- [ ] **Step 1: Generate all 4 GIFs**

Run:
```bash
cd "/Users/charles88/Downloads/HW3_ DQN" && source .venv/bin/activate
for exp in replay_player double_player dueling_player combined_player; do
    python -m src.animate --exp $exp
done
```
Expected: four "GIF written: results/HW3-2/<exp>/dashboard.gif" lines. Each GIF roughly 3–5 MB.

- [ ] **Step 2: Verify each GIF exists and has non-trivial size**

Run:
```bash
ls -la results/HW3-2/*/dashboard.gif
```
Expected: four files, each well over 100 KB.

- [ ] **Step 3: Commit all four GIFs in one commit**

Run:
```bash
cd "/Users/charles88/Downloads/HW3_ DQN"
git add results/HW3-2/replay_player/dashboard.gif \
        results/HW3-2/double_player/dashboard.gif \
        results/HW3-2/dueling_player/dashboard.gif \
        results/HW3-2/combined_player/dashboard.gif
git commit -m "$(cat <<'EOF'
experiment: add dashboard GIFs for 4 HW3-2 runs

Per-snapshot greedy rollouts + synchronized loss curve for each
of the four HW3-2 variants on player mode. Dueling/combined GIFs
exercise the new model_factory dispatch in animate.py.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Write `HW3_2_report.md`

**Files:**
- Create: `HW3_2_report.md`

- [ ] **Step 1: Read all four metrics.json files into memory and write the report**

The report uses real numbers from the four `metrics.json` files. Read them first; then create `/Users/charles88/Downloads/HW3_ DQN/HW3_2_report.md` with the structure below. Substitute the bracketed values with the actual numbers from each file.

```bash
cd "/Users/charles88/Downloads/HW3_ DQN"
for exp in replay_player double_player dueling_player combined_player; do
    echo "=== $exp ==="
    cat results/HW3-2/$exp/metrics.json
done
```

Then create `/Users/charles88/Downloads/HW3_ DQN/HW3_2_report.md`:

```markdown
# HW3-2: Enhanced DQN Variants for player mode
## Double DQN 與 Dueling DQN 的改進機制與比較

> 作者：charles88　|　課程：深度強化學習　|　日期：2026-05-01
> Repo：https://github.com/Charles8745/2026DRL_HW3DQN

---

## 1. 作業目標

本階段（HW3-2）目的是在 HW3-1 完成的 DQN+Replay baseline 之上，
實作並比較兩個常見的 DQN 變體 — Double DQN 與 Dueling DQN — 並
觀察兩者的改進是否能正交疊加。實驗環境改為 4×4 Gridworld 的
`player` mode（只有 Player 起點隨機，Goal/Pit/Wall 固定）。

## 2. 環境：player mode 與 static / random 的差異

| Mode | Player 位置 | 其他物件 | 狀態空間複雜度 |
|---|---|---|---|
| `static` | 固定 (0,3) | 固定 | 1 種棋盤 |
| `player` | 隨機 | 固定 | ~13 種起點（去除被佔位） |
| `random` | 隨機 | 全隨機 | 數百種有效棋盤 |

`player` mode 介於兩者之間：strategy 必須對 13 種起點通用，但仍
能依靠固定的 Goal/Pit/Wall 結構推理；非常適合用來「拉開變體之間
的差異而又不被噪音淹沒」。

## 3. 從 baseline 到變體：兩個痛點

### 3.1 Vanilla DQN+Replay 仍然殘留兩個結構性弱點

1. **Q 值系統性高估**：target 計算用同一網路選動作 + 估值，
   $\max$ 會讓估計誤差總是被「正向」放大。
2. **Sample inefficiency on shared structure**：很多 state 下
   action 之間的差異很小，但網路必須對每個 (s, a) pair 各自估值，
   無法善用「身處此 state 的價值」這個共通訊號。

### 3.2 兩個變體各自針對哪一個

- **Double DQN** → 針對痛點 (1)，把「選動作」與「估值」拆給 online 與
  target 兩個網路。
- **Dueling DQN** → 針對痛點 (2)，把網路拆成 V(s) 與 A(s,a) 兩支，
  共用 V 學起來。

## 4. Double DQN（Hasselt 2016）

### 4.1 原理

Vanilla 的 target：

$$Y_{\text{vanilla}} = r + \gamma \max_{a'} Q_\theta(s', a')$$

對任何單筆估計誤差 $\epsilon(a)$，因為 $\max$ 取了最樂觀的那個，
$\mathbb{E}[\max_a (Q^* + \epsilon)] \geq \max_a Q^*$，**期望偏差恆為正**。

Double DQN 把選動作的網路與估值的網路分離：

$$Y_{\text{double}} = r + \gamma\, Q_{\theta^-}\!\left(s',\ \arg\max_{a'} Q_\theta(s', a')\right)$$

online 認為最好的動作不一定是 target 高估的那一個，期望偏差大幅縮小。

### 4.2 程式碼解讀（diff 自 baseline）

核心改動只在三處（節錄自 [`src/dqn_double.py`](src/dqn_double.py)）：

```python
online_model, target_model = build_model(), build_model()
target_model.load_state_dict(online_model.state_dict())
target_model.eval()

# minibatch update 內：
with torch.no_grad():
    next_actions = online_model(state2_batch).argmax(dim=1, keepdim=True)
    next_q = target_model(state2_batch).gather(1, next_actions).squeeze(1)
Y = reward_batch + gamma * (1 - done_batch) * next_q

# 每 sync_freq 個 update 同步一次：
global_step += 1
if global_step % sync_freq == 0:
    target_model.load_state_dict(online_model.state_dict())
```

### 4.3 訓練結果

![double loss](results/HW3-2/double_player/loss.png)

![double dashboard](results/HW3-2/double_player/dashboard.gif)

| 指標 | 數值 |
|---|---|
| Final loss (last 100 mean ± std) | [從 metrics.json] |
| Win rate（1000 場 test） | [從 metrics.json] |
| 平均勝場步數 | [從 metrics.json] |
| 訓練時間 | [從 metrics.json] |

[一段 ~80 字觀察 / 解釋：勝率比 baseline 高多少、loss 是不是更穩、訓練時間代價]

## 5. Dueling DQN（Wang 2016）

### 5.1 原理

把 Q 拆成「state 自己的價值」 + 「action 相對好壞」：

$$Q(s, a) = V(s) + A(s, a)$$

但這個拆解 $V, A$ 不唯一（任何 $V \leftarrow V + c, A \leftarrow A - c$ 都成立），
網路會學不穩。Wang 2016 解法：強制 advantage 平均為 0：

$$Q(s, a) = V(s) + \left(A(s, a) - \tfrac{1}{|\mathcal A|}\sum_{a'} A(s, a')\right)$$

mean baseline 比 max baseline 更穩定（max 的 indicator function 不光滑，
gradient 噪音較大）。

### 5.2 程式碼解讀

DuelingMLP 結構（節錄自 [`src/model.py`](src/model.py)）：

```python
class DuelingMLP(nn.Module):
    def __init__(self, in_dim=64, hidden1=150, hidden2=100, n_actions=4):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden2, 1)
        self.advantage_head = nn.Linear(hidden2, n_actions)

    def forward(self, x):
        h = self.trunk(x)
        v = self.value_head(h)
        a = self.advantage_head(h)
        return v + (a - a.mean(dim=1, keepdim=True))
```

訓練迴圈相對 baseline 只換一行：`model = build_dueling_model()`。
target 計算與 baseline 完全相同（不加 target network，乾淨拆解架構改進的貢獻）。

### 5.3 訓練結果

![dueling loss](results/HW3-2/dueling_player/loss.png)

![dueling dashboard](results/HW3-2/dueling_player/dashboard.gif)

| 指標 | 數值 |
|---|---|
| Final loss (last 100 mean ± std) | [從 metrics.json] |
| Win rate | [從 metrics.json] |
| 平均勝場步數 | [從 metrics.json] |
| 訓練時間 | [從 metrics.json] |

[一段 ~80 字觀察：學習速度有沒有變快、final win rate 提升幅度、是否符合「mean baseline 比 max baseline 更穩」的論述]

## 6. Double + Dueling 合併

### 6.1 為何能疊加

Double 改的是「target 算法」，Dueling 改的是「網路結構」 — 兩者作用點不重疊，
原 Wang 2016 paper 的實驗也驗證合用最佳。

### 6.2 訓練結果

![combined loss](results/HW3-2/combined_player/loss.png)

![combined dashboard](results/HW3-2/combined_player/dashboard.gif)

| 指標 | 數值 |
|---|---|
| Final loss (last 100 mean ± std) | [從 metrics.json] |
| Win rate | [從 metrics.json] |
| 平均勝場步數 | [從 metrics.json] |
| 訓練時間 | [從 metrics.json] |

[一段 ~80 字：是否優於兩個單一變體 / 是否「1+1 = 2」還是只「1+1 ≈ 1.5」]

## 7. 四組對比

![baseline loss](results/HW3-2/replay_player/loss.png)

| 實驗 | Method | Final Loss (mean ± std) | Win Rate | Avg Steps | Wall time |
|---|---|---|---|---|---|
| baseline | replay | [val] ± [val] | [val] | [val] | [val]s |
| Double | double | [val] ± [val] | [val] | [val] | [val]s |
| Dueling | dueling | [val] ± [val] | [val] | [val] | [val]s |
| Combined | double_dueling | [val] ± [val] | [val] | [val] | [val]s |

**討論**：

- **Double vs baseline**：[一兩句，根據實際數字]
- **Dueling vs baseline**：[一兩句]
- **Combined vs 兩個單一變體**：[一兩句，是否驗證了正交性]
- **訓練時間代價**：[Double/Combined 因多一份網路前向會慢；Dueling 多 1 個 head 但 trunk 共用，差異很小]

## 8. 結論

完成了 HW3 第二階段（Enhanced DQN Variants for player mode）的全部要求：
理解 Q 值高估與 sample inefficiency 兩個 baseline 痛點、實作 Double DQN
與 Dueling DQN 各自針對其中一個痛點、合併兩者驗證正交性；並透過 4 組
dashboard 動畫 + loss 曲線直觀比較四種演算法在 player mode 的表現。

下一階段（HW3-3）將把目前的 PyTorch 實作轉換為 Keras 或 PyTorch Lightning，
並引入 gradient clipping、learning rate scheduling 等 training tricks，
針對 random mode 的不穩定性做進一步的工程化處理。
```

- [ ] **Step 2: Substitute the bracketed `[從 metrics.json]` and `[val]` placeholders with actual numbers**

Open each `results/HW3-2/<exp>/metrics.json` and copy values:
- `final_loss_mean_last_100`, `final_loss_std_last_100` → loss row
- `win_rate` → win-rate cell
- `avg_steps_per_win` → avg-steps cell
- `training_wall_time_sec` → wall-time cell

Replace each bracketed placeholder in `HW3_2_report.md` with the matching value (formatted to 3–4 significant digits).

- [ ] **Step 3: Substitute the qualitative discussion paragraphs**

For each "[一段 ~80 字…]" placeholder in Sections 4.3, 5.3, 6.2, and the "討論" bullets in §7, write a short observation grounded in the actual numbers (e.g. "Double 的 win rate 比 baseline 高 X%，loss std 從 0.0X 降到 0.0Y…"). Stay factual; if the comparison doesn't show the expected pattern, say so honestly.

- [ ] **Step 4: Verify the report is complete and renders correctly**

Run:
```bash
cd "/Users/charles88/Downloads/HW3_ DQN"
grep -n '\[從 metrics\.json\]\|\[val\]\|\[一段 ~80\|\[一兩句' HW3_2_report.md
```
Expected: no matches (all placeholders filled). If any remain, fill them.

- [ ] **Step 5: Commit the report**

Run:
```bash
cd "/Users/charles88/Downloads/HW3_ DQN"
git add HW3_2_report.md
git commit -m "$(cat <<'EOF'
docs: add HW3-2 report (4-variant comparison on player mode)

Chinese understanding report covering Double DQN (overestimation
fix), Dueling DQN (V/A decomposition with mean baseline), and the
combined variant. Embeds loss.png and dashboard.gif from each of
the four experiments, plus a four-way comparison table backed by
actual metrics.json values.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Write `chatlog2.md` and update `README.md`

**Files:**
- Create: `chatlog2.md`
- Modify: `README.md`

- [ ] **Step 1: Create `chatlog2.md` from the HW3-2 conversation history**

Create `/Users/charles88/Downloads/HW3_ DQN/chatlog2.md` following HW3-1's `chatlog.md` format. Reconstruct the conversation that started with the HW3-2 brainstorming through report completion, in chronological turns:

```markdown
# HW3-2: Enhanced DQN Variants for player mode — Chat Log with Claude

> 完整保留 HW3-2 階段與 Claude 對話的紀錄，作為作業執行證據。
> 對話日期：2026-05-01
> 模型：Claude Opus 4.7 (1M context)

---

## Turn 1
**User：** [HW3-2 第一句訊息原文，包含作業需求 #1 / #1_1 / #1_2]

**Claude：** [回應摘要：探索專案脈絡 + 提出第一組釐清問題（experiment matrix）]

---

## Turn 2
**User：** [選擇實驗矩陣的回應]

**Claude：** [回應摘要：第二組釐清問題（training budget）]

---

[繼續到所有 turns，包含設計四段呈現、spec 撰寫與 self-review、plan 撰寫、實作執行、實驗執行、報告撰寫]

---

## Final Summary

完整實作流程：HW3-1 baseline → HW3-2 spec/plan → 4 個實驗訓練 →
4 個 dashboard GIF → HW3_2_report.md → chatlog2.md。共 ~13 個 commits。
```

呈現原則 — 與 HW3-1 chatlog 一致：
1. 保留所有 user 訊息原文（中文 / 錯字保留）
2. 保留 Claude 文字回應原文，**不貼大段 tool output**（只保留執行摘要）
3. 重要 tool 動作標註，例如「執行 `python -m src.dqn_double`，跑了 18 秒」
4. 截圖內容（HW3-3 截圖）以 Markdown 引用還原文字

- [ ] **Step 2: Update `README.md`**

Read the existing `/Users/charles88/Downloads/HW3_ DQN/README.md`. Make these targeted edits (do NOT rewrite the file):

(a) **Fix the stage table** at the bottom. Replace:

```markdown
| **HW3-1**：Naive DQN for static mode | Naive DQN + Experience Replay 對比（**本 README**） | ✅ 已完成 |
| HW3-2：DQN variants | Target Network、Double DQN 等 | ⏳ 規劃中 |
| HW3-3：DQN variants | Dueling DQN、Prioritized Replay 等 | ⏳ 規劃中 |
```

With:

```markdown
| **HW3-1**：Naive DQN for static mode | Naive DQN + Experience Replay 對比 | ✅ 已完成 |
| **HW3-2**：Enhanced DQN Variants for player mode | Double DQN + Dueling DQN + 兩者合併（[`HW3_2_report.md`](HW3_2_report.md)） | ✅ 已完成 |
| HW3-3：Framework conversion + training tricks | PyTorch → Keras / PyTorch Lightning + gradient clipping / lr scheduling | ⏳ 規劃中 |
```

(b) **Add a new HW3-2 section** between the existing "後續階段" heading and the existing HW3-1 sub-content (or as a sibling subsection). The new section mirrors HW3-1's analysis subsection style:

```markdown
## HW3-2：Enhanced DQN Variants for player mode

於 4×4 Gridworld `player` mode（Player 位置隨機、Goal/Pit/Wall 固定）
比較四種 DQN 變體：DQN+Replay（baseline）、Double DQN、Dueling DQN、
Double+Dueling 合併。完整報告與量化討論見 [`HW3_2_report.md`](HW3_2_report.md)。

### 訓練 Loss 曲線

**Baseline (DQN+Replay)**
![baseline loss](results/HW3-2/replay_player/loss.png)

**Double DQN**
![double loss](results/HW3-2/double_player/loss.png)

**Dueling DQN**
![dueling loss](results/HW3-2/dueling_player/loss.png)

**Double + Dueling**
![combined loss](results/HW3-2/combined_player/loss.png)

### 量化指標

| 實驗 | Method | Final Loss (mean) | Win Rate | Avg Steps | 訓練時間 |
|---|---|---|---|---|---|
| baseline | replay | [val] | [val] | [val] | [val]s |
| Double | double | [val] | [val] | [val] | [val]s |
| Dueling | dueling | [val] | [val] | [val] | [val]s |
| Combined | double_dueling | [val] | [val] | [val] | [val]s |

### 策略動畫

| Baseline | Double | Dueling | Combined |
|---|---|---|---|
| ![](results/HW3-2/replay_player/dashboard.gif) | ![](results/HW3-2/double_player/dashboard.gif) | ![](results/HW3-2/dueling_player/dashboard.gif) | ![](results/HW3-2/combined_player/dashboard.gif) |
```

Substitute the `[val]` placeholders with the actual numbers from `metrics.json` (same values as in the report).

(c) **Add HW3-2 commands to the "使用方式 → 執行訓練實驗" section**:

After the three existing HW3-1 training commands, add:

```bash
# HW3-2 (player mode)
python -m src.dqn_replay         --mode player --epochs 3000 --seed 42 \
       --out-dir results/HW3-2/replay_player
python -m src.dqn_double         --mode player --epochs 3000 --seed 42
python -m src.dqn_dueling        --mode player --epochs 3000 --seed 42
python -m src.dqn_double_dueling --mode player --epochs 3000 --seed 42
```

And after the three HW3-1 animate commands:

```bash
python -m src.animate --exp replay_player
python -m src.animate --exp double_player
python -m src.animate --exp dueling_player
python -m src.animate --exp combined_player
```

(d) **Update test count** in the "執行測試" subsection: change "預期 24 個測試全綠" → "預期 33 個測試全綠（HW3-1 24 + HW3-2 9）".

- [ ] **Step 3: Verify README placeholders are filled**

Run:
```bash
cd "/Users/charles88/Downloads/HW3_ DQN"
grep -n '\[val\]' README.md
```
Expected: no matches.

- [ ] **Step 4: Run the full test suite one final time**

Run:
```bash
cd "/Users/charles88/Downloads/HW3_ DQN" && pytest -v
```
Expected: 33 tests PASS.

- [ ] **Step 5: Commit chatlog and README update**

Run:
```bash
cd "/Users/charles88/Downloads/HW3_ DQN"
git add chatlog2.md README.md
git commit -m "$(cat <<'EOF'
docs: add HW3-2 chatlog + update README

Add chatlog2.md preserving the HW3-2 conversation (brainstorming →
spec → plan → implementation → experiments → report). Update
README.md: add HW3-2 section with loss curves, dashboard GIFs, and
the four-way comparison table; fix HW3-3 description in the stage
table (framework conversion + training tricks, not Prioritized
Replay); extend Quick Start commands; bump expected test count
from 24 to 33.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 6: Final review — verify HW3-2 deliverables are all present**

Run:
```bash
cd "/Users/charles88/Downloads/HW3_ DQN"
echo "=== Source ===" && ls src/dqn_double.py src/dqn_dueling.py src/dqn_double_dueling.py
echo "=== Tests ===" && ls tests/test_model_dueling.py tests/test_dqn_double.py tests/test_dqn_dueling.py tests/test_dqn_double_dueling.py
echo "=== Results ===" && ls results/HW3-2/*/{loss.png,dashboard.gif,metrics.json,checkpoint.pth}
echo "=== Docs ===" && ls HW3_2_report.md chatlog2.md docs/superpowers/specs/2026-05-01-hw3-dqn-stage2-design.md docs/superpowers/plans/2026-05-01-hw3-dqn-stage2-implementation.md
echo "=== Git log ===" && git log --oneline -15
```
Expected: every path exists; git log shows the new HW3-2 commits in roughly the order of Tasks 1–9.

---

## Done

When this task list is fully checked off, the HW3-2 deliverables are complete:

- 3 new source files (`dqn_double`, `dqn_dueling`, `dqn_double_dueling`) with smoke tests
- `DuelingMLP` and `build_dueling_model` added to `src/model.py`
- `src/animate.py` extended with `model_factory` dispatch
- 4 trained experiments + GIFs in `results/HW3-2/`
- `HW3_2_report.md` Chinese understanding report
- `chatlog2.md` conversation log
- `README.md` updated (HW3-2 section + corrected HW3-3 description)
- All 33 pytest tests passing
