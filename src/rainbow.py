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
        On done, tail-locks so flush() drains remaining items; if the window
        is already full at done time, the n-step transition is emitted first."""
        self.window.append((s, a, float(r), s_next, bool(done)))
        if done:
            self._terminal_tail = (s_next, True)
            if len(self.window) == self.n:
                return self._make_n_step(self.window)
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
