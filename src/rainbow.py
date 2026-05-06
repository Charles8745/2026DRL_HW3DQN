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


# ============================================================
# Block 6b — train_rainbow + CLI
# ============================================================


def _compute_loss(online: 'DistributionalDuelingMLP',
                  target: 'DistributionalDuelingMLP',
                  batch, weights: torch.Tensor,
                  gamma_n: float, n_atoms: int,
                  v_min: float, v_max: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Distributional cross-entropy loss with PER IS weights and Double DQN
    next-action selection. Returns (weighted_mean_loss, per_sample_ce_detached)."""
    s1, a, R_n, s2, d = batch

    # Compute target first (no-grad). Doing this before the autograd-tracked
    # online forward keeps NoisyLinear's epsilon buffers from being mutated
    # mid-graph (which would trip torch's version checker on backward).
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

    online.reset_noise()
    pred_dist_all = online.forward_dist(s1)                       # (B, n_act, n_atoms)
    a_idx = a.long().view(-1, 1, 1).expand(-1, 1, n_atoms)
    pred_dist = pred_dist_all.gather(1, a_idx).squeeze(1)         # (B, n_atoms)

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
