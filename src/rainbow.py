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
