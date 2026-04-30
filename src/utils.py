"""Shared utilities for HW3-1 DQN training and evaluation."""

import json
import random
from typing import Any

import numpy as np
import torch

ACTION_SET = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}


def set_seed(seed: int) -> None:
    """Seed Python `random`, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def encode_state(game: Any) -> torch.Tensor:
    """Encode Gridworld state to (1, 64) float tensor with small uniform noise.
    Mirrors the book: render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.
    """
    arr = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
    return torch.from_numpy(arr).float()


def epsilon_greedy(qval: torch.Tensor, epsilon: float, n_actions: int = 4) -> int:
    """Return action index. With prob `epsilon`, sample uniformly; else argmax."""
    if random.random() < epsilon:
        return int(np.random.randint(0, n_actions))
    return int(torch.argmax(qval).item())


def running_mean(x: np.ndarray, N: int = 50) -> np.ndarray:
    """Simple moving average of length N. Returns array of length len(x) - N."""
    if len(x) <= N:
        return np.array([])
    c = x.shape[0] - N
    y = np.zeros(c)
    conv = np.ones(N)
    for i in range(c):
        y[i] = (x[i:i + N] @ conv) / N
    return y


def save_metrics(path: str, **kwargs: Any) -> None:
    """Dump kwargs as a JSON object to `path` (UTF-8, indent=2)."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(kwargs, f, indent=2, ensure_ascii=False)
