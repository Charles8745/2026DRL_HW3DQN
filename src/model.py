"""DQN MLP architecture (Listing 3.2 of DRL in Action Ch.3)."""

import torch.nn as nn


def build_model(in_dim: int = 64, hidden1: int = 150,
                hidden2: int = 100, out_dim: int = 4) -> nn.Sequential:
    """Two-hidden-layer MLP with ReLU. Matches Listing 3.2 defaults:
    64 (4-piece × 4×4 grid one-hot, flattened) → 150 → 100 → 4 actions.
    """
    return nn.Sequential(
        nn.Linear(in_dim, hidden1),
        nn.ReLU(),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Linear(hidden2, out_dim),
    )
