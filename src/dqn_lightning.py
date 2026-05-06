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
