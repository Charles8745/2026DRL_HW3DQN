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
