"""Lightning-wrapped Combined DQN with optional training tricks (HW3-3)."""

import argparse
import random
import time
from collections import deque
from pathlib import Path

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
        self.loss_fn = nn.SmoothL1Loss(beta=1.0) if huber else nn.MSELoss()
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


class SnapshotCallback(Callback):
    """Save online model state_dict every N games as ``epoch_NNNN.pth``.

    Tracks game count internally rather than relying on
    ``trainer.current_epoch`` to avoid Lightning version off-by-one nuances.
    Naming matches HW3-1/2 (``epoch_<NNNN>.pth``) so ``animate.py`` works
    unchanged.
    """

    def __init__(self, snapshots_dir: Path, every: int):
        super().__init__()
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
