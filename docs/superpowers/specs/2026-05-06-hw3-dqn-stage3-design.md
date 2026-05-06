# HW3-3: Lightning-converted DQN with Training Tricks for random mode — 設計文件

> **Stage label**：HW3-3: Lightning-converted DQN with Training Tricks for random mode
> **作業來源**：深度強化學習課程 HW3 — DQN and its variants（Stage 3）
> **設計日期**：2026-05-06
> **基底教材**：HW3-2 Combined（Double + Dueling）DQN；PyTorch Lightning 官方文件；Mnih 2015（Huber loss in DQN）；Loshchilov & Hutter 2017（CosineAnnealingLR / SGDR）
> **前置階段**：[HW3-1 spec](2026-04-30-hw3-dqn-stage1-design.md)、[HW3-2 spec](2026-05-01-hw3-dqn-stage2-design.md)

---

## 0. 作業需求對應

| 需求 | 對應交付物 |
|---|---|
| #1 Convert DQN from PyTorch to Keras / PyTorch Lightning | 選 **PyTorch Lightning**：`src/dqn_lightning.py` 內含 `DQNLightningModule` + `RolloutDataset` + `train_lightning(...)` |
| #1_2 PyTorch Lightning 路線 | 重用 `build_dueling_model` (HW3-2)；`Trainer(automatic_optimization=True)` 自動處理 backward / clip / step；checkpoint 兼容 `animate.py` |
| #2 Training tricks（gradient clipping / lr scheduling / etc.） | 三個 tricks：(a) gradient norm clipping max_norm=10.0、(b) CosineAnnealingLR (T_max=epochs, eta_min=1e-5)、(c) Huber loss (`SmoothL1Loss`)；2³ ablation 取 5 cells |
| 主題：random mode | 全部 5 組實驗都跑 `mode='random'`，與 HW3-1 `replay_random` 對照（baseline 85.5% win_rate, loss 0.0613±0.0573） |
| 沿用 HW3-1/2 規格與品質 | 章節結構、metrics schema、commit 風格、測試覆蓋、dashboard GIF 全部對齊 |

---

## 1. 設計原則

1. **random mode 為主軸**：HW3-3 主題「random mode + tricks」呼應 HW3-1 發現的痛點（random mode loss std 與 mean 同量級、win_rate 只有 85.5%）。Player / static mode 不在範圍內。
2. **以 HW3-2 Combined 為起點**：PyTorch baseline 用 HW3-2 最強的 Combined（Double + Dueling）。理由：HW3-2 已驗證 Combined 在 player mode 是 4 變體中 loss 最穩的；移到 random mode 上加 tricks 是自然的進階。
3. **三個 tricks，獨立開關**：gradient clipping / lr scheduling / Huber loss 三者作用點完全正交（gradient / lr / loss function），同一個訓練腳本以 CLI flag 開關，5 組實驗共用 `dqn_lightning.py`。
4. **Lightning 全 declarative**：`automatic_optimization=True`，三個 tricks 各自只用一行 Lightning API：
   - clip → `Trainer(gradient_clip_val=10.0, gradient_clip_algorithm='norm')`
   - sched → `configure_optimizers` 回傳 `{'optimizer': ..., 'lr_scheduler': {'scheduler': CosineAnnealingLR(...), 'interval': 'epoch'}}`
   - huber → `self.loss_fn = nn.SmoothL1Loss() if huber else nn.MSELoss()`
5. **與 HW3-1/2 共用基底**：`gridworld_env`、`utils`、`model.py`（`build_dueling_model`）完全不改邏輯（`animate.py` 只加 `--exp` choices 與 HW3-3 路徑分支）。
6. **單一 seed=42**：對齊 HW3-1/2，不做多 seed 平均；如果結果偏離預期再考慮第二 seed。

---

## 2. 演算法規格

所有 5 組共用：`gamma=0.9, lr=1e-3, mem_size=1000, batch_size=200, max_moves=50, epsilon=0.3, sync_freq=500, seed=42, mode='random', epochs=5000, snapshot_every=250`。

> **超參數對齊**：epochs=5000 與 snapshot_every=250 沿用 HW3-1 `replay_random` 的設定（random mode 所需訓練量比 player 大）。其餘對齊 HW3-2 Combined。

### 2.1 共用骨幹 — Combined（Double + Dueling）on random mode

- 模型：`build_dueling_model()` × 2（online + target）
- Target 計算（與 HW3-2 `dqn_double_dueling` 相同）：

  $$Y = r + \gamma (1-\text{done}) \cdot Q^{\text{dueling}}_{\theta^-}\!\left(s',\ \arg\max_{a'} Q^{\text{dueling}}_{\theta}(s', a')\right)$$
- Target sync：每 500 個 training steps（minibatch updates）硬複製。

### 2.2 Trick A — Gradient Norm Clipping

- **問題**：DQN target $Y$ 偶爾出現大幅變動（target net sync 後跳變、reward 突變），對應的 gradient 也會偶爾變大，破壞 Adam 的 momentum。
- **公式**：每次 `loss.backward()` 後，

  $$g \leftarrow g \cdot \min\!\left(1,\ \frac{c}{\|g\|_2}\right),\quad c = 10.0$$
- **Lightning 實作**（automatic_optimization=True 時直接生效）：

  ```python
  Trainer(gradient_clip_val=10.0, gradient_clip_algorithm='norm', ...)
  ```
- **預期效果**：loss 曲線的尖刺被切平，`final_loss_std_last_100` 降低。對 random mode 尤其重要（loss std 是 mean 的同量級）。
- **參數選擇**：max_norm=10.0 是 DQN 文獻常用值（保守、不破壞學習動能）。1.0 太緊、可能延遲收斂；100.0 太鬆、與不 clip 接近。

### 2.3 Trick B — CosineAnnealingLR

- **問題**：固定 `lr=1e-3` 在訓練後期太大，loss 已收斂但仍被大梯度推來推去 → fine convergence 不到。
- **公式**：

  $$\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_{\max} - \eta_{\min})\!\left(1 + \cos\!\tfrac{t \pi}{T}\right),\quad t = \text{epoch}$$
  with $\eta_{\max} = 1\!\times\!10^{-3}$, $\eta_{\min} = 1\!\times\!10^{-5}$, $T = $ `epochs` (5000).
- **Lightning 實作**：

  ```python
  def configure_optimizers(self):
      opt = torch.optim.Adam(self.online.parameters(), lr=self.hparams.lr)
      if not self.hparams.sched:
          return opt
      sched = torch.optim.lr_scheduler.CosineAnnealingLR(
          opt, T_max=self.hparams.epochs, eta_min=1e-5)
      return {'optimizer': opt, 'lr_scheduler': {
          'scheduler': sched, 'interval': 'epoch'}}
  ```
- **預期效果**：訓練後期 lr 接近 1e-5，loss 微幅再降；`final_loss_mean_last_100` 降低。
- **interval='epoch' vs 'step'**：選 `epoch`（每局 game 結束後 step 一次 scheduler），與 `Trainer(max_epochs=5000)` 對齊；`step` 會讓 scheduler step 次數等於 minibatch updates 數量（依 buffer fill 不固定），不易 reasoning。

### 2.4 Trick C — Huber Loss (Smooth L1)

- **問題**：random mode 的 reward 分布有重尾（HW3-1 random_replay 的 loss std 0.0573 與 mean 0.0613 同量級，明顯有 outlier）。MSE 對 outlier 過度敏感（誤差平方放大），讓單一壞 batch 把網路「踢偏」。
- **公式**：

  $$L_\delta(x) = \begin{cases} \tfrac{1}{2}x^2 & |x| \le \delta \\ \delta(|x| - \tfrac{1}{2}\delta) & |x| > \delta \end{cases},\quad \delta = 1.0$$

  小誤差時與 MSE 相同（保持 smooth gradient），大誤差時退為 L1（gradient 上限固定為 $\delta$）。
- **Lightning 實作**：

  ```python
  self.loss_fn = nn.SmoothL1Loss(beta=1.0) if huber else nn.MSELoss()
  ```
- **預期效果**：loss 曲線整體更平滑，最大尖刺值下降；與 gradient clipping 互補（一個從 loss 端、一個從 gradient 端控 outlier）。
- **DQN 文獻地位**：原 DQN paper（Mnih 2015）描述為「clip TD error to [-1, 1]」，等價於 Huber with $\delta=1$；後續 Rainbow / Double DQN paper 多用此設計。

---

## 3. 檔案結構

```
HW3_DQN/
├── HW3_3_report.md                      # NEW — HW3-3 中文報告
├── chatlog3.md                          # NEW — HW3-3 對話紀錄
├── README.md                            # MODIFY — 加 HW3-3 段落、HW3-3 status → ✅
├── requirements.txt                     # MODIFY — + pytorch-lightning>=2.5,<3
├── src/
│   ├── model.py                         # UNCHANGED
│   ├── utils.py                         # UNCHANGED
│   ├── animate.py                       # MODIFY — 只加 --exp choices + HW3-3 dispatch
│   └── dqn_lightning.py                 # NEW — Lightning 訓練腳本（含 CLI flags 開關 3 tricks）
├── tests/
│   └── test_dqn_lightning.py            # NEW — smoke test (parametrized over 5 trick combos)
├── results/
│   └── HW3-3/                           # NEW
│       ├── baseline_random/             # 5 組訓練產物
│       ├── clip_random/                 #   每組含 loss.png, dashboard.gif,
│       ├── sched_random/                #   metrics.json, checkpoint.pth,
│       ├── huber_random/                #   losses.npy, snapshots/
│       └── full_random/
└── docs/superpowers/
    ├── specs/2026-05-06-hw3-dqn-stage3-design.md   # 本文件
    └── plans/2026-05-06-hw3-dqn-stage3-implementation.md  # 下一步產出
```

`src/dqn_replay.py`、`src/dqn_double.py`、`src/dqn_dueling.py`、`src/dqn_double_dueling.py` 完全保留不動。

---

## 4. 模組介面

### 4.1 `src/dqn_lightning.py`

完整檔案約 200 行（單檔，包含 LightningModule + IterableDataset + train function + CLI）。

```python
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

    **One `__iter__` call = one game = one Lightning "epoch"**. Lightning
    re-creates the iterator each epoch via `iter(loader)`; we play exactly
    one Gridworld game per call, yielding minibatches as the buffer fills.
    The replay buffer (`self.replay`) is an instance attribute so it
    persists across epochs.

    Total games played = `Trainer(max_epochs=...)`. With `interval='epoch'`
    the lr scheduler then steps once per game (T_max=epochs aligned), and
    SnapshotCallback fires once per game.

    The dataset holds a reference to the LightningModule's online net for
    epsilon-greedy action selection (no deep copy). Action selection is in
    `torch.no_grad()`; the network is the same DuelingMLP being optimised.
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
        """Play one game; yield a minibatch after every move where buffer is full."""
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
            self.replay.append(
                (state1, action_idx, reward, state2, done))
            state1 = state2

            if len(self.replay) > self.batch_size:
                minibatch = random.sample(
                    list(self.replay), self.batch_size)
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
    """Combined Double+Dueling DQN, Lightning-wrapped, with optional tricks."""

    def __init__(self, *, lr: float, gamma: float, sync_freq: int,
                 epochs: int, sched: bool, huber: bool):
        super().__init__()
        self.save_hyperparameters()
        self.online = build_dueling_model()
        self.target = build_dueling_model()
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.loss_fn = nn.SmoothL1Loss() if huber else nn.MSELoss()
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
    """Save vanilla state_dict every N games to keep animate.py compatible.

    Tracks game count internally rather than relying on `trainer.current_epoch`
    (which has off-by-one nuances depending on Lightning version). Names the
    files `epoch_NNNN.pth` to match HW3-1/2 conventions where "epoch" = game.
    """

    def __init__(self, snapshots_dir: Path, every: int):
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
    """Train Lightning Combined DQN with optional tricks. Saves the same
    artifact set as HW3-2 variants (checkpoint.pth, snapshots/, losses.npy,
    loss.png, metrics.json) under `out_dir`. Returns metrics dict.
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
        enable_checkpointing=False,        # we save vanilla state_dict ourselves
        logger=False,
        accelerator='cpu',
        devices=1,
    )
    t0 = time.time()
    trainer.fit(module, loader)
    wall_time = time.time() - t0

    # Save final artifacts in HW3-1/2-compatible format
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

    # Auto-name out_dir from active tricks
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
            tag = '_'.join(active)            # e.g. 'clip_huber' (rare)
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
```

### 4.2 `src/animate.py` 修改

`make_dashboard_gif()` 函式邏輯**完全不動**，只動 `main()` 的 CLI dispatch：

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

# Stage dispatch
if args.exp in ('replay_random',) or 'static' in args.exp:
    stage_dir = 'HW3-1'
elif args.exp.endswith('_player'):
    stage_dir = 'HW3-2'
else:
    stage_dir = 'HW3-3'

# Model factory dispatch (Dueling for HW3-2 dueling/combined + all of HW3-3)
hw3_2_dueling = ('dueling_player', 'combined_player')
hw3_3_all = ('baseline_random', 'clip_random', 'sched_random',
             'huber_random', 'full_random')
factory = (build_dueling_model
           if args.exp in hw3_2_dueling + hw3_3_all
           else build_model)
```

> **注意**：HW3-1 既有的 `replay_random` 仍然走 `HW3-1` 分支與 `build_model`（Sequential）；HW3-3 的 `*_random` 都走 `HW3-3` 分支與 `build_dueling_model`。命名空間沒有衝突。

`loss_yscale` 規則維持不變（HW3-3 5 組都不含 `naive` → 全部走 `'linear'`）。

### 4.3 `metrics.json` schema（HW3-3 版）

完全沿用 HW3-1/2 schema，新增兩個欄位：

- `tricks`: `{clip: bool, sched: bool, huber: bool}` — 哪些 tricks 開啟
- `hyperparams.gradient_clip_val` / `hyperparams.lr_scheduler` / `hyperparams.loss_fn` — 三個 tricks 的具體參數（None 表示未啟用）

`method` 欄位的值統一為 `'lightning_combined'`（5 組共用）；以 `experiment` 與 `tricks` 兩欄區分。

範例（`full_random`）：

```json
{
  "stage": "HW3-3: Lightning-converted DQN with Training Tricks for random mode",
  "experiment": "full_random",
  "mode": "random",
  "method": "lightning_combined",
  "tricks": {"clip": true, "sched": true, "huber": true},
  "hyperparams": {
    "epochs": 5000, "gamma": 0.9, "epsilon": 0.3, "lr": 0.001,
    "mem_size": 1000, "batch_size": 200, "max_moves": 50,
    "sync_freq": 500, "seed": 42, "snapshot_every": 250,
    "gradient_clip_val": 10.0,
    "lr_scheduler": "CosineAnnealingLR(eta_min=1e-5)",
    "loss_fn": "SmoothL1Loss"
  },
  "final_loss_mean_last_100": 0.025,
  "final_loss_std_last_100": 0.015,
  "win_rate": 0.92,
  "avg_steps_per_win": 2.4,
  "n_eval_games": 1000,
  "training_wall_time_sec": 30.0
}
```

### 4.4 `requirements.txt` 修改

新增一行：

```
pytorch-lightning>=2.5,<3
```

實際安裝時鎖到具體 minor version（與 torch 2.11 相容的最新 2.x release）。其餘相依不變。

---

## 5. 實驗設計

### 5.1 5 組實驗（all on `mode='random'`, seed=42, epochs=5000）

| # | 名稱 | clip | sched | huber | Out dir | 預估 CPU |
|---|---|---|---|---|---|---|
| 1 | `baseline_random` | ❌ | ❌ | ❌ | `results/HW3-3/baseline_random` | ~25–35s |
| 2 | `clip_random` | ✅ | ❌ | ❌ | `results/HW3-3/clip_random` | ~25–35s |
| 3 | `sched_random` | ❌ | ✅ | ❌ | `results/HW3-3/sched_random` | ~25–35s |
| 4 | `huber_random` | ❌ | ❌ | ✅ | `results/HW3-3/huber_random` | ~25–35s |
| 5 | `full_random` | ✅ | ✅ | ✅ | `results/HW3-3/full_random` | ~25–35s |

> **Wall-time 估計**：HW3-1 `replay_random` 5000 epochs 是 18s（vanilla PyTorch）。Lightning 加 callback 與 dataloader overhead 約 30–50%；Combined 比 replay 多 1 份網路前向 ~10%。最終估 25–35s/run，5 組總計 ~3 分鐘。

### 5.2 執行命令

```bash
source .venv/bin/activate

# 訓練
python -m src.dqn_lightning                                           # baseline
python -m src.dqn_lightning --clip                                    # +clip
python -m src.dqn_lightning --sched                                   # +sched
python -m src.dqn_lightning --huber                                   # +huber
python -m src.dqn_lightning --clip --sched --huber                    # full

# Dashboard GIFs
for exp in baseline_random clip_random sched_random huber_random full_random; do
    python -m src.animate --exp $exp
done
```

### 5.3 預期結果（reality check）

對照 HW3-1 `replay_random`（85.5% win, loss 0.0613±0.0573, 18s）。

| 實驗 | Final Loss (mean ± std) | Win rate | 解釋 |
|---|---|---|---|
| baseline_random | ~0.04 ± 0.04 | ~88–92% | Combined > replay；Lightning 開銷未顯著影響結果 |
| clip_random | ~0.035 ± 0.025 | ~88–92% | std 明顯下降（尖刺被切平），mean 略降 |
| sched_random | ~0.025 ± 0.03 | ~89–93% | mean 降低（後期 lr 小，fine convergence），std 改變不顯著 |
| huber_random | ~0.03 ± 0.02 | ~89–93% | mean 與 std 都降；Huber 對 random 的 reward 噪音最有針對性 |
| full_random | ~0.02 ± 0.012 | ~92–95% | 三 tricks 疊加；最低 mean、最低 std、最高 win_rate |

關鍵 sanity check：
1. **`baseline_random.win_rate ≥ 85.5%`**（HW3-1 replay_random baseline）— 驗證 Lightning 框架轉換沒有破壞訓練。如果低於 85%，先查 RolloutDataset 的 epsilon_greedy 是否在 `torch.no_grad()` 下、target sync timing、seed 是否正確傳到 `pl.seed_everything`。
2. **`full_random` 比 `baseline_random` win_rate 高至少 3 個百分點**、loss std 低至少 30% — 否則 tricks 沒有實質效果，重新檢查超參數（max_norm, eta_min, beta）。
3. **三個單獨 trick 組的 wall_time 應與 baseline 在 ±20% 內**（tricks 都是輕量加法，不是新前向）。

### 5.4 評估方法

完全沿用 `src/utils.evaluate(model, mode='random', n_games=1000)`：1000 場 greedy（無 ε）測試；隨機棋盤每局重抽（`random.seed` 在 evaluate 內不重置 → 1000 場是 1000 個不同 seed 的棋盤；與 HW3-1 random_replay 評估方式一致）。

`module.online` 是 `DuelingMLP` instance；`evaluate` 只用 `model(state)` 介面 → 相容。

---

## 6. 報告 `HW3_3_report.md` 結構

沿用 HW3-1/2 章節風格：

```markdown
# HW3-3: Lightning-converted DQN with Training Tricks for random mode
## PyTorch → Lightning 框架轉換 + 三個訓練技巧的消融研究

> 作者：charles88　|　課程：深度強化學習　|　日期：2026-05-06
> Repo：<github 連結>

1. 作業目標
2. random mode 重訪：HW3-1 留下的痛點
   2.1 win_rate 85.5%、loss std 與 mean 同量級
   2.2 為何 player mode 的 4 變體都 100% — 環境太簡單，看不出 tricks 效果
3. 從 HW3-2 PyTorch Combined 到 PyTorch Lightning
   3.1 自動 vs 手動 optimization
   3.2 RolloutDataset：把 episode 包成 IterableDataset
   3.3 DQNLightningModule + Trainer + SnapshotCallback
   3.4 等價性驗證（baseline_random vs HW3-1 replay_random）
4. Trick A — Gradient Norm Clipping
   4.1 原理（DQN target 跳變導致大 gradient + Adam momentum 被破壞）
   4.2 程式碼（Trainer flag 一行）
   4.3 訓練結果（loss + dashboard）
5. Trick B — CosineAnnealingLR
   5.1 原理（後期 fine convergence + SGDR 文獻）
   5.2 程式碼（configure_optimizers）
   5.3 訓練結果
6. Trick C — Huber Loss
   6.1 原理（random mode reward 重尾；Mnih 2015 的 clip TD error）
   6.2 程式碼（loss_fn 一行）
   6.3 訓練結果
7. 5 組對比
   （表格：Final Loss / Win rate / Avg steps + 一段討論「哪個 trick 最重要」）
8. 結論（總結 HW3 三階段的學習軌跡）
```

長度約 6–8 頁。圖片用相對路徑：`![](results/HW3-3/full_random/loss.png)`。

---

## 7. 測試（pytest）

新增 1 個 test 檔，遵循 HW3-1/2 的 smoke test 模式：

```python
# tests/test_dqn_lightning.py — 5 parametrized smoke tests + 1 module test
import pytest
from src.dqn_lightning import train_lightning, DQNLightningModule

@pytest.mark.parametrize("clip,sched,huber,tag", [
    (False, False, False, 'baseline'),
    (True,  False, False, 'clip'),
    (False, True,  False, 'sched'),
    (False, False, True,  'huber'),
    (True,  True,  True,  'full'),
])
def test_train_lightning_smoke(tmp_path, clip, sched, huber, tag):
    """Each trick combo runs end-to-end; verifies all artifacts produced.
    epochs=2, mem_size=10, batch_size=4, eval_n_games=2 → ~3 seconds.
    """
    out = tmp_path / tag
    metrics = train_lightning(
        epochs=2, mem_size=10, batch_size=4, max_moves=5,
        snapshot_every=1, eval_n_games=2,
        out_dir=str(out), clip=clip, sched=sched, huber=huber,
    )
    assert (out / 'checkpoint.pth').exists()
    assert (out / 'losses.npy').exists()
    assert (out / 'loss.png').exists()
    assert (out / 'metrics.json').exists()
    assert metrics['tricks'] == {'clip': clip, 'sched': sched, 'huber': huber}


def test_lightning_module_construction():
    """LightningModule constructs OK, online & target weights identical at init."""
    m = DQNLightningModule(lr=1e-3, gamma=0.9, sync_freq=500,
                           epochs=10, sched=False, huber=False)
    for p_o, p_t in zip(m.online.parameters(), m.target.parameters()):
        assert (p_o.data == p_t.data).all()
```

預期測試總數：HW3-2 的 33 → HW3-3 後 39 個（+6），全綠通過。

> **Smoke test 為什麼包 5 個 trick combos**：每個 combo 走的 Lightning 路徑略有差異（gradient_clip_val flag、scheduler dict、loss class），用 parametrize 一次覆蓋全部，比寫 1 個 baseline smoke + 信任 Lightning「flag 不會壞」更穩。

---

## 8. Chatlog `chatlog3.md`

完整保留 HW3-3 期間（從本次 brainstorming 開始到報告完成）的對話紀錄，格式同 HW3-1 / HW3-2 的 chatlog：

- 每個 turn 標 `## Turn N`
- `**User：** ...` + `**Claude：** ...`（不貼大段 tool output，只留執行摘要）
- 模型標註：Claude Opus 4.7 (1M context)
- 對話日期：2026-05-06

---

## 9. README.md 更新

HW3-3 commit 時順手修兩處：

1. **後續階段表**：HW3-3 列從「⏳ 規劃中」改為「✅ 已完成」並指向 `HW3_3_report.md`。
2. **新增 HW3-3 段落**：簡介 5 組實驗、嵌入 5 個 loss.png 與 dashboard.gif、附量化指標表（與 HW3-1/2 段落形式對稱）。

---

## 10. Commit 策略

8 個 commits，對應 HW3-2 風格：

1. `docs(spec): add HW3-3 design doc`（本文件）
2. `docs(plan): add HW3-3 implementation plan`（writing-plans 產出）
3. `deps: add pytorch-lightning to requirements`
4. `feat: implement Lightning-wrapped Combined DQN with optional tricks (HW3-3)`
5. `test: cover HW3-3 Lightning smoke tests across 5 trick combos`
6. `experiment: run 5-group ablation on random mode (HW3-3)`
7. `experiment: add dashboard GIFs for 5 HW3-3 runs`
8. `docs: add HW3-3 report + chatlog + README update`

---

## 11. Out of Scope

下列項目不在本作業範圍：

- **跑 Keras 版**（已選 PyTorch Lightning，作業 #1 二選一）
- **重做 HW3-2 player mode 的 Lightning 版**（HW3-3 主題明確指向 random mode）
- **重做 HW3-1 static mode 在 Lightning 上**
- **其他 tricks**：target soft update（Polyak τ）、ε-decay schedule、weight decay、Prioritized Experience Replay、N-step returns、Noisy nets、Double 與 Dueling 個別在 Lightning 上的對照（HW3-3 只用 Combined 為 baseline）
- **Hyperparameter sweep**（max_norm / eta_min / Huber β 不做 grid search，沿用文獻預設）
- **多 seed 平均**（單一 seed=42 對齊 HW3-1/2；如果結果偏離預期再考慮第二 seed）
- **GPU / MPS 加速**（網路太小，CPU 已足夠；`Trainer(accelerator='cpu')` 強制）
- **Lightning 的 W&B / TensorBoard logger**（`logger=False`；保留 `losses.npy` 為唯一 loss 儲存格式，與 HW3-1/2 對齊）
- **Lightning 自動 checkpoint**（`enable_checkpointing=False`；只用 SnapshotCallback 存 vanilla state_dict 給 animate.py）
- **Bit-exact 等價性**（Lightning 與 vanilla PyTorch 隨機數消耗順序不同，不要求逐 step 對齊；只要 baseline_random win_rate ≥ HW3-1 replay_random 的 85.5% 即可）
