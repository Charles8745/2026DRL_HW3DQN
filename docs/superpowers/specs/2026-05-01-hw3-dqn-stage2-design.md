# HW3-2: Enhanced DQN Variants for player mode — 設計文件

> **Stage label**：HW3-2: Enhanced DQN Variants for player mode
> **作業來源**：深度強化學習課程 HW3 — DQN and its variants（Stage 2）
> **設計日期**：2026-05-01
> **基底教材**：[Deep Reinforcement Learning *in Action*](https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction) Chapter 3、Hasselt et al. 2016（Double DQN）、Wang et al. 2016（Dueling DQN）
> **前置階段**：[HW3-1 spec](2026-04-30-hw3-dqn-stage1-design.md)

---

## 0. 作業需求對應

| 需求 | 對應交付物 |
|---|---|
| #1 Implement and compare Double DQN & Dueling DQN | 4 組實驗：`replay_player`、`double_player`、`dueling_player`、`combined_player` |
| #1_1 Double DQN | `src/dqn_double.py` + `results/HW3-2/double_player/` |
| #1_2 Dueling DQN | `src/dqn_dueling.py` + `results/HW3-2/dueling_player/`；網路工廠在 `src/model.py::build_dueling_model` |
| Focus on improvements over basic DQN | 報告 Section 4–6 各自有「原理 → 程式碼解讀（diff 自 baseline）→ 訓練結果」三段結構 |
| 沿用 HW3-1 規格與品質 | 章節結構、metrics schema、commit 風格、測試覆蓋、dashboard GIF 全部對齊 |

---

## 1. 設計原則

1. **Player mode 為主軸**：所有 4 組實驗都跑 `mode='player'`（只有 Player 位置隨機，Goal/Pit/Wall 固定）。State 空間複雜度介於 HW3-1 的 static 與 random 之間。
2. **乾淨拆解每個改進的貢獻**：
   - Baseline = `DQN+Replay`（無 target net）
   - `Double DQN` = baseline + Target Network + 動作選擇/估值分離
   - `Dueling DQN` = baseline + V/A 雙頭網路（**不加** target network）
   - `Combined` = Target Network + 動作選擇/估值分離 + V/A 雙頭網路
3. **與 HW3-1 共用基底**：`gridworld_env`、`utils`、`animate.py` 完全不改邏輯（animate 只加 `--exp` choices）。
4. **每個變體一個獨立檔案**：沿用 HW3-1 「一個變體一個 .py」的慣例（與 `dqn_naive.py` / `dqn_replay.py` 同級）。重複是教學示範性質的可接受代價。
5. **單一 seed=42**：對齊 HW3-1，不做多 seed 平均；如果結果偏離預期再考慮第二 seed。

---

## 2. 演算法規格

所有 4 組共用：`gamma=0.9, lr=1e-3, mem_size=1000, batch_size=200, max_moves=50, epsilon=0.3, seed=42, mode='player', epochs=3000, snapshot_every=150`。

### 2.1 Baseline — DQN + Replay

- 模型：`build_model()` → `64→150→100→4`
- Target：

  $$Y = r + \gamma (1-\text{done}) \cdot \max_{a'} Q_\theta(s', a')$$
- 直接重跑 `src/dqn_replay.py`：

  ```bash
  python -m src.dqn_replay --mode player --epochs 3000 --seed 42 \
         --out-dir results/HW3-2/replay_player
  ```

### 2.2 Double DQN

- 模型：兩份 `build_model()` — online $\theta$ 與 target $\theta^-$
- **核心改動**：target 計算時把「選動作」與「估價值」分離

  $$Y = r + \gamma (1-\text{done}) \cdot Q_{\theta^-}\!\left(s',\ \arg\max_{a'} Q_{\theta}(s', a')\right)$$
- Target sync：每 500 個 **training steps**（global step 計數，不是 epoch），硬複製：

  ```python
  if global_step % sync_freq == 0:
      target_model.load_state_dict(online_model.state_dict())
  ```
- 初始化：訓練開始時 `target.load_state_dict(online.state_dict())`
- 改進原理：Vanilla DQN 的 $\max$ 操作同時用同一網路選動作與估值，會導致系統性高估（overestimation bias）。Double DQN 透過分離兩者，讓 online 選的動作未必是 target 高估的那個，期望偏差更小。

### 2.3 Dueling DQN

- **新模型**：`build_dueling_model()`
  - 共用 trunk：`64→150→100`（與 baseline 相同）
  - Value head：`100→1`
  - Advantage head：`100→4`
  - Forward：

    $$Q(s,a) = V(s) + \left(A(s,a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s,a')\right)$$
- Target 公式：與 baseline 相同（`max_a Q_online(s', a)`），**不使用 target network**
- 改進原理：許多 state 下「身處此 state 的價值」與「選哪個 action 比較好」是兩件事（例如殘局只剩一條路 → V 大但 A 都差不多）。將兩者分開能讓網路學 V 時不被 action 選擇干擾，提升 sample efficiency。Mean baseline 強制 advantage 平均為 0，解 V/A 在 $Q = V + A$ 下的 identifiability 不唯一問題（Wang et al. 2016 §4.2）。

### 2.4 Double + Dueling Combined

- **新模型**：`build_dueling_model()` × 2 份（online + target）
- Target 公式：用 Double 的式子 + Dueling 的網路

  $$Y = r + \gamma (1-\text{done}) \cdot Q^{\text{dueling}}_{\theta^-}\!\left(s',\ \arg\max_{a'} Q^{\text{dueling}}_{\theta}(s', a')\right)$$
- Target sync：同 Double（每 500 steps 硬複製）
- 改進原理：Double 修的是 target 計算，Dueling 改的是網路結構，兩者正交可疊加。

---

## 3. 檔案結構

```
HW3_DQN/
├── HW3_2_report.md                    # NEW — HW3-2 中文報告
├── chatlog2.md                        # NEW — HW3-2 對話紀錄
├── README.md                          # MODIFY — 加 HW3-2 段落、修正 HW3-3 描述
├── src/
│   ├── model.py                       # MODIFY — 加 DuelingMLP + build_dueling_model()
│   ├── animate.py                     # MODIFY — 只加 --exp choices（不改邏輯）
│   ├── dqn_double.py                  # NEW — Double DQN 訓練 + CLI
│   ├── dqn_dueling.py                 # NEW — Dueling DQN 訓練 + CLI
│   └── dqn_double_dueling.py          # NEW — Combined 訓練 + CLI
├── tests/
│   ├── test_model_dueling.py          # NEW — Dueling 模型 forward / aggregation 測試
│   ├── test_dqn_double.py             # NEW — smoke test
│   ├── test_dqn_dueling.py            # NEW — smoke test
│   └── test_dqn_double_dueling.py     # NEW — smoke test
├── results/
│   └── HW3-2/                         # NEW
│       ├── replay_player/           # 從 dqn_replay.py 重跑
│       ├── double_player/
│       ├── dueling_player/
│       └── combined_player/           # 每個含 loss.png, dashboard.gif,
│                                      # metrics.json, checkpoint.pth, losses.npy, snapshots/
└── docs/superpowers/
    ├── specs/2026-05-01-hw3-dqn-stage2-design.md   # 本文件
    └── plans/2026-05-01-hw3-dqn-stage2-implementation.md  # 下一步產出
```

---

## 4. 模組介面

### 4.1 `src/model.py` 新增

```python
import torch
import torch.nn as nn


class DuelingMLP(nn.Module):
    """Dueling network with shared trunk + V/A heads + mean-baseline aggregation."""

    def __init__(self, in_dim: int = 64, hidden1: int = 150,
                 hidden2: int = 100, n_actions: int = 4):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden2, 1)
        self.advantage_head = nn.Linear(hidden2, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        v = self.value_head(h)               # (B, 1)
        a = self.advantage_head(h)           # (B, n_actions)
        return v + (a - a.mean(dim=1, keepdim=True))   # (B, n_actions)


def build_dueling_model(in_dim: int = 64, hidden1: int = 150,
                        hidden2: int = 100, n_actions: int = 4) -> DuelingMLP:
    return DuelingMLP(in_dim, hidden1, hidden2, n_actions)
```

`build_model()` 維持不變。

### 4.2 `src/dqn_double.py`

```python
def train_double(
    *,
    epochs: int = 3000,
    gamma: float = 0.9,
    epsilon: float = 0.3,
    lr: float = 1e-3,
    mem_size: int = 1000,
    batch_size: int = 200,
    max_moves: int = 50,
    sync_freq: int = 500,                 # global training steps
    mode: str = 'player',
    seed: int = 42,
    snapshot_every: int = 150,
    out_dir: str = 'results/HW3-2/double_player',
    eval_n_games: int = 1000,
) -> dict:
    """Double DQN 訓練（Hasselt 2016）。產出與 dqn_replay 相同的檔案集
    （checkpoint, snapshots/, losses.npy, loss.png, metrics.json）。
    metrics.json 帶 method='double'。
    """
```

訓練迴圈相對 `dqn_replay.train_replay` 的 diff（約 10 行）：

```python
# 1) 訓練前：多建一份 target model
online_model = build_model()
target_model = build_model()
target_model.load_state_dict(online_model.state_dict())
target_model.eval()
global_step = 0

# 2) minibatch 更新區塊內（取代原本的 max-Q target Y 計算）：
if len(replay) > batch_size:
    # ... 原本的 minibatch sampling 與 batching ...
    with torch.no_grad():
        online_next = online_model(state2_batch)
        next_actions = online_next.argmax(dim=1, keepdim=True)            # (B, 1)
        target_next = target_model(state2_batch)
        next_q = target_next.gather(1, next_actions).squeeze(1)           # (B,)
    Y = reward_batch + gamma * (1 - done_batch) * next_q
    # ... 原本的 loss 計算 + backward + step ...

    # 3) 完成一次 update 才算一個 training step；sync 在這個 step 計數上做
    global_step += 1
    if global_step % sync_freq == 0:
        target_model.load_state_dict(online_model.state_dict())
```

CLI 完全照 `dqn_replay.py` 形式，外加 `--sync-freq` flag（預設 500）。

### 4.3 `src/dqn_dueling.py`

```python
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
    """Dueling DQN 訓練（Wang 2016）。網路換成 build_dueling_model，
    target 計算與 baseline 相同，無 target network。
    metrics.json 帶 method='dueling'。
    """
```

訓練迴圈相對 `dqn_replay.train_replay` 的 diff（約 1 行）：

```python
model = build_dueling_model()    # 取代 build_model()
# 其餘訓練邏輯完全不變
```

### 4.4 `src/dqn_double_dueling.py`

API 與 `dqn_double.py` 相同，差別只在：

```python
online_model = build_dueling_model()    # 取代 build_model()
target_model = build_dueling_model()
# Double 的 target Y 計算與 sync 邏輯完全沿用
```

`metrics.json` 帶 `method='double_dueling'`。

### 4.5 `src/animate.py` 修改

`make_dashboard_gif()` 函式邏輯**完全不動**，只動 `main()` 的 CLI dispatch：(a) `--exp` 加 4 個新 choices；(b) 依 exp 名字決定 `results/HW3-1/` vs `results/HW3-2/` 路徑前綴。

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', required=True, choices=[
        # HW3-1
        'naive_static', 'replay_static', 'replay_random',
        # HW3-2
        'replay_player', 'double_player', 'dueling_player', 'combined_player',
    ])
    args = parser.parse_args()
    stage_dir = 'HW3-2' if args.exp.endswith('_player') else 'HW3-1'
    make_dashboard_gif(
        exp_dir=f'results/{stage_dir}/{args.exp}',
        loss_yscale='log' if 'naive' in args.exp else 'linear',
    )
```

`loss_yscale` 由現有的 `'naive' in name` 規則決定：HW3-2 4 組都不含 `naive` → 全部走 `'linear'`（與 replay 一致），自動正確，不需新規則。

### 4.6 `metrics.json` schema（HW3-2 版）

完全沿用 HW3-1 schema，`method` 欄位的值擴充至 `'double' | 'dueling' | 'double_dueling'`（HW3-1 用過 `'naive' | 'replay'`）。不額外加新欄位以維持兩階段共通。

```json
{
  "stage": "HW3-2: Enhanced DQN Variants for player mode",
  "experiment": "double_player",
  "mode": "player",
  "method": "double",
  "hyperparams": {
    "epochs": 3000, "gamma": 0.9, "epsilon": 0.3, "lr": 0.001,
    "mem_size": 1000, "batch_size": 200, "max_moves": 50,
    "sync_freq": 500, "seed": 42, "snapshot_every": 150
  },
  "final_loss_mean_last_100": 0.03,
  "final_loss_std_last_100": 0.01,
  "win_rate": 0.94,
  "avg_steps_per_win": 5.2,
  "n_eval_games": 1000,
  "training_wall_time_sec": 22.0
}
```

Baseline 那組（`results/HW3-2/replay_player/metrics.json`）的 `method` 是 `'replay'`、`experiment` 是 `'replay_player'`，由現有 `dqn_replay.py` 直接產出，不需修改任何程式。後續比較表用 `experiment` 欄位辨識最清楚。

---

## 5. 實驗設計

### 5.1 4 組實驗

| # | 名稱 | 演算法 | Out dir | 預估 CPU 時間 |
|---|---|---|---|---|
| 1 | `replay_player` | DQN + Replay | `results/HW3-2/replay_player` | ~12 秒 |
| 2 | `double_player` | Double DQN | `results/HW3-2/double_player` | ~18 秒 |
| 3 | `dueling_player` | Dueling DQN | `results/HW3-2/dueling_player` | ~15 秒 |
| 4 | `combined_player` | Double + Dueling | `results/HW3-2/combined_player` | ~22 秒 |

### 5.2 執行命令

```bash
source .venv/bin/activate

# 訓練
python -m src.dqn_replay         --mode player --epochs 3000 --seed 42 \
       --out-dir results/HW3-2/replay_player
python -m src.dqn_double         --mode player --epochs 3000 --seed 42
python -m src.dqn_dueling        --mode player --epochs 3000 --seed 42
python -m src.dqn_double_dueling --mode player --epochs 3000 --seed 42

# Dashboard GIFs
for exp in replay_player double_player dueling_player combined_player; do
    python -m src.animate --exp $exp
done
```

### 5.3 預期結果（reality check）

Player mode 比 random 簡單（其他物件固定，state 變動只來自 Player 13 種起點），勝率會比 random 的 85.5% 高。

| 實驗 | Final Loss (mean) | Win rate | 解釋 |
|---|---|---|---|
| replay_player | ~0.05 | ~92% | replay 的 ceiling，仍有些起點難收斂 |
| double_player | ~0.03 | ~94% | overestimation 控制 → 略高勝率、loss 更穩 |
| dueling_player | ~0.025 | ~95% | 共用 V(s) 學起來更快 |
| combined_player | ~0.02 | ~96–97% | 兩種改進互補，應達上限 |

如果跑出來偏差超過 ±5% 勝率或 1 個量級 loss，先查：seed、target sync timing、Dueling aggregation 軸是否對、target_model 有沒有 `.eval()`。

### 5.4 評估方法

完全沿用 `src/utils.evaluate(model, mode='player', n_games=1000)`：1000 場 greedy（無 ε）測試，回傳 `{win_rate, avg_steps_per_win, n_games}`。Dueling 與 Combined 回傳的是 `DuelingMLP` instance，因 `evaluate` 只用 `model(state)` 介面，與 `nn.Sequential` 相容。

---

## 6. 報告 `HW3_2_report.md` 結構

沿用 HW3-1 章節風格：

```markdown
# HW3-2: Enhanced DQN Variants for player mode
## Double DQN 與 Dueling DQN 的改進機制與比較

> 作者：charles88　|　課程：深度強化學習　|　日期：2026-05-01
> Repo：<github 連結>

1. 作業目標
2. 環境（player mode 簡介，與 static/random 的差異）
3. 從 baseline 到變體
   3.1 Baseline 的兩個痛點：overestimation + sample efficiency
   3.2 Double 與 Dueling 各自針對哪個痛點
4. Double DQN
   4.1 原理（Hasselt 2016 — overestimation 數學推導 + 解法）
   4.2 程式碼解讀（diff 自 baseline 的 ~10 行）
   4.3 訓練結果（loss + dashboard）
5. Dueling DQN
   5.1 原理（Wang 2016 — V/A 拆解 + identifiability 問題 + mean baseline）
   5.2 程式碼解讀（DuelingMLP 定義 + forward）
   5.3 訓練結果
6. Double + Dueling 合併
   6.1 為何能疊加（兩個改進正交）
   6.2 訓練結果
7. 四組對比
   （表格：Final Loss / Win rate / Avg steps + 一段討論）
8. 結論（連到 HW3-3 — PyTorch → Keras / Lightning 框架轉換 + training tricks
   如 gradient clipping、lr scheduling）
```

長度約 6–8 頁。圖片用相對路徑：`![](results/HW3-2/double_player/loss.png)`。

---

## 7. 測試（pytest）

新增 4 個 test 檔，遵循 HW3-1 的 smoke test 模式（小 epochs / 小 buffer 跑通）：

```python
# tests/test_model_dueling.py — 2 tests
def test_dueling_forward_shape():
    """forward (B=1, 64) → (1, 4)"""

def test_dueling_advantage_zero_mean():
    """forward 後 Q - V 在 action 軸的平均應接近 0
    （驗證 mean-baseline aggregation 數學性質）"""

# tests/test_dqn_double.py — 1 smoke test
def test_train_double_smoke(tmp_path):
    """epochs=2, mem_size=10, batch_size=4 跑通；
    驗證 checkpoint.pth、snapshots/、losses.npy、metrics.json 都產出。"""

# tests/test_dqn_dueling.py — 同上
# tests/test_dqn_double_dueling.py — 同上
```

預期測試總數：HW3-1 的 24 → HW3-2 後 29 個，全綠通過。

---

## 8. Chatlog `chatlog2.md`

完整保留 HW3-2 期間（從本次 brainstorming 開始到報告完成）的對話紀錄，格式同 HW3-1 的 `chatlog.md`：

- 每個 turn 標 `## Turn N`
- `**User：** ...` + `**Claude：** ...`（不貼大段 tool output，只留執行摘要）
- 模型標註：Claude Opus 4.7 (1M context)
- 對話日期：2026-05-01

---

## 9. README.md 更新

HW3-2 commit 時順手修兩處：

1. **後續階段表**：
   - 把 `HW3-3：DQN variants — Dueling DQN、Prioritized Replay` 改為
     `HW3-3：Framework conversion + training tricks（PyTorch → Keras / Lightning + gradient clipping / lr scheduling）`
   - HW3-2 列從「⏳ 規劃中」改為「✅ 已完成」並指向 `HW3_2_report.md`

2. **新增 HW3-2 段落**：簡介 4 組實驗、嵌入 4 個 loss.png 與 dashboard.gif、附量化指標表（與 HW3-1 段落形式對稱）。

---

## 10. Commit 策略

10 個 commits，對應 HW3-1 風格：

1. `docs(spec): add HW3-2 design doc`（本文件）
2. `docs(plan): add HW3-2 implementation plan`（writing-plans 產出）
3. `feat(model): add DuelingMLP with mean-baseline aggregation`
4. `test(model): cover dueling forward shape + advantage zero-mean property`
5. `feat: implement Double DQN training (HW3-2)`
6. `feat: implement Dueling DQN training (HW3-2)`
7. `feat: implement Double + Dueling combined training (HW3-2)`
8. `experiment: run all 4 player-mode experiments (HW3-2)`
9. `experiment: add dashboard GIFs for 4 HW3-2 runs`
10. `docs: add HW3-2 report + chatlog + README update`

---

## 11. Out of Scope

下列項目不在本作業範圍：

- **Prioritized Experience Replay**（按 TD error 加權抽樣）
- **N-step returns**
- **Noisy nets / Rainbow** 類疊加技巧
- 跑 `mode='static'` 或 `mode='random'` 的 4 變體實驗（聚焦 player mode，不做跨 mode 矩陣以免訓練量膨脹 3 倍）
- **Hyperparameter sweep**（lr / sync_freq / mem_size 不做 grid search，沿用文獻預設）
- **Soft target update（Polyak averaging）**（hard sync 已足以呈現 Double DQN 的核心想法）
- **多 seed 平均**（單一 seed=42 對齊 HW3-1；如果結果偏離預期再考慮第二 seed）
- **GPU / MPS 加速**（網路太小，CPU 已足夠）
- **HW3-3 的內容**（PyTorch → Keras / Lightning 框架轉換、gradient clipping、lr scheduling 等 training tricks），留給下一階段
