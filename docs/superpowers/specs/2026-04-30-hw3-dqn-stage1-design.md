# HW3-1: Naive DQN for static mode — 設計文件

> **Stage label**：HW3-1: Naive DQN for static mode
> **作業來源**：深度強化學習課程 HW3 — DQN and its variants（Stage 1）
> **設計日期**：2026-04-30
> **基底教材**：[Deep Reinforcement Learning *in Action*](https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction) Chapter 3

---

## 0. 作業需求對應

| 需求 | 對應交付物 |
|---|---|
| 0_1 Base on DRL in Action repo | `src/gridworld_env.py`、`src/gridboard.py`（adapted from Chapter 3） |
| 0_2 Use updated starter code as baseline | 上述兩個檔案 + 書中 Listing 3.3 / 3.5 改寫成 .py 模組 |
| 1_1 Run the provided code (naive or experience-buffer-replay) | 三組訓練：Naive on static、Replay on static、Replay on random，產物在 `results/HW3-1/` |
| 1_2 Chat with Claude about the code | `chatlog.md`（完整對話紀錄） |
| 1_3 Submit a short understanding report | `report.md`（中文，4–6 頁，章節 1–8） |

---

## 1. 專案結構

repo root = `/Users/charles88/Downloads/HW3_ DQN/`，最終結構：

```
HW3_DQN/
├── README.md                        # placeholder（內容由使用者後續指定）
├── report.md                        # HW3-1 中文報告
├── chatlog.md                       # HW3-1 與 Claude 完整對話紀錄
├── requirements.txt                 # uv pip freeze 鎖檔
├── pyproject.toml                   # uv 專案配置
├── .python-version                  # 鎖 Python 3.12
├── .gitignore
├── LICENSE                          # MIT + 對 DRL in Action 原作者 attribution
├── docs/
│   └── superpowers/
│       └── specs/
│           └── 2026-04-30-hw3-dqn-stage1-design.md  # 本文件
├── src/
│   ├── __init__.py
│   ├── gridboard.py                 # 從 Ch.3 搬，加 attribution 註解
│   ├── gridworld_env.py             # 從 Ch.3 搬，加 attribution 註解
│   ├── model.py                     # build_model() — 共用 MLP
│   ├── utils.py                     # set_seed, encode_state, epsilon_greedy, ACTION_SET
│   ├── dqn_naive.py                 # Naive DQN training (Listing 3.3 重構)
│   ├── dqn_replay.py                # DQN + Replay (Listing 3.5 重構)
│   └── animate.py                   # Dashboard GIF 生成
└── results/
    └── HW3-1/
        ├── naive_static/
        │   ├── loss.png
        │   ├── losses.npy
        │   ├── dashboard.gif
        │   ├── metrics.json
        │   ├── checkpoint.pth
        │   └── snapshots/epoch_NNNN.pth
        ├── replay_static/            # 同上結構
        └── replay_random/            # 同上結構
```

**.gitignore 排除清單**：
```
.venv/
__pycache__/
*.pyc
*.pyo
.DS_Store
DeepReinforcementLearningInAction/   # 原書 repo（如已 clone 在 root，不上傳）
.idea/
.vscode/
```

**Commit 策略（從零到完成）**：
1. `chore: scaffold project structure & dependencies`
2. `feat: add gridworld env adapted from DRL in Action Ch.3`
3. `feat: implement naive DQN training (HW3-1)`
4. `feat: implement DQN with experience replay (HW3-1)`
5. `feat: add dashboard animation generator`
6. `experiment: run all three experiments and commit results`
7. `docs: add HW3-1 report and chatlog`

---

## 2. 環境設定

**Python**：3.12.x（M 系列 Mac 上 uv 自動下載）
**Backend**：CPU（網路太小，MPS overhead 反而慢）

**setup 流程**：
```bash
cd "/Users/charles88/Downloads/HW3_ DQN"
uv venv --python 3.12
source .venv/bin/activate
uv pip install torch numpy matplotlib imageio "imageio[ffmpeg]" tqdm
uv pip freeze > requirements.txt
```

**依賴清單**：
- `torch` — DQN
- `numpy` — state encoding
- `matplotlib` — loss curve & dashboard frames
- `imageio` + `imageio[ffmpeg]` — GIF / MP4 寫出
- `tqdm` — CLI 進度條（取代 notebook 的 `clear_output`）

---

## 3. 程式碼模組介面

### 3.1 `src/gridboard.py`、`src/gridworld_env.py`
**從 DRL in Action Chapter 3 直接搬，邏輯零改動**，僅在檔頭加：
```python
"""
Adapted from Chapter 3 of "Deep Reinforcement Learning in Action"
by Alexander Zai and Brandon Brown (Manning, 2020).
Original source: https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction
"""
```

對外 API（不變）：
- `Gridworld(size=4, mode='static'|'player'|'random')`
- `.makeMove('u'|'d'|'l'|'r')` / `.reward()` / `.display()`
- `.board.render_np()` → `(4, 4, 4)` uint8 one-hot tensor

### 3.2 `src/model.py`
```python
import torch.nn as nn

def build_model(in_dim: int = 64, hidden1: int = 150, hidden2: int = 100,
                out_dim: int = 4) -> nn.Sequential:
    """Returns the MLP from Listing 3.2: in→150→100→out, two ReLU."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden1), nn.ReLU(),
        nn.Linear(hidden1, hidden2), nn.ReLU(),
        nn.Linear(hidden2, out_dim),
    )
```

### 3.3 `src/utils.py`
```python
ACTION_SET = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}

def set_seed(seed: int) -> None: ...           # torch + numpy + random + torch.use_deterministic_algorithms
def encode_state(game) -> torch.Tensor: ...    # game.board.render_np() → (1,64) float + N(0,1e-2)
def epsilon_greedy(qval, epsilon, n_actions=4) -> int: ...
def save_metrics(path: str, **kwargs) -> None: ...   # JSON dump
def running_mean(x, N=50) -> np.ndarray: ...    # 用於 loss 圖平滑

def test_model(model, mode: str, max_steps: int = 15) -> tuple[bool, int]:
    """Listing 3.4 重構。greedy（無 ε）跑一場 game。
    回傳 (won, steps_taken)。動畫腳本與 train 函式都會用到。"""

def evaluate(model, mode: str, n_games: int = 1000) -> dict:
    """重複 test_model n 次，回傳 {'win_rate': ..., 'avg_steps_per_win': ...}。"""
```

### 3.4 `src/dqn_naive.py`
公開函式：
```python
def train_naive(
    *,
    epochs: int = 1000,
    gamma: float = 0.9,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.1,
    lr: float = 1e-3,
    mode: str = 'static',
    seed: int = 42,
    snapshot_every: int = 50,
    out_dir: str = 'results/HW3-1/naive_static',
    stage_label: str = 'HW3-1: Naive DQN for static mode',
) -> dict:
    """Listing 3.3 重構。產出：
        {out_dir}/checkpoint.pth         # 最終 model
        {out_dir}/losses.npy             # 完整 loss 序列
        {out_dir}/loss.png               # 訓練曲線（含 running mean）
        {out_dir}/metrics.json           # final_loss / win_rate / avg_steps / hyperparams / stage
        {out_dir}/snapshots/epoch_NNNN.pth  # 每 snapshot_every epochs 一份
    Returns metrics dict."""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='static')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-dir', default='results/HW3-1/naive_static')
    args = parser.parse_args()
    train_naive(**vars(args))
```

訓練迴圈邏輯逐字對應 Listing 3.3，差異只在：
- 把 `clear_output` 換成 `tqdm` 進度條
- 每 `snapshot_every` epochs 存 model `state_dict`
- 訓練結束跑 1000 場 test games 算 win rate 與平均步數
- 全部 metrics 寫進 JSON

### 3.5 `src/dqn_replay.py`
公開函式：
```python
def train_replay(
    *,
    epochs: int = 5000,
    gamma: float = 0.9,
    epsilon: float = 0.3,                 # 固定（沿用書本）
    lr: float = 1e-3,
    mem_size: int = 1000,
    batch_size: int = 200,
    max_moves: int = 50,
    mode: str = 'random',
    seed: int = 42,
    snapshot_every: int = 250,
    out_dir: str = 'results/HW3-1/replay_random',
    stage_label: str = 'HW3-1: Naive DQN for static mode',
) -> dict: ...
```

邏輯逐字對應 Listing 3.5（replay deque + minibatch sampling + vectorized target）。CLI 介面同 `dqn_naive.py`。

### 3.6 `src/animate.py`
公開函式：
```python
def make_dashboard_gif(
    *,
    exp_dir: str,                         # e.g. 'results/HW3-1/naive_static'
    test_mode: str | None = None,         # 預設讀 metrics.json 的 mode；可覆寫
    fps: int = 5,
    figsize: tuple = (12, 5),
    loss_yscale: str = 'log',             # 'log'（naive）或 'linear'（replay）
    max_test_steps: int = 15,
    out_filename: str = 'dashboard.gif',
) -> str: ...
```

執行邏輯：
1. 讀 `{exp_dir}/snapshots/` 下所有 checkpoint，按 epoch 排序
2. 讀 `{exp_dir}/losses.npy`
3. 對每個 snapshot：
   a. 載入 model
   b. `random.seed(epoch_idx)` 後 `Gridworld(mode=test_mode)` 創建測試棋盤
      （static mode 永遠是同一盤，random/player mode 每個 snapshot 不同盤但可重現）
   c. greedy（無 ε）跑最多 `max_test_steps` 步，每步 render 一張 frame
   d. 結尾加 4 張「結果幀」（Win / Lost）形成停留
4. `imageio.mimsave(out_path, frames, fps=fps, loop=0)`

**Frame 渲染**（單張 frame 製作）：
```python
fig, (ax_grid, ax_loss) = plt.subplots(1, 2, figsize=figsize)

# 左：grid
# 4×4 imshow with discrete colormap; 每格中央 text() 放 P/+/-/W
ax_grid.set_title(f'Epoch {epoch} | Move {move_idx} | Action: {action}')

# 右：loss
ax_loss.plot(losses_full, color='lightgray', linewidth=0.5)         # 完整未來曲線
ax_loss.plot(losses_full[:current_step], color='red', linewidth=1)  # 已訓練的部分
ax_loss.axvline(current_step, color='red', linestyle='--')          # 當前位置
ax_loss.set_yscale(loss_yscale)
ax_loss.set_xlabel('Training step')
ax_loss.set_ylabel('Loss')

fig.suptitle(f'Status: training (epoch {epoch}/{max_epoch}) | Outcome: {outcome}')
```

CLI：
```python
if __name__ == '__main__':
    parser.add_argument('--exp', required=True,
                        choices=['naive_static', 'replay_static', 'replay_random'])
    args = parser.parse_args()
    make_dashboard_gif(exp_dir=f'results/HW3-1/{args.exp}',
                       loss_yscale='log' if 'naive' in args.exp else 'linear')
```

**模組間隔離**：訓練模組與 `animate.py` 之間以 **檔案系統 = 介面**（snapshots dir + losses.npy + metrics.json）。動畫不依賴訓練程式仍在執行，訓練也不知道有人會做動畫。

---

## 4. 實驗設計

### 4.1 三個實驗

| # | 名稱 | Mode | Method | Epochs | 預計 CPU 時間 |
|---|---|---|---|---|---|
| 1 | `naive_static` | static | Naive DQN（Listing 3.3） | 1000 | ~30–60 秒 |
| 2 | `replay_static` | static | DQN + Replay（Listing 3.5） | 1000 | ~1–2 分鐘 |
| 3 | `replay_random` | random | DQN + Replay（Listing 3.5） | 5000 | ~5–10 分鐘 |

實驗 1 vs 實驗 2：在同一個（簡單）環境下對比 Naive 與 Replay。
實驗 2 vs 實驗 3：對比 Replay 在簡單 vs 困難環境的表現。
實驗 1 vs 實驗 3：呈現「Naive 為什麼不能直接拿來做 random」的隱含論點（雖然不會直接跑 Naive on random，因為書本與課程目的都已說明它會失敗）。

### 4.2 Hyperparameters（沿用書本預設）

**實驗 1 — `naive_static`**：
- `epochs=1000, gamma=0.9, lr=1e-3, seed=42`
- ε：1.0 → 0.1 線性衰減（每 epoch 減 `1/epochs`）
- snapshot_every=50（→ 20 個 snapshots + epoch 0）

**實驗 2 — `replay_static`**：
- `epochs=1000, gamma=0.9, epsilon=0.3, lr=1e-3, seed=42`
- `mem_size=1000, batch_size=200, max_moves=50`
- snapshot_every=50（→ 20 個 snapshots + epoch 0）

**實驗 3 — `replay_random`**：
- `epochs=5000, gamma=0.9, epsilon=0.3, lr=1e-3, seed=42`
- `mem_size=1000, batch_size=200, max_moves=50`
- snapshot_every=250（→ 20 個 snapshots + epoch 0）

### 4.3 評估指標（每個實驗都計算）

寫入 `metrics.json`（以下為 `naive_static` 預期值範例）：
```json
{
  "stage": "HW3-1: Naive DQN for static mode",
  "experiment": "naive_static",
  "mode": "static",
  "method": "naive",
  "hyperparams": {
    "epochs": 1000, "gamma": 0.9, "lr": 0.001,
    "epsilon_start": 1.0, "epsilon_end": 0.1, "seed": 42
  },
  "final_loss_mean_last_100": 0.045,
  "final_loss_std_last_100": 0.012,
  "win_rate_1000": 1.000,
  "avg_steps_per_win": 3.0,
  "training_wall_time_sec": 47.3,
  "git_commit": "abcdef1"
}
```

**評估方法**：呼叫 `utils.evaluate(model, mode=..., n_games=1000)`（內部 1000 次 `test_model`，greedy 無 ε）。統計：
- 勝率 = 抵達 Goal 場數 / 1000
- 平均步數 = 勝場的 step 數平均

### 4.4 預期結果（reality check）

| 實驗 | Final loss | Win rate | Avg steps |
|---|---|---|---|
| naive_static | < 0.05 | ~100% | 3（最短路） |
| replay_static | < 0.05 | ~100% | 3 |
| replay_random | < 0.1 | 85–92%（書本 89.4%） | 4–6 |

如果結果與預期偏差超過 ±10%，先 debug：seed、ε 調度、replay buffer warmup、loss reduction 方式等。

### 4.5 動畫產出

每個實驗產一個 `dashboard.gif`，估計：
- 約 20 snapshots × (avg 4 game steps + 4 結果停留幀) ≈ 160 frames
- 5 fps → 32 秒 GIF
- 12×5 inch @ 80 dpi → ~3–5 MB / GIF

如過大，後備方案：snapshot 數降至 10、figsize 降至 8×4、imageio quantizer 使用 `'nq'`。

---

## 5. `report.md` 內容結構

```markdown
# HW3-1: Naive DQN for static mode
## DQN 與 Experience Replay 理解報告

> 作者：[使用者]｜課程：深度強化學習｜日期：2026-04-30
> Repo：<github 連結>

## 1. 作業目標         (~100 字)
## 2. 環境：Gridworld
   2.1 環境簡介
   2.2 三種 mode（直接重現截圖那張表）
## 3. MDP 建模
   3.1 State 表示法
   3.2 Action / Reward / Episode 終止
## 4. DQN 原理
   4.1 從 Q-learning 到 DNN 近似
   4.2 網路架構（64→150→100→4）
   4.3 ε-greedy 探索
## 5. Naive DQN（Listing 3.3）
   5.1 程式碼解讀（核心 inner loop ~25 行 + 行內註解）
   5.2 訓練結果（static mode）— 嵌入 loss.png + dashboard.gif
   5.3 為什麼 Naive 在 static 可以、random 會失敗
        （sample correlation + catastrophic forgetting）
## 6. Experience Replay Buffer（Listing 3.5）
   6.1 想解決的問題（呼應 5.3）
   6.2 程式碼解讀（deque buffer + 隨機 sample + vectorized target）
   6.3 訓練結果
        6.3.1 static mode（與 Naive 對比）
        6.3.2 random mode（Replay 真正發揮的地方）
## 7. 三種實驗對比
   （表格：Final Loss / Win rate / Avg steps + 一段討論）
## 8. 結論                (~80 字，連 Stage 2 預告)
```

**寫作風格決定**：
- 程式碼引用：只貼核心 10–25 行精華片段 + 行內中文註解；完整檔案在 `src/`
- 數字佔位符：`metrics.json` 跑出後用 `scripts/fill_metrics.py`（一個小工具）自動 substitute 模板裡的 `{NAIVE_STATIC_WIN_RATE}` 等 placeholder，避免 copy-paste 錯誤
- 圖片用相對路徑：`![](results/HW3-1/naive_static/loss.png)` — GitHub 與 VS Code preview 都能 render
- 語言：技術中文，必要時夾英文術語（Q-value, replay buffer, catastrophic forgetting...）
- **不含附錄**（使用者要求拿掉）

---

## 6. `chatlog.md` 設計

**檔名**：`chatlog.md`（repo root）

**內容範圍**：完整對話從第一句到最後一句
- 設計階段（腦力激盪）
- 實作階段（執行訓練、產生圖、debug、寫報告）
- 報告 review 階段（修改往返）

**格式**：
```markdown
# HW3-1: Naive DQN for static mode — Chat Log with Claude

> 完整保留與 Claude 對話的紀錄，作為作業 1_2「Chat with Claude
> about the code」的執行證據。
> 對話日期：2026-04-30
> 模型：Claude Opus 4.7 (1M context)

---

## Turn 1
**User：** [使用者訊息原文]
**Claude：** [回應原文，不含 tool output]
*(Claude 執行了 git clone + 讀取 Chapter 3 notebook 與 Gridworld.py)*

---

## Turn 2
...
```

**呈現原則**：
1. 保留所有 user 訊息原文（中文 / 錯字保留）
2. 保留 Claude 的文字回應原文，**不貼大段 tool output**（只保留執行摘要）
3. 截圖內容：使用者一開始貼的截圖，會在 chatlog 裡用 Markdown 表格還原資訊（mode 表格用原內容）
4. 重要 tool 動作仍標註：「執行 `python -m src.dqn_naive`，跑了 47 秒」

**生成時機**：整個作業最後一步才寫 — 因為要包含所有對話。直接從 Claude session 記憶 reconstruct，不編造。

---

## 7. 交付物清單（push 到 GitHub）

1. `src/`（6 個 .py + `__init__.py`）
2. `results/HW3-1/`（3 個實驗的完整產物：loss.png、losses.npy、dashboard.gif、metrics.json、checkpoint.pth、snapshots/）
3. `report.md`（中文，章節 1–8，無附錄）
4. `chatlog.md`（中文，完整對話紀錄）
5. `README.md`（最簡 placeholder，**內容由使用者後續指定**）
6. `requirements.txt`、`pyproject.toml`、`.python-version`、`.gitignore`、`LICENSE`
7. `docs/superpowers/specs/2026-04-30-hw3-dqn-stage1-design.md`（本文件）

---

## 8. Out of scope（Stage 1 不做）

- Target Network（Listing 3.7）— 留給 Stage 2
- Double DQN / Dueling DQN / Prioritized Replay — 留給 Stage 2/3
- 任何 hyperparameter sweep / ablation
- 跑 `mode='player'` 的實驗（需求只要求 static + 隱含的 random for replay）
- GPU / MPS 加速（網路太小，CPU 已足夠）
- 任何超出書本 Listing 3.3 / 3.5 的演算法改動
