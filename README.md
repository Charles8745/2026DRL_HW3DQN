# HW3：Naive DQN → Enhanced DQN Variants 於 Gridworld 環境

> **課程**：深度強化學習 HW3 — DQN and its variants
> **階段**：HW3-1 ✅ + HW3-2 ✅ + **HW3-3 ✅（本次更新）**
> **Repo**：https://github.com/Charles8745/2026DRL_HW3DQN

## 簡介

本專案在 4×4 Gridworld 環境上分階段實作並交叉比較 DQN 系列演算法：

**HW3-1：Naive DQN vs Experience Replay**（於 `static` 與 `random` mode）
- **Naive DQN**（無 Replay）— 每個 transition 立即更新一次
- **DQN + Experience Replay** — 環形 buffer + minibatch 隨機抽樣

**HW3-2：Enhanced DQN Variants for `player` mode**
- **Double DQN**（Hasselt 2016）— online 選動作 + target 估值，控制 Q 值高估
- **Dueling DQN**（Wang 2016）— 網路拆 V(s) + A(s,a) 兩支 + mean-baseline aggregation
- **Double + Dueling 合併** — 兩種改進正交疊加，預期最佳

**HW3-3：Lightning-converted DQN + Training Tricks for `random` mode**（本次更新內容）
- **PyTorch → PyTorch Lightning 移植** — Combined backbone 重用既有 `build_dueling_model`，三個 tricks 全 declarative
- **Trick A：Gradient norm clipping**（max_norm=10.0，作用點：grad）
- **Trick B：CosineAnnealingLR**（eta_min=1e-5，作用點：lr）
- **Trick C：Huber loss / SmoothL1**（作用點：loss function）
- **5 組消融**：baseline / clip / sched / huber / all_tricks，全部跑 random mode

四種 HW3-2 變體共用同一份 baseline（重跑 HW3-1 的 `dqn_replay.py`），保證對照公平；改進分別作用在「target 計算」與「網路結構」兩個獨立維度。

## 三階段一覽

整個 HW3 是一條「baseline → 演算法升級 → 框架升級 + tricks」的進化線。每階段都帶來具體的可量化改進，下表挑出每階段最具代表性的實驗對照：

| Stage | 環境 | 代表實驗 | Win rate | Loss (mean ± std) | 訓練時間 | 一句話定位 |
|---|---|---|---|---|---|---|
| **HW3-1** | `random` | DQN + Replay | 85.5% | 0.0613 ± 0.0573 | 18.11s | 課程 baseline；非平穩棋盤是真痛點 |
| **HW3-2** | `player` | Combined (Double+Dueling) | 100.0% | **0.00032 ± 0.00010** | 12.87s | 4 變體中 loss 最穩；player mode 對勝率太簡單，差異看 loss |
| **HW3-3** | `random` | Lightning Combined（無 tricks） | **88.0%** | **0.0315 ± 0.0174** | 31.87s | Lightning 移植 + Combined 直接把 HW3-1 random 的 loss mean ↓50%、std ↓70% |

從 HW3-1 → HW3-3 在 random mode 上的整體進步：win rate **85.5% → 88.0%**、loss mean **0.061 → 0.032**、loss std **0.057 → 0.017**。這個進步幾乎全來自 HW3-2 的演算法升級（Double + Dueling）+ 框架轉換的工程整合，**不是來自 HW3-3 加的三個 training tricks** — 詳細的反直覺結論見下方 HW3-3 段落與 [`HW3_3_report.md`](HW3_3_report.md)。

### 三段策略動畫對照

每階段挑一個冠軍實驗，左到右依序是 HW3-1、HW3-2、HW3-3：

| HW3-1: Replay (random)<br/>85.5% win rate | HW3-2: Combined (player)<br/>100% win rate | HW3-3: Lightning Combined (random)<br/>88.0% win rate |
|---|---|---|
| ![](results/HW3-1/replay_random/dashboard.gif) | ![](results/HW3-2/combined_player/dashboard.gif) | ![](results/HW3-3/baseline_random/dashboard.gif) |

GIF 左半 = 4×4 棋盤上 agent greedy rollout、右半 = training loss 曲線（紅實線：訓練到當前 epoch、紅虛線：snapshot 位置、灰線：完整未來）。HW3-1 random 的 loss 抖動明顯比 HW3-3 大，HW3-2 player 的 loss 平滑到幾乎看不到 — 三段一目了然。

## 特色

- 從 *Deep Reinforcement Learning in Action*（Manning, 2020）Chapter 3 移植的 Gridworld 環境（邏輯零改動，僅加 attribution）
- 兩個共用網路工廠：`build_model`（HW3-1 / Double DQN，`64→150→100→4`）+ `build_dueling_model`（HW3-2 / Dueling、Combined，trunk + V/A 雙頭 + mean-baseline）
- 五個獨立可執行的訓練腳本（`dqn_naive`、`dqn_replay`、`dqn_double`、`dqn_dueling`、`dqn_double_dueling`），各帶完整 CLI；每個變體一個檔案，diff 自 baseline 一目了然
- 訓練過程每 50 / 150 / 250 epochs 自動儲存 model snapshot；`animate.py` 後端用 model factory dispatch 自動為 Sequential / DuelingMLP 載入對應網路
- 每組實驗自動產出 loss 曲線（PNG）、勝率/步數統計（JSON）、Dashboard 動畫（GIF）
- 49 個 pytest 測試覆蓋環境、模型（含 DuelingMLP zero-mean 性質）、工具、5 種訓練、動畫各模組

## 專案結構

- `src/`：所有原始碼
  - `gridboard.py` / `gridworld_env.py`：Gridworld 環境（從書 Ch.3 移植）
  - `model.py`：`build_model` + `DuelingMLP` / `build_dueling_model`
  - `utils.py`：種子設定、state encoding、ε-greedy、test_model、evaluate
  - `dqn_naive.py`：Naive DQN 訓練 + CLI（HW3-1，Listing 3.3）
  - `dqn_replay.py`：DQN + Experience Replay 訓練 + CLI（HW3-1，Listing 3.5；HW3-2 baseline 也用此檔重跑）
  - `dqn_double.py`：Double DQN 訓練 + CLI（HW3-2）
  - `dqn_dueling.py`：Dueling DQN 訓練 + CLI（HW3-2）
  - `dqn_double_dueling.py`：Double + Dueling 合併訓練 + CLI（HW3-2）
  - `dqn_lightning.py`：PyTorch Lightning 版 Combined DQN + 3 tricks + CLI（HW3-3）
  - `animate.py`：Dashboard GIF 生成（model factory dispatch 支援兩種網路）
- `tests/`：pytest 測試（10 個檔案，49 個測試）
- `results/HW3-1/`：HW3-1 三組實驗的訓練產物（`naive_static/`、`replay_static/`、`replay_random/`）
- `results/HW3-2/`：HW3-2 四組實驗的訓練產物（`replay_player/`、`double_player/`、`dueling_player/`、`combined_player/`）
- `results/HW3-3/`：HW3-3 五組實驗的訓練產物（`baseline_random/`、`clip_random/`、`sched_random/`、`huber_random/`、`full_random/`）
- `docs/superpowers/`：兩階段的設計文件（`specs/`）與實作計畫（`plans/`）
- [`HW3_1_report.md`](HW3_1_report.md)：HW3-1 中文短報告（含完整原理推導與程式碼解讀）
- [`HW3_2_report.md`](HW3_2_report.md)：HW3-2 中文短報告（Double / Dueling / Combined 原理與比較）
- [`HW3_3_report.md`](HW3_3_report.md)：HW3-3 中文短報告（Lightning 移植 + 三個 tricks 消融）
- [`chatlog.md`](chatlog.md)：HW3-1 與 Claude 的完整對話紀錄
- [`chatlog2.md`](chatlog2.md)：HW3-2 與 Claude 的完整對話紀錄
- [`chatlog3.md`](chatlog3.md)：HW3-3 與 Claude 的完整對話紀錄

## HW3-1 分析結果

### 1. 訓練 Loss 曲線

**Naive DQN（static mode）**
![Naive Loss](results/HW3-1/naive_static/loss.png)

**DQN + Replay（static mode）**
![Replay-static Loss](results/HW3-1/replay_static/loss.png)

**DQN + Replay（random mode）**
![Replay-random Loss](results/HW3-1/replay_random/loss.png)

**觀察**：在相同 static 環境下，Naive 的 final loss mean = 0.006、std = 0.010；Replay 版本的 mean = 0.0014（小一個數量級）、std = 0.0003（小 30 倍以上）。Random mode 下因為棋盤分布持續變化，loss 整體偏大（mean 0.0613、std 0.0573），但仍持續收斂。

**解釋**：Replay Buffer 透過 minibatch 隨機抽樣，把單一 transition 帶來的高 variance 平均掉，使梯度估計更穩定。在 random mode 下 loss 偏大是因為網路要同時擬合多種棋盤佈局；最終 loss 仍下降證明 Replay 成功對抗了非平穩分布。

### 2. 學到的策略行為（量化指標）

| 實驗 | Mode | 方法 | Win Rate（1000 場 test） | 平均勝場步數 | Final Loss (mean ± std) | 訓練時間 |
|---|---|---|---|---|---|---|
| 1 | static | Naive DQN | **100.0%** | 7.0（最短路） | 0.006 ± 0.010 | 4.26s |
| 2 | static | DQN + Replay | **100.0%** | 7.0 | 0.0014 ± 0.0003 | 5.26s |
| 3 | random | DQN + Replay | **85.5%** | 2.56 | 0.0613 ± 0.0573 | 18.11s |

**觀察**：在 static mode 下，Naive 與 Replay 皆達到 100% 勝率、且都走出 7 步最短路。Random mode 的 Replay 達到 85.5% 勝率（書本 baseline 89.4%）；平均勝場步數降至 2.56，遠低於 static 的 7.0。

**解釋**：static 下兩種方法表現相同是合理的 — 棋盤永遠相同，沒有樣本相關性問題、也沒有 catastrophic forgetting；Replay 的優勢主要展現在 loss 穩定度而非勝率本身。Random mode 的 `avg_steps = 2.56` 看似「跑得更快」，實則是隨機棋盤常將 Player 與 Goal 放在相鄰格，這些「容易場」拉低平均；真正困難的初始配置（Goal 被 Wall 半包圍、Player 旁邊就是 Pit）多半落在 14.5% 的失敗組裡。

> **註**：static mode 的最短路為 **7 步**而非 3 步，因為 Pit 位於 (0,1) 阻斷了 (0,3)→(0,0) 的直線，agent 必須繞行第 2 列：down→down→left→left→left→up→up。

### 3. 策略動畫（Dashboard GIF）

每個 GIF 同時呈現兩個面向：**左側**是 4×4 Gridworld 上 agent 的實時走位（P 藍、+ 綠、− 紅、W 灰）；**右側**是訓練 loss 曲線（灰線 = 完整未來曲線、紅線 = 已訓練到當前 epoch、紅虛線 = 當前 snapshot 的 epoch 位置）。

**Naive DQN（static mode）**
![Naive Dashboard](results/HW3-1/naive_static/dashboard.gif)

**DQN + Replay（static mode）**
![Replay-static Dashboard](results/HW3-1/replay_static/dashboard.gif)

**DQN + Replay（random mode）**
![Replay-random Dashboard](results/HW3-1/replay_random/dashboard.gif)

**觀察**：static mode 動畫中，agent 在訓練早期（epoch 0–200）會嘗試走捷徑撞到 Pit/Wall，到 epoch 500 後穩定走出 7 步繞行最短路。Random mode 動畫中，每個 snapshot 對應不同的隨機初始棋盤（用 `random.seed(snap_epoch)` 確保可重現），agent 隨訓練進度逐步學會在多種佈局下都能找到 Goal。

**解釋**：動畫直觀呈現了「策略隨訓練進步」的過程：早期 ε 高 → 隨機亂走或誤判；中期 → 局部正確但偶爾走錯；後期 → 穩定取得最大累積 reward。Loss 曲線同步往下走則印證了 Q 函數估計的收斂與策略改善是一體兩面。

## HW3-2 分析結果

### 0. 環境與變體簡介

**`player` mode 為何位於 static / random 之間**：HW3-2 全部 4 組實驗都跑 `mode='player'`，即 Goal/Pit/Wall 三者位置固定不變、僅 Player 起點隨機（共 13 種有效起點）。狀態空間複雜度比 static（單一棋盤）高一個數量級，但比 random（全隨機，數百種有效棋盤）低；非常適合用來「拉開變體之間的差異而又不被噪音淹沒」。

**4 組實驗的對應關係**：

| 實驗 | 演算法 | 改進於 baseline 的點 |
|---|---|---|
| `replay_player` | DQN + Replay（重跑 HW3-1） | 無，作為對照基準 |
| `double_player` | Double DQN | 引入 Target Network；用 online net 選動作、target net 算價值，解 Q 值高估 |
| `dueling_player` | Dueling DQN | 網路拆 V(s) + A(s,a) 兩支 + mean-baseline aggregation，解 sample inefficiency |
| `combined_player` | Double + Dueling | 兩種改進正交疊加 |

所有 4 組共用相同 hyperparameters：`epochs=3000, gamma=0.9, lr=1e-3, mem_size=1000, batch_size=200, max_moves=50, epsilon=0.3, seed=42`；Double / Combined 額外用 `sync_freq=500`（hard target sync）。

### 1. 訓練 Loss 曲線

**Baseline (DQN+Replay)**
![baseline loss](results/HW3-2/replay_player/loss.png)

**Double DQN**
![double loss](results/HW3-2/double_player/loss.png)

**Dueling DQN**
![dueling loss](results/HW3-2/dueling_player/loss.png)

**Double + Dueling**
![combined loss](results/HW3-2/combined_player/loss.png)

**觀察**：Final loss mean 排序為 **Combined (0.00032) < Double (0.00050) < baseline (0.00117) < Dueling (0.00528)** — 跨越一個數量級。Double 比 baseline 低 ~57%、std 從 0.00046 降到 0.00008（少 5.7 倍）；Combined 又比單獨 Double 再低 36%；Dueling 反而比 baseline 高 ~4.5 倍且 std 也最大（0.00500）。

**解釋**：Double DQN 的 target 網路解開「選動作」與「估值」的耦合，避免 max 操作把估計誤差正向放大，因此 loss 平穩 — 這是文獻中標準觀察。Dueling 的 loss 偏大則反映「沒有 target net」+「更深網路」的代價：V/A 雙頭會讓單筆 update 影響更多參數，沒有 target stabilization 時抖動更明顯。Combined 同時取得兩者的好處：target net 提供 update 穩定性，V/A 結構提供 sample efficiency，因此達到最低、最穩定的 loss。

### 2. 學到的策略行為（量化指標）

| 實驗 | Method | Win Rate（1000 場 test） | 平均勝場步數 | Final Loss (mean ± std) | 訓練時間 |
|---|---|---|---|---|---|
| baseline_player | replay | **100.0%** | 4.359 | 0.00117 ± 0.00046 | 9.28s |
| double_player | double | **100.0%** | 4.389 | 0.00050 ± 0.00008 | 10.33s |
| dueling_player | dueling | **100.0%** | **4.317**（最短） | 0.00528 ± 0.00500 | 11.08s |
| combined_player | double_dueling | **100.0%** | 4.393 | **0.00032 ± 0.00010**（最穩） | 12.87s |

**觀察**：4 種方法在 player mode 都收斂到 100% win rate；勝率本身完全分不出變體優劣。差異主要展現在兩個維度 — **loss 穩定度**（Combined 最穩、Dueling 最不穩）與 **avg_steps_per_win**（Dueling 反而最少 4.317，比其它三者少 ~0.07 步）。訓練時間方面 Double / Combined 因多一份網路前向約多 11–39%，Dueling 因 trunk 共用只多 ~19%。

**解釋**：player mode 的 13 種起點對 3000 epochs 而言「太簡單」，所有合理變體都能 100% 解決 — 這提醒我們，在簡單環境下「能不能解決問題」不是好的區分指標，要看 loss 收斂的平穩度與 sample efficiency。Dueling 的「avg_steps 最少」是個有趣副作用：V/A 拆解讓網路在學會「state 價值」之後更積極往高 V 方向走，傾向找到較短路徑；代價是訓練過程的 loss 穩定度下降。Combined 證明了 Double（target 算法層面）與 Dueling（網路結構層面）兩個改進的正交性 — 它們作用點不重疊，可以乾淨疊加。

### 3. 策略動畫（Dashboard GIF）

| Baseline | Double DQN |
|---|---|
| ![](results/HW3-2/replay_player/dashboard.gif) | ![](results/HW3-2/double_player/dashboard.gif) |

| Dueling DQN | Double + Dueling |
|---|---|
| ![](results/HW3-2/dueling_player/dashboard.gif) | ![](results/HW3-2/combined_player/dashboard.gif) |

每個 GIF 顯示 21 個 snapshot（epoch 0、150、300、…、3000）下的 greedy rollout：每個 snapshot 用 `random.seed(snap_epoch)` 重置 player 起點以確保跨變體可比較。

**觀察**：早期 snapshot（epoch 0–450）4 個變體都看似亂走 — 因為網路尚未學到 Q 函數。Combined 在 epoch ~600 即穩定走出最短路，Double 約在 epoch ~750 跟上，baseline 與 Dueling 約在 epoch ~900–1050。後期所有變體都能在多種起點下穩定到達 Goal，與 100% win rate 一致。Loss 曲線（右半部）同步往下：Combined 的曲線最平滑、Dueling 最抖動，與量化指標一致。

**解釋**：動畫直觀呈現了「兩個改進如何加快策略形成」 — Double 的 target 網路讓 Q 值估計更穩定，因此策略提早成形；Dueling 透過共用 V(s) 加速「身處此 state 該往哪個方向走」的學習；兩者疊加時策略最早穩定、最少抖動。同時，動畫也驗證了量化指標的解釋 — Dueling 變體在中期確實有「找到捷徑但偶爾撞牆」的探索行為，這對應其較高的 loss 標準差。

## HW3-3 分析結果

HW3-3 把 HW3-2 最強的 Combined backbone 移植到 PyTorch Lightning，並在 `random` mode 上加三個 training tricks（gradient clipping、CosineAnnealingLR、Huber loss）做 5 組消融。「為何回到 random mode？」因為 player mode 上 4 個 HW3-2 變體都已 100%、看不出差異；要量化 tricks 的價值就得回到還沒解決完的 random mode。

### 1. 訓練 Loss 曲線

| Baseline（無 tricks） | +Gradient Clipping |
|---|---|
| ![Baseline Loss](results/HW3-3/baseline_random/loss.png) | ![Clip Loss](results/HW3-3/clip_random/loss.png) |

| +CosineAnnealingLR | +Huber Loss |
|---|---|
| ![Sched Loss](results/HW3-3/sched_random/loss.png) | ![Huber Loss](results/HW3-3/huber_random/loss.png) |

| +All Tricks | |
|---|---|
| ![Full Loss](results/HW3-3/full_random/loss.png) | |

### 2. 量化指標（5 組對比）

| 變體 | Method | Final Loss (mean ± std) | Win Rate | Avg Steps | Wall time |
|---|---|---|---|---|---|
| baseline | lightning_combined | **0.03151 ± 0.01744** | **88.0%** | 2.62 | 31.87s |
| clip | lightning_combined +clip | 0.25081 ± 0.14592 | 85.2% | 2.61 | 34.13s |
| sched | lightning_combined +sched | 1.52610 ± 0.29979 | 87.4% | 2.64 | 32.28s |
| huber | lightning_combined +huber | 0.03216 ± 0.01878 | 81.0% | 2.64 | 34.77s |
| full | lightning_combined +clip+sched+huber | 0.48854 ± 0.04443 | 86.3% | 2.61 | 36.94s |

> **註**：表中所有數值直接讀自 `results/HW3-3/<tag>_random/metrics.json`。

**重要觀察**：實測結果是 **baseline > 任何加 trick 的版本**。這個與直覺相反的結果出現的原因有三：

1. **Combined backbone 已接近 ceiling**：HW3-1 replay_random 在 random mode 是 85.5%（loss 0.0613±0.0573）；換成 Lightning Combined 直接拉到 88.0%（loss 0.0315±0.0174）— **loss mean ↓50%、std ↓70%**。Double + Dueling 已抑制 Q 值高估與 sample inefficiency 兩個 random mode 的主要痛點，後續 tricks 邊際效用變得很小。
2. **Hyperparameter scale mismatch**：DQN 文獻常用的 `max_norm=10.0` / `eta_min=1e-5` / Huber `β=1.0` 是 Atari CNN（百萬參數）規模的 default；本作業是 30K 參數的小 MLP，gradient 分布、loss landscape 都不同，default 不適合。
3. **5000 epochs 對 cosine 退火不夠長**：CosineAnnealingLR 在最後 1000 epochs 把 lr 退到 1e-5，模型實際上停止學習，loss 在後段累積（這就是 sched 的 loss=1.526 比 baseline 高 50 倍的原因）。

換句話說：**Lightning + Combined 的「框架轉換 + 演算法升級」本身就是 HW3-3 真正的 win**；trick 沒生效不是 bug，是 hyperparameter 沒做尺度調整。

### 3. 策略動畫

| Baseline | +Gradient Clipping |
|---|---|
| ![](results/HW3-3/baseline_random/dashboard.gif) | ![](results/HW3-3/clip_random/dashboard.gif) |

| +CosineAnnealingLR | +Huber Loss |
|---|---|
| ![](results/HW3-3/sched_random/dashboard.gif) | ![](results/HW3-3/huber_random/dashboard.gif) |

| +All Tricks | |
|---|---|
| ![](results/HW3-3/full_random/dashboard.gif) | |

> **詳細分析**：見 [HW3_3_report.md](HW3_3_report.md)。

## 安裝

需要 **Python 3.12**（PyTorch 對 3.13 / 3.14 的支援尚未完整）。本專案使用 [`uv`](https://github.com/astral-sh/uv) 作為套件管理器以加速 venv 建立與相依鎖定。

```bash
# 1. 安裝 uv（若尚未安裝）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 取得專案
git clone https://github.com/Charles8745/2026DRL_HW3DQN.git
cd 2026DRL_HW3DQN

# 3. 建立 venv 並安裝相依套件
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

## 使用方式

### 執行訓練實驗

```bash
source .venv/bin/activate

# HW3-1 (static / random mode)
python -m src.dqn_naive  --mode static --epochs 1000 --seed 42   # Naive DQN, static
python -m src.dqn_replay --mode static --epochs 1000 --seed 42   # DQN + Replay, static
python -m src.dqn_replay --mode random --epochs 5000 --seed 42   # DQN + Replay, random

# HW3-2 (player mode)
python -m src.dqn_replay         --mode player --epochs 3000 --seed 42 \
       --snapshot-every 150 --out-dir results/HW3-2/replay_player
python -m src.dqn_double         --mode player --epochs 3000 --seed 42
python -m src.dqn_dueling        --mode player --epochs 3000 --seed 42
python -m src.dqn_double_dueling --mode player --epochs 3000 --seed 42
```

每組訓練的產物會寫入 `results/HW3-{1,2}/<exp>/`（包含 `loss.png`、`metrics.json`、`checkpoint.pth`、`losses.npy`、`snapshots/`）。

### 生成 Dashboard 動畫 GIF

```bash
# HW3-1
python -m src.animate --exp naive_static
python -m src.animate --exp replay_static
python -m src.animate --exp replay_random

# HW3-2
python -m src.animate --exp replay_player
python -m src.animate --exp double_player
python -m src.animate --exp dueling_player
python -m src.animate --exp combined_player
```

`animate.py` 會根據 exp 名稱自動選擇對應的 model factory（HW3-2 的 dueling / combined 用 `build_dueling_model`，其它用 `build_model`）與輸出目錄前綴。

### 執行測試

```bash
pytest -v
```

預期 49 個測試全綠（HW3-1 24 + HW3-2 9 + HW3-3 16）。

## 設定

主要 hyperparameters 可透過 CLI flag 調整：

**`src/dqn_naive.py`**
- `--mode`：`static` / `player` / `random`
- `--epochs`、`--gamma`、`--lr`、`--seed`、`--snapshot-every`、`--out-dir`
- `--epsilon-start`、`--epsilon-end`（線性衰減 ε）

**`src/dqn_replay.py`**
- 上述全部（除 `--epsilon-start/end` 改為單一 `--epsilon`）
- `--mem-size`、`--batch-size`、`--max-moves`

**`src/dqn_double.py` / `src/dqn_double_dueling.py`**
- 同 `dqn_replay.py`，外加 `--sync-freq`（hard target sync 頻率，預設 500 個 training steps）

**`src/dqn_dueling.py`**
- 同 `dqn_replay.py`，無 `--sync-freq`（不使用 target network）

網路架構由 `src/model.py` 控制：
- `build_model(in_dim, hidden1, hidden2, out_dim)` — Sequential MLP，預設 `64→150→100→4`（HW3-1 / Double DQN 使用）
- `build_dueling_model(in_dim, hidden1, hidden2, n_actions)` — `DuelingMLP` 含共用 trunk + V/A 雙頭 + mean-baseline aggregation（HW3-2 Dueling / Combined 使用）

Gridworld 棋盤大小與獎勵值寫死在 `src/gridworld_env.py` 內（為保持與 DRL in Action Ch.3 一致；如需修改可直接編輯）。

## 後續階段

| Stage | 主題 | 狀態 |
|---|---|---|
| **HW3-1**：Naive DQN for static mode | Naive DQN + Experience Replay 對比（[`HW3_1_report.md`](HW3_1_report.md)） | ✅ 已完成 |
| **HW3-2**：Enhanced DQN Variants for player mode | Double DQN + Dueling DQN + 兩者合併（[`HW3_2_report.md`](HW3_2_report.md)） | ✅ 已完成 |
| **HW3-3**：Framework conversion + training tricks | PyTorch Lightning + grad clipping / cosine sched / Huber loss（[`HW3_3_report.md`](HW3_3_report.md)） | ✅ 已完成 |

## 授權

本專案以 MIT License 授權。`src/gridboard.py` 與 `src/gridworld_env.py` 改編自 *Deep Reinforcement Learning in Action* 第 3 章（Alexander Zai、Brandon Brown，Manning 2020），原作者版權歸原作者所有。詳見 [LICENSE](LICENSE)。
