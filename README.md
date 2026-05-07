# HW3：Naive DQN → Enhanced DQN Variants 於 Gridworld 環境

> **課程**：深度強化學習 HW3 — DQN and its variants
> **階段**：HW3-1 ✅ + HW3-2 ✅ + HW3-3 ✅ + **HW3-4 ✅（本次更新，加分題）**
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

**HW3-3：Lightning-converted DQN + Training Tricks for `random` mode**
- **PyTorch → PyTorch Lightning 移植** — Combined backbone 重用既有 `build_dueling_model`，三個 tricks 全 declarative
- **Trick A：Gradient norm clipping**（max_norm=10.0，作用點：grad）
- **Trick B：CosineAnnealingLR**（eta_min=1e-5，作用點：lr）
- **Trick C：Huber loss / SmoothL1**（作用點：loss function）
- **5 組消融**：baseline / clip / sched / huber / all_tricks，全部跑 random mode

**HW3-4：Rainbow DQN for `random` mode**（本次更新內容，加分題）
- **完整 Rainbow（6 元件）** — Double + Dueling（沿用 HW3-2）+ PER + N-step + C51 distributional + Noisy Nets
- **2 組實驗**：`combined_random`（HW3-3 baseline 重跑）+ `rainbow_random`（主角）
- **意外的負面結果**：完整 Rainbow 收 21.7% win rate，比 baseline 88% 大跌 — 4×4 Gridworld 太小、6 元件互相干擾，反而拖垮基礎演算法（細節見 [`HW3_4_report.md`](HW3_4_report.md)）

四個 HW3-2 變體共用一份 baseline（直接重跑 HW3-1 的 `dqn_replay.py`），這樣 Double 和 Dueling 各自的影響才不會混在一起。

## 四階段一覽

四階段各自跑了不同的東西。下表各挑一個代表實驗對照：

| Stage | 環境 | 代表實驗 | Win rate | Loss (mean ± std) | 訓練時間 | 一句話定位 |
|---|---|---|---|---|---|---|
| **HW3-1** | `random` | DQN + Replay | 85.5% | 0.0613 ± 0.0573 | 18.11s | 課程 baseline；非平穩棋盤是真痛點 |
| **HW3-2** | `player` | Combined (Double+Dueling) | 100.0% | **0.00032 ± 0.00010** | 12.87s | 4 變體中 loss 最穩；player mode 對勝率太簡單，差異看 loss |
| **HW3-3** | `random` | Lightning Combined（無 tricks） | **88.0%** | **0.0315 ± 0.0174** (MSE) | 31.87s | Lightning 移植 + Combined 直接把 HW3-1 random 的 loss mean ↓50%、std ↓70% |
| **HW3-4** | `random` | Rainbow（6 元件全套） | **21.7%** | 0.6409 ± 0.2229 (KL) | 511.82s | 過度工程的反例：4×4 Gridworld 太小、Rainbow 反而崩塌 |

同樣 random mode，HW3-1 → HW3-3：win rate 85.5% → 88.0%、loss mean 0.061 → 0.032。HW3-3 → HW3-4 嘗試再加 4 個元件（PER + N-step + C51 + Noisy）反而跌到 21.7%——比隨機亂走（~25–30%）還差。完整原因分析在 [`HW3_4_report.md`](HW3_4_report.md)，但一句話：**演算法複雜度與任務規模要匹配**，Atari 上 6 元件互相加乘的 Rainbow，搬到 4×4 Gridworld 上反而互相干擾。

### 四段策略動畫對照

每階段挑一個代表實驗：

| HW3-1: Replay (random)<br/>85.5% win rate | HW3-2: Combined (player)<br/>100% win rate |
|---|---|
| ![](results/HW3-1/replay_random/dashboard.gif) | ![](results/HW3-2/combined_player/dashboard.gif) |

| HW3-3: Lightning Combined (random)<br/>88.0% win rate | HW3-4: Rainbow (random)<br/>21.7% win rate |
|---|---|
| ![](results/HW3-3/baseline_random/dashboard.gif) | ![](results/HW3-4/rainbow_random/dashboard.gif) |

GIF 左半是 agent 在 4×4 棋盤上的 greedy rollout，右半是 loss 曲線（紅實線是訓練到當前 epoch、紅虛線是 snapshot 位置、灰線是整段曲線）。HW3-1 random 那條明顯抖、HW3-2 player 那條幾乎是直線、HW3-3 random 介於兩者之間，HW3-4 那條 KL loss 跟前三條 MSE loss 量綱不可直接比，但 agent 在棋盤上撞牆走 pit 的行為就明顯不是學成的策略。

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
  - `rainbow.py`：完整 Rainbow DQN（NoisyLinear + DistributionalDuelingMLP + SumTree + PER + NStepBuffer + categorical projection + 訓練迴圈 + CLI，單檔 ~700 LOC）（HW3-4）
  - `animate.py`：Dashboard GIF 生成（model factory dispatch 支援三種網路）
- `tests/`：pytest 測試（11 個檔案，71 個測試）
- `results/HW3-1/`：HW3-1 三組實驗的訓練產物（`naive_static/`、`replay_static/`、`replay_random/`）
- `results/HW3-2/`：HW3-2 四組實驗的訓練產物（`replay_player/`、`double_player/`、`dueling_player/`、`combined_player/`）
- `results/HW3-3/`：HW3-3 五組實驗的訓練產物（`baseline_random/`、`clip_random/`、`sched_random/`、`huber_random/`、`full_random/`）
- `results/HW3-4/`：HW3-4 兩組實驗的訓練產物（`combined_random/`、`rainbow_random/`）
- `docs/superpowers/`：四階段的設計文件（`specs/`）與實作計畫（`plans/`）
- [`HW3_1_report.md`](HW3_1_report.md)：HW3-1 中文短報告（含完整原理推導與程式碼解讀）
- [`HW3_2_report.md`](HW3_2_report.md)：HW3-2 中文短報告（Double / Dueling / Combined 原理與比較）
- [`HW3_3_report.md`](HW3_3_report.md)：HW3-3 中文短報告（Lightning 移植 + 三個 tricks 消融）
- [`HW3_4_report.md`](HW3_4_report.md)：HW3-4 中文報告（完整 Rainbow + 失敗模式分析）
- [`chatlog.md`](chatlog.md)：HW3-1 與 Claude 的完整對話紀錄
- [`chatlog2.md`](chatlog2.md)：HW3-2 與 Claude 的完整對話紀錄
- [`chatlog3.md`](chatlog3.md)：HW3-3 與 Claude 的完整對話紀錄
- [`chatlog4.md`](chatlog4.md)：HW3-4 與 Claude 的完整對話紀錄

## HW3-1 分析結果

### 1. 訓練 Loss 曲線

**Naive DQN（static mode）**
![Naive Loss](results/HW3-1/naive_static/loss.png)

**DQN + Replay（static mode）**
![Replay-static Loss](results/HW3-1/replay_static/loss.png)

**DQN + Replay（random mode）**
![Replay-random Loss](results/HW3-1/replay_random/loss.png)

**觀察**：static 環境下 Naive 的 final loss mean = 0.006、std = 0.010；Replay 版本 mean = 0.0014（小一個數量級）、std = 0.0003（小 30 倍以上）。Random mode 下棋盤分布每局都變，loss 整體偏大（mean 0.0613、std 0.0573），但仍一路往下走。

**解釋**：Replay 從 buffer 隨機抽 minibatch、抽多了，單筆 transition 帶來的 noise 自然被平均掉。Random mode 的 loss 偏大則是因為網路得同時 fit 各種棋盤；loss 仍有下降代表 Replay 在非平穩分布上仍然 work。

### 2. 學到的策略行為（量化指標）

| 實驗 | Mode | 方法 | Win Rate（1000 場 test） | 平均勝場步數 | Final Loss (mean ± std) | 訓練時間 |
|---|---|---|---|---|---|---|
| 1 | static | Naive DQN | **100.0%** | 7.0（最短路） | 0.006 ± 0.010 | 4.26s |
| 2 | static | DQN + Replay | **100.0%** | 7.0 | 0.0014 ± 0.0003 | 5.26s |
| 3 | random | DQN + Replay | **85.5%** | 2.56 | 0.0613 ± 0.0573 | 18.11s |

**觀察**：static mode 下 Naive 跟 Replay 都到 100% 勝率，也都走出 7 步最短路。Random mode 的 Replay 拿到 85.5% 勝率（書本 baseline 是 89.4%），平均勝場步數降到 2.56，比 static 的 7.0 短很多。

**解釋**：static 下兩個方法表現一樣其實沒什麼意外 — 棋盤永遠固定，本來就沒有樣本相關性、沒有 catastrophic forgetting；Replay 的好處主要在 loss 穩定度而不是勝率。Random mode 的 `avg_steps = 2.56` 看起來「跑得更快」其實是錯覺：隨機棋盤常常把 Player 跟 Goal 放在相鄰格，那些 1–2 步就贏的「送分題」把平均拉低；真正難的局面（Goal 被 Wall 半包圍、Player 旁邊就是 Pit）多半進了那 14.5% 的失敗組。

> **註**：static mode 的最短路為 **7 步**而非 3 步，因為 Pit 位於 (0,1) 阻斷了 (0,3)→(0,0) 的直線，agent 必須繞行第 2 列：down→down→left→left→left→up→up。

### 3. 策略動畫（Dashboard GIF）

每個 GIF 同時呈現兩個面向：**左側**是 4×4 Gridworld 上 agent 的實時走位（P 藍、+ 綠、− 紅、W 灰）；**右側**是訓練 loss 曲線（灰線 = 完整未來曲線、紅線 = 已訓練到當前 epoch、紅虛線 = 當前 snapshot 的 epoch 位置）。

**Naive DQN（static mode）**
![Naive Dashboard](results/HW3-1/naive_static/dashboard.gif)

**DQN + Replay（static mode）**
![Replay-static Dashboard](results/HW3-1/replay_static/dashboard.gif)

**DQN + Replay（random mode）**
![Replay-random Dashboard](results/HW3-1/replay_random/dashboard.gif)

**觀察**：static mode 動畫裡，agent 早期（epoch 0–200）會嘗試走捷徑、被 Pit / Wall 擋下，epoch 500 後就穩定走出 7 步繞行最短路。Random mode 動畫每個 snapshot 對應不同的隨機初始棋盤（用 `random.seed(snap_epoch)` 確保可重現），agent 隨訓練進度逐步學會在多種佈局下都能找到 Goal。

**解釋**：動畫看下來就是「策略慢慢長出來」的過程 — 早期 ε 大、亂走或被 Pit 騙到；中期局部正確、偶爾還是走錯；後期就穩了。Loss 曲線同步往下走，Q 函數估計的收斂跟策略好不好其實是同一件事的兩個面向。

## HW3-2 分析結果

### 0. 環境與變體簡介

**`player` mode 為什麼挑這個**：HW3-2 4 組都跑 `mode='player'`，Goal / Pit / Wall 三者位置固定，只有 Player 起點隨機（共 13 種起點）。狀態空間比 static（一張棋盤）大一個數量級，比 random（幾百張）小一個數量級 — 剛好夠拉開變體之間的差異又不會被 noise 蓋過。

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

**觀察**：Final loss mean 從低到高是 **Combined (0.00032) < Double (0.00050) < baseline (0.00117) < Dueling (0.00528)**，最高跟最低差一個數量級。Double 比 baseline 低 57%、std 從 0.00046 降到 0.00008（小 5.7 倍）；Combined 又比 Double 再低 36%；Dueling 反而比 baseline 高 4.5 倍、std 也最大（0.00500）。

**解釋**：Double 把「選動作」跟「估值」拆給 online 跟 target 兩個網路，max 操作就不會把同一份估計誤差放大成 Q 高估，loss 自然平穩（這就是 Hasselt 2016 描述的標準效應）。Dueling 的 loss 偏大反映另一個 trade-off：沒有 target network、又把網路加深成 V/A 雙頭，每筆 update 影響更多參數、抖動更明顯。Combined 把兩個改進疊起來，target net 處理穩定性、V/A 結構處理 sample efficiency，loss 自然就是 4 組最低也最穩。

### 2. 學到的策略行為（量化指標）

| 實驗 | Method | Win Rate（1000 場 test） | 平均勝場步數 | Final Loss (mean ± std) | 訓練時間 |
|---|---|---|---|---|---|
| baseline_player | replay | **100.0%** | 4.359 | 0.00117 ± 0.00046 | 9.28s |
| double_player | double | **100.0%** | 4.389 | 0.00050 ± 0.00008 | 10.33s |
| dueling_player | dueling | **100.0%** | **4.317**（最短） | 0.00528 ± 0.00500 | 11.08s |
| combined_player | double_dueling | **100.0%** | 4.393 | **0.00032 ± 0.00010**（最穩） | 12.87s |

**觀察**：4 個方法在 player mode 都收 100% win rate，勝率完全分不出差異。能看出差異的是 loss 穩定度（Combined 最穩、Dueling 最抖）跟 avg_steps_per_win（Dueling 反而最短，4.317，比其它 3 個少 0.07 步）。訓練時間 Double 跟 Combined 因為多跑一份網路前向多了 11–39%，Dueling 因為 trunk 共用、只多 19%。

**解釋**：player mode 的 13 種起點對 3000 epochs 來說太簡單，所有合理變體都能 100% 解掉，所以勝率不是好的區分指標 — 看 loss 穩定度跟 sample efficiency 才有差異。Dueling 的「avg_steps 最少」是個意料之外的副作用：V/A 拆解讓網路一旦學到「state 價值」就更積極往高 V 方向走、自然找到比較短的路；代價是 loss 抖動。Combined 把 Double（改 target 算法）跟 Dueling（改網路結構）兩個改進疊起來剛好可以，因為作用點根本不重疊。

### 3. 策略動畫（Dashboard GIF）

| Baseline | Double DQN |
|---|---|
| ![](results/HW3-2/replay_player/dashboard.gif) | ![](results/HW3-2/double_player/dashboard.gif) |

| Dueling DQN | Double + Dueling |
|---|---|
| ![](results/HW3-2/dueling_player/dashboard.gif) | ![](results/HW3-2/combined_player/dashboard.gif) |

每個 GIF 顯示 21 個 snapshot（epoch 0、150、300、…、3000）下的 greedy rollout：每個 snapshot 用 `random.seed(snap_epoch)` 重置 player 起點以確保跨變體可比較。

**觀察**：早期 snapshot（epoch 0–450）4 個變體都還在亂走，網路根本還沒學到 Q 函數。Combined 在 epoch ~600 就穩定走出最短路，Double 約 epoch ~750 跟上，baseline 跟 Dueling 約 epoch ~900–1050 才到位。後期所有變體都能在多種起點下穩定到達 Goal，跟 100% win rate 對得上。右半的 loss 曲線同步往下走：Combined 最平滑、Dueling 最抖，跟量化指標相符。

**解釋**：動畫把「兩個改進怎麼加快策略形成」演了出來 — Double 的 target 網路讓 Q 估計穩定、策略提早成形；Dueling 共用 V(s)、加速「在這個 state 該往哪走」的學習；兩個一起用時策略最早穩、最少抖。Dueling 在中期常常「找到捷徑但偶爾撞牆」，這也跟它較高的 loss std 對得上。

## HW3-3 分析結果

HW3-3 把 HW3-2 最強的 Combined 移植到 PyTorch Lightning，再在 `random` mode 上加三個 training tricks（gradient clipping、CosineAnnealingLR、Huber loss）做 5 組消融。回到 random mode 是因為 player mode 上 4 個 HW3-2 變體都已經 100% 了，要量化 trick 的影響就只能回到還沒解完的 random mode。

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

結果跟預期相反：baseline 比任何加 trick 的版本都好。第一輪數據出來時我以為是實作 bug，重讀 metrics 跟訓練 log 才整理出原因，大致是三件事交織在一起。

一、Combined 大概已經摸到 4×4 Gridworld 的 ceiling 了。HW3-1 replay_random 在 random mode 是 85.5% / loss 0.0613±0.0573；換成 Lightning Combined 直接到 88.0% / loss 0.0315±0.0174，loss mean 降一半、std 降到三成。Double + Dueling 已經把 random mode 兩個主要痛點（Q 高估、sample inefficiency）處理掉，後面再加 trick 能撈的邊際本來就小。

二、超參尺度不對。`max_norm=10.0`、`eta_min=1e-5`、Huber `β=1.0` 都是 DQN 文獻寫給 Atari CNN（百萬參數）那種網路用的；這裡是 30K 參數的小 MLP，gradient norm 跟 loss landscape 完全不在同個量級。10.0 的 norm clip 對小網路反而會限制正常更新；1e-5 的 eta_min 在 1e-3 起步的小 lr 上幾乎是直接歸零。

三、5000 epochs 對 cosine 退火太短。CosineAnnealingLR 把最後 1000 個 epoch 的 lr 推到 1e-5，模型那段時間幾乎沒更新、loss 一路累積成那條 1.526 的曲線（看 sched 那欄就知道）。

換個角度看，HW3-3 真正的進步其實在 baseline 那行 — 同樣 random mode，loss 從 0.0613 砍到 0.0315、win rate 從 85.5% 到 88.0%。Lightning 移植加 Combined 演算法本身就把 baseline 拉起來了，trick 想再錦上添花、但這次沒添到。

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

## HW3-4 分析結果

HW3-4 把 Hessel et al. 2018 的 Rainbow（6 元件）完整實作出來，疊在 HW3-3 baseline 之上：除了 HW3-2 已經有的 Double + Dueling，再加 PER + N-step + C51 distributional + Noisy Nets。實驗共 2 組，全部跑 random mode、seed=42、5000 epochs。

### 1. 量化指標（2 組對比）

| 變體 | Method | Win Rate | Avg Steps | Final Loss (mean ± std) | Wall time |
|---|---|---|---|---|---|
| `combined_random` | lightning_combined | **88.0%** | 2.62 | 0.0315 ± 0.0174 (MSE) | 31.47s |
| `rainbow_random` | rainbow | **21.7%** | 2.05 | 0.6409 ± 0.2229 (KL) | 511.82s |

`combined_random` 是 HW3-3 Lightning Combined 直接重跑，重現 HW3-3 的 88.0% win rate，作為 HW3-4 的對比 baseline。`rainbow_random` 完整跑 6 元件，意外地 win rate 跌到 21.7%——不只低於 baseline 的 88%，連完全隨機行動的 ~25–30% 都不如。

### 2. Loss 曲線

| `combined_random`（MSE） | `rainbow_random`（KL） |
|---|---|
| ![](results/HW3-4/combined_random/loss.png) | ![](results/HW3-4/rainbow_random/loss.png) |

兩條曲線都單調下降（KL 從 ~4 收到 0.6），看起來訓練「正常」——但 win rate 顯示 Rainbow 收斂到一個錯的目標分佈。診斷發現 Q 值在不同動作之間幾乎一致（全部聚在 −9.85 附近）、return 分佈把 60–80% 機率質量壓在 z = −10 那個 atom 上，等於學會「不論做什麼下場都是死」。

### 3. 策略動畫

| `combined_random`（88.0%） | `rainbow_random`（21.7%） |
|---|---|
| ![](results/HW3-4/combined_random/dashboard.gif) | ![](results/HW3-4/rainbow_random/dashboard.gif) |

`combined_random` 跟 HW3-3 baseline 看起來一樣（早期亂走、後期穩定走最短路）；`rainbow_random` 則整個訓練過程中都在撞牆、走進 pit、或徘徊到 max_moves 截止——loss 在收斂、策略卻沒有收斂。

> **完整失敗模式分析、3 組 hyperparameter 嘗試、為什麼 Rainbow 在 4×4 Gridworld 失效的理論解釋**：見 [HW3_4_report.md](HW3_4_report.md)。

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

# HW3-4 (random mode)
python -m src.dqn_lightning --mode random --epochs 5000 --seed 42 \
       --snapshot-every 250 --out-dir results/HW3-4/combined_random
python -m src.rainbow                                         # rainbow defaults
```

每組訓練的產物會寫入 `results/HW3-{1,2,3,4}/<exp>/`（包含 `loss.png`、`metrics.json`、`checkpoint.pth`、`losses.npy`、`snapshots/`）。

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

# HW3-4
python -m src.animate --exp combined_random
python -m src.animate --exp rainbow_random
```

`animate.py` 會根據 exp 名稱自動選擇對應的 model factory（HW3-4 rainbow 用 `build_rainbow_model`，HW3-2 dueling / combined 與 HW3-3 / HW3-4 combined_random 用 `build_dueling_model`，其它用 `build_model`）與輸出目錄前綴。

### 執行測試

```bash
pytest -v
```

預期 71 個測試全綠（HW3-1 24 + HW3-2 9 + HW3-3 16 + HW3-4 22）。

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

**`src/dqn_lightning.py`** (HW3-3)
- 同 `dqn_double_dueling.py`，外加 `--clip` / `--sched` / `--huber` 三個 tricks 的開關 flag

**`src/rainbow.py`** (HW3-4)
- `--mode`、`--epochs`、`--gamma`、`--lr`、`--mem-size`、`--batch-size`、`--max-moves`、`--sync-freq`、`--seed`、`--snapshot-every`、`--out-dir`
- Rainbow 特有：`--n-step`（n-step 步數）、`--n-atoms` / `--v-min` / `--v-max`（C51 distributional support）、`--alpha` / `--beta-start` / `--beta-end`（PER 抽樣參數）、`--sigma-init`（NoisyLinear 噪音初值）

網路架構由 `src/model.py` 與 `src/rainbow.py` 控制：
- `build_model(in_dim, hidden1, hidden2, out_dim)` — Sequential MLP，預設 `64→150→100→4`（HW3-1 / Double DQN 使用）
- `build_dueling_model(in_dim, hidden1, hidden2, n_actions)` — `DuelingMLP` 含共用 trunk + V/A 雙頭 + mean-baseline aggregation（HW3-2 Dueling / Combined、HW3-3 Lightning Combined、HW3-4 combined_random 使用）
- `build_rainbow_model(...)` — `DistributionalDuelingMLP`：plain trunk + NoisyLinear V/A 雙頭，C51 51 atoms 在 `[-10, +10]`，per-atom dueling aggregation；`forward(state)` 回 expected Q 對齊 HW3-1/2/3 介面（HW3-4 rainbow_random 使用）

Gridworld 棋盤大小與獎勵值寫死在 `src/gridworld_env.py` 內（為保持與 DRL in Action Ch.3 一致；如需修改可直接編輯）。

## 後續階段

| Stage | 主題 | 狀態 |
|---|---|---|
| **HW3-1**：Naive DQN for static mode | Naive DQN + Experience Replay 對比（[`HW3_1_report.md`](HW3_1_report.md)） | ✅ 已完成 |
| **HW3-2**：Enhanced DQN Variants for player mode | Double DQN + Dueling DQN + 兩者合併（[`HW3_2_report.md`](HW3_2_report.md)） | ✅ 已完成 |
| **HW3-3**：Framework conversion + training tricks | PyTorch Lightning + grad clipping / cosine sched / Huber loss（[`HW3_3_report.md`](HW3_3_report.md)） | ✅ 已完成 |
| **HW3-4**：Rainbow DQN（加分題） | 完整 Rainbow（PER + N-step + C51 + Noisy）；負面結果分析（[`HW3_4_report.md`](HW3_4_report.md)） | ✅ 已完成 |

## 授權

本專案以 MIT License 授權。`src/gridboard.py` 與 `src/gridworld_env.py` 改編自 *Deep Reinforcement Learning in Action* 第 3 章（Alexander Zai、Brandon Brown，Manning 2020），原作者版權歸原作者所有。詳見 [LICENSE](LICENSE)。
