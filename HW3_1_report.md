# HW3-1: Naive DQN for static mode
## DQN 與 Experience Replay 理解報告

> 作者：charles88　|　課程：深度強化學習　|　日期：2026-04-30
> Repo：（push 後填入）

---

## 1. 作業目標

本階段（HW3-1）目的是理解 Deep Q-Network（DQN）的最基本形式，
並補上經驗回放（Experience Replay Buffer）這個讓 DQN 真正可訓練的
關鍵元件。實作平台採用 *Deep Reinforcement Learning in Action*
第 3 章提供的 Gridworld 環境，並在三組實驗下驗證程式碼行為與書本
結果一致。

## 2. 環境：Gridworld

### 2.1 環境簡介

4×4 棋盤，包含四種棋子：
- **P (Player)**：智能體，由 DQN 控制
- **+ (Goal)**：到達即勝（reward = +10）
- **− (Pit)**：踏到即敗（reward = −10）
- **W (Wall)**：障礙，無法穿越

動作集合 `{u, d, l, r}`。每一步未碰到 Goal 或 Pit 時 reward = −1
（鼓勵快速完成）。

### 2.2 三種 Mode

| Mode | 說明 | Player 位置 | 其他物件位置 | 適用情境 |
|---|---|---|---|---|
| `static` | 完全靜態配置，所有物件位置固定 | 固定 (0,3) | 固定：Goal→(0,0) Pit→(0,1) Wall→(1,1) | 測試邏輯正確性、可重現結果 |
| `player` | 只有 Player 隨機，其它物件固定 | 隨機位置 | 固定（同上） | 模擬不同起點，測試策略泛化能力 |
| `random` | 所有物件 (Player, Goal, Pit, Wall) 位置全隨機 | 隨機 | 隨機 | 訓練更強健的策略，提升泛化能力 |

## 3. MDP 建模

### 3.1 State 表示法

每一個棋子在棋盤上佔一個 `(4, 4)` 的 channel，one-hot 編碼。四個棋子
堆疊成 `(4, 4, 4)` tensor，再 flatten 成長度 64 的向量，並加上微小
雜訊 `np.random.rand(1, 64) / 100`：

```python
state = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100
```

雜訊的目的是避免完全相同的 state 反覆出現導致網路 over-confident，
並讓梯度更新更穩定。

### 3.2 Action / Reward / Episode 終止

- **Action**：`{0:u, 1:d, 2:l, 3:r}`
- **Reward**：到達 Goal +10、踏入 Pit −10、其他 −1
- **Episode 終止條件**：reward ≠ −1（即遊戲結果出爐）；Replay 版本
  另加 `max_moves=50` 的步數上限避免亂走

## 4. DQN 原理

### 4.1 從 Q-learning 到 DNN 近似

Tabular Q-learning 的更新式：

$$Q(s,a) \leftarrow Q(s,a) + \alpha\left[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right]$$

當 state space 大到無法以表格列舉時，改用神經網路 $Q_\theta(s, a)$
近似 Q 函數。把 update 重寫為**最小化 MSE**的監督式學習問題：

$$\mathcal{L}(\theta) = \mathbb{E}\left[\left(Q_\theta(s, a) - \left(r + \gamma \max_{a'} Q_\theta(s', a')\right)\right)^2\right]$$

target 端對 $\theta$ 不取梯度（detach），這就是 DQN 的核心。

### 4.2 網路架構

```
Input (1, 64)
    │  Linear(64 → 150) + ReLU
Hidden 1 (1, 150)
    │  Linear(150 → 100) + ReLU
Hidden 2 (1, 100)
    │  Linear(100 → 4)
Output Q-values (1, 4)
```

實作於 [`src/model.py`](src/model.py)。Optimizer 採 Adam (`lr=1e-3`)，
Loss 採 `MSELoss`。

### 4.3 ε-greedy 探索

訓練時以機率 ε 隨機選動作（探索），其餘 1−ε 機率選 argmax(Q)（利用）。
- Naive DQN 用 **線性衰減** ε：1.0 → 0.1（每 epoch 減 `1/epochs`）
- Replay 版用 **固定** ε = 0.3（沿用書本，因 buffer 提供額外多樣性）

## 5. Naive DQN（Listing 3.3）

### 5.1 程式碼解讀

核心訓練 inner loop（節錄自 [`src/dqn_naive.py`](src/dqn_naive.py)）：

```python
qval = model(state)                              # 1) 前向算 Q
action_idx = epsilon_greedy(qval, epsilon)       # 2) ε-greedy 選動作
game.makeMove(ACTION_SET[action_idx])            # 3) 執行
state2 = encode_state(game)                      # 4) 取下一個 state
reward = game.reward()
with torch.no_grad():
    newQ = model(state2)
maxQ = torch.max(newQ)
Y = reward + (gamma * maxQ) if reward == -1 else reward   # 5) target
X = qval.squeeze()[action_idx]                   # 6) 預測值
loss = loss_fn(X, torch.tensor([Y]).detach().squeeze())
optimizer.zero_grad(); loss.backward(); optimizer.step()  # 7) 更新
```

要點：
- **每一個 transition 立即做一次 update**，沒有 buffer。
- target Y 對網路參數 detach，避免梯度流到 target。
- terminal step（reward ≠ −1）不加 γ·maxQ，因為下個 state 的價值無意義。

### 5.2 訓練結果（static mode）

![naive_static loss](results/HW3-1/naive_static/loss.png)

![naive_static dashboard](results/HW3-1/naive_static/dashboard.gif)

| 指標 | 數值 |
|---|---|
| Final loss (last 100 mean ± std) | 0.006 ± 0.010 |
| Win rate（1000 場 test） | 100.0% |
| 平均勝場步數 | 7.0 |
| 訓練時間 | 4.26 秒 |

訓練約 1000 epochs 後，agent 在 static mode 已能走出最短路徑直達 Goal。
最短路徑為 7 步（因為 Pit 位於 (0,1) 阻斷了 (0,3)→(0,0) 的直線，agent
必須繞行第 2 列）。

### 5.3 為什麼 Naive 在 static 可以、在 random 會失敗？

Naive DQN 之所以能在 static mode 收斂，是因為棋盤永遠是同一盤，agent
其實在學一個非常窄的狀態子集合的 Q 表。一旦換成 random mode，下面
兩個問題會放大：

1. **樣本高度相關**：每個 update 用的 (s, a, r, s') 都是上一步直接接著的
   transition，違反 SGD 的 IID 假設，梯度估計有偏。
2. **Catastrophic forgetting**：網路只看最近的 state distribution，
   學會新棋盤的同時忘掉舊棋盤；隨機重置會讓網路不停在不同分布之間擺盪。

這兩個問題在 static mode 下都不存在 → Naive DQN 也能收斂。

## 6. Experience Replay Buffer（Listing 3.5）

### 6.1 想解決的問題

呼應 5.3 — Replay Buffer 用「過去經驗的隨機抽樣」打破時序相關性、
並讓網路反覆看舊資料避免遺忘。

### 6.2 程式碼解讀

核心結構（節錄自 [`src/dqn_replay.py`](src/dqn_replay.py)）：

```python
replay = deque(maxlen=mem_size)                  # 環形 buffer

# 每步：
replay.append((state1, action_idx, reward, state2, done))   # 1) 儲存

if len(replay) > batch_size:
    minibatch = random.sample(list(replay), batch_size)     # 2) 隨機抽樣
    state1_batch = torch.cat([s1 for s1,_,_,_,_ in minibatch])
    # ... 同樣抽出 action / reward / state2 / done batches

    Q1 = model(state1_batch)                                 # 3) 向量化前向
    with torch.no_grad():
        Q2 = model(state2_batch)
    Y = reward_batch + gamma * (1 - done_batch) * torch.max(Q2, dim=1)[0]
    X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(1)).squeeze()
    loss = loss_fn(X, Y.detach())                            # 4) 向量化 loss
```

關鍵差異：
- 每步**做兩件事**：(a) 把 transition 存入 buffer、(b) 從 buffer 隨機抽 200
  筆做 mini-batch update。
- target 計算用 `(1 - done)` mask 取代 if/else，向量化整個 batch 的更新。
- buffer 大小 1000：相當於最近約 50–200 場遊戲的 transitions。

### 6.3 訓練結果

#### 6.3.1 Static mode（與 Naive 直接對比）

![replay_static loss](results/HW3-1/replay_static/loss.png)

![replay_static dashboard](results/HW3-1/replay_static/dashboard.gif)

| 指標 | Naive | Replay |
|---|---|---|
| Final loss (last 100 mean) | 0.006 | 0.0014 |
| Final loss (last 100 std) | 0.010 | 0.0003 |
| Win rate | 100.0% | 100.0% |
| 平均步數 | 7.0 | 7.0 |
| 訓練時間（秒） | 4.26 | 5.26 |

在 static mode 下兩種方法皆達到 100% 勝率與 7 步最短路；觀察重點是
loss 曲線的**穩定度**：Replay 版本的 loss mean 比 Naive 小一個數量級
（0.0014 vs 0.006），std 更是少了 30 倍以上（0.0003 vs 0.010），因為
batch 統計減少單筆 transition 帶來的 variance。代價是 mini-batch
sampling 拉長了每 epoch 的 wall time（4.26 → 5.26 秒）。

#### 6.3.2 Random mode（Replay 真正發揮的地方）

![replay_random loss](results/HW3-1/replay_random/loss.png)

![replay_random dashboard](results/HW3-1/replay_random/dashboard.gif)

| 指標 | 數值 |
|---|---|
| Final loss (last 100 mean ± std) | 0.0613 ± 0.0573 |
| Win rate（1000 場 test） | 85.5% |
| 平均勝場步數 | 2.56 |
| 訓練時間 | 18.11 秒 |

書本 baseline 89.4%，本次實作 85.5%（n=1000 test games），略低於書本
但仍在 random mode 一般可達區間內。`avg_steps_per_win = 2.56` 看起來
偏小，原因是 random 模式下隨機初始化常將 Player 與 Goal 放在相鄰格，
能 1–2 步勝出的場次拉低了平均；而真正困難的初始配置（Goal 被 Wall
或 Pit 半包圍）多半會落在 14.5% 的失敗組裡。從 dashboard GIF 可以
看到，agent 在訓練早期（epoch 0–1000）幾乎是亂走，到 epoch 3000 後
已能在多數隨機初始棋盤上找到 Goal。

## 7. 三種實驗對比

| 實驗 | Mode | Final Loss (mean) | Win Rate | Avg Steps | 訓練時間 |
|---|---|---|---|---|---|
| Naive DQN | static | 0.006 | 100.0% | 7.0 | 4.26s |
| DQN + Replay | static | 0.0014 | 100.0% | 7.0 | 5.26s |
| DQN + Replay | random | 0.0613 | 85.5% | 2.56 | 18.11s |

**討論**：

- **Naive vs Replay 在 static**：勝率持平（皆 100%），最短路也都是 7 步，
  但 Replay 的 final loss mean 小一個數量級、std 小兩個數量級，明顯
  更平穩，代價是 wall time 略長。在 static 這個簡單情境下 Replay 的
  好處不明顯，但也沒有壞處。
- **Replay 在 static vs random**：環境複雜度從「固定」變「全隨機」後，
  win rate 從 100% 降到 85.5%，這完全合理 — 全隨機棋盤包含許多無解
  或極困難的初始配置（例如 Goal 被 Wall 包圍、Player 起手就在 Pit
  附近）。
- **未跑 Naive on random 的隱含論點**：書本與多份文獻已證實 Naive DQN
  在 random mode 無法穩定收斂；本次將計算資源優先用於完整的 Replay
  baselines。

## 8. 結論

完成了 HW3 第一階段（Naive DQN for static mode）的全部要求：理解
Q-learning → DQN 的演進、實作 Naive DQN 與 Experience Replay 兩個
版本、在 static 與 random 模式下交叉驗證；並透過 dashboard 動畫直觀
呈現 agent 學習過程。下一階段（HW3-2）將加入 Target Network
（Listing 3.7）並比較對 random mode 訓練穩定性的影響，逐步過渡到
Double DQN 與 Dueling DQN 等變體。
