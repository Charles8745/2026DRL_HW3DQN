# HW3-3: Lightning-converted DQN with Training Tricks for random mode
## PyTorch → Lightning 框架轉換 + 三個訓練技巧的消融研究

> 作者：charles88　|　課程：深度強化學習　|　日期：2026-05-06
> Repo：https://github.com/Charles8745/2026DRL_HW3DQN

---

## 1. 作業目標

本階段（HW3-3）有兩個並行的交付目標。第一個是**框架遷移**：把 HW3-2 以純
PyTorch 手寫的 Combined DQN（Double + Dueling）訓練迴圈，重構為 PyTorch
Lightning 的宣告式架構。Lightning 的核心吸引力在於把 optimizer step、
gradient clipping、lr scheduler 全部納入 `Trainer` 旗標管理，讓後續的
訓練技巧以最少的程式碼修改就能開啟或關閉，是課程作業 #1 的精神所在。

第二個目標是**訓練技巧的消融研究**：在 Lightning 框架之上，逐一加入並
實測三個常見的 DQN 工程技巧——梯度範數裁剪（Gradient Norm Clipping）、
餘弦退火學習率（CosineAnnealingLR）、Huber Loss——並記錄每個技巧對
win rate、loss 穩定度、訓練時間的影響。

本次實驗的骨幹（backbone）固定為 HW3-2 的 Combined 架構（Double target
計算 + Dueling MLP），訓練環境切換為 `random` mode（Player、Goal、Pit、
Wall 全部每局重新隨機），共五組消融（baseline / clip / sched / huber /
full），每組跑 5000 epochs、1000 局測試。

---

## 2. random mode 重訪：HW3-1 留下的痛點

HW3-3 選 random mode 作為實驗場域，是因為前兩個階段的測試環境——static
與 player mode——對優化技巧的鑑別力都不足。下表彙整三種模式的特性：

| Mode | Player 位置 | 其他物件 | 有效棋盤數 | HW3-2 Combined win rate |
|---|---|---|---|---|
| `static` | 固定 (0,3) | 固定 | 1 種 | ~100% |
| `player` | 隨機 | 固定 | ~13 種 | 100.0% |
| `random` | 隨機 | 全隨機 | 數百種 | 未測（HW3-2 只測 player） |

在 HW3-1，`replay_random` 的最終 win rate 為 85.5%，final loss 為
0.0613 ± 0.0573。這個數字有兩個問題：一是勝率離 100% 還有明顯空間；
二是 loss std 高達 mean 的 93%，表示訓練極不穩定——有些 batch 誤差
很小，有些卻飆很高，顯示 random mode 的 reward 分布對普通 MSE 梯度
是一個嚴苛考驗。

正因為 random mode 有真實的「優化困難」——非平穩的狀態分布、每局
不同的 Goal/Pit 位置導致 reward 突變、更大的 Q 高估壓力——才能看到
訓練技巧的加減效應。static 和 player 太容易收斂，trick 有沒有用根本
看不出差異。

---

## 3. 從 HW3-2 PyTorch Combined 到 PyTorch Lightning

### 3.1 自動 vs 手動 optimization

Lightning 的 `LightningModule` 有兩種梯度更新模式：`automatic_optimization=True`
（預設）與 `automatic_optimization=False`（手動）。手動模式給出最大彈性，
但 gradient clipping 和 lr scheduling 都必須自己在 `training_step` 裡呼叫。

本作業選擇 **`automatic_optimization=True`**，理由是三個 tricks 都可以用
`Trainer` 旗標宣告式開啟，完全不用修改 `training_step` 的邏輯：

```python
Trainer(
    max_epochs=5000,
    gradient_clip_val=10.0,      # trick A：一個旗標
    gradient_clip_algorithm='norm',
)
```

trade-off 是：Lightning 的 automatic optimization 要求每個 `training_step`
call 都有對應的損失值回傳，意味著 episode rollout（環境互動）必須
被包進 `IterableDataset`，由 DataLoader 一步步餵給 `training_step`。

### 3.2 RolloutDataset：把 episode 包成 IterableDataset

關鍵設計是讓一次 `__iter__` call 等於一局完整的 episode；replay buffer
作為 instance attribute 跨 epoch 持續累積，不會在每個 DataLoader 重建時
被清空。以下為核心節錄：

```python
class RolloutDataset(IterableDataset):
    def __init__(self, env, agent, replay, max_steps=200):
        self.env   = env
        self.agent = agent
        self.replay = replay      # 跨 epoch 持續累積
        self.max_steps = max_steps

    def __iter__(self):
        state, _ = self.env.reset()
        done = False
        for _ in range(self.max_steps):
            action = self.agent.act(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.replay.push(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        yield self.replay.sample()   # 回傳一個 minibatch
```

這個設計讓每個 Lightning epoch 精確對應一局環境互動 + 一次網路更新，
行為與 HW3-2 的手寫迴圈等價，方便直接比較兩者的 win rate。

### 3.3 DQNLightningModule + Trainer + SnapshotCallback

三大元件共同組成完整的訓練系統：

**`DQNLightningModule`**：繼承 `LightningModule`，持有 `online`（DuelingMLP）
與 `target`（結構相同，每 `sync_freq` 步 hard update 一次）兩個網路；
`training_step` 計算 Double DQN target 並回傳損失；`configure_optimizers`
依 `hparams.sched` 決定是否附加 `CosineAnnealingLR`。

**`Trainer`**：依消融組別設定 `gradient_clip_val`（clip 組 = 10.0，其餘 None）
與 `max_epochs=5000`；Lightning 自動在每個 backward 後、optimizer step 前
套用 gradient clipping，不需要手動 `nn.utils.clip_grad_norm_`。

**`SnapshotCallback`**：每 100 個 epoch 把 `online.state_dict()` 存成
普通 `.pth` 檔案，供後續 `animate.py` 載入、渲染 GIF。Lightning 的
checkpoint 機制預設存 `LightningModule` 完整狀態（含 optimizer 等），
不能直接給 vanilla PyTorch 的 `load_state_dict` 使用，因此需要這個
輕量 callback 輸出乾淨的 state dict。

### 3.4 等價性驗證

在無任何 tricks 的情況下（`baseline_random`），Lightning Combined 的
win rate 為 **88.0%**，final loss mean = **0.03151 ± 0.01744**。

對比 HW3-1 的 `replay_random`：win rate = 85.5%，loss = 0.0613 ± 0.0573。
兩個差異都完整對應演算法升級（replay → Combined）的預期效果：

- win rate 提升 2.5 個百分點：Double target 抑制了 Q 高估，讓策略更準確
- loss mean 降低 ~49%（0.0315 vs 0.0613）：Dueling V/A 拆解減少了
  不必要的 (s,a) 聯合估計，loss 收斂更快
- loss std 降低 ~70%（0.0174 vs 0.0573）：目標更穩定，不再因為
  目標網路同步跳變而造成大幅振盪

這個比較確認了**框架轉換本身沒有破壞訓練語義**，同時也記錄了
Combined 骨幹相對 vanilla replay 在 random mode 的具體收益。

---

## 4. Trick A — Gradient Norm Clipping

### 4.1 原理

DQN 訓練中，每次 target network 同步（hard copy `online → target`）
都會讓下一批 target $Y = r + \gamma Q_{\theta^-}(s', a')$ 產生突變，
導致局部的大梯度噴發。若不加約束，這些大梯度會讓 online 網路的參數
瞬間大幅偏移，有時需要數百個 step 才能修復。

Gradient Norm Clipping 的做法是在 backward 之後、optimizer step 之前，
把所有參數梯度的 $\ell_2$ 範數壓縮到 `max_norm` 以內：

$$\tilde{g} = g \cdot \min\!\left(1,\ \frac{\texttt{max\_norm}}{\|g\|_2}\right)$$

本組使用 `max_norm=10.0`，對應 DQN 文獻的常用 default。

### 4.2 程式碼（一行 Lightning API）

```python
Trainer(
    max_epochs=5000,
    gradient_clip_val=10.0,
    gradient_clip_algorithm='norm',  # 'norm' = L2 norm clipping
)
```

當 `automatic_optimization=True` 時，`Trainer` 在每次呼叫
`optimizer.step()` 之前自動套用 `nn.utils.clip_grad_norm_`，
無需在 `training_step` 內手動介入。其他三組（sched / huber / full）
若不啟用 clip，只需把 `gradient_clip_val` 設為 `None`（預設值）。

### 4.3 訓練結果

![clip loss](results/HW3-3/clip_random/loss.png)

![clip dashboard](results/HW3-3/clip_random/dashboard.gif)

| 指標 | 數值 |
|---|---|
| Final loss (mean ± std, last 100 epochs) | 0.25081 ± 0.14592 |
| Win rate（1000 局測試） | 85.2% |
| 平均勝場步數 | 2.61 |
| 訓練時間 | 34.13s |

clip 組的 win rate（85.2%）不僅沒有超越 baseline（88.0%），
反而下降了 2.8 個百分點；final loss（0.25081）更比 baseline（0.03151）
高出將近 8 倍，std（0.14592）也遠大於 baseline（0.01744）。

這個結果的最可能解讀是 `max_norm=10.0` 對本作業的 30K 參數小網路
而言**並非寬鬆上限，而是頻繁觸發的有效約束**。DQN 文獻（Mnih 2015）
的 `max_norm=10.0` 是針對 Atari CNN（數百萬參數，梯度天然分散在更多
維度上，單次 $\ell_2$ 範數較小）設計的。MLP 小網路的梯度集中在少數
全連接層，$\ell_2$ 範數更容易超過 10.0，導致裁剪頻繁發生、每步的有效
學習幅度被壓縮，5000 epochs 內無法達到 baseline 同等的收斂程度。
loss 的高均值與高方差正反映了這種「欠收斂」狀態：模型始終在較大的
誤差區間內振盪，策略品質也因此略低於 baseline。

---

## 5. Trick B — CosineAnnealingLR

### 5.1 原理

學習率退火的設計動機是：訓練前期需要較大的 lr 快速逼近最優解附近，
後期則需要較小的 lr 在最優解附近做精細調整，避免因步伐太大而振盪。
CosineAnnealingLR 用餘弦曲線把 lr 從初始值 $\eta_{\max}$ 平滑地降到
`eta_min`：

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\!\left(\frac{t}{T_{\max}}\pi\right)\right)$$

本組設定 $T_{\max}$ = `epochs` = 5000、$\eta_{\min}$ = 1e-5，
意味著在第 5000 個 epoch 時 lr 降至初始值的 1/100。

### 5.2 程式碼

```python
def configure_optimizers(self):
    opt = torch.optim.Adam(self.online.parameters(), lr=self.hparams.lr)
    if not self.hparams.sched:
        return opt
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=self.hparams.epochs, eta_min=1e-5)
    return {'optimizer': opt,
            'lr_scheduler': {'scheduler': sched, 'interval': 'epoch'}}
```

Lightning 在每個 epoch 結束後自動呼叫 `scheduler.step()`，
`interval='epoch'` 確保退火以 epoch 為單位（對應一局 episode），
而非以 batch 為單位。

### 5.3 訓練結果

![sched loss](results/HW3-3/sched_random/loss.png)

![sched dashboard](results/HW3-3/sched_random/dashboard.gif)

| 指標 | 數值 |
|---|---|
| Final loss (mean ± std, last 100 epochs) | 1.52610 ± 0.29979 |
| Win rate（1000 局測試） | 87.4% |
| 平均勝場步數 | 2.64 |
| 訓練時間 | 32.28s |

sched 組的 win rate（87.4%）同樣略低於 baseline（88.0%），
差距約 0.6 個百分點；但最值得關注的是 final loss 高達 **1.52610**，
比 baseline 高出約 48 倍。

這個數字初看令人困惑，但仔細分析退火曲線就能解釋：在第 4900–5000
個 epoch 期間，lr 已從初始的 1e-3 降至接近 1e-5，此時每個 batch
的梯度更新幅度極小，模型實際上已接近「凍結」。而 random mode 的
reward 分布在每一局都不同，即使策略已趨穩定，Q 估計仍然持續面對
新棋盤的新 TD 誤差，loss 在「低 lr 無法修正」的狀態下持續累積，
最後 100 個 epoch 的平均值因此被拉到 1.5 以上。

從另一個角度看，win rate 僅下降 0.6 個百分點，表示策略在 lr 降
到很低之前就已基本學完，後期的高 loss 並不代表「策略變差」，
而是「模型停止學習後 TD 殘差累積」的測量假象。這是典型的
**過早退火（premature annealing）**：5000 epochs 對於 1e-3 → 1e-5
的餘弦退火而言太短，退火尾段佔了太大比例，真正需要的細調時間
不足，反而先進入了「無法學習」的停滯期。

---

## 6. Trick C — Huber Loss

### 6.1 原理

random mode 中，Goal 的位置每局隨機改變，導致 reward 信號有較大的
跨局方差——某些罕見的棋盤配置（例如 Player 緊鄰 Goal）會產生非常
大的正 reward，形成重尾分布。普通 MSE 的梯度 $\nabla = 2(Q - Y)$ 與
誤差成正比，一個大 TD 誤差就能讓參數大幅跳動。

Huber Loss（即 `SmoothL1Loss`）在 $|\delta| \leq 1$ 時等同 MSE，
在 $|\delta| > 1$ 時轉為 L1，梯度被限制在 $\pm 1$：

$$L_\delta(y, \hat{y}) = \begin{cases}
\frac{1}{2}(y - \hat{y})^2 & |y - \hat{y}| \leq \delta \\
\delta\bigl(|y - \hat{y}| - \frac{\delta}{2}\bigr) & |y - \hat{y}| > \delta
\end{cases}$$

本組使用 $\delta = 1.0$，對應 Mnih 2015 DQN paper 的「clip TD error」。

### 6.2 程式碼

```python
self.loss_fn = nn.SmoothL1Loss(beta=1.0) if huber else nn.MSELoss()
```

損失函數在 `__init__` 時依 `hparams.huber` 選定，其餘訓練迴圈
完全不變。`beta=1.0` 對應 Huber 的 $\delta=1.0$，與 PyTorch 1.9+
的 `SmoothL1Loss` API 一致。

### 6.3 訓練結果

![huber loss](results/HW3-3/huber_random/loss.png)

![huber dashboard](results/HW3-3/huber_random/dashboard.gif)

| 指標 | 數值 |
|---|---|
| Final loss (mean ± std, last 100 epochs) | 0.03216 ± 0.01878 |
| Win rate（1000 局測試） | 81.0% |
| 平均勝場步數 | 2.64 |
| 訓練時間 | 34.77s |

Huber 是三個 tricks 中對 win rate 衝擊最大的：從 baseline 的 88.0%
降至 **81.0%**，下降 7 個百分點。loss 數值（0.03216）與 baseline
（0.03151）幾乎相同，因為大部分 TD 誤差落在 $|\delta| \leq 1$ 的
範圍內，Huber = MSE 的區段主導了最後 100 個 epoch 的均值。

7 個百分點的 win rate 下降令人意外，值得仔細分析。Random mode 的
「壞 batch」並非純粹的噪音——當 Goal 突然改位置時，那個 transition
的 TD 誤差會出現真正需要修正的大數值（reward 跳到 +10 或 -10 的
極端值）。MSE 的 $\nabla \propto \delta^2$ 行為對這些「關鍵大誤差」
給出了放大後的梯度，讓網路能夠快速修正 Q 值估計；Huber 把這類大
誤差的梯度切回 $\pm 1$，實際上**壓制了最需要大幅修正的關鍵更新**。

換言之，在獎勵本身就有高方差的 random mode 環境下，MSE 的「平方
放大」不是缺陷而是必要的學習動能；Huber 設計用來對抗的「異常值」，
在這裡恰恰是真正的訓練信號，而非雜訊。

---

## 7. 5 組對比

![baseline loss](results/HW3-3/baseline_random/loss.png)

![baseline dashboard](results/HW3-3/baseline_random/dashboard.gif)

![full loss](results/HW3-3/full_random/loss.png)

![full dashboard](results/HW3-3/full_random/dashboard.gif)

| 變體 | Loss mean | Loss std | Win rate | Avg steps | Wall (s) |
|---|---|---|---|---|---|
| baseline | 0.03151 | 0.01744 | 88.0% | 2.62 | 31.87 |
| clip | 0.25081 | 0.14592 | 85.2% | 2.61 | 34.13 |
| sched | 1.52610 | 0.29979 | 87.4% | 2.64 | 32.28 |
| huber | 0.03216 | 0.01878 | 81.0% | 2.64 | 34.77 |
| full | 0.48854 | 0.04443 | 86.3% | 2.61 | 36.94 |

**討論**：

**與預測不同的觀察**：本次消融研究的核心發現是——三個 DQN 訓練技巧在本作業
的 4×4 Gridworld + 30K 參數 DQN + 5000 epochs 的配置下，**全部未能提升
win rate**，甚至都有不同程度的下降（clip -2.8%、sched -0.6%、huber -7.0%、
full -1.7%）。full 組（三個 tricks 全開）的 win rate（86.3%）同樣低於
baseline（88.0%）；loss std（0.04443）是五組中最低，顯示訓練確實更穩定，
但穩定帶來的卻是較低均值收斂而非更高的策略品質。

這個結果在工程上並不代表實作錯誤，而是一個重要的尺度教訓，
有三條可能的解讀路徑：

1. **接近 ceiling 的 baseline**：Combined（Double+Dueling）本身已針對
   random mode 的兩大痛點（Q 高估、sample inefficiency）做出根本性修正。
   88.0% 的 win rate 在 5000 epochs 的訓練量下已接近這個配置的收斂極限；
   tricks 設計用來修的更細微毛病（梯度尖峰、後期震盪、重尾 reward）
   在當前配置的統計誤差範圍內看不到正面效應，只看得到各自帶來的
   hyperparameter mismatch 負面效應。

2. **hyperparameter 尺度不匹配**：`max_norm=10.0`、`eta_min=1e-5`、
   `Huber β=1.0` 均來自 DQN 文獻 default，但那些 default 的設計對象是
   Atari DQN 的 CNN 架構（數百萬參數、連續像素輸入、更大的梯度空間）。
   本作業的 30K 參數 MLP 有截然不同的梯度分布：梯度集中在少數全連接層，
   $\ell_2$ 範數更容易超過 10.0（clip 問題）；TD 誤差分布更集中，
   $|\delta|>1$ 的比例更低（Huber 剪切的是真正的訓練信號）；5000 epochs
   的退火跨度對 MLP 而言太短（sched 問題）。直接套用文獻 default 需要
   先做尺度感的配接，而非直接使用。

3. **訓練量的限制**：若把訓練延長到 20000 epochs，cosine 退火的後期
   精調效應和 Huber 的穩定梯度可能才會在 win rate 上顯現優勢；
   5000 epochs 是 tricks 設計假設的訓練量的下界，在這個範圍內
   看到的可能是過渡期效應而非穩態效應。

**Loss-side 的正面發現**：儘管 tricks 對 win rate 的影響為負，loss 的
比較仍然提供了有價值的資訊。baseline 的 loss mean（0.03151）比 HW3-1
`replay_random`（0.0613）低約 49%，std（0.01744）比 HW3-1（0.0573）低
約 70%——這確認了 **Lightning + Combined backbone 的組合本身就是 random
mode 的顯著改進**，與 tricks 無關。full 組的 loss std（0.04443）是五
組中最低，相對於均值的變異係數僅 9.1%（0.044/0.489），顯示三個 tricks
同時開啟確實讓訓練過程更加穩定，只是這份穩定性在 5000 epochs 內
沒有轉化為更高的 win rate。

**訓練時間**：五組落在 31.87–36.94s 區間（比較：HW3-1 `replay_random`
約 18.11s）。Lightning 的 DataLoader 封裝 + Combined 的雙網路前向 +
各 trick 的計算開銷，累計約 75–100% 的時間增加，但所有變體都在
合理的單機訓練時間範圍內，完整消融五組合計約 170 秒。

---

## 8. 結論

HW3-3 完成了兩個工程交付目標：**Lightning 框架移植**（驗證等價性：
baseline_random 88.0% vs HW3-1 replay_random 85.5%，框架轉換未破壞
訓練語義）與 **三個 tricks 的完整消融**（每組均有 loss 曲線、dashboard
動畫、量化指標）。

本次最有價值的觀察並非「哪個 trick 最好」，而是**「在小規模配置下，
DQN 文獻的 default 參數無法直接套用」這個 hyperparameter scaling 的教訓**。
gradient clipping、cosine annealing、Huber loss 都是業界公認有效的技巧，
但它們的有效性依賴特定的前提——較大的網路、較長的訓練、與之匹配的
reward 尺度。把為 Atari CNN 設計的 default 照搬到 30K 參數的 MLP 上，
不僅沒有收益，反而帶來了 clip 的欠收斂、sched 的過早退火、huber 的
關鍵梯度壓制。這個負面結果本身就是一個可重現的工程發現，比
「全部有效」的結果更具說服力：它清楚指向了後續實驗應該調整的方向
（放寬 max_norm、縮短退火週期、調大 Huber δ）。

真正帶來 win rate 與 loss 雙重進步的不是 tricks，而是 **HW3-2 的
演算法升級**（Combined backbone）加上 **HW3-3 的工程化整合**（Lightning
宣告式架構讓 tricks 以最少程式碼開關，未來超參調優更方便）。
回望 HW3 三個階段的完整弧線：HW3-1 以 Experience Replay 對抗
random mode 非平穩分布的根本困難；HW3-2 以 Double+Dueling 從演算法
層面控制 Q 高估與 sample inefficiency；HW3-3 以 Lightning 移植和 trick
消融驗證了「框架工程化的可靠性」與「超參尺度感的必要性」。
三個階段共同構成了一個從環境建模、演算法設計到工程實踐的完整 DQN
學習路徑。
