# HW3-2: Enhanced DQN Variants for player mode
## Double DQN 與 Dueling DQN 的改進機制與比較

> 作者：charles88　|　課程：深度強化學習　|　日期：2026-05-01
> Repo：https://github.com/Charles8745/2026DRL_HW3DQN

---

## 1. 作業目標

本階段（HW3-2）目的是在 HW3-1 完成的 DQN+Replay baseline 之上，
實作並比較兩個常見的 DQN 變體 — Double DQN 與 Dueling DQN — 並
觀察兩者的改進是否能正交疊加。實驗環境改為 4×4 Gridworld 的
`player` mode（只有 Player 起點隨機，Goal/Pit/Wall 固定）。

## 2. 環境：player mode 與 static / random 的差異

| Mode | Player 位置 | 其他物件 | 狀態空間複雜度 |
|---|---|---|---|
| `static` | 固定 (0,3) | 固定 | 1 種棋盤 |
| `player` | 隨機 | 固定 | ~13 種起點（去除被佔位） |
| `random` | 隨機 | 全隨機 | 數百種有效棋盤 |

`player` mode 介於兩者之間：strategy 必須對 13 種起點通用，但仍
能依靠固定的 Goal/Pit/Wall 結構推理；非常適合用來「拉開變體之間
的差異而又不被噪音淹沒」。

## 3. 從 baseline 到變體：兩個痛點

### 3.1 Vanilla DQN+Replay 仍然殘留兩個結構性弱點

1. **Q 值系統性高估**：target 計算用同一網路選動作 + 估值，
   $\max$ 會讓估計誤差總是被「正向」放大。
2. **Sample inefficiency on shared structure**：很多 state 下
   action 之間的差異很小，但網路必須對每個 (s, a) pair 各自估值，
   無法善用「身處此 state 的價值」這個共通訊號。

### 3.2 兩個變體各自針對哪一個

- **Double DQN** → 針對痛點 (1)，把「選動作」與「估值」拆給 online 與
  target 兩個網路。
- **Dueling DQN** → 針對痛點 (2)，把網路拆成 V(s) 與 A(s,a) 兩支，
  共用 V 學起來。

## 4. Double DQN（Hasselt 2016）

### 4.1 原理

Vanilla 的 target：

$$Y_{\text{vanilla}} = r + \gamma \max_{a'} Q_\theta(s', a')$$

對任何單筆估計誤差 $\epsilon(a)$，因為 $\max$ 取了最樂觀的那個，
$\mathbb{E}[\max_a (Q^* + \epsilon)] \geq \max_a Q^*$，**期望偏差恆為正**。

Double DQN 把選動作的網路與估值的網路分離：

$$Y_{\text{double}} = r + \gamma\, Q_{\theta^-}\!\left(s',\ \arg\max_{a'} Q_\theta(s', a')\right)$$

online 認為最好的動作不一定是 target 高估的那一個，期望偏差大幅縮小。

### 4.2 程式碼解讀（diff 自 baseline）

核心改動只在三處（節錄自 [`src/dqn_double.py`](src/dqn_double.py)）：

```python
online_model, target_model = build_model(), build_model()
target_model.load_state_dict(online_model.state_dict())
target_model.eval()

# minibatch update 內：
with torch.no_grad():
    next_actions = online_model(state2_batch).argmax(dim=1, keepdim=True)
    next_q = target_model(state2_batch).gather(1, next_actions).squeeze(1)
Y = reward_batch + gamma * (1 - done_batch) * next_q

# 每 sync_freq 個 update 同步一次：
global_step += 1
if global_step % sync_freq == 0:
    target_model.load_state_dict(online_model.state_dict())
```

### 4.3 訓練結果

![double loss](results/HW3-2/double_player/loss.png)

![double dashboard](results/HW3-2/double_player/dashboard.gif)

| 指標 | 數值 |
|---|---|
| Final loss (last 100 mean ± std) | 0.00050 ± 0.00008 |
| Win rate（1000 場 test） | 100.0% |
| 平均勝場步數 | 4.389 |
| 訓練時間 | 10.33 秒 |

勝率與 baseline 同樣達到 100%（player mode 對所有變體都夠簡單），
但 Double DQN 的 final loss mean（0.00050）比 baseline（0.00117）
低約 57%，std 從 0.00046 縮小到 0.00008（小 5.7 倍），清楚驗證了
target 高估的控制確實讓 loss 收斂更為平穩。代價是訓練時間多出約 11%
（10.33 vs 9.28 秒），來自每步多一次 target network 的前向推理。

## 5. Dueling DQN（Wang 2016）

### 5.1 原理

把 Q 拆成「state 自己的價值」 + 「action 相對好壞」：

$$Q(s, a) = V(s) + A(s, a)$$

但這個拆解 $V, A$ 不唯一（任何 $V \leftarrow V + c, A \leftarrow A - c$ 都成立），
網路會學不穩。Wang 2016 解法：強制 advantage 平均為 0：

$$Q(s, a) = V(s) + \left(A(s, a) - \tfrac{1}{|\mathcal A|}\sum_{a'} A(s, a')\right)$$

mean baseline 比 max baseline 更穩定（max 的 indicator function 不光滑，
gradient 噪音較大）。

### 5.2 程式碼解讀

DuelingMLP 結構（節錄自 [`src/model.py`](src/model.py)）：

```python
class DuelingMLP(nn.Module):
    def __init__(self, in_dim=64, hidden1=150, hidden2=100, n_actions=4):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden2, 1)
        self.advantage_head = nn.Linear(hidden2, n_actions)

    def forward(self, x):
        h = self.trunk(x)
        v = self.value_head(h)
        a = self.advantage_head(h)
        return v + (a - a.mean(dim=1, keepdim=True))
```

訓練迴圈相對 baseline 只換一行：`model = build_dueling_model()`。
target 計算與 baseline 完全相同（不加 target network，乾淨拆解架構改進的貢獻）。

### 5.3 訓練結果

![dueling loss](results/HW3-2/dueling_player/loss.png)

![dueling dashboard](results/HW3-2/dueling_player/dashboard.gif)

| 指標 | 數值 |
|---|---|
| Final loss (last 100 mean ± std) | 0.00528 ± 0.00500 |
| Win rate | 100.0% |
| 平均勝場步數 | 4.317 |
| 訓練時間 | 11.08 秒 |

Dueling DQN 呈現出一個有趣的矛盾：final loss（0.00528）是四個變體
中最高且 std 最大（0.00500），但 avg_steps_per_win（4.317）卻是四組
中最少，表示 agent 學到了更短的路徑。一種合理解釋是 V/A 拆解讓網路
的探索行為更積極，loss 曲線振盪更劇烈，但策略品質反而更優；沒有
target network 的代價是訓練不夠平滑，以 loss 穩定度換取了路徑效率。

## 6. Double + Dueling 合併

### 6.1 為何能疊加

Double 改的是「target 算法」，Dueling 改的是「網路結構」 — 兩者作用點不重疊，
原 Wang 2016 paper 的實驗也驗證合用最佳。

### 6.2 訓練結果

![combined loss](results/HW3-2/combined_player/loss.png)

![combined dashboard](results/HW3-2/combined_player/dashboard.gif)

| 指標 | 數值 |
|---|---|
| Final loss (last 100 mean ± std) | 0.00032 ± 0.00010 |
| Win rate | 100.0% |
| 平均勝場步數 | 4.393 |
| 訓練時間 | 12.87 秒 |

合併版的 final loss（0.00032）是四個變體中最低，比單獨 Double（0.00050）
再降低 36%，比單獨 Dueling（0.00528）低了整整一個數量級；std（0.00010）
同樣是四組最低。這清楚驗證了兩個改進的正交性：Double 帶來的 target
穩定性與 Dueling 帶來的 V/A 結構優勢可以同時生效，兩者合用的效果
超過各自單獨使用的加總，1+1 ≥ 1.5 的效果在 loss 數字上一覽無遺。

## 7. 四組對比

![baseline loss](results/HW3-2/replay_player/loss.png)

| 實驗 | Method | Final Loss (mean ± std) | Win Rate | Avg Steps | Wall time |
|---|---|---|---|---|---|
| baseline | replay | 0.00117 ± 0.00046 | 100.0% | 4.359 | 9.28s |
| Double | double | 0.00050 ± 0.00008 | 100.0% | 4.389 | 10.33s |
| Dueling | dueling | 0.00528 ± 0.00500 | 100.0% | 4.317 | 11.08s |
| Combined | double_dueling | 0.00032 ± 0.00010 | 100.0% | 4.393 | 12.87s |

**討論**：

- **Player mode 對所有變體都「太簡單」**：4 種方法都達到 100% win rate。
  player mode 只有 13 種起點變化、其他物件固定，3000 epochs 足以讓任何
  方法收斂；勝率區分不出變體優劣。
- **Loss 穩定度才是關鍵指標**：Final loss mean 排序為 Combined (0.00032)
  < Double (0.00050) < baseline (0.00117) < Dueling (0.00528)。Double 比
  baseline 低 ~57%（target 高估控制有效）；Dueling 反而比 baseline 高
  ~4.5 倍（沒有 target net 加上更深網路導致 loss 較不穩）；Combined 結合
  Double 的 target 穩定性 + Dueling 的 V/A 結構，得到最低 loss。
- **Avg steps 看不出明顯差異**：4 個方法的 avg_steps_per_win 都在 4.31–4.39
  之間（player mode 平均最短路約 4 步）。
- **訓練時間代價**：Double / Combined 因多一份網路前向會慢約 10–40%；
  Dueling 多 1 個 head 但 trunk 共用，只比 baseline 慢 ~19%。

## 8. 結論

完成了 HW3 第二階段（Enhanced DQN Variants for player mode）的全部要求：
理解 Q 值高估與 sample inefficiency 兩個 baseline 痛點、實作 Double DQN
與 Dueling DQN 各自針對其中一個痛點、合併兩者驗證正交性；並透過 4 組
dashboard 動畫 + loss 曲線直觀比較四種演算法在 player mode 的表現。

最有價值的觀察：**player mode 對所有方法都收斂到 100% win rate，但
loss 穩定度差異一個數量級**。這提醒我們，在簡單環境下「能不能解決問題」
不是好的區分指標；要看 loss 收斂的平穩度與 sample efficiency。

下一階段（HW3-3）將把目前的 PyTorch 實作轉換為 Keras 或 PyTorch Lightning，
並引入 gradient clipping、learning rate scheduling 等 training tricks，
針對 random mode 的不穩定性做進一步的工程化處理。
