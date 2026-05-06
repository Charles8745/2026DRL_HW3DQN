# HW3-4: Rainbow DQN for random mode — 設計文件

> **Stage label**：HW3-4: Rainbow DQN for random mode（加分題）
> **作業來源**：深度強化學習課程 HW3 — DQN and its variants（Stage 4 / bonus）
> **設計日期**：2026-05-06
> **基底教材**：HW3-3 Lightning Combined（Double + Dueling）；Hessel et al. 2018 (Rainbow)；Schaul et al. 2016 (PER)；Bellemare et al. 2017 (C51)；Fortunato et al. 2018 (Noisy Nets)；Sutton & Barto Ch. 7 (n-step bootstrapping)
> **前置階段**：[HW3-1 spec](2026-04-30-hw3-dqn-stage1-design.md)、[HW3-2 spec](2026-05-01-hw3-dqn-stage2-design.md)、[HW3-3 spec](2026-05-06-hw3-dqn-stage3-design.md)

---

## 0. 作業需求對應

| 需求 | 對應交付物 |
|---|---|
| #1 分析 | 本文件 §1（為何 Rainbow / 4 個新元件數學摘要 / 與 HW3-3 baseline 對照） + `HW3_4_report.md` 第 1–3 章 |
| #2 講解實作 + 實作 | 本文件 §2–§7 + `src/rainbow.py`（NoisyLinear + DistributionalDuelingMLP + PrioritizedReplayBuffer + NStepBuffer + projection + train_rainbow + CLI） |
| #3 訓練過程動畫 | `results/HW3-4/{combined,rainbow}_random/dashboard.gif`（沿用 `src/animate.py`，加一條 dispatch 即可） |
| 主題：Random Mode | 全部 2 組實驗都跑 `mode='random'`，對照 HW3-3 `baseline_random`（88.0% / loss 0.0315±0.0174）與 HW3-1 `replay_random`（85.5% / loss 0.0613±0.0573） |
| 沿用 HW3-1/2/3 規格 | 章節結構、metrics schema、commit 風格、測試覆蓋、dashboard GIF 全部對齊 |

---

## 1. 設計原則

1. **完整 Rainbow，不打折**：Hessel 2018 的 6 個元件全部疊上去——Double（HW3-2 已做）、Dueling（HW3-2 已做）、PER、N-step、Distributional / C51、Noisy Nets。少做任何一個都不能叫 Rainbow。
2. **Random mode 為主軸**：HW3-4 的對照是「HW3-3 Combined → HW3-4 Rainbow，把 4×4 random 從 88% 推到 90%+」。Player / static mode 不在範圍內（HW3-2 顯示 player 上 4 變體都 100%，看不出 Rainbow 的提升空間）。
3. **改回 vanilla PyTorch**：HW3-3 已經拿到 Lightning 框架轉換的分數；Rainbow 的 PER priority update 需要在 training step 之外回寫 TD error，跟 Lightning 的 `training_step` 純 loss 抽象有摩擦——回 vanilla 反而比較乾淨。HW3-2 的 `dqn_double_dueling.py` 訓練迴圈是更合適的起點。
4. **單檔 `rainbow.py`**：所有 Rainbow 特有邏輯（NoisyLinear、Distributional 網路、PER buffer、N-step buffer、Categorical projection、訓練迴圈、CLI）集中在一個檔。理由：4 個新元件高度耦合，分散到多個檔案會讓 import 與資料流動很難追。檔案會是 ~600 LOC，可以接受。
5. **Forward 介面對齊既有腳本**：`DistributionalDuelingMLP.forward(state) → (B, n_actions)` 直接回 expected Q（內部把 distribution 算 expectation）；distribution 透過獨立 `forward_dist(state) → (B, n_actions, n_atoms)` 取得。這樣 `evaluate()` / `test_model()` / `animate.py` 完全不用改，只需要在 `animate.py` 加一條 model factory dispatch。
6. **單一 seed=42**：對齊 HW3-1/2/3。
7. **2 組實驗**：`combined_random`（重跑 HW3-3 Combined，當 HW3-4 baseline）+ `rainbow_random`（主角）。不做元件 ablation——4 個元件高度互依（C51 改 loss、PER 改抽樣、Noisy 換掉 ε、N-step 改 target），單獨拆開跑 wall time 翻 5–6 倍卻看不出多少額外資訊（HW3-3 已經告訴我們：4×4 Gridworld 上單 trick ablation 經常看不出邊際效應）。

---

## 2. 演算法規格

所有 hyperparameters：`gamma=0.9, lr=1e-4, mem_size=10000, batch_size=200, max_moves=50, sync_freq=500, seed=42, mode='random', epochs=5000, snapshot_every=250`。

> **與 HW3-3 的差異**：
> - `lr` 從 1e-3 降到 1e-4：Rainbow paper 與 distributional / noisy 文獻都用較小 lr（distributional output 的 cross-entropy gradient 比 MSE 大，搭配 noisy params 的學習動能 → 1e-3 容易發散）
> - `mem_size` 從 1000 增到 10000：PER 在小 buffer 上 priority distribution 過度集中，原文用 1M（Atari），這裡按 4×4 規模等比縮放
> - **沒有 epsilon**：Noisy Nets 完全取代 ε-greedy，CLI 移除 `--epsilon`
> - **沒有 huber/clip flag**：Rainbow 用 cross-entropy（已經對 outlier 不敏感），且 PER 的 IS weight 自帶 importance scaling

### 2.1 Component 1 — Double DQN（從 HW3-2 沿用）

Action 選擇用 online net、value 估計用 target net：
$$a^*_{t+n} = \arg\max_{a'} Q_\theta(s_{t+n}, a')$$
然後拿 target net 對 $(s_{t+n}, a^*_{t+n})$ 的 distribution 做 projection。

### 2.2 Component 2 — Dueling Networks（從 HW3-2 沿用，但操作於 atoms）

Distributional + Dueling 的合成形式：對每個 atom $i$ 分開 aggregate
$$q_{\text{logits}}(s, a, i) = V(s, i) + A(s, a, i) - \tfrac{1}{n_{\text{actions}}} \sum_{a'} A(s, a', i)$$
然後 softmax over atoms 得 $p_i(s, a)$；expected Q $= \sum_i z_i \cdot p_i(s, a)$。

V head 輸出 shape `(B, 1, n_atoms)`、A head 輸出 shape `(B, n_actions, n_atoms)`，broadcast 相加。

### 2.3 Component 3 — Prioritized Experience Replay (Schaul 2016)

**抽樣機率**：
$$P(i) = \frac{p_i^\alpha}{\sum_j p_j^\alpha}, \quad p_i = |\delta_i| + \epsilon_p$$

其中 $\delta_i$ 是該 transition 上次計算的 KL loss（Rainbow 用 cross-entropy 當 priority signal，不是 TD error），$\epsilon_p = 1\text{e-}6$ 確保所有 transition 至少有非零機率。新加入的 transition 給 max priority 確保至少被抽一次。

**Importance sampling weights**：
$$w_i = \left(\frac{1}{N} \cdot \frac{1}{P(i)}\right)^\beta \Big/ \max_j w_j$$
β 從 0.4 線性退火到 1.0（over `epochs=5000`）；分母的 `max w_j` 確保 w 在 `(0, 1]`、不會把 gradient 放大。

**資料結構：Sum-tree**
- 二元樹陣列，葉子存 priority、內部節點存子節點 sum。
- `add(priority, transition)`: O(log N)
- `sample(s)` for s in [0, total_p): O(log N) — 從 root 走到對應的葉子
- `update(idx, new_priority)`: O(log N)
- batch 抽樣分 batch_size 等距區間，每區間 sample 一次（stratified）

**超參數**：α=0.5（原文 0.5–0.7 之間，4×4 task 取較小的 0.5 避免過度集中）；β_start=0.4、β_end=1.0；ε_p=1e-6。

### 2.4 Component 4 — N-step Returns (n=3)

**目的**：1-step TD 在 random mode 太短視（Player 距 Goal 多步時，reward signal 大部分時間是 -1 的 step penalty），加長 horizon 讓 reward 訊號傳遞更快。

**N-step transition**：把連續 n 個 (s, a, r, s', d) 累積成 (s_t, a_t, R^{(n)}_t, s_{t+n}, d_{t+n})，其中
$$R^{(n)}_t = \sum_{k=0}^{n-1} \gamma^k r_{t+k}$$
若中途 done（reward != -1），就提前 truncate（只累積到 done step），並把 `s_{t+n}` 設為 done state、`d_{t+n}=True`。

**N-step buffer**：`collections.deque(maxlen=n)`，每收到一個 1-step transition 就 push 進去；當佇列滿（或遇到 done）時 pop 最舊的並算 n-step return，丟入 PER buffer。Episode 結束時清空殘留（剩下 1, 2, ..., n-1 個 transition 都用 truncated n-step push 進 buffer）。

**Bellman target**：
$$Y_t^{(n)} = R^{(n)}_t + \gamma^n (1 - d_{t+n}) \cdot Q_{\theta^-}(s_{t+n}, a^*_{t+n})$$
這裡的 $Q_{\theta^-}$ 是 distributional projected target（見 §2.5）。Double 用在 $a^*_{t+n}$ 的選擇。

**超參數**：n=3（Rainbow 原文）。

### 2.5 Component 5 — Distributional RL / C51 (Bellemare 2017)

**Support**：n_atoms=51，$V_{\min}=-10$、$V_{\max}=+10$，$\Delta z = (V_{\max} - V_{\min})/(n_{\text{atoms}} - 1) = 0.4$。Atoms 為 $z_i = V_{\min} + i \cdot \Delta z$, $i=0, \ldots, 50$。

> Gridworld reward 範圍：goal=+10, pit=-10, step=-1, max_moves=50 → 單 episode return 上界 +10、下界約 -50。但 n-step return（n=3）含 γ=0.9 折扣的範圍是 [-1·(1+γ+γ²) + γ³·V_target, ... +10] ≈ [-2.71 + γ³·V_min, 10]。V_min=-10 / V_max=+10 對 4×4 task 夠用，且對齊 Rainbow paper 的 [V_min=-10, V_max=10]。

**Categorical projection（Bellemare Algorithm 1）**：
1. 對每個 atom $z_j$，計算 projected support $\hat{z}_j = \text{clip}(R^{(n)} + \gamma^n \cdot z_j, V_{\min}, V_{\max})$
2. 計算 $\hat{z}_j$ 在原 support 上的位置 $b_j = (\hat{z}_j - V_{\min})/\Delta z$，下界 $l = \lfloor b \rfloor$、上界 $u = \lceil b \rceil$
3. 把 target distribution 的機率 $p_j$ 線性分配到 $z_l$ 與 $z_u$：$m_l \mathrel{+}= p_j \cdot (u - b)$、$m_u \mathrel{+}= p_j \cdot (b - l)$
4. 對 done state，target 是 $\delta(\hat{z} = \text{clip}(R, V_{\min}, V_{\max}))$（point mass）

實作上 batch 化（向量化所有 51 個 atoms × 200 個 batch sample）。

**Loss**：每個 sample 的 cross-entropy
$$L_i = -\sum_j m_j(i) \log p_j(s_i, a_i)$$
乘 PER 的 $w_i$，再求 batch mean：
$$L = \tfrac{1}{B} \sum_i w_i \cdot L_i$$

**Priority update**：使用 per-sample $L_i$（無 IS weight）作為新 priority，回寫 sum-tree。

### 2.6 Component 6 — Noisy Nets (Fortunato 2018)

**Linear layer 替換**：Dueling 的 V/A 兩個 head 各兩層 Linear → 替換為 NoisyLinear。Trunk 不換（trunk 不需要 exploration noise，純 representation learning，省參數）。

**Factorised Gaussian noise**（Atari 版，比 independent 噪音省參數）：
- learnable: $\mu_W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$、$\sigma_W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$、$\mu_b \in \mathbb{R}^{d_{\text{out}}}$、$\sigma_b \in \mathbb{R}^{d_{\text{out}}}$
- noise: $\varepsilon_p \in \mathbb{R}^{d_{\text{in}}}$、$\varepsilon_q \in \mathbb{R}^{d_{\text{out}}}$（每次 forward 重抽）
- 構造：$\varepsilon_W = f(\varepsilon_q) \otimes f(\varepsilon_p)$、$\varepsilon_b = f(\varepsilon_q)$，其中 $f(x) = \text{sign}(x) \sqrt{|x|}$
- forward: $y = (\mu_W + \sigma_W \odot \varepsilon_W) x + (\mu_b + \sigma_b \odot \varepsilon_b)$

**初始化**（與原文一致）：
- $\mu_W, \mu_b \sim \mathcal{U}(-1/\sqrt{d_{\text{in}}}, +1/\sqrt{d_{\text{in}}})$
- $\sigma_W, \sigma_b = \sigma_0 / \sqrt{d_{\text{in}}}$，$\sigma_0 = 0.5$（factorised 版的原文預設）

**reset_noise**：每次 batch sample 重抽 `ε_p, ε_q`（online net 與 target net 各自獨立）。

**Eval 行為（μ-only）**：`NoisyLinear.forward` 根據 `self.training` 決定要不要加噪音——training 時 `y = (μ_W + σ_W ⊙ ε_W) x + ...`，eval 時直接 `y = μ_W x + μ_b`（純 μ，無 ε）。理由：（a）eval 確定性 → win_rate 可重現；（b）`utils.py` 不用動。`train_rainbow` 會在呼叫 `evaluate(online, ...)` 前 `online.eval()`、後 `online.train()`。

> **與 Rainbow 原文的差異**：原文 eval 也保留 noise（exploration 是 policy 一部分）。這裡選 μ-only 是基於可重現性與「win_rate 是評估指標而非 policy 行為展示」的權衡；σ 收斂後兩者數值差異很小。

**Action 選擇**：直接 `argmax_a Q(s, a)`，沒有 ε。Buffer 收集時是 training mode（有 noise）→ 自動 exploration；eval 時是 eval mode（μ-only）→ 純 greedy。

---

## 3. 檔案結構

```
HW3_DQN/
├── HW3_4_report.md                          # NEW — HW3-4 中文報告
├── chatlog4.md                              # NEW — HW3-4 對話紀錄
├── README.md                                # MODIFY — 加 HW3-4 段、HW3-4 status → ✅
├── src/
│   ├── model.py                             # UNCHANGED
│   ├── utils.py                             # UNCHANGED
│   ├── animate.py                           # MODIFY — +rainbow_random/combined_random choices, dispatch
│   └── rainbow.py                           # NEW — Rainbow 全套實作 + CLI
├── tests/
│   └── test_rainbow.py                      # NEW — sum-tree / NoisyLinear / projection / smoke train
├── results/
│   └── HW3-4/                               # NEW
│       ├── combined_random/                 # 重跑 HW3-3 Combined 作 baseline
│       └── rainbow_random/                  # 主角
└── docs/superpowers/
    ├── specs/2026-05-06-hw3-dqn-stage4-design.md       # 本文件
    └── plans/2026-05-06-hw3-dqn-stage4-implementation.md  # 下一步產出
```

`src/dqn_replay.py`、`src/dqn_double.py`、`src/dqn_dueling.py`、`src/dqn_double_dueling.py`、`src/dqn_lightning.py` 完全保留不動。

---

## 4. 模組介面

### 4.1 `src/rainbow.py` 結構

完整檔案約 600 LOC，分為 6 個區塊：

```python
"""Rainbow DQN for Gridworld random mode (HW3-4)."""

# ============================================================
# Block 1: NoisyLinear (factorised Gaussian noise)
# ============================================================
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5): ...
    def reset_parameters(self): ...
    def reset_noise(self): ...               # 重抽 ε_p, ε_q
    def forward(self, x): ...                # μ + σ⊙ε

# ============================================================
# Block 2: DistributionalDuelingMLP (Noisy + Dueling + C51)
# ============================================================
class DistributionalDuelingMLP(nn.Module):
    def __init__(self, n_atoms=51, v_min=-10.0, v_max=10.0,
                 in_dim=64, hidden1=150, hidden2=100, n_actions=4): ...
    def forward(self, x):                    # 回 expected Q (B, n_actions) — 跟 HW3-1/2/3 接口對齊
        return (self.forward_dist(x) * self.support).sum(dim=-1)
    def forward_dist(self, x):               # 回 distribution (B, n_actions, n_atoms)
        # trunk → V/A heads (NoisyLinear) → dueling aggregation per atom → softmax
        ...
    def reset_noise(self):                   # propagate to all NoisyLinear
        ...

def build_rainbow_model(n_atoms=51, v_min=-10.0, v_max=10.0,
                        in_dim=64, hidden1=150, hidden2=100, n_actions=4):
    """Factory used by animate.py for snapshot loading."""
    return DistributionalDuelingMLP(...)

# ============================================================
# Block 3: SumTree (binary heap-style array, O(log N))
# ============================================================
class SumTree:
    def __init__(self, capacity): ...
    def add(self, priority, data): ...
    def sample(self, s): ...                 # walk from root, returns (idx, priority, data)
    def update(self, idx, priority): ...
    @property
    def total(self): ...

# ============================================================
# Block 4: PrioritizedReplayBuffer (alpha, beta, IS weights)
# ============================================================
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.5, beta_start=0.4, beta_end=1.0,
                 epsilon=1e-6): ...
    def push(self, transition): ...          # 給 max priority
    def sample(self, batch_size, frac): ...  # frac in [0,1] for β anneal
                                             # returns (transitions, indices, IS weights)
    def update_priorities(self, indices, priorities): ...
    def __len__(self): ...

# ============================================================
# Block 5: NStepBuffer (n=3 default)
# ============================================================
class NStepBuffer:
    def __init__(self, n=3, gamma=0.9): ...
    def append(self, s, a, r, s_next, done): ...
        # returns Optional[(s, a, R^(n), s_{t+n}, d_{t+n})] — None if not enough yet
    def flush(self): ...
        # called at episode end; yields remaining truncated n-step transitions

# ============================================================
# Block 6: Categorical projection + train_rainbow + CLI
# ============================================================
def project_distribution(next_dist, rewards, dones, gamma_n,
                         support, v_min, v_max, n_atoms):
    """Bellemare Algorithm 1, batch-vectorised. Returns (B, n_atoms)."""
    ...

def train_rainbow(*, epochs=5000, gamma=0.9, lr=1e-4, mem_size=10000,
                  batch_size=200, max_moves=50, sync_freq=500,
                  n_step=3, n_atoms=51, v_min=-10.0, v_max=10.0,
                  alpha=0.5, beta_start=0.4, beta_end=1.0,
                  sigma_init=0.5,
                  mode='random', seed=42,
                  snapshot_every=250,
                  out_dir='results/HW3-4/rainbow_random',
                  eval_n_games=1000): ...

def main(): ...                              # CLI
```

### 4.2 訓練迴圈骨架

```python
def train_rainbow(...):
    set_seed(seed)
    online = build_rainbow_model(...)
    target = build_rainbow_model(...)
    target.load_state_dict(online.state_dict())
    target.eval()

    optimizer = torch.optim.Adam(online.parameters(), lr=lr)
    per = PrioritizedReplayBuffer(mem_size, alpha=alpha,
                                   beta_start=beta_start, beta_end=beta_end)
    losses, global_step = [], 0
    snapshot 'epoch_0000.pth'

    for epoch in tqdm(range(epochs)):
        game = Gridworld(size=4, mode=mode)
        n_step_buf = NStepBuffer(n=n_step, gamma=gamma)
        s1 = encode_state(game)
        for mov in range(max_moves):
            online.reset_noise()                                    # 抽探索噪音
            with torch.no_grad():
                qval = online(s1)                                   # expected Q
            a = int(qval.argmax(dim=1).item())                      # 無 ε
            game.makeMove(ACTION_SET[a])
            s2 = encode_state(game)
            r = game.reward()
            done = (r != -1)
            ready = n_step_buf.append(s1, a, r, s2, done)
            if ready is not None:
                per.push(ready)
            s1 = s2
            if done: break

            if len(per) > batch_size:
                frac = epoch / epochs
                batch, idxs, w = per.sample(batch_size, frac)
                # ... compute KL loss + IS weight
                # ... update online
                # ... per.update_priorities(idxs, per_sample_loss)
                losses.append(loss_mean)
                global_step += 1
                if global_step % sync_freq == 0:
                    target.load_state_dict(online.state_dict())
        for tail in n_step_buf.flush():                             # episode 結束清空
            per.push(tail)
        if (epoch + 1) % snapshot_every == 0:
            torch.save(online.state_dict(), snapshots_dir / f'epoch_{epoch+1:04d}.pth')

    # ... save metrics, plot, evaluate
```

### 4.3 KL loss 計算（向量化）

```python
def compute_loss(online, target, batch, weights, gamma, n_step,
                 support, v_min, v_max, n_atoms):
    s1, a, R_n, s2, d = batch                          # all tensors, batch_size
    # online prediction: distribution at chosen action
    online.reset_noise()
    pred_dist_all = online.forward_dist(s1)            # (B, n_actions, n_atoms)
    pred_dist = pred_dist_all.gather(
        1, a.long().view(-1, 1, 1).expand(-1, 1, n_atoms)
    ).squeeze(1)                                        # (B, n_atoms)

    # target: Double DQN action selection from online
    with torch.no_grad():
        online.reset_noise()
        next_q = online(s2)                             # (B, n_actions) expected Q
        next_a = next_q.argmax(dim=1)                   # (B,)
        target.reset_noise()
        target_dist_all = target.forward_dist(s2)       # (B, n_actions, n_atoms)
        target_dist = target_dist_all.gather(
            1, next_a.view(-1, 1, 1).expand(-1, 1, n_atoms)
        ).squeeze(1)                                    # (B, n_atoms)
        m = project_distribution(target_dist, R_n, d, gamma**n_step,
                                  support, v_min, v_max, n_atoms)

    # Cross-entropy per sample
    log_pred = torch.log(pred_dist.clamp(min=1e-8))
    per_sample_ce = -(m * log_pred).sum(dim=1)          # (B,)
    weighted_loss = (weights * per_sample_ce).mean()

    return weighted_loss, per_sample_ce.detach()         # 第二個用於 PER priority update
```

### 4.4 `src/animate.py` 修改

`make_dashboard_gif()` 邏輯**完全不動**。`main()` 加 4 處小改：

```python
parser.add_argument('--exp', required=True, choices=[
    # HW3-1
    'naive_static', 'replay_static', 'replay_random',
    # HW3-2
    'replay_player', 'double_player', 'dueling_player', 'combined_player',
    # HW3-3
    'baseline_random', 'clip_random', 'sched_random',
    'huber_random', 'full_random',
    # HW3-4
    'combined_random', 'rainbow_random',
])

# Stage dispatch: HW3-4 cells live under results/HW3-4/
hw3_3 = {'baseline_random', 'clip_random', 'sched_random',
         'huber_random', 'full_random'}
hw3_4 = {'combined_random', 'rainbow_random'}
if args.exp in hw3_4:
    stage_dir = 'HW3-4'
elif args.exp in hw3_3:
    stage_dir = 'HW3-3'
elif args.exp.endswith('_player'):
    stage_dir = 'HW3-2'
else:
    stage_dir = 'HW3-1'

# Model factory dispatch
if args.exp == 'rainbow_random':
    from src.rainbow import build_rainbow_model
    factory = build_rainbow_model
else:
    dueling_exps = {'dueling_player', 'combined_player', 'combined_random'} | hw3_3
    factory = build_dueling_model if args.exp in dueling_exps else build_model
```

> **注意**：HW3-4 的 `combined_random` 用 `build_dueling_model`（重跑 HW3-3 Combined），`rainbow_random` 用 `build_rainbow_model`。`forward(state) → expected Q` 介面跟 HW3-1/2/3 一致，`animate.py` 內的 `qval = model(state); action_idx = argmax(qval)` 直接 work。

### 4.5 `metrics.json` schema（HW3-4 版）

完全沿用 HW3-1/2/3 schema，新增三個欄位：

- `components`: `{double, dueling, per, n_step, distributional, noisy}` — 6 個 bool（baseline 為 `{double:T, dueling:T, per:F, n_step:F, distributional:F, noisy:F}`，rainbow 全 True）
- `hyperparams.n_atoms` / `v_min` / `v_max` / `n_step` / `alpha` / `beta_start` / `beta_end` / `sigma_init`
- `final_loss_mean_last_100` 改為 KL loss（已乘 IS weight 並 mean over batch）；HW3-4 `combined_random` 仍是 MSE，數值不可直接比 HW3-4 `rainbow_random`，需在 report 中說明

範例（`rainbow_random`）：

```json
{
  "stage": "HW3-4: Rainbow DQN for random mode",
  "experiment": "rainbow_random",
  "mode": "random",
  "method": "rainbow",
  "components": {
    "double": true, "dueling": true, "per": true,
    "n_step": true, "distributional": true, "noisy": true
  },
  "hyperparams": {
    "epochs": 5000, "gamma": 0.9, "lr": 0.0001,
    "mem_size": 10000, "batch_size": 200, "max_moves": 50,
    "sync_freq": 500, "seed": 42, "snapshot_every": 250,
    "n_step": 3, "n_atoms": 51, "v_min": -10.0, "v_max": 10.0,
    "alpha": 0.5, "beta_start": 0.4, "beta_end": 1.0,
    "sigma_init": 0.5
  },
  "final_loss_mean_last_100": 1.85,
  "final_loss_std_last_100": 0.45,
  "win_rate": 0.93,
  "avg_steps_per_win": 2.55,
  "n_eval_games": 1000,
  "training_wall_time_sec": 60.0
}
```

---

## 5. 實驗設計

### 5.1 2 組實驗（all on `mode='random'`, seed=42, epochs=5000）

| # | 名稱 | 演算法 | Out dir | 預估 CPU |
|---|---|---|---|---|
| 1 | `combined_random` | HW3-3 Lightning Combined（重跑作 HW3-4 baseline） | `results/HW3-4/combined_random` | ~30s |
| 2 | `rainbow_random` | Rainbow（6 元件全套） | `results/HW3-4/rainbow_random` | ~60s |

> **Wall-time 估計**：HW3-3 baseline_random 約 32s。Rainbow 加 PER（O(log N) sample → +10%）、C51（output dim ×51 → +5%、KL projection → +20%）、Noisy（reset_noise → +5%）、N-step（O(1) → 0%），合計約 1.5–2× → 50–65s。2 組總計 ~1.5 分鐘。

### 5.2 執行命令

```bash
source .venv/bin/activate

# 訓練
python -m src.dqn_lightning --mode random --epochs 5000 \
       --out-dir results/HW3-4/combined_random                       # baseline 重跑
python -m src.rainbow                                                 # rainbow（預設值即 HW3-4 配置）

# Dashboard GIFs
python -m src.animate --exp combined_random
python -m src.animate --exp rainbow_random
```

### 5.3 預期結果（reality check）

對照 HW3-3 baseline_random（88.0% / loss 0.0315±0.0174 / wall 31.87s）。

| 實驗 | Final Loss (mean ± std) | Win rate | Avg steps | Wall time | 備註 |
|---|---|---|---|---|---|
| combined_random | ~0.030 ± 0.017 | ~88% | ~2.6 | ~30s | 應該重現 HW3-3 baseline_random（同演算法、同 seed）|
| rainbow_random | KL ~1.5–3 | ~91–95% | ~2.5 | ~50–65s | 主目標：win_rate ≥ 90%、明顯高於 baseline |

關鍵 sanity check：
1. **`combined_random.win_rate ∈ [86%, 90%]`** — 與 HW3-3 baseline_random 88.0% 在 ±2 pp 內。如果差太多，先查 seed 是否正確、是否真的是同一段程式（Lightning vs vanilla 隨機消耗順序略有差別，±2 pp 可接受）。
2. **`rainbow_random.win_rate ≥ 90%`** — 至少 2 pp 高於 baseline。如果沒達到：
   - 先檢查 PER buffer 是否真的有給 hard transition 高 priority（log 一下 priority 分佈）
   - 檢查 Noisy 是否真的有 noise（σ 沒有訓到全 0）
   - 檢查 C51 distribution 是否合理（不該是 uniform 也不該全壓在一個 atom）
3. **`rainbow_random` loss 跟 baseline 不可直接比**：cross-entropy/KL 跟 MSE 量綱不同。需在 report 中說明。Reality check 用 expected-Q 變化或 win rate 而非 raw loss number。

### 5.4 評估方法

完全沿用 `src/utils.evaluate(model, mode='random', n_games=1000)`，**`utils.py` 不動**。

`build_rainbow_model()` 回 `DistributionalDuelingMLP` instance；`evaluate` 只用 `model(state) → (B, n_actions)` 介面 → 因為我們 forward 直接回 expected Q，相容。

**Eval 確定性處理（在 `train_rainbow` 內，不動 `utils.py`）**：

```python
online.eval()                    # NoisyLinear 切換到 μ-only（self.training=False）
eval_result = evaluate(online, mode=mode, n_games=eval_n_games)
online.train()                   # 還原
```

對 dashboard GIF 同樣有意義：`animate.py` 走 `model = factory(); model.load_state_dict(sd); model.eval()`（既有邏輯第 145 行就有 `model.eval()`），所以 GIF rollout 也是 μ-only deterministic。HW3-4 不需要動 `animate.py` 這部分邏輯。

---

## 6. 報告 `HW3_4_report.md` 結構

沿用 HW3-1/2/3 章節風格：

```markdown
# HW3-4: Rainbow DQN for random mode（加分題）
## Hessel 2018 的 6 元件整合：在 4×4 Gridworld 上的完整 reproduce

> 作者：charles88　|　課程：深度強化學習　|　日期：2026-05-06
> Repo：<github 連結>

1. 作業目標 + 為什麼做 Rainbow
   1.1 HW3-3 留下的 12% 失敗：random mode 的剩餘痛點
   1.2 Rainbow = Double + Dueling + PER + N-step + Distributional + Noisy
2. 4 個新元件原理摘要
   2.1 Prioritized Experience Replay（sum-tree、α/β、IS weights）
   2.2 N-step returns（n=3）
   2.3 Distributional RL / C51（categorical projection）
   2.4 Noisy Networks（factorised Gaussian noise）
3. 整合架構：DistributionalDuelingMLP
   3.1 V/A heads × n_atoms × softmax
   3.2 Dueling aggregation per atom
   3.3 Forward 介面如何對齊既有 evaluate / animate
4. 實作說明（程式碼導讀）
   4.1 NoisyLinear
   4.2 SumTree + PrioritizedReplayBuffer
   4.3 NStepBuffer
   4.4 Categorical projection
   4.5 訓練迴圈整合
5. 實驗結果（2 組對比）
   5.1 量化指標表
   5.2 Loss 曲線（注意 KL vs MSE 不可比）
   5.3 Dashboard GIF
6. 結論
   6.1 哪個元件貢獻最大（推測，因為沒做 ablation）
   6.2 HW3-1 → HW3-4 整體軌跡（85.5% → 100% / 88% → 93%）
   6.3 4×4 Gridworld 太小做不出 Rainbow 全貌的 caveat
```

長度約 8–10 頁。圖片用相對路徑：`![](results/HW3-4/rainbow_random/loss.png)`。

---

## 7. 測試（pytest）

新增 1 個 test 檔，遵循 HW3-1/2/3 的 smoke test 模式：

```python
# tests/test_rainbow.py — 8 個測試
import pytest, torch
from src.rainbow import (
    NoisyLinear, DistributionalDuelingMLP, build_rainbow_model,
    SumTree, PrioritizedReplayBuffer, NStepBuffer,
    project_distribution, train_rainbow,
)


def test_noisy_linear_forward_shape():
    """NoisyLinear forward 形狀正確、reset_noise 真的改變輸出。"""

def test_distributional_model_shapes():
    """forward(state) → (B, n_actions); forward_dist(state) → (B, n_actions, n_atoms);
    每個 (B, a) softmax 機率總和 ≈ 1。"""

def test_dueling_aggregation_zero_mean_per_atom():
    """A - mean A 的 mean over actions 應該為 0（per atom）。"""

def test_sum_tree_basic():
    """add → total 累加正確；sample(s) 對 prefix sum 取 prefix 的 idx 正確；update 反映在 total。"""

def test_per_buffer_sampling_and_priority():
    """新加入用 max priority 抽到；update_priorities 後權重變化反映在抽樣分佈。"""

def test_n_step_buffer():
    """append n 個 1-step → 第一個 n-step transition 出現；
    R^(n) = sum γ^k · r_k 正確；
    early done → truncated n-step 正確。"""

def test_project_distribution_basic():
    """target distribution 在 V_min..V_max 範圍內 → projection 機率總和 ≈ 1；
    done state → projection 是 point mass at clip(R, v_min, v_max)。"""

def test_train_rainbow_smoke(tmp_path):
    """Smoke：epochs=5, mem_size=50, batch_size=8, max_moves=5, eval_n_games=2 → 全 artifact 落地。"""
    out = tmp_path / 'smoke'
    metrics = train_rainbow(
        epochs=5, mem_size=50, batch_size=8, max_moves=5,
        snapshot_every=2, eval_n_games=2,
        out_dir=str(out),
    )
    assert (out / 'checkpoint.pth').exists()
    assert (out / 'losses.npy').exists()
    assert (out / 'loss.png').exists()
    assert (out / 'metrics.json').exists()
    assert metrics['method'] == 'rainbow'
    assert metrics['components']['per'] is True
    assert metrics['components']['noisy'] is True
```

預期測試總數：HW3-3 的 49 → HW3-4 後 57 個（+8），全綠通過。

---

## 8. Chatlog `chatlog4.md`

完整保留 HW3-4 期間（從本次 brainstorming 開始到報告完成）的對話紀錄，格式同 HW3-1/2/3 chatlog：

- 每個 turn 標 `## Turn N`
- `**User：** ...` + `**Claude：** ...`（不貼大段 tool output，只留執行摘要）
- 模型標註：Claude Opus 4.7 (1M context)
- 對話日期：2026-05-06

---

## 9. README.md 更新

HW3-4 commit 時順手修兩處：

1. **後續階段表**：HW3-4 列從「⏳ 規劃中」改為「✅ 已完成」並指向 `HW3_4_report.md`。
2. **新增 HW3-4 段落**：Rainbow 簡介、2 組實驗對比表、嵌入 loss.png 與 dashboard.gif、與 HW3-1/2/3 段落形式對稱。
3. **三段策略動畫對照表**：擴成四段（HW3-1 / HW3-2 / HW3-3 / HW3-4），讓「演算法逐階段升級」軌跡完整呈現。
4. **三階段一覽表**：擴成四階段一覽。

---

## 10. Commit 策略

8 個 commits，對應 HW3-3 風格：

1. `docs(spec): add HW3-4 design doc`（本文件）
2. `docs(plan): add HW3-4 implementation plan`（writing-plans 產出）
3. `feat(rainbow): implement NoisyLinear + DistributionalDuelingMLP`
4. `feat(rainbow): implement SumTree + PrioritizedReplayBuffer + NStepBuffer`
5. `feat(rainbow): implement train_rainbow loop with categorical projection + CLI`
6. `test(rainbow): cover noisy/dueling/sumtree/per/nstep/projection/smoke`
7. `experiment(hw3-4): run combined + rainbow + dashboard GIFs`
8. `docs(hw3-4): add report + chatlog + README update`

> 細分到 3-4-5 三個 feat commit 是因為單一 commit 加 600 LOC 不利 review；Rainbow 的 6 個 sub-modules 內部相對獨立，分 3 commit 剛好對應「網路 / 資料結構 / 訓練迴圈」三層。

---

## 11. Out of Scope

下列項目不在本作業範圍：

- **元件 ablation**（PER 單獨、N-step 單獨、C51 單獨、Noisy 單獨）— 4 個元件高度互依（C51 改 loss、PER 改抽樣、N-step 改 target、Noisy 換掉探索），單獨拆開的對照組合理但 wall time 翻 5 倍且邊際資訊在 4×4 Gridworld 上很有限。報告會在「結論」一段討論個別元件的預期貢獻但不做實驗驗證。
- **多 seed 平均**（單一 seed=42 對齊 HW3-1/2/3）。
- **Hyperparameter sweep**（α/β/sigma_init/V_min/V_max/n_atoms 全部沿用文獻預設，不做 grid search）。
- **GPU 加速**（小網路 CPU 已足夠；強制 `device='cpu'`）。
- **TensorBoard / W&B logger**（`losses.npy` 已是唯一 loss 儲存格式）。
- **Player / static mode 的 Rainbow 對照**（HW3-2 已顯示 player 4 變體都 100%，看不出 Rainbow 額外提升空間；static mode HW3-1 Naive 就 100%，更看不出）。
- **Lightning 移植**（HW3-3 已交付框架轉換；HW3-4 改回 vanilla 是基於演算法整合的可讀性）。
- **Dueling-only Rainbow / Distributional-only Rainbow**（這些都是 paper 中的中間態；HW3-4 直接交付完整 Rainbow）。
- **Rainbow paper 之外的 modern variants**（IQN、QR-DQN、Munchausen DQN、APE-X 等不在範圍）。
