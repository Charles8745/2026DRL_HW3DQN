# HW3-4: Rainbow DQN for random mode（加分題）— Chat Log with Claude

> 完整保留 HW3-4 階段與 Claude 對話的紀錄，作為作業執行證據。
> 對話日期：2026-05-06 → 2026-05-07
> 模型：Claude Opus 4.7 (1M context)

---

## Phase 1 — Brainstorming（設計階段）

### Turn 1

**User：**

> 使用 superpower brainstorming
> 以下是我這次深度強化學習作業的第四階段(加分題)要求，主題是使用 Rainbow DQN 解 Random Mode GridWorld
> #1 先分析
> #2 接著講解怎麼實作，並實作
> #3 一樣要附上訓練過程的動畫

**Claude：**

啟動 `superpowers:brainstorming` skill。先建立 todos、探索專案脈絡（讀取 HW3-1/2/3 的 spec、plan、report、src/、tests/、metrics.json、現有 GIF），確認可重用 baseline。Auto Mode 啟動：minimize interruptions、prefer action over planning，但 brainstorming HARD-GATE 要求設計 → 用戶批准 → 才能進實作。

接著輸出三段：

**一、分析**：
- 解釋為什麼是 Rainbow——HW3-3 baseline 已經到 88%，剩下 12% 失敗來自三個 Combined 沒處理的方向：(1) 重要 transition 學得不夠多 → PER；(2) TD bootstrap 太短視 → N-step；(3) value 估計過度確定 → Distributional / C51；(4) ε-greedy 不適配 state → Noisy Nets。
- Rainbow = 6 元件 = HW3-2 Double + Dueling + 4 個新元件。
- 4 個新元件的數學摘要（PER 的 sum-tree + α/β、N-step return 公式、C51 的 categorical projection、Noisy Nets 的 factorised Gaussian noise）。

**二、實作方案**：
- 對比 3 個方案：A. 完整 Rainbow（推薦）/ B. Rainbow-lite（只加 PER + N-step） / C. 6 元件 ablation（過於耗時）。
- 推薦 A：HW3-1/2/3 都採取「實作完整、實驗量適中」風格。少做任何元件都不能叫 Rainbow。
- 2 組實驗：`combined_random`（重跑 HW3-3 Combined 作 baseline）+ `rainbow_random`（主角）。
- 改回 vanilla PyTorch（不用 Lightning），因為 PER 的 priority writeback 跟 Lightning 的 `training_step` 純 loss 抽象有摩擦。
- 單檔 `src/rainbow.py` ~600 LOC，6 個 block 集中。
- Forward 介面對齊既有腳本：`forward(state)` → expected Q，跟 HW3-1/2/3 接口完全相容。

**三、待確認設計細節**：列出 5 個決定點（方案選 A/B/C、C51 atoms 數、N-step 的 n、Noisy σ_init、是否同步寫 report/chatlog/README），請用戶選擇或回「都用你推薦的」。

---

### Turn 2

**User：**

> 都用你推薦的

**Claude：**

確認方案 A（完整 6 元件 Rainbow）+ C51 51 atoms + n=3 + σ_init=0.5 + 同步寫所有文件。

寫 spec 到 `docs/superpowers/specs/2026-05-06-hw3-dqn-stage4-design.md`（640 行，11 個 section：作業需求對應、設計原則、6 元件演算法規格、檔案結構、模組介面、實驗設計、報告大綱、測試、chatlog、README、commit 策略、Out of Scope）。Self-review 修了 3 處：
1. §2.6 NoisyLinear 加 eval 行為條款（`self.training` 旗標切換 μ-only / 加噪音）
2. §4.4 把未定義的 `hw3_3_cells` 改成 `hw3_3` 集合
3. §5.4 確認沿用 `evaluate()` 與 `animate.py:146` 的 `model.eval()`，`utils.py` 完全不動

Commit `44a15b9`。請用戶 review。

---

### Turn 3

**User：**

> ok

**Claude：**

啟動 `superpowers:writing-plans` skill。寫計畫到 `docs/superpowers/plans/2026-05-06-hw3-dqn-stage4-implementation.md`（2095 行、15 個 task、14 commits target、~22 個測試）：

| Task | 內容 |
|---|---|
| 1 | Module skeleton + import test |
| 2 | NoisyLinear（4 tests） |
| 3 | DistributionalDuelingMLP + factory（4 tests） |
| 4 | SumTree（4 tests） |
| 5 | PrioritizedReplayBuffer（3 tests） |
| 6 | NStepBuffer（3 tests） |
| 7 | Categorical projection（2 tests） |
| 8 | train_rainbow + CLI + smoke test（1 test） |
| 9 | animate.py dispatch |
| 10 | combined_random 實驗 |
| 11 | rainbow_random 實驗 |
| 12 | HW3_4_report.md |
| 13 | README.md 4-stage 擴展 |
| 14 | chatlog4.md |
| 15 | Final verification |

Self-review：所有 spec section 都對應到 task；signature 跨 task 一致；無 placeholder。Commit `4651c3b`。請用戶選 Subagent-Driven 或 Inline Execution。

---

## Phase 2 — Implementation（實作階段，subagent-driven）

### Turn 4

**User：**

> Subagent-Driven

**Claude：**

啟動 `superpowers:subagent-driven-development` skill。每個 task 派一個 fresh subagent，task 之間做兩階段 review（spec compliance → code quality），全程在 main branch 上連續執行。

**Task 1（scaffold）** — sonnet implementer 在 `4ba1602` 寫 module skeleton（imports + STAGE_LABEL）+ 1 個 import smoke test。haiku spec reviewer ✅、sonnet quality reviewer ✅。

**Task 2（NoisyLinear）** — sonnet implementer 在 `d684047` 實作 factorised Gaussian noise，4 tests 全綠（forward shape、reset_noise、eval determinism、train mode noise）。reviewers ✅，注意到 `self.training` 旗標讓 eval 自動 deterministic、`weight_epsilon` 是 register_buffer 跟 `.to(device)` / state_dict 走。

**Task 3（DistributionalDuelingMLP）** — sonnet implementer 在 `f9f70eb` 實作網路：plain trunk + NoisyLinear V/A 雙頭、per-atom dueling aggregation、softmax over atoms、`forward()` 回 expected Q 跟 HW3-1/2/3 對齊。4 tests 全綠（output shape、distribution validity、`forward == sum(dist*support)` 一致性、`reset_noise` 傳遞到所有 NoisyLinear）。reviewers ✅。

**Task 4（SumTree）** — sonnet implementer 在 `181082f` 實作 binary heap-as-array。`add` / `update` / `sample(s)` 都 O(log N)。4 tests 全綠（total 累加、prefix sum 抽樣、update propagation、circular overwrite）。reviewers ✅。

**Task 5（PER）** — sonnet implementer 在 `edc2861` 實作 stratified PER。`(N * P(i)) ** -beta / max_w` IS weights、新 transition 給 max priority 確保至少抽到一次、β 線性退火。3 tests 全綠。reviewers ✅。

**Task 6（NStepBuffer）** — sonnet implementer 在 `f67b2b6` 實作滑動視窗。**implementer 抓到 spec 裡的 bug**：原本 spec 說 `done=True` 時 `append` 統一 return None，但 test #1 要求「window 滿 + done=True 時要立刻 emit」，否則無法通過。修正成正確的：window 滿且 done → emit。3 tests 全綠。reviewers ✅，認為這個修正其實比 spec 設計更好（PER 可以連續 receive transition、不必等到 episode end）。

**Task 7（categorical projection）** — sonnet implementer 在 `bc91c18` 實作 Bellemare 2017 Algorithm 1 的向量化版本。`Tz = clip(R + γ^n z, V_min, V_max)`、線性內插到相鄰 atom、`eq_mask` 補丁處理 Tz 剛好落在 atom 上的邊界。2 tests 全綠（mass preservation、done state point mass）。reviewers ✅。

**Task 8（train_rainbow + smoke test）** — opus implementer 在 `dc0312d` 整合 6 個元件成完整訓練迴圈。**implementer 抓到並修了 autograd version-mismatch bug**：原本 spec 順序是 online 先 forward → 進 no_grad block 又 reset_noise（in-place 修改 weight_epsilon buffer）→ backward 報錯。修正：把 no_grad block（target 分佈計算）放最前面，再做 autograd-tracked online forward。Smoke test 跑 epochs=4 全綠，full suite 71/71 全綠。spec reviewer ✅；code quality reviewer ✅，特別肯定這個 autograd fix 是「正確且最小手術」。

**Task 9（animate.py dispatch）** — sonnet implementer 在 `218a77c` 加 HW3-4 stage + factory dispatch（lazy import `build_rainbow_model`，HW3-1/2/3 invocations 不會 import `src.rainbow`）。`combined_random` 走 `build_dueling_model`、`rainbow_random` 走 `build_rainbow_model`。reviewers ✅。

---

### Turn 5（Task 10：combined_random 實驗）

跑 `python -m src.dqn_lightning --mode random --epochs 5000 --seed 42 --out-dir results/HW3-4/combined_random`：**31.47s, win_rate=88.0%**，重現 HW3-3 baseline 的數值（同 seed、同程式）。21 個 snapshot、loss.png、metrics.json、dashboard.gif 全產出。Commit `b69ca49`。

---

### Turn 6（Task 11：rainbow_random 實驗，多次失敗）

第一次跑 `python -m src.rainbow`（spec 預設 lr=1e-4, mem_size=10000）：**516s, win_rate=21.7%**，遠遠低於 baseline 88% 也低於隨機亂走 ~25–30%。Plan 的 reality-check 規則說 win_rate < 0.85 要 DONE_WITH_CONCERNS。

進入診斷：
- Loss 從 3.83 收到 0.64（單調下降，optimizer 沒卡）
- 取 5 個隨機棋盤的 Q 值：全部聚在 −9.85 附近、不分動作；最戲劇的是「Goal 在右鄰格、最佳動作應該選 right、但 Q(right) 跟 Q(left) 一樣」
- Distribution 把 60–80% 機率質量壓在 z = −10
- 訓練全程 win rate 在 17–22% 振盪、從沒收斂

**Hypothesis 1：lr 太低。** 改 lr=1e-3（對齊 HW3-3 baseline）重跑：**429s, win_rate=19.3%**，更糟。

**Hypothesis 2：mem_size 太大。** 加上 mem_size=1000（對齊 HW3-3）重跑：**499s, win_rate=8.6%**，更更糟。

三組設定都 < 25%，確認**不是 hyperparameter 問題**，是 algorithmic 整合問題。

最終決定：**revert 到 spec 原設定（lr=1e-4, mem_size=10000）**，作為「按 Rainbow paper 與 Atari 文獻 default」的記錄；負面結果忠實記錄到 report 裡。re-run 重現 21.7%（seed=42 deterministic），生成 dashboard.gif，commit `86bd4d5`。

---

### Turn 7（Task 12：寫 HW3_4_report.md）

寫報告（404 行、7 個章節）：作業目標 → 4 個新元件原理 → 整合架構 → 程式碼導讀（5 個關鍵片段）→ 實驗結果（含失敗模式診斷四點）→ 為什麼 Rainbow 在 4×4 Gridworld 失效（4 個技術原因 + 規模 mismatch 解釋）→ HW3 四階段學習軌跡總結。

特別處理：
- 不掩飾 21.7% 的負面結果，直接放在第 1 章開頭「先講結論」
- 第 5 章列出 3 組 hyperparameter 嘗試結果
- 第 6 章解釋為什麼 distributional dueling 的 softmax 容易塌縮、PER 反而強化錯誤學習、Noisy σ 衰減過快
- 第 7 章把 HW3-1/2/3/4 win rate 軌跡列表，總結「演算法複雜度與任務規模要匹配」

Commit `e78b561`。

---

### Turn 8（Task 13：更新 README.md）

把 README 從三階段擴成四階段：
- 標題列：HW3-4 ✅（本次更新，加分題）
- 簡介：加入 HW3-4 段落，誠實標註負面結果
- 「四階段一覽」表格：加入 Rainbow 行（21.7% / 0.6409 KL）
- 動畫對照：改成 2x2 layout，加入 HW3-4 dashboard
- 新增「## HW3-4 分析結果」section（鏡像 HW3-3 結構：量化指標表 → loss 曲線 → 策略動畫 → 指向 report）
- 專案結構：加 rainbow.py、results/HW3-4/、HW3_4_report.md、chatlog4.md
- 使用方式：加 HW3-4 訓練 + animate 指令
- 設定：加 `src/rainbow.py` CLI flag 列表 + `build_rainbow_model` 描述
- 後續階段表：HW3-4 → ✅ 已完成
- 測試數：49 → 71

Commit `ead9b9b`。

---

### Turn 9（Task 14：寫本檔 chatlog4.md）

User：「繼續」。Claude 繼續執行 Task 14：寫 chatlog4.md（即本檔），記錄完整對話歷程，格式對齊 chatlog3.md（## Phase 標題、## Turn N、**User：** + **Claude：** 摘要）。

---

## Phase 3 — Final Verification（驗證階段）

### Turn 10（Task 15）

執行最終驗證：
- `pytest -v` 71/71 全綠
- `git status` 工作樹乾淨
- `git log --oneline` 確認 commit 序列完整
- 確認 results/HW3-4/{combined,rainbow}_random/ 兩組產物（checkpoint, losses, loss.png, metrics.json, snapshots/, dashboard.gif）都到位

HW3-4 完成。

---

## 過程中遇到的兩個技術 fix（值得記錄）

### Fix #1：NStepBuffer 的 done+full-window 邏輯（Task 6）

Spec 原版：
```python
if done:
    self._terminal_tail = (s_next, True)
    return None
```

但 test 要求 window 滿 + done=True 時要立刻 emit：
```python
if done:
    self._terminal_tail = (s_next, True)
    if len(self.window) == self.n:
        return self._make_n_step(self.window)
    return None
```

Implementer 在實作時抓到、修正、commit；reviewer 認為這個修正比 spec 更好。

### Fix #2：`_compute_loss` 的 noise reset 順序（Task 8）

Spec 原版：
```python
online.reset_noise(); pred_dist = online.forward_dist(s1)   # autograd-tracked
with torch.no_grad():
    online.reset_noise(); next_q = online(s2)               # in-place 修改 weight_epsilon!
    target.reset_noise(); target_dist = target.forward_dist(s2)
    m = project_distribution(...)
loss = -(m * log(pred_dist)).sum().mean()                    # 報錯：weight_epsilon version mismatch
```

PyTorch autograd 在 forward 時把 `weight_epsilon` 的版本捕捉進 graph，然後 no_grad block 裡 in-place 修改它的版本，backward 時版本不一致 → 報錯。

修正：把 no_grad block（target 分佈計算）放最前面，再做 autograd-tracked online forward：
```python
with torch.no_grad():                                # 先做 target，no_grad
    online.reset_noise(); next_q = online(s2)
    target.reset_noise(); target_dist = target.forward_dist(s2)
    m = project_distribution(...)
online.reset_noise()                                 # 後做 online pred，autograd
pred_dist = online.forward_dist(s1)
loss = -(m * log(pred_dist)).sum().mean()
```

這個 bug 在所有單元測試（22 個）都過了之後才在 smoke test 浮現——是「整合測試 catch unit test 漏的」教科書例子。

---

## Commits 序列

```
ead9b9b docs(readme): add HW3-4 Rainbow section + extend to 4-stage comparison
e78b561 docs(hw3-4): add HW3-4 report with negative-result analysis
86bd4d5 experiment(hw3-4): rainbow_random main result (full Rainbow DQN)
b69ca49 experiment(hw3-4): combined_random baseline (HW3-3 Lightning Combined re-run)
218a77c feat(animate): wire HW3-4 combined_random + rainbow_random dispatch
dc0312d feat(rainbow): implement train_rainbow loop + CLI
bc91c18 feat(rainbow): implement categorical projection (Bellemare 2017 Algorithm 1)
f67b2b6 feat(rainbow): implement NStepBuffer for n-step bootstrapping
edc2861 feat(rainbow): implement PrioritizedReplayBuffer with stratified sampling
181082f feat(rainbow): implement SumTree for O(log N) prioritized sampling
f9f70eb feat(rainbow): implement DistributionalDuelingMLP + build_rainbow_model
d684047 feat(rainbow): implement NoisyLinear with factorised Gaussian noise
4ba1602 feat(rainbow): scaffold src/rainbow.py module skeleton
4651c3b docs(plan): add HW3-4 Rainbow DQN implementation plan
44a15b9 docs(spec): add HW3-4 Rainbow DQN design doc
```

15 個 commit 對應 15 個 task（spec → plan → 8 個 feat（rainbow + animate）→ 2 個 experiment → 2 個 docs）。本檔（chatlog4.md）會在下一個 commit 中加入。

---

## 重點數字回顧

| 項目 | 數值 |
|---|---|
| `combined_random` win rate | **88.0%** |
| `rainbow_random` win rate | **21.7%** |
| Combined ↔ Rainbow 差距 | **−66.3 pp** |
| Rainbow 訓練時間 | 511.82s（vs Combined 31.47s，~16×） |
| Rainbow loss (KL) | 0.6409 ± 0.2229 |
| 程式碼新增 | `src/rainbow.py` ~700 LOC |
| 測試新增 | 22 個（總計 49 → 71） |
| Commits | 15 |
| 最大教訓 | 演算法複雜度與任務規模要匹配 |
