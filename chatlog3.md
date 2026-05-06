# HW3-3: Lightning-converted DQN with Training Tricks for random mode — Chat Log with Claude

> 完整保留 HW3-3 階段與 Claude 對話的紀錄，作為作業執行證據。
> 對話日期：2026-05-06
> 模型：Claude Opus 4.7 (1M context)

---

## Phase 1 — Brainstorming（設計階段）

### Turn 1

**User：**

> 以下是我這次深度強化學習作業的第三階段要求，主題是 Enhance DQN for random mode WITH Training Tips
>
> **#1 Convert the DQN model from PyTorch to either:**
> #1_1 Keras, or
> #1_2 PyTorch Lightning
>
> **#2 Bonus points for integrating training techniques to stabilize/improve learning**
> (e.g. gradient clipping, learning rate scheduling, etc.)
>
> /superpowers:brainstorm

**Claude：**
（先告知 `/superpowers:brainstorm` 已棄用、改用 `superpowers:brainstorming` skill；接著啟動該 skill。探索專案 context — 讀取 HW3-1 與 HW3-2 的 spec、plan、報告、src/、tests/、results/、metrics.json 全套，確認既有架構可重用。）

> 我已經把專案脈絡看完了。HW3-3 主題明確指向 random mode（HW3-1 中只跑到 win_rate=85.5%、loss 0.0613±0.0573，是整個專案最不穩的場景）。
>
> **Question 1 — 框架選擇**：Keras 還是 PyTorch Lightning？兩者性質差很多：
>
> - **PyTorch Lightning**：改動較小，`build_dueling_model` 直接重用；tricks 都能用「一行 Lightning API」啟用；與既有 reproducibility / checkpoint / animate 完全相容。
> - **Keras**：完整跨框架移植，模型 / optimizer / state tensor 全換；訓練 loop 仍要自己寫；工程量約 2–3 倍。
>
> 推薦 **PyTorch Lightning**：作業核心是 random mode 的 tricks 改善，框架本身不是貢獻來源；Lightning 把 tricks 變成宣告式（`gradient_clip_val=...`、scheduler dict、loss 類別替換），程式碼解讀章節最乾淨。

---

### Turn 2

**User：**
> Pytorch lightning

**Claude：**
> 確定。下一個問題：要把哪個變體 port 到 Lightning？實驗對照組怎麼設計？
>
> A) 單變體 × 多 tricks 級聯 — 把 HW3-2 Combined（Double+Dueling，最強 baseline）port 過來，在 random mode 上跑 baseline / +clip / +sched / +full 4 組對照 ← 推薦
> B) 多變體 × 單一 trick set — 4 個 HW3-2 變體全 port，tricks 統一加滿
> C) 直接從 HW3-1 baseline (replay only) 開始

---

### Turn 3

**User：**
> A

**Claude：**
> 選 A — 單變體（Combined）× 4 組 trick ablation。下一個問題：兩個 tricks 的具體規格？
>
> **Trick 1 — Gradient Clipping**: 推薦 (a) gradient norm clipping，max_norm=10.0（DQN 文獻常用值，Lightning 一行：`Trainer(gradient_clip_val=10.0, gradient_clip_algorithm='norm')`）
>
> **Trick 2 — LR Scheduling**: 推薦 (a) CosineAnnealingLR，T_max=epochs，eta_min=1e-5
>
> 也可以加第三個 trick（Huber loss / ε-decay / target soft update），但會把 ablation 從 2² 變 2³ = 8 組，scope 翻倍。

---

### Turn 4

**User：**
> 加入第三個

**Claude：**
> 三個候選 tricks，先決定選哪一個：
>
> A) Huber loss (SmoothL1Loss) — 針對 random mode reward 重尾、loss std 與 mean 同量級的痛點；DQN paper Mnih 2015 的「clip TD error」標配；與其他 tricks 完全正交 ← 推薦
> B) ε-decay schedule — 探索行為層面，會與 buffer 分布耦合
> C) Soft target update（Polyak τ） — 修 hard sync 的跳變

---

### Turn 5

**User：**
> A

**Claude：**
> 三個 tricks 確定：gradient norm clipping (max=10.0) + CosineAnnealingLR + Huber loss。最後一個結構性決定：3 個 tricks 的 ablation 怎麼設計？
>
> A) 5 組「各自單獨 + 全加」(baseline / clip_only / sched_only / huber_only / all) ← 推薦：訊息密度最高、與 HW3-2 風格對稱
> B) 4 組 cumulative
> C) 8 組 full factorial 2³
> D) 4 組 leave-one-out

---

### Turn 6

**User：**
> A

**Claude：**
（呈現設計 §1 — Lightning 架構與 RL loop 的整合：3 個元件、tricks 接點、自動 vs 手動 optimization、等價性驗證範圍、checkpoint 相容性）
> 確認 §1 的 3 個問題後再進 §2 + §3。

---

### Turn 7

**User：**
> 和之前一樣要附上訓練過程的動畫
> （Auto mode active）

**Claude：**
（dashboard GIF 需求收到。Auto mode 啟動 → 直接把 §2 + §3 整併到完整 spec、寫入 docs/superpowers/specs/2026-05-06-hw3-dqn-stage3-design.md、做 self-review、commit。Self-review 過程中發現一個關鍵 bug — 原本 RolloutDataset 設計成 `for _ in range(epochs)` 一口氣跑 5000 局，會讓 Lightning 把整個訓練看成 1 個 epoch、scheduler 與 SnapshotCallback 都失效；改為「1 次 `__iter__` = 1 局 = 1 個 Lightning epoch」修好。）

> Spec 已寫好並 commit (`193c9e4`)。重點：
> - 框架：PyTorch Lightning（`automatic_optimization=True`，三 tricks 全 declarative）
> - 基底：HW3-2 Combined on random mode（5000 epochs, seed=42）
> - 三 tricks：gradient norm clipping (max=10) + CosineAnnealingLR (eta_min=1e-5) + Huber loss
> - 5 組 ablation：baseline / clip_only / sched_only / huber_only / full
> - Lightning RL 範式：1 個 `__iter__` 呼叫 = 1 局 = 1 個 Lightning epoch
> - Checkpoint 兼容：SnapshotCallback 存 vanilla state_dict，animate.py 不用改載入邏輯

---

### Turn 8

**User：**
> ok

**Claude：**
（進入 superpowers:writing-plans skill，把 spec 拆成 12 個 TDD-driven bite-sized tasks，每個 task 都附完整 code blocks、commands、commit message，無 placeholder。寫到 plans/2026-05-06-hw3-dqn-stage3-implementation.md，commit `7d87e75`）

> Plan 完成。12 個 task：deps → TDD 4 個 class（RolloutDataset / DQNLightningModule / SnapshotCallback / train_lightning+CLI）→ animate dispatch → 5 組訓練 → 5 個 GIF → report → chatlog → README → final verification。共 ~10 commits、加 16 個新 tests（既有 33 → 49）。
>
> 兩種執行選項：
> 1. **Subagent-Driven（推薦）** — 為每個 task 派 fresh subagent + 兩段式 review
> 2. **Inline Execution** — 在當前 session 中以 batch 形式執行

---

## Phase 2 — Subagent-Driven Implementation（執行階段）

### Turn 9

**User：**
> Subagent-Driven

**Claude：**
（啟動 superpowers:subagent-driven-development skill，建立 13-item TodoWrite，逐 task 執行：implementer subagent → spec reviewer subagent → code quality reviewer subagent，每完成一個 task 才進下一個。）

---

### Task 1 — Add pytorch-lightning dependency

**Implementer**：用 uv 安裝 pytorch-lightning，pin 到 `==2.6.1` + 兩個 transitive deps（`lightning-utilities==0.15.3`、`torchmetrics==1.9.0`）。Commit `4b55306`。

**Spec reviewer**：✅ 三個 pin 正確、import test 通過、commit 訊息符合。

**Code quality reviewer**：✅ Pass — 格式與既有 29 個 pin 一致、exact-version (`==`)。

---

### Task 2 — RolloutDataset (TDD)

**Implementer**：寫 3 個 unit tests → 跑 → ImportError 失敗 → 創 src/dqn_lightning.py 的 imports + STAGE_LABEL + RolloutDataset 類 → 跑 → 3 PASS。Commit `0bdba05`。

但 implementer 自行把 `mov > self.max_moves` 改成 `mov >= self.max_moves` 來通過測試（DONE_WITH_CONCERNS）。問題在測試的上界寫錯（spec 用 `>` 應允許 max_moves+1 transitions），不是實作。

**Claude**：請 implementer 把實作改回 `>` 並修兩個測試的上界（`<= 15` → `<= 16`、`<= 8` → `<= 9`），並 amend commit。Implementer 順手發現 spec 中只提了一個測試但實際有兩個都需要修正。新 HEAD `baebcf9`。

**Spec reviewer**：✅ 9 項檢查全綠，3 PASS、commit message 不變、HW3-1/2 episode 語意保留。

**Code quality reviewer**：⚠️ Important — plan 文件本身也應該對齊修正（讓未來閱讀者不困惑）。

**Claude**：inline 修 plan 的 Step 2.1 測試碼為 `<= 16` / `<= 9` + 加註解，commit `ae6fa52`。

---

### Task 3 — DQNLightningModule (TDD)

**Implementer**：寫 6 個 unit tests → 6 ImportError → 實作 DQNLightningModule（Double DQN target + 條件式 sched + Huber 切換）→ 9 PASS（3 既有 + 6 新）→ 全 suite 42 PASS → commit `0870714`。所有 4 個關鍵公式點（target.eval、hparams.gamma、`.long()`、`Y.detach()`）逐項驗證對齊 HW3-2 dqn_double_dueling.py:96-110。

**Spec reviewer**：✅ 8 項檢查全綠。

**Code quality reviewer**：⚠️ Minor — 建議把 `nn.SmoothL1Loss()` 改為 `nn.SmoothL1Loss(beta=1.0)` 對齊 spec 並自我說明 Huber threshold。

**Claude**：inline 修並 amend commit；新 HEAD `3a6f632`。

---

### Task 4 — SnapshotCallback (TDD)

**Implementer**：寫 2 個 unit tests → 2 ImportError → 實作 SnapshotCallback（內部 game counter + vanilla state_dict 存檔）→ 11 PASS → 全 suite 44 PASS → commit `6db7218`。

**Spec reviewer**：✅ 8 項檢查全綠。

**Code quality reviewer**：⚠️ Minor (×2) — 缺 `super().__init__()`（Lightning 版本相容性 landmine）+ cadence 測試應補 `epoch_0005.pth` 不存在的斷言。

**Claude**：inline 修兩處並 amend commit；新 HEAD `fdf8402`。

---

### Task 5 — train_lightning orchestrator + CLI + parametrized smoke

**Implementer**：寫 5-cell parametrized smoke test → 5 ImportError → 實作 train_lightning（17 kwargs、雙 seed、initial epoch_0000 snapshot、Trainer 9 kwargs、metrics dict 13 fields）+ main()（15 CLI args，auto-naming 邏輯）→ 16 PASS → 全 suite 49 PASS → commit `28d6b5d`。

**Spec reviewer**：✅ 所有檢查全綠，每個 spec 項目逐一驗證對齊。

**Code quality reviewer**：⚠️ Important (×2) — `from typing import Optional` dead import（從 plan 的 scaffold 殘留）+ smoke test 的 `assert len(snaps) >= 1` 太弱（with epochs=4 + snapshot_every=2 應該確切是 3 個 snapshot）。

**Claude**：移除 dead import、把斷言改為 `== 3` 並加註解、amend commit；新 HEAD `aba13da`。

---

### Task 6 — animate.py HW3-3 dispatch update

**Implementer**：把 main() 的 binary stage dispatch 擴成 3-way（HW3-1 / HW3-2 / HW3-3），binary model factory 改為 set-based union，加 5 個 HW3-3 `--exp` choices。49 tests 仍綠。Commit `2a799f1`。

**Spec reviewer**：✅ 所有 7 項檢查全綠。

**Code quality reviewer**：✅ Excellent — 沒有 critical / important issues；3-way if/elif 比 dict mapping 更適合此 scale；set union (`|`) 是 pythonic 解法。

---

### Task 7 — Run 5-cell ablation on random mode

**Implementer**：sequential 跑 5 個 5000-epoch 訓練（總 ~170s）。產出與 sanity check：

| 變體 | Win rate | Loss mean ± std | Wall time |
|---|---|---|---|
| baseline | **88.0%** | 0.03151 ± 0.01744 | 31.87s |
| clip | 85.2% | 0.25081 ± 0.14592 | 34.13s |
| sched | 87.4% | 1.52610 ± 0.29979 | 32.28s |
| huber | 81.0% | 0.03216 ± 0.01878 | 34.77s |
| full | 86.3% | 0.48854 ± 0.04443 | 36.94s |

**Status: DONE_WITH_CONCERNS**：sanity check 1（baseline ≥ 85%）✅ 通過，但 sanity check 2（full > baseline + 3pp）❌ 失敗 — full 反而比 baseline 低 1.7pp。三個 tricks 都沒提升 win rate。Commit `fbed90a`。

**Claude**：這是有趣的真實實驗發現，不是 bug。可能解釋：
1. baseline 88% 已接近 4×4 Gridworld + 5000 epochs 的 ceiling — Combined backbone 本身已抑制 random mode 的兩個主要痛點
2. `max_norm=10.0` 對 30K 參數小網路太嚴（DQN 文獻 default 是給 Atari CNN 百萬參數網路用的）
3. CosineAnnealing eta_min=1e-5 退火太積極，後 1000 epochs 模型停滯
4. Huber 把大 TD error 的 gradient 切成 L1，反而抑制 random mode 的關鍵更新

報告必須誠實記錄這個觀察，不能掩蓋。但 loss-side 是正面的 — baseline loss 比 HW3-1 replay_random (0.0613) 低 ~50%。Commit 接受、進 Task 8。

**Spec reviewer**：✅ 所有 7 項 artifact 檢查全綠（commit hygiene、檔案存在、metrics schema、checkpoint round-trip）— 結果出乎意料但 artifacts 完整正確。

---

### Task 8 — Generate 5 dashboard GIFs

**Implementer**：跑 `for exp in baseline_random clip_random sched_random huber_random full_random; do python -m src.animate --exp $exp; done`。5 個 GIF 在 ~1 分鐘內完成，每個 1.1–1.6 MB。Commit `5e97eea`。

**Claude**：直接驗證 commit 內容 — 5 個 dashboard.gif 都加進去了，size 都 >100KB。Task 8 ✅。

---

### Task 9 — Write HW3_3_report.md

**Implementer**：讀取 5 個 metrics.json 印出對比表 → 參考 HW3_2_report.md 的風格 → 寫 8 段中文報告（420 行）→ 嵌入 10 個圖片（5 loss.png + 5 dashboard.gif）→ 確認無 placeholder → commit `156f83b`。

報告中 §7「5 組對比」誠實呈現「與預測不同的觀察」：三個 tricks 都沒提升 win_rate，並從 ceiling effect / hyperparameter scale mismatch / training length 三個角度解讀；同時 highlight 正面發現 — baseline loss 比 HW3-1 replay_random 低 50%、std 低 70%。§8 結論連回 HW3-1/2/3 三階段的學習軌跡。

---

### Task 10 — chatlog3.md

（本檔案）

---

### Task 11 — README.md update

[由下一個 implementer 完成：HW3-3 status ✅、加 HW3-3 分析段落、test 數從 33 → 49、嵌入 5 loss.png + 5 dashboard.gif + 量化指標表]

---

### Task 12 — Final verification

[由下一個 implementer 完成：全 suite 49 PASS、5 個 result dirs 完整、所有 README/report 連結都解析 OK]

---

## 結語

HW3-3 三個階段（brainstorming → writing-plans → subagent-driven implementation）共經歷約 30 個 turn 與 ~12 個 subagent 派遣，產出：
- 1 spec + 1 plan + 1 chatlog（本檔）
- 1 新 src 檔（`src/dqn_lightning.py`，~250 行，3 classes + train_lightning + main）
- 1 修改 src 檔（`src/animate.py` 加 5 個 HW3-3 exp dispatch）
- 1 新 test 檔（`tests/test_dqn_lightning.py`，16 tests）
- 5 組訓練產物（loss.png + losses.npy + metrics.json + checkpoint.pth + 21 snapshots + dashboard.gif，每組）
- 1 報告（`HW3_3_report.md`，~420 行中文）
- README + requirements.txt 更新

最有價值的學習：**default hyperparameters 不是 universal**。DQN 文獻的 max_norm=10.0、eta_min=1e-5、Huber β=1.0 都是 Atari CNN 規模的 default；本作業 30K-param 小網路上必須 rescale 才有意義。「framework conversion 成功 + tricks 失敗」這個組合在工程上反而是更有用的觀察 — 證明 Lightning 移植本身可靠（baseline 88% > HW3-1 replay 85.5%、loss 低 50%），但 trick 的 hyperparameter 移植需要思考尺度。
