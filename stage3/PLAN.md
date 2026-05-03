# Stage 3 — High-Quality SFT v2 (Plan, NOT executing for now)

> **状态**：本次复现项目**未执行 Stage 3**，决定走 Stage 2-v2 → Stage 4 (DPO) 路径。
> 本文档保留 Stage 3 的完整计划，供未来扩展使用。
> **2026-05 更新**：基于 v2 final eval 实测数字重新评估，调整 Stage 3 优先级。

> **如果之后想做 Stage 3**：基于 `stage2-v2/` 复制改造即可。

---

## 📊 v2 final eval 实测（写于 2026-05）

| 指标 | v2 实测 | LLaVA-1.5-7B | gap | 决定 Stage 3 该优先什么 |
|---|---|---|---|---|
| **RefCOCO val Acc@0.5** | **78.1%** ⭐ | 30% | **+48 (我们大胜)** | 已饱和，**不再是优先级** |
| RefCOCO testB Acc@0.5 | 70.7% | 28% | +43 | 已饱和 |
| TextVQA acc | 61.7% ⭐ | 58.2% | +3.5 (我们略胜) | 已优秀，**保持即可** |
| NoCaps avg_len | 135 | ~80 | (incomparable, 我们更长) | 持平，保持 |
| NoCaps rep_rate | 0.5% | ~0% | -0.5 | 微调即可 |
| **POPE F1** | **0.76** | 0.86 | -10 | 中等优先 — **DPO 主战场** |
| **POPE Yes-ratio** | **75%** ⚠️ | ~53% | -22 | **核心问题** — DPO 一击致命 |
| **VQAv2 acc** | **56.5%** ⚠️ | 78.5% | **-22** ⚠️ | **最大短板** — Stage 3 主战场 |

**关键归因**（基于上面三个变量框架）：
- 大胜的项目（RefCOCO/TextVQA）：`SigLIP2 + 充足训练数据 + LoRA r=64` 三者协同
- 短板的项目（VQAv2）：`1.5B LLM 推理深度限制 + 没直接 expose VQAv2 train + LLaVA-Instruct 被稀释` 三者叠加

---

## 🎯 目标（v2 数据修订版）：专攻 VQA + 修 yes-bias，保住其他

Stage 2-v2 后**真实存在**的 4 类问题（按 gap 大小排）：

| # | 问题 | gap (vs LLaVA-1.5-7B) | Stage 3 解决思路 |
|---|---|---|---|
| 1 | **VQAv2 56.5%（差 22 点）** ⚠️ | -22 | **加 VQAv2 train 100K + GQA + A-OKVQA + OK-VQA** |
| 2 | **POPE Yes-bias 75%（差 22 点 vs 53%）** ⚠️ | -22 | LRV-Instruction（hard negatives）+ Stage 4 DPO 联手 |
| 3 | POPE F1 0.76（差 10 点） | -10 | 同上 |
| 4 | NoCaps rep_rate 0.5%（轻微）| 持平 | 持续监控，多任务保持 |

**已经达成的目标（不再追求）**：
- ✅ RefCOCO val 78%（v1 21% → v2 78%，大胜 LLaVA-1.5-7B 30%）
- ✅ TextVQA 62%（v1 ~25% → v2 62%，超过 LLaVA-1.5-7B 58%）
- ✅ NoCaps 长 caption 能力（135 词，rep 0.5%）

**不做的事**：
- ❌ 改架构（继续 LoRA + ProjectorWithNorm）
- ❌ DPO/RLHF（留给 Stage 4）
- ❌ 加大模型（继续 1.5B；如果想换 3B 是单独的实验）
- ❌ 继续堆 grounding 数据（v2 已饱和，再加边际收益 < 1 点）

---

## 📦 数据组成（v2 实测调优后版）

**核心调整**：grounding 减半（从 v2 的 126K → 60K，已经饱和），释放配额给 VQA 类。

```
LLaVA-Instruct-150K          150K  (27%)   ⭐ 加回 v1 量（v2 削成 100K 是误判）
🆕 VQAv2 train               100K  (18%)   ⭐⭐ 直接攻 VQAv2 短板（最重要）
ShareGPT4V_instruct          100K  (18%)   ⭐ 换文件 (216 词 detailed caption)
🆕 GQA (compositional)        50K  (9%)    新增：场景图组合推理
🆕 LLaVA-RLHF chosen (SFT)    50K  (9%)    新增：高质量推理对话
🆕 LRV-Instruction (negative) 30K  (5%)    新增：降 yes-bias
TextVQA                       28K  (5%)    保留：OCR (v2 已优秀，持平即可)
RefCOCO                       20K  (4%)    削减：v2 grounding 已饱和
RefCOCO+                      20K  (4%)    削减
RefCOCOg                      20K  (4%)    削减
🆕 A-OKVQA (knowledge)        17K  (3%)    新增：常识 VQA
🆕 OK-VQA                     14K  (3%)    新增：外部知识
─────────────────────────────────
TOTAL                        ~599K
```

**v2 vs Stage 3 数据对比**：

| 类别 | v2 占比 | **Stage 3 占比** | 变化 |
|---|---|---|---|
| Grounding | 36% (126K) | **10% (60K)** | **-26 点** ⬇️ 已饱和 |
| General VQA | 36% (128K) | **62% (377K)** | **+26 点** ⬆️ 主战场 |
| Long Caption | 28% (100K) | 17% (100K) | 持平绝对量 |
| OCR | 8% (28K) | 5% (28K) | 持平 |
| Hard Negative (新) | 0% | 5% (30K) | 攻 yes-bias |
| **总样本** | **354K** | **599K** | +69% |

### 各新增数据集

| 数据集 | HF repo | train 大小 | 字段格式 | 教什么 |
|---|---|---|---|---|
| **VQAv2 train** ⭐ | `HuggingFaceM4/VQAv2` | 444K | `{image, question, answers (10 个)}` | **直接攻 v2 短板** |
| **GQA** | `lmms-lab/GQA` | 943K | `{image, question, answer}` | 场景图衍生组合问题 |
| **A-OKVQA** | `HuggingFaceM4/A-OKVQA` | 17K | `{image, question, choices, rationale}` | 需要常识推理 |
| **OK-VQA** | `lmms-lab/OK-VQA` | 9K + 5K | `{image, question, answers}` | 需外部知识（地理 / 品牌等）|
| **LRV-Instruction** | `Lin-Chen/LRV-Instruction` | 400K（取 30K）| 含正反例 ("Is there X?" → "No, ...") | 专治 yes-bias |
| **LLaVA-RLHF SFT** | `zhiqings/LLaVA-RLHF-Data` | ~50K SFT subset | 多轮高质量对话 | 比合成数据更好 |

---

## ⚙️ 训练配置

### 起点
```
Stage 1 ckpt-11500
       ↓
Stage 2-v2 final (LoRA r=64 + projector)
       ↓
Stage 3 final (在 v2 基础上继续训)
```

### 推荐选项 A：继续 LoRA r=64

| 参数 | 值 | 备注 |
|---|---|---|
| 起点 ckpt | Stage 2-v2 final | LoRA + projector + base 全继承 |
| LoRA | r=64 (继续) | 不换结构，保留 v2 学到的内容 |
| LR | **5e-5** | Stage 2 的 1/4，refinement 不是 bulk learning |
| Batch | 8 × grad_accum 4 = 32 | 跟 v2 一致 |
| Epoch | 1 | 449K 够 |
| `--no_gradient_checkpointing` | ✅ | Blackwell 上省 30% 时间 |
| Held-out | 200/dataset | 训前抽出来 |

### 时长预估

```
599K samples / 32 = ~18,700 iters
18,700 × 2.77s (Blackwell, --no_gradient_checkpointing) = ~14.4h
+ overhead = ~15h
```

如果想缩短：n_vqav2 削到 50K → 总样本 549K → ~13h。

---

## 📊 预期效果（基于 v2 实测重新校准）

| 指标 | **v2 实测** | **Stage 3 预期** | 增益来源 |
|---|---|---|---|
| RefCOCO val Acc@0.5 | **78.1%** | 73-78% | grounding 减半 → 略降 1-5 点（可接受） |
| RefCOCO testB Acc@0.5 | 70.7% | 64-70% | 同上 |
| **VQAv2 acc** ⭐ | **56.5%** | **68-75%** ⭐⭐ | **VQAv2 train 直接 expose + 多任务 VQA 互补 = 主升点** |
| **POPE F1** | 0.76 | **0.79-0.83** | LRV 降假阳性 + 多任务再平衡 |
| **POPE Yes-ratio** ⭐ | **75%** | **58-65%** | **LRV-Instruction + 平衡数据修偏** |
| **GQA acc** | — | **55-65%** | 新能力 |
| **A-OKVQA acc** | — | **45-55%** | 新能力 |
| TextVQA acc | **61.7%** | 60-65% | 持平（v2 已优秀） |
| NoCaps avg_len | 135 | 130-160 | 用 instruct 文件可能略变 |
| NoCaps rep_rate | 0.5% | 0-0.5% | 持平 |
| NoCaps word_recall | 28.5% | 30-35% | 用 instruct 略升 |

**Stage 3 净效果**：
- ✅ VQAv2 +12-18 点（从最大短板变成"还行"）
- ✅ POPE Yes-bias 大幅缓解 -10~-17 点（但仍不如 DPO 干净）
- ✅ GQA / A-OKVQA 新能力（v2 完全没这些）
- ⚠️ RefCOCO 略降 0-5 点（grounding 已饱和的代价，可接受）

**关键认知**（来自 v2 实测）：
- **VQAv2 60-65% 接近 1.5B 天花板** —— 想突破到 75%+ 需要换 3B 模型
- **POPE Yes-bias 用 SFT 修不彻底** —— 真正攻坚还得靠 DPO（Stage 4）
- **所以即使做完 Stage 3，Stage 4 (DPO) 仍有价值**

---

## 📂 实施需要的代码（基于 stage2-v2 改造）

```
stage3/
├── setup.sh                      ← 同 stage2-v2
├── 01_prepare_data.py            ← 加 GQA / A-OKVQA / OK-VQA / LRV / LLaVA-RLHF SFT 下载
├── _common3.py                   ← 加 5 个新 dataset 类
├── 03_train_stage3.py            ← 继续 LoRA r=64，从 v2 ckpt 起步
└── 04_eval_stage3.py             ← v2 eval 基础上加 GQA / A-OKVQA evaluator
```

工程量预估：4-6h 编码（5 个新 dataset 类是大头）。

---

## ⚠️ 选择 Stage 4 (DPO) 而非 Stage 3 的理由（v2 实测后修订）

我们决定走 Stage 4 路径，因为：

1. **时间成本低**（DPO 4.5h vs SFT 15h；DPO 训练本身只 30-50 min）
2. **教育价值独特** —— DPO 是现代 LLM 训练 standard，跟 ChatGPT/Claude 训练管线对齐
3. **POPE Yes-bias 修复更彻底**（DPO 预计 48-55% vs Stage 3 SFT 58-65%）
4. **不需要担心灾难性遗忘**（DPO LR 极小 1e-6，相比 SFT 5e-5）
5. **v2 grounding/OCR 已超 LLaVA-1.5-7B** —— SFT 的"加新数据"价值递减
6. **VQAv2 短板的本质是 LLM 容量**（1.5B 天花板 ~65%）—— Stage 3 SFT 也只能从 56% 推到 70%，**真正打破天花板要换 3B+ 模型**，是单独的实验

详见 `../stage4-dpo/PLAN.md`。

---

## 🎯 何时回头做 Stage 3？

基于 v2 实测，**真正值得做 Stage 3 的场景**：

| 触发条件 | 优先做的事 |
|---|---|
| Stage 4 DPO 后 POPE Yes-ratio 仍 > 60% | 加 LRV-Instruction (Stage 3 mini) |
| 想看到 GQA / A-OKVQA 数字（v2 完全没 expose）| 跑 Stage 3 完整版 |
| 时间预算允许 ≥ 2-3 天 | Stage 3 完整版 |
| 想冲 VQAv2 70%+ | **换 3B 模型 + Stage 3 SFT 双管齐下**（不是单独 Stage 3）|

**Stage 3 mini 版（仅修 yes-bias）**：
- 只加 LRV-Instruction 30K + 削 grounding 50%
- LR 5e-5, 1 epoch, ~5h 训完
- 预期：POPE Yes-ratio 75% → 60-65%
- 当 DPO 不够用时的备选

**Stage 3 完整版（加新能力 + 修 yes-bias）**：
- 上面整个数据 mix
- ~15h on Blackwell
- 预期：VQAv2 +12-18 点，POPE Yes 修到 58-65%

**Stage 3 + 换 3B 模型组合**：
- 同时做"换模型"和"加 SFT 数据"两件事
- 预期：VQAv2 70-75%，逼近 LLaVA-1.5-7B
- 但工程成本高（要重训 stage1 + stage2 + stage3）= ~3-5 天

---

## 🪞 v2 教训（对未来 Stage 3/4 都适用）

来自 v2 实测对原 plan 预测的验证（这些教训直接影响 Stage 3 的优先级排序）：

### 教训 1：Grounding 数据收益**远超线性**
- 我预测 RefCOCO val 40-46%，实测 78%
- 5× grounding 数据 + 4× LoRA 容量 = ~4× 实际增益（不是 2×）
- **结论**：v2 之后，grounding 数据 marginal utility 接近 0

### 教训 2：OCR 一次到位
- 我预测 TextVQA 40-50%，实测 62%
- 加 28K TextVQA 直接超越 LLaVA-1.5-7B (58%)
- **结论**：专项任务数据"少而精"就能突破

### 教训 3：VQAv2 的天花板就是 LLM 容量
- v1 (3-task mix): 57.2%
- v2 (6-task mix, less LLaVA-Instruct): 56.5%
- 即使 v2 加了 100K LLM 数据 + 4× LoRA 也没涨
- **结论**：要突破 60%+ 需要换更大 LLM，纯加数据不行

### 教训 4：LoRA SFT 治不了 yes-bias
- v1 POPE Yes-ratio 67% → v2 75% (反而更糟)
- 多任务 SFT 倾向"给肯定回答"
- **结论**：yes-bias 要靠 DPO 这种偏好对齐方法
