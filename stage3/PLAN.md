# Stage 3 — High-Quality SFT v2 (Plan, NOT executing for now)

> **状态**：本次复现项目**未执行 Stage 3**，决定走 Stage 2-v2 → Stage 4 (DPO) 路径。
> 本文档保留 Stage 3 的完整计划，供未来扩展使用。

> **如果之后想做 Stage 3**：基于 `stage2-v2/` 复制改造即可。

---

## 🎯 目标：针对 v2 短板做"加新能力 + 修补"

Stage 2-v2 (Phase 1+) 后仍存在的 5 个问题：

| # | 问题 | Stage 3 解决思路 |
|---|---|---|
| 1 | POPE Yes-bias 60-65% | 加 LRV-Instruction（hard negatives）|
| 2 | VQAv2 64% 落后 LLaVA-7B 78.5% | 加 A-OKVQA + OK-VQA（需常识）|
| 3 | 缺组合推理（"left of red, what color"）| 加 GQA（场景图衍生）|
| 4 | ShareGPT4V 选错文件 (用了 share-captioner 而非 instruct) | 这次切换到 sharegpt4v_instruct |
| 5 | 没有真 held-out splits | 训练前抽 200/dataset 作 holdout |

**不做的事**：
- ❌ 改架构（继续 LoRA + ProjectorWithNorm）
- ❌ DPO/RLHF（留给 Stage 4）
- ❌ 加大模型（继续 1.5B）

---

## 📦 数据组成

```
LLaVA-Instruct-150K          100K  (22%)   保留：多轮 VQA 基础
ShareGPT4V_instruct           80K  (18%)   ⭐ 换文件 (216 词 detailed caption)
RefCOCO + RefCOCO+ + RefCOCOg 80K  (18%)   保留：grounding 能力
TextVQA                       28K  (6%)    保留：OCR
🆕 GQA (compositional)        50K  (11%)   新增：场景图推理
🆕 A-OKVQA (knowledge)        17K  (4%)    新增：需常识 VQA
🆕 LRV-Instruction (negative) 30K  (7%)    新增：降幻觉
🆕 OK-VQA                     14K  (3%)    新增：外部知识
🆕 LLaVA-RLHF chosen (SFT)    50K  (11%)   新增：高质量推理
─────────────────────────────────
TOTAL                        449K
```

### 各新增数据集

| 数据集 | HF repo | train 大小 | 字段格式 | 教什么 |
|---|---|---|---|---|
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
449K samples / 32 = ~14,000 iters
14,000 × 2.77s (Blackwell) = ~10.8h
+ overhead = ~11.5h
```

---

## 📊 预期效果

| 指标 | v2 预期 | **Stage 3 预期** | 增益来源 |
|---|---|---|---|
| RefCOCO val Acc@0.5 | 40-46% | 42-48% | grounding 已饱和，微升 |
| POPE F1 | 79-82% | 82-85% | LRV 降假阳性 |
| **POPE Yes-ratio** | 60-65% | **52-58%** | **LRV-Instruction 核心修复** |
| VQAv2 acc | 62-66% | 66-70% | A-OKVQA + OK-VQA 帮助 |
| **GQA acc** | — | **55-65%** | 新能力 |
| **A-OKVQA acc** | — | **45-55%** | 新能力 |
| TextVQA acc | 40-50% | 42-52% | 持平 |
| NoCaps avg_len | 100-130 | **130-160** | 用 instruct 文件 |
| NoCaps rep_rate | 0% | 0% | 持平 |

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

## ⚠️ 选择 Stage 4 (DPO) 而非 Stage 3 的理由

我们决定走 Stage 4 路径，因为：
1. **时间成本低**（10-12h vs 16h）
2. **教育价值独特** —— DPO 是 RLHF 核心，跟现代 LLM 训练管线对齐
3. **POPE Yes-bias 修复更彻底**（DPO 48-55% vs SFT 52-58%）
4. **不需要担心灾难性遗忘**（DPO LR 极小，1e-6）
5. **v2 已经够丰富** —— 再 SFT 一次教育价值有限（v1→v2 已经做过）

详见 `../stage4-dpo/PLAN.md`。

---

## 🎯 何时回头做 Stage 3？

如果 Stage 4 (DPO) 之后发现：
- 模型在某些任务上**根本能力不足**（不是输出风格问题，而是不会做）
- 需要新增任务类型（如 chart 理解、math 视觉推理）
- 时间预算允许（>= 2 天专注做这件事）

→ 那时回来执行 Stage 3，把上面的代码框架实现出来即可。
