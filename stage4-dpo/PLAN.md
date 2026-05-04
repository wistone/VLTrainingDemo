# Stage 4 — DPO (Direct Preference Optimization)

在 Stage 2-v2 (Phase 1+) 之上做偏好对齐，**主要目标是修复 POPE Yes-bias** (60-65% → 48-55%)
和提升整体输出质量。是整个 LLaVA-style 复现项目的最后一阶段。

---

## 🎯 实验目的

DPO 的本质：**用偏好对 (chosen, rejected) 教模型 "在两个候选答案里选哪个"**。
不是教新知识，是**精修已有知识的输出风格**。

| Stage 2-v2 后仍存在的问题 | DPO 能解决吗 |
|---|---|
| **POPE Yes-bias 67%** | ✅ **核心修复目标** — 给 DPO 一堆 (chosen=No, rejected=Yes) 的偏好对 |
| 长 caption 偶尔幻觉 | ✅ 可以修 — chosen=准确简短，rejected=过度发挥 |
| 输出冗长/啰嗦 | ✅ 可以修 — chosen=简洁，rejected=冗长 |
| 不会 GQA 推理 | ❌ DPO 加不了**模型本身没有的能力** |
| 不会 OCR 复杂图表 | ❌ 同上 |
| 缺常识知识（A-OKVQA）| ❌ 同上 |

**DPO 的天花板 = 模型已有能力的 best of N 输出**。所以要先有 Stage 2-v2 的多任务能力打底。

---

## 🧮 DPO 数学直觉（30 秒版）

```
Loss = -log σ( β × [logπ(chosen|x) - logπ(rejected|x)
                    - (logπ_ref(chosen|x) - logπ_ref(rejected|x))] )

  π     : 当前训练的模型（active model）
  π_ref : 冻结的参考模型（reference, 通常是 SFT 后的初始权重）
  β     : KL 约束强度，典型 0.1-0.5
```

直觉：让 active 模型相对于 reference 模型，给 chosen 的概率比给 rejected 的概率提升得多。
β 越大 → 对参考模型的约束越强 → 偏离起始权重越保守。

**两个模型同时在内存**：active (训练中) + reference (frozen)。我们 1.5B 不算太大，
Blackwell 102GB 装得下。

---

## 📦 数据来源

### Primary: `zhiqings/LLaVA-RLHF-10K` ⭐ 推荐

跟 LLaVA 系列原生兼容的偏好数据：
- ~10K preference pairs
- Chosen / rejected 都是基于真实 LLaVA 模型生成的（不是合成）
- 涵盖 hallucination、verbosity、accuracy 等多维度偏好
- 论文：Sun et al., "Aligning Large Multimodal Models with Factually Augmented RLHF"

字段：
```python
{
  "image": "...",
  "conversations": [{from: human, value: "..."}, {from: gpt, value: "..."}],
  "chosen_score": ...,
  "rejected": "...",  # 较差的回答
  ...
}
```

### Secondary (optional): `MMInstruction/VLFeedback`

- ~80K 多模态偏好对
- 来自 12 种不同 VLM 的输出 + GPT-4V judge
- 多样性高但对 LLaVA 适配度未必一致

### Backup: 自生成

如果上面都不行：
- 用 v2 模型对 1000 个 prompt 各采样 4 个回答
- 用 GPT-4o 当 judge 选 best/worst
- 成本 ~$30-50 USD 的 API 调用

**计划**：先尝试 LLaVA-RLHF-10K，如果效果好就完事；不够再加 VLFeedback。

---

## ⚙️ 训练配置

### 关键超参（DPO 必须谨慎）

| 参数 | 推荐值 | 理由 |
|---|---|---|
| LR | **5e-7 ~ 5e-6** | DPO 极敏感，比 SFT LR 低 100×；高 LR 会 model collapse |
| **β (DPO temperature)** | **0.1** | 控制偏离 reference 的强度 |
| Batch (per device) | 4 | DPO 内存翻倍（chosen + rejected）|
| Grad accum | 8 | 保持 effective batch 32 |
| Epoch | 1 | 10K pairs 不需要多 epoch |
| LoRA | **继续 v2 的 r=64** | 在 v2 final adapter 基础上继续训，不重置 |
| Reference model | Stage 2-v2 frozen | 冻结一份，用来算 π_ref |
| Optimizer | AdamW | 同 SFT |
| Warmup | 10% steps | DPO 训练步数少，warmup 短一点 |

### 显存预估

```
Active model (Qwen2.5-1.5B + ViT + projector + LoRA r=64): ~5 GB (bf16)
Reference model (frozen, same): ~5 GB (bf16, 不需要 grad)
Activations (chosen):  ~25 GB (batch=4, no grad checkpointing)
Activations (rejected): ~25 GB (同上)
Optimizer state (LoRA only ~78M):  ~1 GB
Misc / fragmentation: ~10 GB
─────────────────────────
Total: ~71 GB / 102 GB
```

应该能装下，余量 30 GB。如果 OOM 备选方案：
- 关 grad_checkpointing → 重开（省 50% 激活）
- batch=2, grad_accum=16
- LoRA r=32（省优化器状态）

---

## 📊 预期效果

| 指标 | v2 baseline | **DPO 后预期** | 增益 |
|---|---|---|---|
| **POPE F1** | 80-82% | **82-84%** | +2 |
| **POPE Yes-ratio** | 60-65% | **48-55%** ⭐⭐ | **-10 ~ -12 pt 大胜** |
| POPE precision | 0.68 | **0.78-0.82** | +10-14 |
| POPE recall | 0.91 | 0.85-0.90 | -1 ~ -6 (略降) |
| POPE accuracy | 74% | **78-82%** | +4-8 |
| VQAv2 acc | 62-66% | 64-68% | +2 |
| RefCOCO val | 40-46% | 41-46% | 持平 |
| TextVQA | 40-50% | 41-50% | 持平 |
| NoCaps avg_len | 100-130 | 90-120 | **-10**（DPO 偏好简洁）|
| NoCaps word_recall | 27-37% | **30-40%** | +3 |
| NoCaps rep_rate | 0% | 0% | 持平 |

### POPE 改善的"双刃剑"

DPO 主要让模型学会**说"No"**（"图里没有 X" 时），所以：
- ✅ Yes-ratio 显著降低
- ✅ Precision 大涨（不再瞎说 yes）
- ⚠️ Recall 可能略降（偶尔该说 yes 也变保守）
- 🎯 F1 综合提升（precision 上升幅度大于 recall 下降）

---

## ⏱ 训练时长

```
LLaVA-RLHF-10K, 10K pairs / 32 (effective batch) = 312 iters
312 × ~6s/iter (DPO 双 forward 比 SFT 慢) = 1872s = ~31 min
```

等等 —— **这么短？**

是的，DPO 训练数据量少（10K vs SFT 354K）。但每个 step 比 SFT 慢，因为：
- Chosen forward + Rejected forward (2 次)
- Reference model forward (1 次，但可以缓存)
- 实际单 iter 大概是 SFT 的 2× 慢

**完整时间估算**：
| 阶段 | 时间 |
|---|---|
| 加载两个 model | ~5 min |
| 数据 tokenize 预处理 | ~2 min |
| 训练 (10K pairs, 1 epoch) | **~35-50 min** |
| 保存 ckpt | ~2 min |
| Eval (full) | ~3.5h |
| **总计** | **~4.5h** ⭐ |

**DPO 训练本身只要 30-50 min**！这是它最大的优势 —— **得益于偏好数据量小 + LR 极低**。
真正花时间的是 eval。

---

## 🛠️ 工程实现

### 文件结构

```
stage4-dpo/
├── PLAN.md                  ← 本文档
├── setup.sh                 ← 装 TRL
├── 01_prepare_dpo_data.py   ← 下偏好数据 + 标准化格式
├── _common_dpo.py           ← DPO dataset 类 + collator
├── 03_train_dpo.py          ← TRL DPOTrainer 接入（待写）
└── 04_eval_dpo.py           ← 沿用 stage2-v1/04_eval_stage2.py（不需要重写）
```

### 关键技术点

#### 1. TRL `DPOTrainer` 接入

```python
from trl import DPOTrainer, DPOConfig

dpo_config = DPOConfig(
    output_dir=...,
    learning_rate=1e-6,        # 极小！
    beta=0.1,                  # KL 强度
    max_length=1500,
    max_prompt_length=1024,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    warmup_ratio=0.1,
    bf16=True,
)

trainer = DPOTrainer(
    model=peft_model,                  # 我们的 LoRA model
    ref_model=reference_model,         # frozen 副本
    args=dpo_config,
    train_dataset=dpo_dataset,
    processing_class=tokenizer,
)
```

#### 2. Reference model 处理

两种方式：
- **方式 A**：复制一份完整 model 当 reference（最干净，但内存翻倍）
- **方式 B**：PEFT 的 `disable_adapter()` —— 训练时关掉 LoRA 就是 reference，开 LoRA 就是 active
  - 省内存（共用 base weights）
  - TRL DPOTrainer 内置支持这个 trick

我们用**方式 B**，更省内存。

#### 3. Vision-language DPO 注意

TRL 较新版本（>= 0.10）支持 vision-language DPO，但要：
- 数据 collator 同时处理 chosen 和 rejected 的 input_ids
- pixel_values 在 chosen 和 rejected 之间共享（图片相同，只回答不同）
- attention_mask 分别处理

封装好的话，调用模式跟普通 DPO 一致。

#### 4. 数据格式标准化

我们的 dataset class 输出：
```python
{
  "prompt": "<image>\nIs there a dog in the image?",
  "chosen": "No, I see a cat sitting on a chair.",
  "rejected": "Yes, there's a small brown dog playing.",
  "image": <PIL.Image>,
}
```

DPOTrainer 自动构造 input_ids[chosen] 和 input_ids[rejected]。

---

## ⚠️ 风险 & 备用方案

### 风险 1：DPO 容易 model collapse
- 现象：训几百步后 loss 突然变 NaN，模型完全失效
- 原因：LR 太大、β 太小、偏好对质量低
- 防御：LR 1e-6 起步，监控 logits 数值

### 风险 2：Reference model 内存爆
- 现象：DPOTrainer 启动时 OOM
- 备用：用 LoRA disable_adapter trick (方式 B)，省一半内存

### 风险 3：TRL 多模态支持不全
- 现象：DPOTrainer 报错"vision input not supported"
- 备用：手动写 DPO loss（用 PyTorch 实现，~50 行代码）

### 风险 4：训完反而退化
- 现象：DPO 后 RefCOCO 突然变差，POPE Yes-ratio 反而升高
- 原因：偏好数据质量差，或 β 太小（KL 约束不够）
- 调试：增 β 到 0.3-0.5 重训

---

## 🎯 KPI / 衡量"DPO 训练成功"

| KPI | 目标 |
|---|---|
| **POPE Yes-ratio** | < 58% (vs v2 60-65%) ⭐ 核心 |
| **POPE F1** | ≥ v2 (不能退化) |
| RefCOCO val Acc@0.5 | ≥ v2 - 1pt（不能严重退化）|
| VQAv2 acc | ≥ v2 - 1pt |
| NoCaps rep_rate | 0% (保持) |
| 训练时长 | < 1.5h |

---

## 📅 时间线

```
今天:    等 v2 训完 + sanity eval
明天:    v2 full eval (~3.5h) + 我写完 stage4-dpo 训练代码
明天晚:  下 LLaVA-RLHF 数据 (10K pairs, ~50MB)
明天晚:  跑 DPO 训练 (~1h)
后天上午: DPO eval (~3.5h)
后天下午: 整理 4-stage 完整对比 + 最终 README
```

---

## 📝 现在的进度

✅ Stage 4 PLAN 已写（本文档）
✅ Stage 4 数据下载脚本 (`01_prepare_dpo_data.py`)
⏳ Stage 4 setup.sh + DPO 训练代码（等 v2 跑完后写）

下一步：等 v2 全程跑完，看 v2 final eval 数字 → 决定是否启动 DPO。
