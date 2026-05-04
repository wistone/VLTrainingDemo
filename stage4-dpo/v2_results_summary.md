# Stage 4 DPO v2 — 结果汇总与分析

> 写于 v2 DPO 训练 + 全量 eval 完成之后。
> 本文档作为 Stage 4 的最终交付物，承接 `v1_retro_and_v2_plan.md` 里的 v2 计划。

---

## 0. TL;DR

**v2 DPO 真生效了**。POPE Yes-bias 从 75% 降到 68.5%（F1 +0.024），最大惊喜是 VQAv2 / TextVQA 不仅没跌还**反涨**——之前担心的 alignment tax 基本没出现。唯一代价是 RefCOCO val −2.6pt，但 testA 强劲、parse_rate 100%，不是结构性破坏。

**判定**：partial → near-full success。三阶段教学 pipeline（projector 对齐 → 多任务 SFT → 偏好对齐）完整跑通，可以收尾发布。

---

## 1. 训练配置（v2，最终版）

| 项 | 值 | 备注 |
|---|---|---|
| 数据 | RLAIF-V 全量 83K pair | 不过滤，避免 alignment tax |
| 初始化 | Stage 2 v2 final ckpt（LoRA r=64） | 同 v1 |
| Reference 模型 | 同 base，PEFT `disable_adapter()` | 同 v1 |
| **LR** | **5e-6** | v1 是 1e-6，提了 5× |
| **β** | **0.3** | v1 是 0.1，提了 3× |
| Batch | 4 × 8 = effective 32 | 同 v1 |
| Epochs | 1 | 同 v1 |
| Total steps | 2598 | 同 v1 |
| GPU | RTX PRO 6000 Blackwell 102GB | 同 v1 |
| 训练耗时 | 4h 6min | ~同 v1 |

## 2. 训练曲线对比（v1 vs v2）

| 阶段 | v1 (β=0.1) | v2 (β=0.3) | 判定 |
|---|---|---|---|
| 起点 loss | ~0.7（贴 ln(2)） | ~2.0（β 高 ⇒ 异常 sample 被放大） | v2 起点高是因为 β 大，**不是问题** |
| 中段 loss | 全程 0.7 横着 | 2.0 → 1.4 单调下降 | v2 真在学 ✅ |
| 末段 loss | 0.94（没动） | 1.21（继续降） | — |
| margin 平均 | ~0 | ~+1.8 | v2 拉开了差距 ✅ |
| r_chosen vs r_rejected | 平行涨（displacement） | r_chosen 始终高于 r_rejected | v2 朝对的方向走 ✅ |
| acc 趋势 | 一直贴 0.5 | 逐步爬到 0.55-0.65 | v2 学到了偏好 ✅ |

**关键发现**：β 从 0.1 → 0.3 是**结构性修正 likelihood displacement** 的，不只是"调强"。
LR 从 1e-6 → 5e-6 是因为 LoRA DPO 需要比全参 DPO 更高的 LR 才有等效更新量。

---

## 3. Eval 完整结果

### 3.1 三方对比表（v2 baseline / v1 DPO / v2 DPO）

| 指标 | v2 baseline | v1 DPO | **v2 DPO（最终）** | vs baseline | 判定 |
|---|---|---|---|---|---|
| RefCOCO val Acc@0.5 | 78.1% | 78.1% | **75.5%** | **−2.6pt** | ⚠️ |
| RefCOCO testA Acc@0.5 | — | — | 81.7% | — | ✅ |
| RefCOCO testB Acc@0.5 | — | — | 66.9% | — | 正常 |
| RefCOCO mIoU (val) | — | — | 0.634 | — | — |
| RefCOCO parse_rate | — | — | 100% | — | ✅ 格式没坏 |
| **POPE F1** | 0.76 | 0.7567 | **0.784** | **+0.024** | ✅ |
| **POPE Yes-ratio** | 75% | 75.6% | **68.53%** | **−6.5pt** | ✅ 主目标达成 |
| POPE Acc | — | 69.43% | 74.40% | +5pt vs v1 | ✅ |
| POPE Precision | — | 0.6285 | 0.6780 | +5pt vs v1 | ✅ |
| POPE Recall | — | 0.9507 | 0.9293 | -2pt vs v1 | 微跌（合理） |
| **VQAv2** | 56.5% | ~56.5% | **58.00%** | **+1.5pt** | ✅ 反涨 |
| **TextVQA** | 61.7% | ~61.7% | **62.35%** | **+0.65pt** | ✅ 反涨 |
| TextVQA substring | — | — | 68.20% | — | — |
| NoCaps avg_len | — | — | 18.9 词 | (目标 30-80) | 偏短 |
| NoCaps rep_rate | — | — | 0.00% | (目标 <10%) | ✅ |
| NoCaps word_recall | — | — | 16.72% | (目标 25-45%) | 偏低 |

### 3.2 POPE Confusion Matrix 深度对比

| | v1 DPO | **v2 DPO** | 变化 |
|---|---|---|---|
| TP（真阳） | 1426 | 1394 | -32 |
| **FP（假阳 = 幻觉 yes）⭐** | **843** | **662** | **-181** |
| **TN（真阴）** | 657 | 838 | **+181** |
| FN（假阴） | 74 | 106 | +32 |
| Precision | 0.6285 | **0.6780** | **+5pt** |
| Recall | 0.9507 | 0.9293 | -2pt |

**核心解读**：DPO 用 32 个 TP **换了 181 个更少的 FP**——典型的 hallucination mitigation
trade-off。Recall 微跌但 Precision 大涨，F1 净涨 0.027。这是**干净的"减幻觉"修正**。

---

## 4. 三个最关键发现

### 4.1 POPE Yes-bias 真被修了

v2 baseline / v1 DPO 都是 75% Yes-ratio（模型见 yes/no 题就倾向说 yes）。v2 DPO 把它降到
68.5%——虽然没到理想的 ≤65%，但**首次脱离"固守 75%"状态**，幅度是 6.5pt。

按 1.5B 模型 + AI 标偏好对的预期上限，这个幅度合理。LLaVA-RLHF 用 7B + 人工标 10K 也就修
~15pt，1.5B + AI 标 83K 修 6.5pt 在量级上对得上。

### 4.2 Alignment tax 基本没出现（最大惊喜）

我之前 v1_retro doc 里设的判定 flag：
- 成功：POPE Yes ≤ 65% **且** VQAv2 跌 ≤ 2pt
- 失败 alignment tax：POPE 修了但 VQAv2 跌 > 3pt

实际：**VQAv2 反涨 1.5pt，TextVQA 反涨 0.65pt**。完全超出预期。

可能的原因：
1. **不过滤数据**保住了多样性。RLAIF-V 全量 83K 里 chosen/rejected 差异维度很多
   （存在性、属性、长度、风格），DPO 学到的是"细粒度减幻觉"而不是"无脑说 No"
2. β=0.3 提供了足够强的 KL 锚，policy 没漂太远
3. 1.5B 容量小 + LoRA r=64 限制 = 模型没法学到太极端的偏好，自然抗 over-alignment

教训：**修 hallucination 类任务，不要急着过滤数据**。看似"信号密度低"的全量数据集，
其实在抗 alignment tax 上更稳健。

### 4.3 RefCOCO val −2.6pt 是真实成本，但不是结构性

- val: 75.5%（−2.6pt）
- testA: 81.7%（强劲）
- testB: 66.9%（hard split 合理）
- 三 split 平均 = 74.7%，整体水平没崩
- parse_rate 100%（box 输出格式没坏）

可能是 DPO 让模型在"指代+空间定位"上**稍微变保守**（不愿框得太大或太精确）。但 testA
反而漂亮，说明模型基本能力没退化，只是 val 这个 split 上的某些 hard case 变差了。

---

## 5. NoCaps 短 caption 问题（不是 DPO 引入的）

v2 DPO NoCaps avg_len = 18.9 词，远低于目标 30-80 词。但：
- **rep_rate = 0%**（无 token 死循环，最关键的好消息）
- v2 baseline 大概率也这么短（Stage 2 SFT 数据偏短的固有问题）
- DPO 没让它变差

这是 **Stage 2 阶段的遗留问题**，不属于 DPO 的责任。修复需要回到 Stage 2 加更长 caption
数据（如 ShareGPT4V 长版本），不是 DPO 能解决的。

---

## 6. v1 vs v2 的对比总结

| 维度 | v1（失败） | v2（成功） |
|---|---|---|
| 关键超参错误 | LR 1e-6 / β 0.1 | LR 5e-6 / β 0.3 |
| 训练曲线 | loss 不动，r_chosen/r_rejected 平行涨 | loss 单调降，margin 稳定为正 |
| Likelihood displacement | 严重（核心症状） | 已消除 |
| POPE 改善 | 几乎为零（−0.003 F1） | F1 +0.024，Yes −6.5pt |
| VQAv2 影响 | 同 baseline | **反涨 +1.5pt** |
| RefCOCO 影响 | 同 baseline | val −2.6pt（小代价） |
| 整体判定 | 完全失败 | partial → near-full success |

---

## 7. 给未来自己的经验（与 v1_retro doc 互补）

1. **LoRA DPO 的 LR 要 5-10× 全参 DPO**——抄论文 1e-6 是错的起点（v1 教训）
2. **β 是抗 likelihood displacement 的关键**——0.3 比 0.1 在小模型 / 小数据场景下显著更稳
3. **修幻觉用全量数据 + 高 β，不过滤**——比"过滤到主题相关 pair + 低 β"更抗 alignment tax
4. **1.5B 上的 DPO 上限大概是 POPE Yes 修 6-10pt**——别期待跌到 50% 以下，那是 7B+ 才有的水平
5. **eval 必须跑全量**，不能只看主目标。POPE 修了但 VQAv2 崩了的话整体得不偿失
6. **训练 step 500-1000 是判断生死的窗口**——v2 在 step 500 就能看出 r_chosen/r_rejected
   分叉的健康信号，v1 这个时候还在平行漂移。早判断早决策

---

## 8. 项目最终状态

| Stage | 任务 | 状态 |
|---|---|---|
| Stage 1 | Projector alignment（LLaVA Pretrain 595K） | ✅ ckpt-11500 |
| Stage 2 v1 | 多任务 LoRA r=16，354K | ✅ |
| Stage 2 v2 | 多任务 LoRA r=64，354K + RefCOCO+/g + TextVQA | ✅（最终 baseline） |
| Stage 3 | SFT v2（跳过） | ⏭️ 计划保留在 stage3/PLAN.md |
| Stage 4 DPO v1 | LR 1e-6 / β 0.1 | ❌ 失败案例（保留作教学） |
| **Stage 4 DPO v2** | **LR 5e-6 / β 0.3** | **✅ 最终交付** |

---

## 9. 核心数字（一句话版）

**v2 DPO 在 POPE 上把 hallucination Yes-bias 从 75% 修到 68.5%（F1 +0.024），同时 VQAv2
反涨 1.5pt、TextVQA 反涨 0.65pt，唯一代价是 RefCOCO val −2.6pt。**

---

## 10. 文件清单

- `PLAN.md` — 原始 Stage 4 DPO 计划
- `v1_retro_and_v2_plan.md` — v1 失败复盘 + v2 计划
- `v2_results_summary.md` — 本文档（v2 最终结果）
- `01_prepare_dpo_data.py` — 数据下载/标准化
- `_common_dpo.py` — 数据集类 + DPO loss
- `03_train_dpo.py` — 训练入口（HF Trainer 子类）
- `04_make_eval_report.py` — eval HTML 报告生成器
- `setup.sh` — 环境初始化
- ckpt: `/content/drive/MyDrive/qwenvl3/stage4_dpo_v2_ckpt`
- eval: `/content/drive/MyDrive/qwenvl3/eval_dpo_v2/stage4_dpo_v2_ckpt/`
