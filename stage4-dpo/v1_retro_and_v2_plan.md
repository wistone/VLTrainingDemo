# Stage 4 DPO: v1 复盘 & v2 训练计划

> 写于 v1 训练完成 + POPE eval 出结果之后。
> v1 没生效，本文档记录失败原因分析和 v2 调整方案。

---

## 0. TL;DR

- **v1 结果**：训练完成（4h 8min），但 POPE 跟 v2 baseline 几乎完全相同，DPO 没生效。
- **失败原因**：LR (1e-6) 对 LoRA 来说太低 + β (0.1) 太松，导致 **likelihood displacement**（chosen / rejected 的 logp 平行漂移而不是拉开差距）。
- **v2 方案**：保持全量 83K RLAIF-V 数据不动（避免过滤带来的 alignment tax），只把 LR 提到 5e-6、β 提到 0.3。预计 ~4h。
- **备选**：v2 还崩 → 切 SimPO（reference-free，结构上消除 displacement）。

---

## 1. v1 训练复盘

### 1.1 v1 配置

| 项 | 值 |
|---|---|
| 数据 | RLAIF-V 全量 83K pair |
| 初始化 | Stage 2 v2 final ckpt（LoRA r=64） |
| Reference 模型 | 同 base，PEFT `disable_adapter()` 切换 |
| LR | 1e-6（cosine schedule，warmup 10%） |
| β | 0.1 |
| Batch | per_device 4 × grad_accum 8 = effective 32 |
| Epochs | 1 |
| Total steps | 2598 |
| GPU | RTX PRO 6000 Blackwell 102GB |
| Gradient checkpointing | 关 |
| 训练耗时 | 4h 8min |

### 1.2 训练曲线症状（DPO 没学到的烟雾枪）

最后报告 `train_loss = 0.9367`，但更关键的是过程指标：

| 指标 | 实际表现 | 应该长什么样 |
|---|---|---|
| loss | 整轮在 0.5~1.5 震荡，最终 0.94 | 从 ln(2)≈0.69 缓慢降到 0.4~0.6 |
| rewards/accuracies | 整轮贴着 0.5 ± 0.1 | 应爬到 0.65+ |
| rewards/chosen | 0 → +2~4 | 应稳步上涨 |
| rewards/rejected | 0 → +2~4（**和 chosen 平行涨**） | 应在 0 附近或下降 |
| rewards/margins | -10 ~ +10 大幅震荡，均值 ≈ 0 | 应稳定为正且单调增（→ 0.5~1.5） |
| grad_norm | 4~7（健康） | — |

**核心症状：r_chosen 和 r_rejected 平行往上爬。**

按 DPO loss 几何，这意味着 policy 整体抬升了 logp（变得更"自信"），但**没有学会区分 chosen 和 rejected**——margin 没拉开。这在 DPO 文献里有名字：**likelihood displacement**。

### 1.3 v1 Eval 结果（POPE only）

| 指标 | v2 baseline (DPO 前) | v1 DPO 后 | 变化 |
|---|---|---|---|
| POPE F1 | 0.76 | 0.7567 | -0.003（噪声内） |
| POPE Yes-ratio | 75% | 75.63% | +0.6%（甚至略恶化） |
| POPE Acc | — | 69.43% | — |
| POPE Precision | — | 0.6285 | — |
| POPE Recall | — | 0.9507 | — |

Confusion matrix（n=3000）：
- TP=1426, FP=843, TN=657, FN=74
- 1500 真阳里 1426 答对（recall 95%）
- 1500 真阴里有 843 答错成 yes（这就是幻觉病——超过一半的"应该说 no"被答成"yes"）

跟 v2 baseline 模式完全一致，4h 训练几乎没在权重上留下痕迹。

### 1.4 失败原因诊断（按可能性排序）

#### 主因 1：LR 1e-6 对 LoRA 而言太低

- DPO 原论文用 5e-7 ~ 1e-6 是针对**全参数微调**的尺度
- LoRA 只动 ~1% 参数，**等效 LR 通常要 5~10×**才能产生可比更新量
- LoRA DPO 实践中常见 LR 5e-6 ~ 1e-5
- 我们用了 1e-6 = 标准 LoRA DPO 的 1/5~1/10 → 每步更新太小，2598 步内累积量不够撼动 r=64 的 v2 LoRA

#### 主因 2：β=0.1 让 KL 锚太松，加剧 displacement

- β 小 = 允许 policy 偏离 reference 远 = 训练自由度大
- 自由度大但**信号弱**时，最容易出现"policy logp 整体漂移"而不是"按偏好方向选择性漂移"
- 上面的训练曲线（chosen/rejected 平行涨）就是典型 displacement
- DPO-RLHF 实践中 β 常用 0.1~0.5，**修偏差任务越想见效就越靠近 0.5**

#### 次因 3：RLAIF-V 信号密度

- RLAIF-V 是 AI 自标，pair 类型很杂（存在性、属性、长度、风格都有）
- "存在性幻觉修复"信号大概只占 30~40%
- 这个因素改不了（除非过滤数据，但过滤会引入 alignment tax，见 v2 决策讨论）

#### 次因 4：v2 LoRA r=64 已经过 354K SFT 训练，有一定饱和

- 小步调更新难以撼动已经训过的方向
- 主因 1（提 LR）即可缓解

#### 不是问题

- grad_norm 4~7 全程稳定 → 不是优化崩坏
- 没有 NaN → 不是数值问题
- LR cosine 跑完 → 不是中断

---

## 2. v2 训练计划

### 2.1 设计思路

v1 的核心问题是**超参不对**，不是数据或代码逻辑问题。v2 只调超参，**不动数据、不改 loss、不改模型结构**——单变量验证。

### 2.2 关键决策：保持全量 83K，不过滤

考虑过过滤 RLAIF-V 到"存在性判断"pair（30K）以提高信号密度，但**主动否决了**。理由：

- **Alignment tax 风险**：只训"chosen=no / rejected=yes"会让模型学到"遇到不确定就说 no"的 prior
- 这会拖累 VQAv2（55:45 yes:no 题型，No-bias 会扣分）和 NoCaps（描述变保守）
- LLaVA-RLHF 论文实证：人工标 hallucination 数据训完，POPE 涨但 VQAv2 跌 1~2 pt

全量 83K 让 RLAIF-V 自然分布塑造偏好，**避免人为引入单向偏置**。代价是信号密度低，但这是用 LR / β 来补的——超参调好了，full 83K 也能榨出效果。

### 2.3 v2 配置（vs v1 diff）

| 项 | v1 | v2 | 变化 |
|---|---|---|---|
| LR | 1e-6 | **5e-6** | 5× |
| β | 0.1 | **0.3** | 3× |
| 数据 | 83K | 83K | 不变 |
| Epochs | 1 | 1 | 不变 |
| Batch / grad_accum | 4 / 8 | 4 / 8 | 不变 |
| Warmup | 0.1 | 0.1 | 不变 |
| Output dir | `stage4_dpo_ckpt` | **`stage4_dpo_v2_ckpt`** | 别覆盖 v1 |
| Run name | `stage4-dpo` | **`stage4-dpo-v2-lr5e6-beta03`** | wandb 区分 |

#### 为什么两个超参一起调

- **只调 LR**：动得快了，但 displacement 反而可能加剧（policy 漂得更猛但仍 chosen/rejected 一起漂）
- **只调 β**：KL 锚紧了但更新还是太弱，可能直接训不动
- **两个一起调才对症**：LR 提供更新动能，β 提供方向约束（强迫拉开 chosen/rejected 差距）

### 2.4 启动命令

```bash
!python /content/QwenVL3/stage4-dpo/03_train_dpo.py \
    --stage2_ckpt /content/drive/MyDrive/qwenvl3/stage2_v2_ckpt \
    --stage1_ckpt /content/drive/MyDrive/qwenvl3/stage1_projector_ckpt \
    --processor_dir /content/drive/MyDrive/qwenvl3/stage1_projector_ckpt \
    --dpo_data_dir /content/drive/MyDrive/qwenvl3/data/dpo/rlaif_v \
    --output_dir /content/drive/MyDrive/qwenvl3/stage4_dpo_v2_ckpt \
    --lr 5e-6 \
    --beta 0.3 \
    --num_epochs 1 \
    --batch_size 4 \
    --grad_accum 8 \
    --warmup_ratio 0.1 \
    --logging_steps 10 \
    --no_gradient_checkpointing \
    --run_name stage4-dpo-v2-lr5e6-beta03
```

注意：路径以你 v1 训练实际用的为准，复用即可。Jupyter cell 里 `\` 行尾**不能跟 inline 注释**（comment 会破坏行续接），所以这里不写注释。

### 2.5 早期监控信号（关键，前 1h 决定要不要 kill）

Warmup 占 ~260 步。前 200 步看不出趋势是正常的。但到 **step 500**（约 1h）应该看到：

#### 健康（继续跑完）

```
step 500:
  loss: 0.55 ~ 0.65   (从 0.69 降下来一点)
  rewards/accuracies: > 0.55
  rewards/chosen:   正且增长
  rewards/rejected: 负或接近 0（不应该和 chosen 一起涨）
  rewards/margins: > 0.2 且单调增

step 1500:
  loss: 0.40 ~ 0.55
  rewards/accuracies: > 0.65
  rewards/margins: > 0.5
```

#### 又翻车（果断 kill，省 3h）

```
step 500:
  loss: 0.65 ~ 0.75 (没降)
  rewards/accuracies: ~0.5
  rewards/chosen 和 rewards/rejected: 平行往上爬
  rewards/margins: 0 附近震荡
```

如果 step 500 仍是这个模式 → 直接 Ctrl+C，跳到 §4 SimPO 备选方案。

### 2.6 训完跑全量 eval（不只 POPE）

v1 复盘时只跑了 POPE，但 v2 必须跑全量评测才能判定有没有 alignment tax：

```bash
python stage2/04_eval_stage2.py \
  --stage2_ckpt /content/drive/MyDrive/qwenvl3/stage4_dpo_v2_ckpt \
  --stage1_ckpt /content/drive/MyDrive/qwenvl3/stage1_projector_ckpt \
  --eval_data_root /content/drive/MyDrive/qwenvl3/data/eval \
  --stage2_data_root /content/drive/MyDrive/qwenvl3/data/stage2 \
  --out_dir /content/drive/MyDrive/qwenvl3/eval_dpo_v2 \
  --skip nocaps stage1_regression \
  --eval_batch_size 8
```

NoCaps / Stage1 regression 可以跳，主要看 POPE / RefCOCO / VQAv2 / TextVQA。

### 2.7 判定标准

| 状态 | 条件 | 行动 |
|---|---|---|
| **成功** | POPE Yes ≤ 65% **且** VQAv2 跌 ≤ 2pt | 收尾，写 README |
| **部分成功** | POPE Yes 65~70% **且** VQAv2 跌 ≤ 2pt | 可接受，收尾 |
| **Alignment tax** | POPE 修了但 VQAv2 跌 > 3pt | 回退到 v2 baseline，或切 SimPO |
| **没学到** | POPE 几乎没动（跟 v1 同模式） | 切 SimPO（§4） |

参考目标值：

| 指标 | v2 baseline | v1 DPO | v2 DPO 目标 | 容忍下限 |
|---|---|---|---|---|
| RefCOCO val | 78.1% | 78.1% | ≥ 77% | -1pt |
| POPE F1 | 0.76 | 0.7567 | ≥ 0.82 | — |
| POPE Yes-ratio | 75% | 75.63% | 58~68% | — |
| VQAv2 | 56.5% | ~56.5% | ≥ 54.5% | -2pt |
| TextVQA | 61.7% | ~61.7% | ≥ 60% | -1.5pt |

---

## 3. 概率评估（直说）

| 配置 | POPE Yes-ratio 预期 | 信心 |
|---|---|---|
| v1（已发生） | 75% | — |
| v2（LR 5e-6 + β 0.3） | 60~68% | 中高 |
| v2 + 切 SimPO（备选） | 58~63% | 高 |

**1.5B 模型 + AI 标 83K 数据，预期上限是 POPE Yes 跌到 ~58%**。指望跌到 50% 以下不现实——LLaVA-RLHF 用 7B + 人工标 10K 也只修了 15pt 左右，1.5B 的容量上限在这。

---

## 4. 备选方案：SimPO（如果 v2 还失败）

### 4.1 触发条件

v2 训完仍出现 likelihood displacement（chosen/rejected 平行漂移），或者 step 500 监控信号仍是 v1 那个崩坏模式。

### 4.2 为什么 SimPO 能解决

- **结构上消除 displacement**：SimPO 不用 reference model，loss 直接基于 length-normalized policy logp 差。policy 没法靠"整体抬 logp"作弊
- **附加好处**：不再 forward reference model → 训练快 ~2×（4h → 2h）
- **代价**：失去 reference model 的正则锚 → 需要靠 loss 里的 margin 项 (γ) 来防止训崩

### 4.3 改动量

只动 `_common_dpo.py` 里 `dpo_loss` 函数，外面的训练循环不用改：

```python
def simpo_loss(chosen_logp, rejected_logp,
               chosen_lens, rejected_lens,
               beta=2.0, gamma=1.0):
    """
    SimPO: reference-free, length-normalized.
    """
    chosen_logp_norm = chosen_logp / chosen_lens
    rejected_logp_norm = rejected_logp / rejected_lens

    logits = beta * (chosen_logp_norm - rejected_logp_norm) - gamma
    loss = -F.logsigmoid(logits).mean()

    accuracy = (chosen_logp_norm > rejected_logp_norm).float().mean()
    margin = (chosen_logp_norm - rejected_logp_norm).mean()

    return loss, {
        "accuracy": accuracy.item(),
        "margin": margin.item(),
        "chosen_logp_norm": chosen_logp_norm.mean().item(),
        "rejected_logp_norm": rejected_logp_norm.mean().item(),
    }
```

并在 `compute_loss` 里跳过 reference forward。

### 4.4 SimPO 推荐超参

- β = 2.0 ~ 2.5（注意 SimPO 的 β 跟 DPO 的 β 不同尺度，更大）
- γ = 1.0（margin 超参）
- LR 可以保持 5e-6 或略降
- 数据照旧 83K

---

## 5. 给未来的自己的几条经验

1. **LoRA DPO 的 LR 要比全参 DPO 高 5~10×**——抄论文的 1e-6 是错的起点
2. **β 不光是 KL 强度，还是抗 displacement 的关键**——小数据 / 小模型尤其要 0.3+
3. **r_chosen / r_rejected 要分别看，不能只看 loss**——loss 在 0.6~0.7 区间时区分不出"健康下降"和"displacement 假象"
4. **早 kill 比硬扛 4h 好**——step 500 看不出健康趋势就重启，不要赌
5. **修 hallucination 类任务必须跑全量 eval**——POPE 单项分高没用，要看 alignment tax 的代价
6. **教学项目里"失败案例"也是产出**——v1 这次没生效本身就是宝贵的真实经验，记录下来比假装成功更有价值

---

## 6. 文件清单

- `PLAN.md` — 原始 stage 4 DPO 计划
- `v1_retro_and_v2_plan.md` — 本文档（v1 复盘 + v2 计划）
- `01_prepare_dpo_data.py` — 数据下载/标准化
- `_common_dpo.py` — 数据集类 + DPO loss
- `03_train_dpo.py` — 训练入口（HF Trainer 子类）
- `setup.sh` — 环境初始化
