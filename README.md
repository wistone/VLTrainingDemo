# QwenVL3 — LLaVA-style 教学复现:1.5B + 单卡 A100

在 Google Colab 单卡 A100 / Blackwell 上,**端到端复现一个 LLaVA-style 多模态大模型的完整训练管线**。
基座 Qwen2.5-1.5B-Instruct,视觉塔 SigLIP2-SO400M,通过四个独立阶段把"会续写文字的 LLM"变成"会看图、会做 VQA、会指物体、会被偏好对齐的 VL 助手"。

> **目标不是**复刻 Qwen-VL / LLaVA-1.5 的 benchmark 数字、训练自研 ViT、追求 SOTA。
> **目标是**亲手跑完每个阶段,理解 VL 训练里 4 个独立的核心机制 —— 视觉/语言对齐、多任务统一、(SFT 跳过)、偏好对齐。

---

## 🎯 TL;DR

- **完成阶段**:Stage 1 (Projector 对齐) → Stage 2-v1 (多任务 LoRA) → Stage 2-v2 (Phase 1+ 改进) → Stage 4 DPO v2 (偏好对齐)
- **跳过阶段**:Stage 3 (高质量 SFT) —— 基于 v2 实测数据决定,详见 [stage3/PLAN.md](stage3/PLAN.md)
- **总训练时长**:约 4 个 Colab/Blackwell session,合计 ~30h GPU 时间
- **最终模型**:1.5B base + 78M 可训练 LoRA 参数(占比 3.8%),在 7 项 benchmark 上跑出可与 7B SFT 同档比较的数字
- **核心交付**:每阶段的训练脚本 / eval 脚本 / 数据 prep 脚本 + 详细 README 踩坑笔记 + 自包含 HTML 评测报告
- **教学价值**:每阶段都暴露了 1-2 个 silent bug 或非平凡决策,作为一个"四阶段反 pattern 标本库"很完整

### 各阶段最终数字一览


| 阶段                 | 主要指标                                                              | 数值                                | 状态                                                   |
| ------------------ | ----------------------------------------------------------------- | --------------------------------- | ---------------------------------------------------- |
| **Stage 1**        | with-image loss / Δloss(image vs no-image)                        | **1.727 / +3.437**                | ✅ projector 对齐成功                                     |
| **Stage 2-v1**     | RefCOCO val [Acc@0.5](mailto:Acc@0.5) / POPE F1 / NoCaps rep_rate | 21.2% / 77.9% / **0%**            | ✅ chat + 多任务跑通,但 grounding 数据有 silent bug            |
| **Stage 2-v2**     | RefCOCO val / POPE F1 / TextVQA / VQAv2                           | **78.1% / 76.0% / 61.7% / 56.5%** | ✅ grounding 数据修了,LoRA r=64 翻倍                        |
| **Stage 4 DPO v2** | POPE F1 / Yes-ratio / VQAv2 / TextVQA                             | **78.4% / 68.5% / 58.0% / 62.4%** | ✅ Yes-bias 修 -6.5pt,VQAv2/TextVQA 反涨,无 alignment tax |


---

## 🗺️ 训练管线总览

```
┌─────────────────────────────────────────────────────────────────────┐
│  Qwen2.5-1.5B-Instruct (frozen)  +  SigLIP2-SO400M-patch14-384     │
│                                                                      │
│  Stage 1:  ProjectorWithNorm 5M  →  视觉特征 ↔ LLM embedding 对齐   │
│            数据:LLaVA-Pretrain 558K (单轮 caption)                 │
│            产出:stage1_ckpt_v3/checkpoint-11500                     │
│                                                                      │
│  Stage 2 v1:  + LoRA r=16 (q/k/v/o + gate/up/down) 18M              │
│             数据:LLaVA-Instruct 150K + RefCOCO + ShareGPT4V         │
│                  259K total (实际加载,silent bug 让 RefCOCO 缩到 8.8K) │
│             产出:stage2_ckpt(8088 steps,12h on A100 40GB)          │
│                                                                      │
│  Stage 2-v2:  ↑ LoRA 升 r=64 (78M),数据扩到 6 任务 355K             │
│             修复 RefCOCO silent fallback bug,加 RefCOCO+/g + TextVQA │
│             产出:stage2_v2_ckpt(11091 steps,8.5h on Blackwell 102GB)│
│                                                                      │
│  [Stage 3:  跳过 —— v2 实测后决定走 DPO,完整 plan 保留待用]        │
│                                                                      │
│  Stage 4 DPO v2:  RLAIF-V 83K 偏好对,LR 5e-6 / β 0.3 / 1 epoch     │
│             v1 (LR 1e-6 / β 0.1) 失败案例保留作教学                 │
│             产出:stage4_dpo_v2_ckpt(2598 steps,4h on Blackwell)    │
└─────────────────────────────────────────────────────────────────────┘
```

每阶段在 Drive 留 ckpt,下一阶段从前一阶段 ckpt 起步。整个管线**不重训前序阶段**,只在末端追加新能力。

---

## 📚 历经过程与决策记录

按时间顺序串起来,每节都给出"为什么这么做"和"踩了什么坑":

### Stage 1 · Projector 对齐(最初的目标)

**目的**:让冻结的 SigLIP2 输出能让冻结的 Qwen2.5-1.5B 听懂。**只训 5M 参数的 MLP+LayerNorm**。

**核心决策 — 给 projector 加 LayerNorm**:Qwen2.5 的 token embedding L2 norm ≈ 0.78(远小于 LLaMA-2 的 ~5),
默认 LlavaMultiModalProjector 输出会被训练放大到 norm 800+,完全淹没 LLM 残差流的文字信号。
现象是 loss 卡 ~9.5 不下降、image-token ablation Δloss ~0。**加 LayerNorm 让 projector 输出 norm 稳定到 √1536 ≈ 39**,
loss 立刻能从 ln(152K) ≈ 11.93 降到 1.7。详见 `stage1/_common.py` 里的 `ProjectorWithNorm`。

**踩过的坑**:

1. 视觉权重和 LLM 权重根本没装载(`from_pretrained` 后 missing keys 都是 448 个,因为 transformers ≥4.50 把组件挂在 `model.model.`* 而不是 `model.*`)
2. `<image>` 占位符 1 token vs 视觉 feature 729 个 → 手动展开成 729 个 image_token_id
3. OOM at batch=32 (Qwen vocab 152K, fp32 logits 单 tensor ~15GB) → 降 batch=8 grad_accum=4
4. Stage 1 末期会 token 死循环("Subaru subaru subaru..." / "0 0 0 0..."),裸 caption 续写范式解决不了 → Stage 2 chat template 自然修复

**最终结果**:`with-image loss = 1.727`,`Δloss = +3.437`(视觉信号确实"接进了"LLM)。
20 张 holdout 抽样人工 review,9 张改善、5 张持平、3 张混合、3 张退化。详见 [stage1/README.md](stage1/README.md)。

### Stage 2 v1 · 多任务 LoRA(第一次接触多任务)

**目的**:在 Stage 1 的 caption 模型上加 LoRA 适配,学会聊天格式 + VQA / Grounding / 长 caption 三类输出。

**核心决策 — LoRA r=16 + projector 全参微调**:LLM 仍冻结基座,只走 LoRA;projector 继续训练让它适应新分布。

**第一次发现 silent default 反 pattern**(本项目最重要的方法论收获):

> v1 训完跑 eval 时性能比预期低,反查代码发现两个 silent bug:
>
> 1. `**lmms-lab/RefCOCO` 没有 train split** —— 只有 val/testA/testB。代码 `for split in ["train", "validation", "val"]` 静默 fallback 到 val,实际训了 8,811 条而不是声明的 50K。**没报错、没 warning,启动日志只有一行 `[task] refcoco: 8811 样本`,看一眼以为对了**。
> 2. `**sorted(json_files)[0]` 取了 share-captioner 而不是 sharegpt4v_instruct** —— 字母序错位,数据本身仍 OK 但不是想要的那个文件。
>
> **教训**:任何 fallback / 自动选择都必须 print,最好对非首选还要 warn。两个 bug 催生了 v2,也成了贯穿整个项目的座右铭。

**最终结果**:RefCOCO val [Acc@0.5](mailto:Acc@0.5) 21.2% / POPE F1 77.9% / NoCaps rep_rate **0%**(Stage 1 死循环彻底治好)/ VQAv2 57.2%。详见 [stage2/README.md](stage2/README.md)。

### Stage 2-v2 · Phase 1+ 改进(修 silent bug + 翻倍容量)

**目的**:修 v1 暴露的两个数据漏洞 + 把 LoRA 容量加到 r=64,重训得到一个"严肃的 baseline"。

**关键改动**:


| 维度           | v1                     | v2                           |
| ------------ | ---------------------- | ---------------------------- |
| RefCOCO 数据源  | lmms-lab val (8.8K) ⚠️ | jxu124 train (42K) ✅         |
| RefCOCO+     | ❌                      | ✅ UNC GitHub 原版 pickle (42K) |
| RefCOCOg     | ❌                      | ✅ jxu124 train (42K)         |
| TextVQA      | 仅 eval                 | ✅ 进训练 (28K, consensus≥3 过滤)  |
| LoRA rank    | 16                     | **64** (4× 容量)               |
| 总样本          | 259K                   | **355K**                     |
| Grounding 占比 | 3% (实际)                | **36%** ⭐                    |


**踩到的额外坑**:RefCOCO+ 在 HF 上**没有任何 train repo**(所有候选 401)→ 写 `RefCOCOPickleTaskDataset` 解析 UNC 原版 pickle。
jxu124 字段格式跟 lmms-lab 不同(xyxy 像素 vs xywh,不 bundle bytes 要从 COCO zip 查) → 加多源 `bbox_format` 适配。

**最终结果**(stage2-v2 final eval):

- RefCOCO val [Acc@0.5](mailto:Acc@0.5): **78.1%**(v1 21% → v2 78%,**大胜 LLaVA-1.5-7B 30%**)
- RefCOCO testA/B: 83.8% / 70.7%
- POPE F1 0.76,但 **Yes-ratio 75%**(yes-bias 反而更糟,LoRA SFT 治不了)
- TextVQA 61.7%(**超过 LLaVA-1.5-7B 58.2%**)
- VQAv2 56.5%(短板,见下文 Stage 3 决策)
- NoCaps avg_len 135 词 / rep_rate 0.5%

详见 [stage2-v2/README.md](stage2-v2/README.md) 和 [stage2-v2/stage2_v2_inspect_samples.html](stage2-v2/stage2_v2_inspect_samples.html)(自包含 HTML 报告,带样本图卡)。

### Stage 3 · 决定**跳过**(基于 v2 实测的判断)

v2 final eval 出来后,跟原 PLAN 的预期对照,做了一次重大决策调整:**不做 Stage 3,直接走 Stage 4 DPO**。


| v2 实测 vs LLaVA-1.5-7B         | gap        | Stage 3 SFT 能修吗                             | DPO 能修吗                 |
| ----------------------------- | ---------- | ------------------------------------------- | ----------------------- |
| RefCOCO val 78% vs 30%        | +48 ⭐      | 已饱和,加数据边际收益 < 1pt                           | 不需要修                    |
| TextVQA 62% vs 58%            | +4         | 已优秀                                         | 不需要修                    |
| **POPE Yes-ratio 75% vs 53%** | **-22 ⚠️** | SFT 治不了(LRV-Instruction 也只能修到 60-65%)       | **DPO 主战场,预计修到 48-55%** |
| **VQAv2 56% vs 78%**          | **-22 ⚠️** | 加 VQAv2 train 能从 56% 推到 70%,但本质是 1.5B 容量天花板 | DPO 加不了新能力              |


判断:**Yes-bias 用 SFT 修不彻底,VQAv2 短板的本质是模型容量(1.5B 天花板 ~65%)** —— 单做 Stage 3 性价比不高(15h SFT 换 +12pt VQAv2 + 不彻底的 yes-bias 修复),
不如直接做 Stage 4 DPO(4h 训完,Yes-bias 一击致命,且不需要担心灾难遗忘)。完整决策推理见 [stage3/PLAN.md](stage3/PLAN.md)(完整计划保留,未执行)。

### Stage 4 DPO · 偏好对齐(本项目终点)

**目的**:用 DPO 修 v2 留下的 POPE Yes-bias 75% 问题,同时不破坏其他能力(no alignment tax)。

#### v1(失败案例,保留作教学)


| 项             | v1 配置               | 结果                             |
| ------------- | ------------------- | ------------------------------ |
| 数据            | RLAIF-V 全量 83K pair | —                              |
| LR            | 1e-6(抄论文)           | ❌ 太低                           |
| β             | 0.1(抄论文)            | ❌ 太松                           |
| 训练时长          | 4h 8min             | —                              |
| POPE F1 / Yes | 0.7567 / 75.6%      | **跟 v2 baseline 几乎相同,DPO 没生效** |


**症状**:训练曲线显示 r_chosen 和 r_rejected **平行往上爬**(经典 likelihood displacement),margin ≈ 0,DPO accuracy 整轮贴 0.5。
完整复盘见 [stage4-dpo/v1_retro_and_v2_plan.md](stage4-dpo/v1_retro_and_v2_plan.md)。

#### v2(成功)


| 项      | v2 配置               | 改动理由                                            |
| ------ | ------------------- | ----------------------------------------------- |
| 数据     | RLAIF-V 全量 83K(不过滤) | 过滤会损失多样性,加重 alignment tax                       |
| **LR** | **5e-6**(v1 的 5×)   | LoRA DPO 需要比全参 DPO 更高的 LR 才有等效更新量               |
| **β**  | **0.3**(v1 的 3×)    | 抗 likelihood displacement 的关键 —— 不只是"调强",是结构性修正 |
| 训练时长   | 4h 6min(同 v1)       | —                                               |


**v2 训练曲线**:loss 单调下降(2.0 → 1.4 → 1.21),r_chosen 始终高于 r_rejected,margin 稳定 +1.8,DPO accuracy 爬到 0.55-0.65。

**v2 最终结果**:


| 指标                                    | v2 baseline | v1 DPO | v2 DPO     | vs baseline | 判定                              |
| ------------------------------------- | ----------- | ------ | ---------- | ----------- | ------------------------------- |
| **POPE F1** ⭐                         | 0.76        | 0.7567 | **0.7840** | **+0.024**  | ✅ 主目标达成                         |
| **POPE Yes-ratio** ⭐                  | 75.0%       | 75.6%  | **68.5%**  | **−6.5pt**  | ✅ Yes-bias 修了                   |
| POPE Accuracy                         | —           | 69.4%  | 74.4%      | —           | +5pt vs v1                      |
| **VQAv2 acc**                         | 56.5%       | ~56.5% | **58.0%**  | **+1.5pt**  | ✅ 反涨!没 alignment tax            |
| **TextVQA acc**                       | 61.7%       | ~61.7% | **62.4%**  | **+0.65pt** | ✅ 反涨!                           |
| RefCOCO val [Acc@0.5](mailto:Acc@0.5) | 78.1%       | 78.1%  | 75.5%      | −2.6pt      | ⚠️ 唯一代价(testA 81.7% 强劲,不是结构性破坏) |


**POPE confusion matrix(DPO 真正在干什么)**:DPO **用 32 个 TP 换了 181 个更少的 FP** —— 经典 hallucination mitigation trade-off。Recall 微跌 2pt 但 Precision 涨 5pt,F1 净涨 0.027。

完整报告(带 12 张真实样本图卡):[stage4-dpo/stage4_dpo_v2_report.html](stage4-dpo/stage4_dpo_v2_report.html)
详细分析:[stage4-dpo/v2_results_summary.md](stage4-dpo/v2_results_summary.md)

---

## 📊 完成的结果(终值汇总 + 与业界对比)

### 完整 metric 表(v2 DPO,最终交付物)


| 任务            | 指标                        | 我们 1.5B + LoRA 78M | 对照 LLaVA-1.5-7B | 对照 Qwen-VL-7B | 对照 Qwen2.5-VL-72B (SOTA) |
| ------------- | ------------------------- | ------------------ | --------------- | ------------- | ------------------------ |
| RefCOCO val   | [Acc@0.5](mailto:Acc@0.5) | **75.5%**          | 30%             | 88%           | 94%                      |
| RefCOCO testA | [Acc@0.5](mailto:Acc@0.5) | **81.7%**          | 32%             | 92%           | 94%                      |
| RefCOCO testB | [Acc@0.5](mailto:Acc@0.5) | **66.9%**          | 28%             | 84%           | 91%                      |
| POPE          | F1                        | **0.784**          | 0.86            | 0.87          | 0.89                     |
| POPE          | Yes-ratio                 | **68.5%**          | ~53%            | —             | —                        |
| VQAv2         | Accuracy                  | **58.0%**          | 78.5%           | 78.8%         | 84%                      |
| TextVQA       | Accuracy                  | **62.4%**          | 58.2%           | 63.1%         | 84.7%                    |
| NoCaps        | avg_gen_length            | 18.9 词             | ~80             | —             | (用 CIDEr 不可直比)           |
| NoCaps        | repetition_rate           | **0.00%**          | ~0%             | —             | —                        |


### gap 怎么读

我们用 **2% 的参数量(78M LoRA / 7B 全参)+ ~0.02% 的训练数据(355K SFT + 83K DPO / 7B 模型 1.4 万亿 token)**,
做到了 SOTA 30-90% 的能力水平。这个 gap 主要不是架构问题,而是:

1. **参数量差 4-50×**(1.5B vs 7B-72B)
2. **训练数据差 1000-10000×**
3. **VQAv2 22pt gap 部分是 fairness 假象**:LLaVA-1.5 训过 VQAv2 train(half-in-distribution),我们是纯 zero-shot OOD —— 22pt gap 里约 10pt 来自这个不公平,10-12pt 来自 LLM 容量差,2pt 来自其他

某些指标我们**反超 7B 全参 SFT 的 LLaVA-1.5**(RefCOCO 三 split 全胜、TextVQA +4pt) —— 因为我们专项数据干净 + LoRA r=64 + Stage 4 DPO 修了 yes-bias。这印证了**"参数量不是唯一变量,数据质量和训练流程同样关键"**。

---

## 🪞 跨阶段方法论收获

四阶段共同的反 pattern 标本:

### 1. Silent default / silent fallback 是最危险的 bug

- Stage 2-v1 的 lmms-lab RefCOCO silent fallback 让训练数据少了 5×
- Stage 2-v1 的 sorted()[0] 选错 ShareGPT4V 文件
- 教训:**任何 try/except 失败、任何"找不到首选"的 fallback,都必须 print + 对非首选 warn**。
v2 已把所有 dataset loader 改成强制打印 split 名 + 大字号 warning。

### 2. Caption-only 范式会 token 死循环,chat template 是结构性修复

- Stage 1 末期 BPE 罕见词(Subaru / UAG)预测错首个 subword 后会自我复读
- Stage 2 加 chat template 后 NoCaps rep_rate 从 ~10% → **0%**
- 教训:**生成质量问题不一定是数据问题,有时是范式问题**

### 3. 加了 normalization 才能在 Qwen 这种 vocab 154K + embed norm 0.78 的怪基座上对齐

- 默认 LlavaMultiModalProjector 输出 norm 800+,完全淹没文字 token(norm ~0.78)
- LayerNorm 把 projector 输出钉到 √1536 ≈ 39,跟文字 token 同尺度
- 教训:**抄基础架构前先量一下 embedding norm,Qwen / LLaMA / Mistral 完全不同**

### 4. LoRA DPO 的 LR 要 5-10× 全参 DPO

- v1 抄论文 1e-6 → likelihood displacement,白训 4h
- v2 提到 5e-6 + β 从 0.1 → 0.3 → 一击成功
- 教训:**DPO 论文超参基本都是全参 fine-tune,LoRA 场景要按等效更新量重新调**

### 5. 修幻觉用全量数据 + 高 β,不要急着过滤

- v1/v2 都用 RLAIF-V 全量 83K(没过滤"主题相关 pair")
- 反而 VQAv2 / TextVQA 都**反涨**,没出现 alignment tax
- 教训:**看似"信号密度低"的全量数据集,在抗 alignment tax 上更稳健 —— 多样性 > 纯度**

### 6. 训练 step 500-1000 是 DPO 判断生死的窗口

- v2 在 step 500 就能看出 r_chosen / r_rejected 分叉的健康信号
- v1 这个时候还在平行漂移,但当时没及时止损,白跑了 2000 步
- 教训:**DPO 不要等收敛,前 1000 步看 r_chosen vs r_rejected 是不是分叉**

### 7. 大显存的卡关 gradient_checkpointing 反而更快

- Stage 2-v2 在 Blackwell 102GB 上关 gc → 8.5h(同 batch effective 32 比 A100 40GB 的 12h 少 30%)
- 但 batch=16 + 关 gc 在 cross_entropy 阶段会 OOM(fp32 logits 单 tensor 9.5GB)→ 降到 batch=8
- 教训:**gc 是低显存策略,大显存关 gc 跑得快,但要小心 logits fp32 内存**

---

## 📂 目录结构

```
QwenVL3/
├── PLAN.md                      ← 全程总计划(Stage 1/2/3 详细任务拆分)
├── README.md                    ← 本文档
│
├── stage1/                      ← Projector 对齐
│   ├── README.md                ← 详细记录(踩坑 #1-#8)
│   ├── 01_prepare_data.py       ← LLaVA-Pretrain-558K 下载
│   ├── 02_assemble_model.py     ← Qwen2.5 + SigLIP + ProjectorWithNorm 装配
│   ├── 03_train_projector.py    ← HF Trainer 训练循环
│   ├── 04_eval_stage1.py        ← caption + image-token ablation
│   ├── 05_compare_eval.py       ← 两个 ckpt 对比 → HTML
│   └── _common.py               ← ProjectorWithNorm 类 + transformers 兼容层
│
├── stage2-v1/                      ← v1 多任务 LoRA(完成,留作 v1 baseline)
│   ├── README.md                ← v1 详细记录 + 数据漏洞剖析
│   ├── 01_prepare_data.py       ← LLaVA-Instruct + COCO + RefCOCO + ShareGPT4V
│   ├── 02_baseline_eval.py      ← 训前 caption-only / chat 模式 baseline
│   ├── 03_train_stage2.py       ← LoRA r=16 训练
│   ├── 04_eval_stage2.py        ← OOD eval: RefCOCO + POPE + VQAv2 + NoCaps
│   ├── 04_download_eval_data.py ← OOD 评测数据下载
│   ├── 05_sample_training_data.py
│   ├── 06_inspect_eval_samples.py ← eval 结果分层抽样 → HTML
│   ├── stage2_baseline_at_8000_caption.html
│   ├── stage2_ckpt_step8088_inspect_samples.html  ← v1 final eval 报告(带图)
│   └── stage2_training_data.html
│
├── stage2-v2/                   ← Phase 1+ 改进(实际最终的 SFT baseline)
│   ├── README.md                ← v2 改动 + 三大 silent bug 复盘
│   ├── 01_prepare_data.py       ← + jxu124 RefCOCO/g + UNC pickle RefCOCO+ + TextVQA
│   ├── 03_train_stage2.py       ← LoRA r=64 + 6 任务 + 显式 split 日志
│   ├── _common2.py              ← +RefCOCOPickleTaskDataset / RefCOCOTaskDataset bbox_format / TextVQATaskDataset
│   └── stage2_v2_inspect_samples.html  ← v2 final eval 报告(带图)
│
├── stage3/                      ← 跳过(plan 保留)
│   └── PLAN.md                  ← v2 实测后的决策推理 + 完整 Stage 3 mini/full 备选方案
│
├── stage4-dpo/                  ← DPO 偏好对齐(终点)
│   ├── PLAN.md                  ← 原始 Stage 4 计划
│   ├── v1_retro_and_v2_plan.md  ← v1 失败复盘 + v2 调整方案
│   ├── v2_results_summary.md    ← v2 最终结果汇总
│   ├── 01_prepare_dpo_data.py   ← RLAIF-V 83K 下载 + 标准化
│   ├── 03_train_dpo.py          ← TRL DPOTrainer 子类
│   ├── 04_make_eval_report.py   ← eval JSON → HTML 报告生成器
│   ├── _common_dpo.py           ← DPO dataset 类 + collator
│   └── stage4_dpo_v2_report.html ← 最终交付报告(带 12 张样本图卡)
```

---

## 🚀 复现路径(最小可行)

环境:**Google Colab Pro+ A100 40GB**(或更好,Blackwell 102GB 推荐做 Stage 2-v2 / Stage 4)+ Google Drive 持久化(每阶段 ckpt 都存 Drive)。

按顺序 4 个阶段,每个阶段 1 个 Colab session 即可:

```bash
# 1. Stage 1 (~12h on A100 40GB)
bash stage1/setup.sh
python stage1/01_prepare_data.py    # 下 LLaVA-Pretrain 558K → /content
python stage1/02_assemble_model.py   # Qwen + SigLIP + ProjectorWithNorm → Drive/stage1_init
python stage1/03_train_projector.py  # 训 → Drive/stage1_ckpt_v3
python stage1/04_eval_stage1.py      # caption + Δloss ablation

# 2. Stage 2-v2 (~8.5h on Blackwell 102GB,推荐;或 ~12h on A100 40GB)
bash stage2-v2/setup.sh
python stage2-v2/01_prepare_data.py --only_phase1plus
# 手动下 RefCOCO+ UNC pickle(见 stage2-v2/README.md §🛠️ 启动命令)
python stage2-v2/03_train_stage2.py --no_gradient_checkpointing --batch_size 8 --grad_accum 4
python stage2/04_eval_stage2.py      # 复用 stage2 的 eval(同接口)

# 3. Stage 4 DPO v2 (~4h on Blackwell)
bash stage4-dpo/setup.sh
python stage4-dpo/01_prepare_dpo_data.py    # RLAIF-V 83K → Drive
python stage4-dpo/03_train_dpo.py --beta 0.3 --learning_rate 5e-6
python stage2/04_eval_stage2.py              # 同 eval pipeline
python stage4-dpo/04_make_eval_report.py     # → HTML 报告
```

每阶段的 setup.sh / Colab session 边界 / Drive 挂载逻辑见各子 README 的"🛠️ 启动命令"段。

---

## 🎓 这个项目对我的意义

做这个项目的初衷,不是为了出 SOTA 数字,而是想**弥补"读论文"和"亲手跑训练"之间的 gap**。读 LLaVA / DPO 那些论文时总觉得自己懂了,但真把代码摊开一行行写、训练曲线一个个盯,才发现中间有大量论文从来不提的工程细节和直觉。完整跑下来,我有几个收获是只看论文绝不可能拿到的:

- **"对齐"从抽象概念变成了具体的数字调试**:Stage 1 加上 LayerNorm 让 projector 输出 norm 从 800 降到 √1536 ≈ 39,loss 立刻从 ~9.5 跳到 1.7 —— 才真正理解视觉/语言对齐的本质是**两个分布的 norm 必须同尺度**,不是什么玄学
- **训练曲线变成了能读的东西**:Stage 4 v1 训了 4 小时一无所获,看着曲线才意识到 r_chosen / r_rejected **平行往上爬**就是 likelihood displacement —— 这个名词以前只是论文里飘过的字,现在是我能在 wandb 一眼认出的图形
- **对 silent default / silent fallback 形成了本能警惕**:Stage 2 v1 因为 `for split in ["train", "validation", "val"]: try... break` 静默 fallback,实际训了 8.8K 而不是 50K,白跑 12 小时 —— 这种坑论文从来不写,自己踩过一次以后写代码会下意识 print 每个分支
- **学会了基于实测做 ROI 判断**:v2 final eval 之后,把每项 metric 跟 LLaVA-1.5-7B 拉对照,意识到 VQAv2 22pt gap 里有 10pt 是**fairness 假象**(他们见过 train split,我们是纯 zero-shot)、Yes-bias **SFT 治不了**只能靠 DPO —— 才有信心**推翻原计划跳过 Stage 3**。这种判断只能从实测里长出来,论文不会教
- **抄论文超参跑通 ≠ 理解原理**:DPO 论文给的 LR 1e-6 / β 0.1 是全参 finetune 的数字,LoRA 场景要按等效更新量调到 5e-6 / 0.3,这种细节没人讲过 —— 是自己在 v1 失败案例里摸出来的

如果你也想沿着这条路走一遍,代码 + 每个阶段的 README + 失败复盘 doc 都按时间顺序留在仓库里,可以把它当成一份"踩坑过程实录"来读。**不适合**:想直接拿 ckpt 上线、想出 SOTA、想跳过坑赶进度。

---

## 📌 局限性 & 没做的事

- **Stage 3(高质量 SFT)未执行** —— 完整 plan 保留在 [stage3/PLAN.md](stage3/PLAN.md),v2 实测后判断 ROI 不够
- **NoCaps avg_gen_length 18.9 词偏短** —— Stage 2 SFT 数据(share-captioner)风格短的固有问题,DPO 没引入但也没修,需要回 Stage 2 加 ShareGPT4V 长版本
- **VQAv2 58% 是 1.5B 容量天花板** —— 加数据只能推到 ~70%,要破 75%+ 必须换 3B+ 模型重训整个 pipeline,是单独的实验
- **数据全英语** —— 中文 VL 能力没覆盖
- **没做安全对齐 / 内容过滤** —— DPO 数据是 RLAIF-V(主要修幻觉),没专门处理 unsafe content

---

## 📎 关键参考资料

- [PLAN.md](PLAN.md) — 项目最初的全程计划(2026-05-01)
- [stage1/README.md](stage1/README.md) — Projector 对齐详细复盘(8 个工程坑)
- [stage2/README.md](stage2/README.md) — v1 多任务 LoRA + 数据漏洞剖析
- [stage2-v2/README.md](stage2-v2/README.md) — Phase 1+ 改进 + RefCOCO+ UNC pickle 加载
- [stage3/PLAN.md](stage3/PLAN.md) — Stage 3 完整计划(未执行,基于 v2 实测的决策推理)
- [stage4-dpo/PLAN.md](stage4-dpo/PLAN.md) — DPO 原始计划
- [stage4-dpo/v1_retro_and_v2_plan.md](stage4-dpo/v1_retro_and_v2_plan.md) — DPO v1 失败复盘 + v2 调整
- [stage4-dpo/v2_results_summary.md](stage4-dpo/v2_results_summary.md) — DPO v2 最终结果分析

**HTML 评测报告**(自包含,直接浏览器打开):

- `stage1/holdout_eval_compare.html` — Stage 1 ckpt-4500 vs ckpt-11500 对比
- `stage2/stage2_ckpt_step8088_inspect_samples.html` — Stage 2 v1 final(3.6MB,带样本图)
- `stage2-v2/stage2_v2_inspect_samples.html` — Stage 2-v2 final(5MB,带样本图)
- `stage4-dpo/stage4_dpo_v2_report.html` — Stage 4 DPO v2 最终报告(611KB,12 张真实样本图卡)

