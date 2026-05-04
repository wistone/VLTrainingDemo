# Stage 2 — Multitask LoRA Training (v1, completed 2026-05)

LLaVA 复现的第二阶段：在 Stage 1 训好的 projector 基础上，**给 LLM 加 LoRA 适配**，
学习 chat template + 三种下游任务（VQA / grounding / 长 caption）。

> ⚠️ 这是 **v1 版本**，已训练完成（8088 steps，最终 ckpt 见
> `/content/drive/MyDrive/qwenvl3/stage2_ckpt`）。
> v2 (`../stage2-v2/`) 是 Phase 1+ 改进版，加入了 RefCOCO+/g、TextVQA、LoRA r=64
> 等改动 —— 主要为了修复 v1 训练后才发现的数据漏洞。

---

## 🎯 实验目的

让 Stage 1 训好的 caption 模型学会三件事：

1. **聊天格式**：Qwen2.5 chat template (`<|im_start|>...<|im_end|>`)
2. **多任务输出**：
   - 开放式 VQA（"What is the man wearing?" → "A red jacket"）
   - 视觉定位（"Where is the cat?" → `<box>(x1,y1),(x2,y2)</box>`）
   - 段落级长描述
3. **避免 token 死循环**（Stage 1 的痛点 — UAG tank、Subaru 等死循环案例）

| 模块 | 状态 | 可训练参数 |
|---|---|---|
| SigLIP2 ViT | ❄️ 冻结 | 0 |
| ProjectorWithNorm | ✅ 全参（继续微调） | ~5M |
| Qwen2.5-1.5B LLM | 🔧 LoRA r=16 (q/k/v/o + gate/up/down) | ~18M |
| **总可训练** | | **22.6M (1.13% of total)** |

---

## 📦 数据组成（实际训练时加载到的数量）

| 数据集 | 默认 limit | **实际加载** | 占比 | 教什么 |
|---|---|---|---|---|
| **LLaVA-Instruct-150K** | 150K | **150,000** ✅ | 58% | 多轮 VQA + 推理 |
| **RefCOCO** (lmms-lab) | 50K | **8,811** ⚠️ | 3% | 视觉定位 → bbox 输出 |
| **ShareGPT4V** (share-captioner) | 100K | **100,000** ⚠️ | 39% | 段落 caption（实际是 share-captioner 不是 instruct，见漏洞 #1）|
| **总样本数** | (300K target) | **258,811** | | (缺口 14% 全在 RefCOCO) |

**图源**：COCO train2017.zip (~18GB)，由 LLaVA-Instruct 和 ShareGPT4V 共用。

**No held-out splits**：LLaVA-Instruct 和 ShareGPT4V 都用 `[:limit]` 切，没保留
holdout。所以 eval 时不能用「训练数据采样」，必须用 OOD 公开 benchmark
（POPE / VQAv2 / NoCaps）。

---

## 🚨 训练后才发现的数据漏洞（重要！）

v1 训完跑评测对比 SOTA 时，性能比预期低，反查代码发现了 **2 个 silent default 类
bug**，都是"看 log 一眼以为对了，仔细查代码才发现不对"的隐蔽错误。

### 🔴 漏洞 #1：RefCOCO 实际只训了 8,811 条 (val split)，不是 50K (train split)

**根因**：`stage2-v1/03_train_stage2.py` 加载 RefCOCO 时：
```python
for split in ["train", "validation", "val"]:
    try:
        hf_ds = load_dataset(rc_dir, split=split, trust_remote_code=True)
        break        # 拿到第一个能加载的 split 就 break，不报告是哪个
    except Exception:
        hf_ds = None
```

而 `lmms-lab/RefCOCO` 是**评测专用 dataset，不含 train split**：
- ❌ `split="train"` → 抛异常被吞掉
- ❌ `split="validation"` → lmms-lab 用 `val` 不是 `validation`
- ✅ `split="val"` → 拿到 8,811 条
- 然后 `RefCOCOTaskDataset(hf_ds, ..., limit=50000)` → `min(8811, 50000) = 8811`

**没报错，没 warning**，启动日志只有一行 `[task] refcoco: 8811 样本`，看一眼以为对了。

**双重影响**：
1. **训练数据严重不足**：实际 8.8K 是预期的 17%
2. **eval 污染**：v1 evaluation 用同一个 lmms-lab/RefCOCO val split，
   **训练数据 ≈ 评测前 1000 题**，理论上是数据泄露

**实际损失（实测验证）**：
- val (训练里见过) Acc@0.5 = **21.2%**
- testA (没训过) Acc@0.5 = **23.3%**
- 如果污染严重，val 应该 >> testA。**实际 val < testA**，
  说明 LoRA r=16 + 1 epoch 没记住 val 具体内容，污染影响 < 2 个点
- 但**方法论上不严谨**，发论文/严肃报告需要重训

**v2 修复**：换数据源到 `jxu124/refcoco` (42K 真 train split) +
`jxu124/refcocog` (42K) + 从 GitHub 手动下 RefCOCO+ UNC 原版 pickle (42K)。
v2 同时给 `_try_load_hf_dataset` 加了**强制打印 split 名 + 非 train fallback 时大字号 warning**，
避免再发生 silent fallback。

### 🟡 漏洞 #2：ShareGPT4V 实际用了 share-captioner，不是 sharegpt4v_instruct

**根因**：`01_prepare_data.py` 下载了 ShareGPT4V repo 的两个 json：
| 文件 | 样本数 | caption 平均长度 | 字母序 |
|---|---|---|---|
| `share-captioner_coco_lcs_sam_1246k_1107.json` | 1.2M | 155 词 | 第 1 (`-` ASCII 45 < `h` 104) |
| `sharegpt4v_instruct_gpt4-vision_cap100k.json` | 102K | 216 词 | 第 2 |

而代码 `sorted(sg_dir.rglob("*.json"))[0]` 取**字母序第 1 个**，正好选了
`share-captioner_*` —— 数据本身也 OK（GPT-4V 标注的 1.2M 大池子），
但**不是真正想要的 sharegpt4v_instruct**（更结构化、更长、更精选）。

**没报错**，启动日志 `[task] sharegpt4v: 100000 样本（已过滤为 COCO 子集）`
看一眼也以为对了。

**实际损失**：
- 比理想数据短 ~28% (155 vs 216 词)
- 比理想数据少 "instruct 风格"（"In the center of the image..." 这种结构）
- 但数据**仍然是高质量 GPT-4V caption**，远不是垃圾数据
- NoCaps 实测 avg_len 131 词，rep_rate 0%，**模型确实学到了长 caption 能力**

**v3+ 修复**：`commit 49dbec6` 加 `SHAREGPT4V_PREFERENCE` 显式偏好列表，
明确按 `[sharegpt4v_instruct, share-captioner]` 顺序找文件。**v1/v2 都是
share-captioner 训出来的，对比时是公平的**。

### 🔵 共同模式：silent default 是反 pattern

两个漏洞都是同一种反 pattern：**程序自动选了一个"看似合理的默认"，但没说自己选了什么**。

教训：任何 fallback / 自动选择都应该 print，最好对非首选还要 warn。
v2 已把 `_try_load_hf_dataset` 和 `SHAREGPT4V_PREFERENCE` 都加了显式日志。

---

## 📊 评测结果（Final, ckpt 8088 steps）

跑命令（默认参数 full eval）：
```bash
python stage2-v1/04_eval_stage2.py \
    --stage2_ckpt /content/drive/MyDrive/qwenvl3/stage2_ckpt \
    --stage1_ckpt /content/drive/MyDrive/qwenvl3/stage1_ckpt_v3 \
    --processor_dir /content/drive/MyDrive/qwenvl3/stage1_init \
    --eval_data_root /content/drive/MyDrive/qwenvl3/data/eval \
    --stage2_data_root /content/drive/MyDrive/qwenvl3/data/stage2 \
    --stage1_data_root /content/drive/MyDrive/qwenvl3/data/llava-pretrain
```

### 主指标

| 任务 | 指标 | **Final 值** | sanity (ckpt-5000) | 解读 |
|---|---|---|---|---|
| **RefCOCO val** | Acc@0.5 | **21.20%** | 20.00% | ⚠️ val 在训练里（污染但实际差距 <2 点）|
| RefCOCO val | Acc@0.7 | 6.30% | 2.00% | 精确定位仍弱 |
| RefCOCO val | mean IoU | 0.310 | 0.292 | 大概区域 OK，框不准 |
| RefCOCO val | parse_rate | 100% | 100% | bbox 格式学透了 |
| **RefCOCO testA** | Acc@0.5 | **23.30%** | 20.00% | 没在训练里，干净 |
| RefCOCO testA | Acc@0.7 | 7.10% | 7.00% | — |
| **RefCOCO testB** | Acc@0.5 | **15.40%** | 11.00% | 物体多样，难 |
| RefCOCO testB | Acc@0.7 | 4.20% | 2.00% | — |
| **POPE** | F1 | **77.93%** | 78.10% | ⭐ 接近 LLaVA-1.5-7B (86%) |
| POPE | accuracy | 74.17% | 74.20% | — |
| POPE | precision | 0.6803 | — | 假阳性多 |
| POPE | recall | 0.9120 | — | 召回好 |
| POPE | yes_ratio | 67.03% | 67.80% | ⚠️ 持续 yes-bias |
| **VQAv2** | accuracy | **57.18%** | 55.92% | 略低于预期 (60-64%) |
| **NoCaps** | repetition_rate | **0.00%** ⭐⭐ | 0.00% | Stage 1 死循环彻底治好 |
| NoCaps | avg_gen_length | 131.27 词 | 136.40 | 健康长度 |
| NoCaps | avg_word_recall | 27.57% | 27.00% | 内容覆盖偏底部 |
| NoCaps | distinct_word_ratio | 12.52% | 25.64% | n=200 vs n=30 的 Heaps' law 差异，非退化 |

### vs 业界主流 VL 模型

| 指标 | 我们 (1.5B + LoRA 23M) | LLaVA-1.5-7B (7B 全参) | Qwen-VL-7B | Qwen2.5-VL-72B (SOTA) |
|---|---|---|---|---|
| RefCOCO val Acc@0.5 | **21%** | 30% | 88% | 94% |
| POPE F1 | **78%** | 86% | 87% | 89% |
| VQAv2 | **57%** | 78.5% | 78.8% | 84% |

**这个 gap 主要不是模型架构问题，而是**：
- 参数量小 4-50× (1.5B vs 7B-72B)
- 训练数据少 1000-10000× (300K vs 1.4 万亿 token)
- LoRA r=16 vs 全参 finetune
- 我们 RefCOCO 实际只训了 8.8K（漏洞 #1）

### 关键观察

✅ **最大胜利**：NoCaps repetition_rate 从 Stage 1 的 ~10% 降到 **0%**。
chat template + 多任务训练**根本性修复了 token 死循环**这个 Stage 1 痛点。
NoCaps avg_len 131 词也证明长 caption 能力 work。

✅ **POPE F1 77.93%** 达 LLaVA-1.5-7B 的 91% 水平，1.5B 参数能到这个数字非常好。

⚠️ **RefCOCO 仅 21%** — 部分是漏洞 #1（实际只训 8.8K），部分是 LoRA r=16
表达力不够。**v2 同时解决这两点**（127K grounding 数据 + LoRA r=64）。

⚠️ **POPE Yes-bias 67%** —— 健康范围 45-55%。这个用 LoRA 多任务训练解决不了，
需要 DPO 后训。Stage 3 可以做。

⚠️ **ckpt-5000 → final 几乎没涨** (RefCOCO val 20% → 21%, POPE F1 78.1% → 77.9%)。
说明 v1 在 5000 步基本饱和了 —— 数据不够多 + LoRA 容量不够大，后面 38% 训练
只是精修。**这本身就是 v2 必要性的证据**。

---

## ⏱ 实验时长

| 配置 | 数值 |
|---|---|
| GPU | Colab Pro+ A100 40GB |
| Batch | 8 × grad_accum 4 = effective 32 |
| LoRA | r=16, alpha=32 |
| 总 iters | 8088 |
| 单 iter | ~5.45 s |
| **训练时长** | **~12h** |
| 中途崩过几次 | 是 |

### 中途崩坏复盘
- **首次崩溃**：训到 5300 步后凌晨 5:28 崩，wandb 显示 "Crashed"
- **原因**：浏览器 tab 被挂起 → Colab heartbeat 失败 → runtime 被 kill
  - 不是 Pro+ background execution（那是自动的）
  - 不是 24h 上限（才训 ~7h）
  - 主要是 Chrome 没把 colab 加白名单 + macOS 没跑 caffeinate
- **resume 后**：从 checkpoint-5000 续训，wandb 用 `WANDB_RESUME=allow` +
  `WANDB_RUN_ID=udk1ya9o` 让曲线接续。剩 2800 步用 ~3h 跑完

---

## 🐛 工程踩过的坑（除了上面的数据漏洞）

### 1. torchao 0.10 与 PEFT 不兼容（启动期就崩）
- Colab 预装 torchao 0.10，PEFT ≥0.13 的 `dispatch_torchao` 检查 ≥0.16，硬 ImportError
- 修复：训练脚本启动时 `_ensure_torchao_compat()` subprocess uninstall torchao
- `commit 75c6ed4` "Make Stage 2 training script self-heal torchao incompatibility"

### 2. Multi-process ZipFile CRC 错误
- `dataloader_num_workers > 0` 时多进程并发读同一个 zip → BadZipFile CRC error
- 原因：ZipFile 对象不能跨进程共享
- 修复：CocoZipLoader 用 per-PID cache，每个 worker 进程懒加载自己的 ZipFile
- `commit 712dc89` "Fix CocoZipLoader for multi-process DataLoader workers"

### 3. RefCOCO 字段名不一致
- lmms-lab/RefCOCO 用 `answer` 字段存 referring expression（不是 `sentences`）
- 修复：`_extract_ref()` 加宽松字段匹配
- `commit 8599baa` "Fix RefCOCO field mapping for lmms-lab/RefCOCO"

### 4. Stage 1 ckpt 路径过期
- `save_total_limit=2` 持续删旧 ckpt，硬编码 `--stage1_ckpt .../checkpoint-XXXX` 经常无效
- 修复：`resolve_stage1_ckpt()` 自动 fallback 到最新可用 ckpt
- `commit be65fb6` "Auto-resolve stale Stage 1 checkpoint in stage2 training script"

### 5. 中间 ckpt 缺 image_processor
- HF Trainer 的 `processing_class=tokenizer` 只管 tokenizer，不存 image_processor
- 现象：eval 时 `AutoImageProcessor.from_pretrained(ckpt)` 报 "preprocessor_config.json not found"
- 修复：eval 脚本独立寻找 tokenizer 和 image_processor 目录
- `commit 2596ed0` "Fix Stage 2 checkpoint loading for projector and image_processor"

### 6. Projector 加载 silently 走 random init（最阴险）
- ProjectorSaverCallback 存的 keys 没前缀（`linear_1.weight`），但
  install_custom_projector 默认按 `multi_modal_projector.linear_1.weight` 前缀过滤
- 影响：所有 Stage 2 中间 checkpoint 的 eval **都会用随机 projector**，但只 print
  一行 warning "[warn] 没找到 multi_modal_projector 权重，使用随机初始化"
- 修复：detect 文件名 `multi_modal_projector.safetensors` 直接装载所有 keys
- `commit 2596ed0` 同一次提交修复

### 7. wandb resume 需要单独 set API key
- `WANDB_RESUME=allow` + `WANDB_RUN_ID=...` 只告诉 wandb "怎么 resume"
- 不告诉 "用谁的账号" → 重新启动 session 后 `~/.netrc` 没了又跳出登录
- 解决：要么 `wandb login --relogin` 一次，要么再加 `WANDB_API_KEY=...`

---

## 📂 文件说明

| 文件 | 用途 |
|---|---|
| `setup.sh` | Stage 2 环境初始化（同 Stage 1） |
| `01_prepare_data.py` | 下载 LLaVA-Instruct + COCO + RefCOCO + ShareGPT4V + TextVQA |
| `_common2.py` | ChatFormatter + Multitask Dataset 类 + LoRA target 解析 |
| `02_baseline_eval.py` | 训前 baseline eval（caption-only / chat 两种 prompt 模式） |
| `03_train_stage2.py` | 训练脚本：base + ProjectorWithNorm + LoRA + Trainer |
| `04_eval_stage2.py` | 训后 OOD eval：RefCOCO val/testA/testB + POPE + VQAv2 + NoCaps + Stage 1 regression |
| `04_download_eval_data.py` | 下 OOD 评测数据（POPE / MME / NoCaps / VQAv2） |
| `05_sample_training_data.py` | 从训练数据抽样生成 HTML 可视化（每个 task 20 张图） |
| `06_inspect_eval_samples.py` | eval 结果分层抽样 (best/random/worst) 渲染 HTML |

---

## ✅ 衡量"Stage 2 v1 训练成功"的标准

| KPI | 目标 | **Final 实测** | 评分 |
|---|---|---|---|
| RefCOCO val Acc@0.5 ≥ 20% | 20% | **21.2%** | ✅ 刚达标（受漏洞 #1 限制）|
| RefCOCO testB Acc@0.5 ≥ 12% | 12% | **15.4%** | ✅ 干净数据上达标 |
| POPE F1 ≥ 75% | 75% | **77.9%** | ✅ 超预期 |
| VQAv2 ≥ 50% | 50% | **57.2%** | ✅ 健康 |
| NoCaps rep_rate < 5% | <5% | **0.00%** | ✅✅ 大胜 |
| NoCaps avg_len ≥ 50 词 | ≥50 | **131 词** | ✅ 长 caption 能力到位 |

**整体评价**：v1 跑通了完整 LLaVA-style multi-task LoRA 训练流水线，
**chat template 和 token 循环这两个最核心问题都解决了**。但 RefCOCO 数据有
重大 bug（实际只训了 8.8K，本应 50K）+ ShareGPT4V 选错文件，这两个发现
催生了 v2。v1 的数字作为「教育复现 baseline」是有意义的，作为「论文级 reference」
则需要 v2 那样修复后重训。

---

## 🔄 v1 → v2 改动概览

| 维度 | v1 | v2 (Phase 1+) |
|---|---|---|
| 训练 dataset 数 | 3 | **6** |
| RefCOCO 来源 | lmms-lab val (8.8K) ⚠️ | jxu124 train (42K) ✅ |
| RefCOCO+ | ❌ | ✅ UNC 原版 pickle (42K) |
| RefCOCOg | ❌ | ✅ jxu124 train (42K) |
| TextVQA | 仅 eval | ✅ 进训练 (28K, OCR 专项) |
| ShareGPT4V 文件选择 | share-captioner ⚠️ | share-captioner（同 v1，便于对比；patch 已 commit 但未应用）|
| LoRA rank | 16 | **64** (4× 容量) |
| 总样本 | 258K | **355K** |
| Grounding 占比 | 3% (实际) | **36%** (10× 提升) |
| `_try_load_hf_dataset` | silent fallback | 显式 print + warning |

详见 `../stage2-v2/README.md`。
