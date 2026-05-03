# Stage 2 — Multitask LoRA Training (v1)

LLaVA 复现的第二阶段：在 Stage 1 训好的 projector 基础上，**给 LLM 加 LoRA 适配**，
学习 chat template + 三种下游任务（VQA / grounding / 长 caption）。

> ⚠️ 这是 **v1 版本**。v2 (`../stage2-v2/`) 是 Phase 1+ 改进版，加入了 RefCOCO+/g、
> TextVQA、LoRA r=64 等改动。

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
| **总可训练** | | **~23M (1.1% of total)** |

---

## 📦 数据组成

| 数据集 | 默认样本数 | 占比 | 教什么 |
|---|---|---|---|
| **LLaVA-Instruct-150K** | 150K | 50% | 多轮 VQA + 推理 |
| **RefCOCO** (lmms-lab) | 50K (实际 8.8K ⚠️) | 17% | 视觉定位 → bbox 输出 |
| **ShareGPT4V** | 100K | 33% | 段落长 caption |
| **Total** | 300K | | |

**图源**：COCO train2017.zip (~18GB)，由 LLaVA-Instruct 和 ShareGPT4V 共用。

⚠️ **lmms-lab/RefCOCO 是 eval-only 数据集**（只有 val/testA/testB 没有 train），
v1 训练时**静默 fallback 到 val 数据**，实际只训了 ~8.8K grounding 样本。
这个 bug 在 v2 里修复（参见 `stage2-v2/README.md`）。

**No held-out splits**：LLaVA-Instruct 和 ShareGPT4V 都用 `[:limit]` 切，没保留
holdout。所以 eval 时不能用「训练数据采样」，必须用 OOD 公开 benchmark
（POPE / VQAv2 / NoCaps）。

### TextVQA 也下了但没用上
`commit 0715a40` "Replace OCR-VQA with TextVQA in Stage 2 data prep" —— 当初想
拿 TextVQA 顶替下载失败的 OCR-VQA 进训练，但代码里 `build_task_datasets()`
最终没接上。**TextVQA 仅作 baseline eval 对照**。Phase 1+ (v2) 真接进去训了。

---

## 📊 评测结果

### Sanity test (ckpt-5000, 62% 训练完成)

跑命令：`04_eval_stage2.py --n_refcoco 100 --n_pope 500 --n_vqav2 200 --n_nocaps 30`

| 任务 | 指标 | 实测 | 解读 |
|---|---|---|---|
| **RefCOCO val** | Acc@0.5 | **20%** | 远低于业界（LLaVA-1.5-7B 30%, Qwen-VL-7B 88%） |
| RefCOCO val | Acc@0.7 | 2% | 几乎从来不精准 → LoRA r=16 + 少量数据的天花板 |
| RefCOCO val | mean IoU | 0.292 | 知道大概区域，框不准 |
| RefCOCO val | parse_rate | **100%** | bbox 格式学透了 |
| **POPE** | F1 | **78.1%** ⭐ | 接近 LLaVA-1.5-7B (86%) |
| POPE | Yes-ratio | 67.8% ⚠️ | 明显 yes-bias（健康范围 45-55%） |
| **VQAv2** | accuracy | **55.9%** | 落在 50-60% 预期中段 |
| **NoCaps** | rep_rate | **0.0%** ⭐⭐ | Stage 1 死循环问题彻底修复！ |
| NoCaps | avg_len | 136 词 | 长 caption 能力 work |
| NoCaps | word_recall | 27% | 内容覆盖底部健康范围 |

**最大胜利**：NoCaps repetition_rate 从 Stage 1 的 ~10% 降到 **0%**。
chat template + 多任务训练**根本性修复了 token 死循环**这个 Stage 1 痛点。

**最大短板**：RefCOCO 仅 20% — 因为实际 grounding 训练数据只有 8.8K（lmms-lab
没有 train split），且 LoRA r=16 表达力有限。**这是 Stage 2-v2 要解决的核心
问题**。

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

## 🐛 主要踩过的坑

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

### 7. 没有 held-out split
- LLaVA-Instruct + ShareGPT4V 训练时 `[:limit]` 直接切，eval 时 `random.sample` 又
  从相同池子里抽 → 评测污染
- v1 通过**只用 OOD eval (POPE / VQAv2 / NoCaps)** 绕开
- v2 也没改这个，但 Stage 3 SFT 时必须修

### 8. wandb resume 需要单独 set API key
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

---

## ✅ 衡量"Stage 2 v1 训练成功"的标准

| KPI | 目标 | 实测 | 评分 |
|---|---|---|---|
| RefCOCO val Acc@0.5 ≥ 20% | 20% | 20% | ✅ 刚到底线 |
| POPE F1 ≥ 75% | 75% | 78.1% | ✅ 超预期 |
| VQAv2 ≥ 50% | 50% | 55.9% | ✅ 健康 |
| NoCaps rep_rate < 5% | <5% | 0% | ✅✅ 大胜 |
| Stage 1 caption 不退化 | rep_rate <15% | (未跑) | ⏸ |

**整体评价**：v1 跑通了完整 LLaVA-style multi-task LoRA 训练流水线，**chat template
和 token 循环这两个最核心问题都解决了**。但 RefCOCO 数据有重大 bug（实际只
训了 8.8K，本应 50K），这个发现催生了 v2。

---

## 🔄 v1 → v2 改动概览

| 维度 | v1 | v2 (Phase 1+) |
|---|---|---|
| 训练 dataset 数 | 3 | **6** |
| RefCOCO 来源 | lmms-lab (val) | jxu124 (train) |
| RefCOCO+ | ❌ | ✅ UNC 原版 pickle |
| RefCOCOg | ❌ | ✅ jxu124 train |
| TextVQA | 仅 eval | ✅ 进训练 |
| LoRA rank | 16 | **64** |
| 总样本 | 300K | 354K |
| Grounding 占比 | 17% | **36%** |

详见 `../stage2-v2/README.md`。
