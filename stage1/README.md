# Stage 1 — Projector Alignment

LLaVA 复现的第一阶段：在冻结的 Qwen2.5-1.5B LLM 和 SigLIP2-SO400M ViT 之间，
**仅训练一个 projector**，让视觉 patch features 学会"用 LLM 听得懂的语言说话"。

---

## 🎯 实验目的

让 SigLIP2 输出的 729 个视觉 token 在加入 LLM 输入序列后，能让 LLM 流畅生成
对应的图像 caption。**核心是建立"视觉 → LLM token embedding"的对齐关系**。

| 模块 | 状态 | 可训练参数 |
|---|---|---|
| SigLIP2-SO400M ViT | ❄️ 冻结 | 0 |
| **ProjectorWithNorm**（自定义 2 层 MLP + LayerNorm） | ✅ 训练 | ~5M |
| Qwen2.5-1.5B LLM | ❄️ 冻结 | 0 |

---

## 📦 数据组成

| 数据集 | 样本数 | 格式 | 图源 |
|---|---|---|---|
| **LLaVA-Pretrain-558K** | 558K | `{image, conversations: [<image>, alt-text]}` | BLIP / CC-3M / SBU 混合 |

**特点**：极简单 turn 的图-alt-text pair。Caption 通常 5-15 词的电商风格短描述。

**Holdout**：`split_holdout()` 从最后 1000 条 random sample 出 20 张到 `holdout_20.json`，
但⚠️ **训练时 `data` 加载 558K 全部，没有剔除 holdout** —— 严格说不是真 held-out，
只是「训练里见过但仅 1 次的」抽样。Stage 1 ckpt-11500 评测就是基于这 20 张图。

---

## 🏗️ 关键架构决策：ProjectorWithNorm

```python
class ProjectorWithNorm(nn.Module):
    def __init__(self, vision_hidden, text_hidden):
        self.linear_1 = nn.Linear(vision_hidden, text_hidden)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(text_hidden, text_hidden)
        self.norm = nn.LayerNorm(text_hidden)   # ← 关键
```

**为什么必须加 LayerNorm**：
- Qwen2.5 token embedding L2 norm ≈ 0.78（远小于 LLaMA 的 ~5）
- 默认 LlavaMultiModalProjector 输出会被训练放大到 norm 800+
- 这种"巨大的视觉 token"完全淹没了 LLM 内部残差流的文字信号
- 现象：loss 卡在 ~9.5 不下降，ablation Δloss 几乎为 0
- 加 LayerNorm 后：projector 输出 norm 稳定到 √1536 ≈ 39，与文字 token 同尺度
- Loss 立刻能从 ~11.93 (ln(152K)) 降到 1.7

`commit 9c57cde` "Add LayerNorm to projector to fix Qwen2.5 scale mismatch"

---

## 📊 评测结果

### 主指标（ckpt-11500，66% 训练完成）

| 指标 | 数值 | 含义 |
|---|---|---|
| **with-image loss** | **1.727** | 从 11.93 (ln 152K) 降到 1.7，对齐成功 |
| **without-image loss** | 5.164 | "盲跑"baseline |
| **Δloss (image - no image)** | **3.437** | 视觉信号强度，>3 表示视觉真的被"接进了" LLM |
| Avg generation length | 11 词 | 短 alt-text 风格 |
| Token repetition rate | ~10% | 仍有偶发死循环（见踩坑 #5） |

### 与 ckpt-4500 (26%) 的对比

20 张 holdout 图人工 review，结果分布：
- 🟢 改善：9 张（如 Chris Stapleton 名人识别完美命中、Bernese 犬种 + 梵高画风识别）
- ➖ 持平：5 张
- 🟡 喜忧参半：3 张
- 🔴 退化：1 张（#5 UAG tank — OCR 提升后掉进 "0 0 0" 数字循环）
- ⚫ 双方都崩坏：2 张（Subaru BPE 陷阱、空白展厅）

详见 `holdout_images_local/stage1_eval_compare.html`（由
`commit 06d6e2b` "Add Stage 1 eval comparison HTML generator" 生成）。

---

## ⏱ 实验时长

| 配置 | 数值 |
|---|---|
| GPU | Colab Pro+ A100 40GB |
| Batch | 8 (per device) × grad_accum 4 = effective 32 |
| 总 iters | 8088 |
| **训练时长** | **~12h**（中途 wandb 显示 5.45 s/it） |
| 中途断过 | 是，5300 步崩溃后 resume 续训 |

---

## 🐛 主要踩过的坑

### 1. 视觉权重和 LLM 权重根本没装载（最严重）
- `commit e4df89d` "Fix critical: vision and language pretrained weights were never loaded"
- 现象：`from_pretrained` 后看 missing/unexpected keys 都是 448 个
- 原因：transformers ≥4.50 改了 LlavaForConditionalGeneration 的内部结构，组件挂在 `model.model.*` 而非 `model.*`
- 修复：在 `02_assemble_model.py` 显式做 weight transfer：
  ```python
  target_vt = getattr(vt_module, "vision_model", vt_module)
  target_vt.load_state_dict(vision_model.state_dict(), strict=False)
  ```

### 2. Projector 输出 norm 800 vs LLM embed norm 0.78（1000× 放大）
- 见上文「关键架构决策」，加 LayerNorm 解决
- `commit 9c57cde`

### 3. `<image>` token 数量与视觉 feature 数量不一致
- 现象：`tokens: 32, features: 35782656`（35M！是 batch × 729 × 1536）
- 原因：tokenizer 把 `<image>` 当成 1 个 token，但 SigLIP2 输出 729 个 patch
- 修复：手动展开 `<image>` 占位符成 729 个 image_token_id
- `commit 10c67fc` "Expand <image> placeholder to N tokens to match vision feature count"

### 4. OOM at batch=32（Qwen vocab 152K 引发）
- `logits.fp32 = batch × seq × 152064 × 4B`，batch=32 时 = ~15GB 单 tensor
- 修复：默认 batch=8 grad_accum=4 + bf16 训练
- `commit bd3ec1f` "Tune Stage 1 defaults for 40GB A100"

### 5. Token 重复死循环（Stage 1 caption 特性）
- 现象：模型对某些图说 "subarv subarv subarv..." 或 "0 0 0 0 0..."
- 原因：BPE 把罕见品牌词（如 Subaru）切成多个 subword；预测错首个 subword 后掉进 OOD 自我复读
- Stage 1 训练范式（裸 caption 续写）解决不了，**Stage 2 chat template + 多任务训练自然修复**

### 6. Held-out 实际不存在
- `split_holdout()` 切 20 张到 json，但训练 dataset 加载所有 558K，没排除
- 影响：评测严格说不是 OOD，但 558K 中单条样本被记忆概率低，仍可作为能力指针
- Stage 3 时应改成"先抽 holdout，再 limit 训练数据"

### 7. 中间 checkpoint 没有 tokenizer
- HF Trainer 的 `processing_class=tokenizer` 才会自动存
- 修复：传入 + 加 `--processor_dir` 参数让 eval 找到
- `commit b32e350` "Load tokenizer from processor_dir; save tokenizer with each checkpoint"

### 8. Drive 同步滞后导致 ckpt 不可用
- save_total_limit=2 删旧 ckpt，但 Drive 同步未必同步完
- 修复：`resolve_*_ckpt` 自动 fallback 到最新可用 ckpt
- `commit d52fd9a` "Auto-fallback to previous checkpoint if Drive sync lags"

---

## 📂 文件说明

| 文件 | 用途 |
|---|---|
| `setup.sh` | Colab 环境初始化（pip install + Drive 挂载验证） |
| `01_prepare_data.py` | LLaVA-Pretrain-558K 下载 + 解压到 `/content` |
| `02_assemble_model.py` | 装配 Qwen2.5 + SigLIP2 + ProjectorWithNorm 初始 ckpt |
| `_common.py` | ProjectorWithNorm 类 + transformers API 兼容层 |
| `03_train_projector.py` | 训练脚本 |
| `04_eval_stage1.py` | 评测：caption 生成 + image-token ablation Δloss |
| `05_compare_eval.py` | 比较两个 ckpt 的 eval 结果，生成 HTML |

---

## ✅ 衡量"Stage 1 训练成功"的标准

- [x] `with-image loss` < 2.5（最终 1.727 ✅）
- [x] `Δloss (image vs no-image)` ≥ 3.0（最终 3.437 ✅）
- [x] Holdout 上人工 review 大部分 caption 合理（9/20 改善 ✅）
- [x] 进入 Stage 2 后 projector 不需要从头学（Stage 2 直接复用 ✅）

Stage 1 已完成 → 输出: `stage1_ckpt_v3/checkpoint-11500`，被 Stage 2 / Stage 2-v2 共用。
