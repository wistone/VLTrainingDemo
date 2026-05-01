# VL 模型训练复现计划

## 项目目标

在 Google Colab 单卡 A100（80GB 优先）上，**端到端复现一个 LLaVA-style VL 模型的三阶段训练流程**，基座使用 Qwen2.5-1.5B。

**目标不是**：复刻 Qwen2.5-VL 的 benchmark 数字、训练自研 ViT、或追求 SOTA。
**目标是**：通过亲手跑完三个阶段，内化 VL 训练中三个独立的核心概念——视觉-语言对齐、多任务统一、指令对齐。

**最终交付物**：
- 一个能对自然图像做 VQA、OCR、简单 grounding 的 1.5B 量级 VL 模型
- 三阶段完整训练日志和评估报告
- 一份"踩坑笔记"，记录每阶段的实际困难与解决方案

---

## 全局技术选型

| 组件 | 选型 | 备注 |
|---|---|---|
| 基座 LLM | Qwen2.5-1.5B-Instruct | 备选 0.5B（显存吃紧时降级） |
| 视觉塔 | SigLIP-SO400M-patch14-384 | 全程冻结 |
| 投影层 | 2-layer MLP（hidden=2048，gelu） | 视觉特征 → LLM embedding |
| 精度 | bf16 + gradient checkpointing | |
| 训练框架 | LLaMA-Factory（首选）/ ms-swift | 内置 VL collator，省事 |
| LoRA 配置 | r=16, alpha=32, target=q/k/v/o + gate/up/down | Stage 2/3 用 |
| Checkpoint 策略 | 每 200 步存到 Google Drive | 防 session 断开 |
| 数据格式 | webdataset (tar shards) | Drive I/O 优化 |

---

## Stage 1 · Projector 对齐

### 目标
让冻结的 SigLIP 输出能被冻结的 Qwen2.5-1.5B 理解。**只训练 MLP 投影层**（~20M 参数）。

### 关键学习项
1. **多模态 collator 怎么写**：图像预处理 → patch → ViT → projector → 与文本 token 交错拼接
2. **`<image>` 占位符的 token 替换机制**：input_ids 中 `<image>` 被替换为 N 个视觉 token embedding
3. **为什么必须冻结 LLM**：projector 冷启动权重接近随机，回传梯度会污染 LLM
4. **冷启动的训练稳定性**：观察前 100 步 loss 是否 NaN、是否需要 warmup
5. **视觉 token 与文本 token 的尺度差异**：projector 输出的 norm 应与 LLM embedding norm 量级匹配

### 数据
- **LLaVA-Pretrain-558K**（LAION/CC/SBU 清洗后子集，单轮 caption 任务）
- 数据量：约 558,000 图文对
- 大小：图像 ~25GB + json ~500MB

### 训练配置
- batch size: 32（80GB）/ 16（40GB）
- learning rate: 1e-3（projector 高 lr，因为只训它）
- epochs: 1
- 总步数：约 17,500 步

### 估计耗时
- A100 80GB：**4–6 小时**
- A100 40GB：**6–9 小时**（需要降 batch + grad accumulation）

### 衡量标准（Pass / Fail）
| 指标 | 通过线 | 怎么测 |
|---|---|---|
| Loss 收敛 | 从 ~6.0 降到 ≤ 2.5 | 看 wandb / tensorboard |
| 定性 caption 质量 | 20 张 held-out COCO 图，至少 15 张描述抓到主体 | 人工 review |
| 不崩 | 无 NaN，无显存爆炸 | 看日志 |
| 视觉 token 利用率 | 输入图像 vs 不输入图像，loss 差 ≥ 1.0 | 自己写 ablation 脚本 |

**自我验证**：随便贴一张猫图，模型应该能说出"a cat sitting on..."级别的描述。如果还在生成无关内容，说明 projector 没对齐成功。

---

## Stage 2 · 多任务预训练（轻量版）

### 目标
在统一的 next-token 目标下，让模型见过多种 VL 任务格式（VQA / caption / OCR / grounding）。**LLM 加 LoRA + projector 全参微调**，ViT 仍冻结。

### 关键学习项
1. **任务混合配比怎么定**：不同任务样本数差异大，等比例 vs 平衡采样的影响
2. **统一接口怎么设计**：grounding 框如何编码成文本（`<box>(x1,y1),(x2,y2)</box>`）、OCR 如何用 prompt 触发
3. **长尾任务被淹没现象**：占比 5% 的任务 loss 是否能降下来
4. **LoRA vs 全参的取舍**：为什么 1.5B 这个量级 LoRA 够用
5. **Projector 是否需要继续训**：观察 Stage 1 训好的 projector 在新数据分布下是否需要适应

### 数据组合
| 子集 | 样本数 | 任务类型 |
|---|---|---|
| LLaVA-Instruct-150K | 150K | 多轮 VQA |
| GQA 子集 | 200K | 推理类 VQA |
| OCR-VQA 子集 | 200K | 文字识别 VQA |
| RefCOCO + Visual Genome | 100K | Grounding（带框） |
| DocVQA 子集 | 50K | 文档 VQA |
| ShareGPT4V 子集 | 100K | 长 caption |
| **合计** | **~800K** | |

### 训练配置
- batch size: 16（80GB）/ 8（40GB）
- learning rate: LLM LoRA 2e-4，projector 1e-4
- epochs: 1
- 总步数：约 50,000 步

### 估计耗时
- A100 80GB：**12–16 小时**（**需要拆 2 个 Colab session**，靠 checkpoint 续训）
- A100 40GB：**18–24 小时**（拆 3 session）

### 衡量标准
| 指标 | 通过线 | 怎么测 |
|---|---|---|
| 各子任务 loss 单独收敛 | 每个任务 loss 都从初始降 ≥ 30% | 按任务标记日志 |
| VQA 准确率 | GQA 1K held-out > 40% | 跑 evaluator |
| OCR 字符匹配 | OCR-VQA 1K held-out exact match > 35% | 自写 |
| Grounding IoU | RefCOCO val 100 样本 mean IoU > 0.4 | 解析 box 后算 IoU |
| Stage 1 能力没退化 | caption 质量定性不变差 | 人工对比 |

**自我验证**：能正确定位"图中的红色汽车在哪"（输出合理的 bounding box），能读出图中招牌文字。

---

## Stage 3 · SFT 指令对齐

### 目标
把"会看图的基座"变成"听话的多模态助手"。LoRA on LLM + projector 全参，ViT 冻结。

### 关键学习项
1. **Chat template 的放大效应**：同样的数据，套不套 `<|im_start|>` 模板效果差异
2. **多轮 vs 单轮分布**：多轮对话占比对长对话能力的影响
3. **灾难遗忘**：Stage 2 学的 grounding/OCR 能力在 SFT 后还在吗？
4. **System prompt 设计**：不同 system prompt 对回答风格的塑造
5. **采样参数**：temperature / top_p 对推理质量的实际影响（推理时观察）

### 数据
- **LLaVA-1.5-mix-665K** 或 ShareGPT4V（~200K 子集起步）
- 推荐：先用 200K 子集快速跑通，再决定是否扩到 665K

### 训练配置
- batch size: 16
- learning rate: 1e-4（比 Stage 2 低）
- epochs: 1（数据量大）/ 2（数据量小）
- 总步数：12,500（200K 子集）/ 41,000（665K）

### 估计耗时
- 200K 子集 @ A100 80GB：**3–4 小时**
- 665K @ A100 80GB：**8–12 小时**

### 衡量标准
| 指标 | 通过线 | 怎么测 |
|---|---|---|
| 对话连贯性 | 50 prompt 多轮对话，无明显跑题 | 人工 + LLM-as-judge |
| 指令跟随 | "用一句话总结这张图" 类指令服从率 > 80% | 50 prompt 抽检 |
| Stage 2 能力保留 | grounding / OCR 准确率下降 < 20% | 复用 Stage 2 evaluator |
| 拒答合理性 | 模糊/不安全提问能合理拒答 | 准备 10 个 trick prompt |

**自我验证**：能多轮聊一张图、能按格式输出（"列出图中所有物体并编号"）、能拒答幻觉性问题。

---

## 全程时间预算

| 阶段 | 80GB 单卡 | 40GB 单卡 | session 数 |
|---|---|---|---|
| 环境搭建 + 数据准备 | 4–6h | 4–6h | 1 |
| Stage 1 | 4–6h | 6–9h | 1 |
| Stage 2 | 12–16h | 18–24h | 2–3 |
| Stage 3（200K 子集） | 3–4h | 5–7h | 1 |
| 评估 + 调试余量 | 6–8h | 6–8h | 1–2 |
| **合计** | **30–40h** | **40–55h** | **6–8 个 Colab session** |

按每天 1 个 session 节奏，**总周期 1–2 周**。

---

## 风险与备选

| 风险 | 触发条件 | 备选方案 |
|---|---|---|
| Colab 抢不到 A100 | Pro+ 都没卡 | 降级到 L4，基座换 0.5B，跳过 Stage 2 |
| Stage 2 训不完 | 单 session 12h 不够 | 数据砍半到 400K，或 LoRA rank 降到 8 |
| Stage 1 loss 不下降 | 前 1000 步 loss 不动 | 检查 projector 初始化、warmup、lr 太大 |
| Drive I/O 瓶颈 | GPU 利用率 < 50% | 数据预先打包成 webdataset tar，本地 SSD 缓存 |
| 灾难遗忘严重 | Stage 3 后 grounding 全废 | 在 SFT 数据里混 5–10% Stage 2 数据 |

---

## 已确定的决策（2026-05-01）

- **基座**：Qwen2.5-1.5B-Instruct
- **框架**：ms-swift，**仅用于 Stage 2/3**
- **Stage 1 框架例外**：用 HuggingFace transformers 直接写训练循环。原因：ms-swift 假设输入是已对齐的 VL 模型，不擅长"从零组装 LLaVA 架构 + 只训 projector"。Stage 1 训完后导出为 LLaVA 标准格式 checkpoint，Stage 2/3 切回 ms-swift

---

## Stage 1 详细工作拆分

### 任务列表

| # | 任务 | 输出 | 耗时 | 依赖 |
|---|---|---|---|---|
| 1.1 | Colab 环境初始化（GPU 验证 / pip install / wandb 登录） | 可运行环境 | 0.5h | — |
| 1.2 | 数据下载与解压（LLaVA-Pretrain-558K → /content/data） | 558K 图文对在本地盘 | 1–1.5h | 1.1 |
| 1.3 | 模型组装（Qwen2.5-1.5B + SigLIP + 随机初始化 MLP → LlavaForConditionalGeneration） | `stage1_init/` 在 Drive | 0.5h | 1.1 |
| 1.4 | 烟雾测试：在 100 样本子集上跑 50 步，验证 loss 下降、显存够用 | 烟雾测试通过 | 0.5h | 1.2, 1.3 |
| 1.5 | 全量训练 1 epoch（558K 样本 / batch 32 / lr 1e-3 / cosine） | `stage1_ckpt/` 在 Drive | 4–6h | 1.4 |
| 1.6 | 评估：20 张 held-out caption 质量 + image-token ablation（有图 vs 无图 loss 差） | `eval_stage1/` 报告 | 1h | 1.5 |
| 1.7 | 决策点：通过则进 Stage 2；不通过则回到 1.5 调超参 | — | — | 1.6 |

**总耗时**：80GB 单卡 ~7–10h，建议 1 个 Colab session 内完成（烟雾测试 + 全量训练 + 评估）。

### 脚本清单（stage1/ 目录）

| 文件 | 作用 | 何时跑 |
|---|---|---|
| `setup.sh` | 安装依赖 + 验证 GPU | 每次新 Colab session 第一步 |
| `01_prepare_data.py` | 下载 LLaVA-Pretrain-558K，解压，校验 5 个随机样本 | 任务 1.2，每次新 session 都要跑（/content 不持久） |
| `02_assemble_model.py` | 组装 Qwen2.5-1.5B + SigLIP + 随机 MLP，存到 Drive | 任务 1.3，**只跑一次** |
| `03_train_projector.py` | Stage 1 主训练脚本（freeze ViT + LLM，只训 projector，HF Trainer） | 任务 1.4 + 1.5 |
| `04_eval_stage1.py` | Caption 质量评估 + image-token ablation | 任务 1.6 |

### Pass/Fail 衡量线（再次列出）

| 指标 | 通过线 | 测法 |
|---|---|---|
| Loss 收敛 | 从 ~6.0 降到 ≤ 2.5 | wandb 曲线 |
| Caption 质量 | 20 张 held-out 至少 15 张抓到主体 | 人工 review |
| 视觉 token 利用率 | 有图 vs 无图 loss 差 ≥ 1.0 | `04_eval_stage1.py` ablation |
| 训练稳定性 | 无 NaN，显存 < 75GB | 训练日志 |
