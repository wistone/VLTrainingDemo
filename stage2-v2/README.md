# Stage 2-v2 — Phase 1+ 改进版

在 v1 基础上**大幅扩充 grounding 数据 + 加入 OCR 任务 + 升级 LoRA 容量**，
目标是把 RefCOCO 从 v1 的 ~30% 拉到 40%+，同时新增 TextVQA OCR 能力。

---

## 🎯 实验目的：解决 v1 暴露的三个问题

### Problem 1：v1 的 RefCOCO 训练数据是个 bug
v1 用 `lmms-lab/RefCOCO` —— 这个 repo **只有 val/testA/testB，没有 train split**。
v1 代码 `for split in ["train", "validation", "val"]` 静默 fallback，**实际训了
val 数据 (~8.8K)，而不是声明的 50K**。这一方面让 grounding 信号严重不足，另一
方面**会污染未来评测**（model 训过的数据你又用来 eval 它）。

### Problem 2：缺 OCR 专项训练
v1 没接 TextVQA，模型对图中文字的识别能力靠 LLaVA-Instruct 顺带学到，水平很弱。

### Problem 3：LoRA r=16 容量不足
v1 用 r=16，对 grounding 这种"精确空间投影"任务表达力受限（Acc@0.7 仅 2%）。

---

## ⚙️ 关键改动 vs v1

| 维度 | v1 | **v2 (Phase 1+)** |
|---|---|---|
| 训练 dataset 数 | 3 | **6** |
| RefCOCO 来源 | lmms-lab (val 8.8K) | **jxu124/refcoco train (42K)** |
| RefCOCO+ | ❌ | ✅ **UNC 原版 pickle (42K)** |
| RefCOCOg | ❌ | ✅ **jxu124/refcocog train (42K)** |
| TextVQA | 仅 eval | ✅ **进训练 (28K)** |
| LoRA rank | 16 | **64** (4× capacity) |
| LoRA alpha | 32 | **128** (保持 alpha/r=2) |
| 可训练参数 | 23M (1.1%) | **78M (3.8%)** |
| LLaVA-Instruct | 150K | 100K (削减以平衡) |
| 总样本 | 300K | **354,908** |
| Grounding 占比 | 17% | **36%** ⭐ |
| OCR 专项 | 0% | **8%** ⭐ |

---

## 📦 数据组成（实际加载数）

```
[mix] 总数据组成:
      llava_instruct     100,000  (28.2%)
      refcoco             42,404  (11.9%)   jxu124/refcoco train (全部)
      refcoco_plus        42,278  (11.9%)   UNC pickle (50K cap → 42K bbox 命中)
      refcocog            42,226  (11.9%)   jxu124/refcocog train (全部)
      sharegpt4v         100,000  (28.2%)
      textvqa             28,000  (7.9%)    consensus≥3 过滤
      TOTAL              354,908
```

**按能力维度分**：
- Grounding (3 个 RefCOCO 总和): 127K = **36%** ⭐ 主提升点
- General VQA (LLaVA + TextVQA): 128K = 36%
- Long Caption (ShareGPT4V): 100K = 28%
- 其中 OCR 专项 (TextVQA): 28K = 8% ⭐ 新增

---

## 🏗️ 数据来源细节（这一阶段最有教育意义的部分）

### RefCOCO（jxu124/refcoco）

`commit e34fa20` "Switch stage2-v2 to jxu124 RefCOCO repos for actual train splits"

发现 lmms-lab 没 train split 后，换到 `jxu124/refcoco`：
- **42,404 train samples**
- 字段：`{ref_id, ann_id, image_id, file_name, sentences, bbox (xyxy 像素), captions}`
- 关键差异：jxu124 用 **xyxy 像素**（不是 lmms-lab 的 xywh）
- 图片不 bundle，要从 COCO `train2017.zip` 通过 `image_id` 查找
- 多个 captions 训练时**随机选一个**作天然 3× 数据增强

### RefCOCO+（UNC 原版 pickle）

`commit 83c8235` "Add RefCOCOPickleTaskDataset for UNC-original RefCOCO+ data"

jxu124 没 refcoco-plus，所有 HF candidates 全 401。**从 GitHub 下载原版**：
- `refs(unc).p`: pickle，含所有 ref_id, ann_id, sentences, split
- `instances.json`: COCO format annotations，含所有 ann_id → bbox (xywh) 映射
- 通过 `RefCOCOPickleTaskDataset` 自定义类加载：filter split=train → join ann_id → 查图
- Drive 持久化：
  ```
  /content/drive/MyDrive/qwenvl3/data/stage2/refcoco_plus_train/
    ├── refs.p             31MB
    └── instances.json    115MB
  ```

⚠️ **15% 的 train refs 找不到 bbox** → 50K limit 实际加载 42,278。原因可能是
`refcoco+/instances.json` 是个子集（Feb 2016 版本），不含所有 RefCOCO+ 引用的
ann_id。可接受。

### TextVQA（lmms-lab/textvqa, consensus 过滤）

字段：`{image, question, answers (10 个标注员)}`。

`TextVQATaskDataset` 实现：
- **多数投票** 选 GT answer
- **Consensus filter**：要求至少 3 人同意，否则跳过该样本（去除歧义样本）
- 实时过滤（`__getitem__` 时检查），不预扫
- 28K (target) → ~24-26K (实际，过滤后)

---

## 📊 评测结果

### ⏳ 待 eval（训练 ~7h 时撰写）

实际数字会在训练完后用 `stage2/04_eval_stage2.py` 跑出来。基于 ckpt-5000 (v1)
和数据扩量后的预期：

| 指标 | v1 实测 | **v2 预测** | 增幅来源 |
|---|---|---|---|
| RefCOCO val Acc@0.5 | 20% | **40-46%** ⭐ | +RefCOCO+/g, LoRA r=64, 真 train data |
| RefCOCO testA Acc@0.5 | 20% | 40-46% | 同上 |
| RefCOCO testB Acc@0.5 | 11% | **28-35%** ⭐ | +RefCOCO+/g 物体多样性 |
| RefCOCO Acc@0.7 | 2% | 14-20% | LoRA r=64 提精度 |
| POPE F1 | 78.1% | 79-82% | 持平 |
| POPE Yes-ratio | 67.8% | 55-62% | 多任务平衡 |
| VQAv2 acc | 55.9% | 62-66% | TextVQA 间接帮助 |
| **TextVQA acc** | **~25%**(估) | **40-50%** ⭐ | **+OCR 专项** |
| NoCaps rep_rate | 0% | 0% | 已饱和 |
| NoCaps avg_len | 136 词 | 100-130 词 | 略减更精炼 |
| NoCaps word_recall | 27% | 33-40% | 视觉理解更强 |

---

## ⏱ 实验时长

| 配置 | 数值 |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition (102GB) |
| Batch | 8 × grad_accum 4 = effective 32 |
| LoRA | r=64, alpha=128 |
| `--no_gradient_checkpointing` | ✅ 开启（102GB 显存够用，省 30% 时间） |
| 总 iters | 11,091 |
| 单 iter | **~2.77 s/it** |
| **训练时长** | **~8.5h** ⭐ |
| GPU 显存峰值 | 72 / 95 GB (75%) |

### vs v1
- v1 (A100 40GB, batch=8 grad=4, with checkpoint): 12h, 300K samples
- v2 (Blackwell 102GB, batch=8 grad=4, **no** checkpoint): **8.5h**, 354K samples
- 数据多 18%，但训练时间反而少 30% —— 全靠关 gradient_checkpointing

---

## 🐛 主要踩过的坑（这次新发现的）

### 1. lmms-lab 数据集是 eval-only，**没有 train split**
最大坑。v1 偷偷训了 ~8.8K 而不是 50K 还浑然不知。修复：
- 修了 `_try_load_hf_dataset` 加 split 显式 logging + 强制 train 检查
- 换数据源到 jxu124（而 jxu124 又没 refcoco-plus，进一步扩展到 UNC pickle）

### 2. jxu124 字段格式跟 lmms-lab 不兼容
- `bbox`: jxu124 是 xyxy 像素，lmms-lab 是 xywh COCO 格式
- `image`: jxu124 不 bundle bytes（要从 COCO zip 查），lmms-lab bundle
- `captions`: jxu124 是 list[str]（多个 ref），lmms-lab 是单个 `answer`
- 修复：`RefCOCOTaskDataset` 加 `bbox_format` 参数 + multi-source `_extract_*` 方法

### 3. RefCOCO+ HF 上根本没 train repo
- 所有候选（jxu124-plus, lmms-lab/refcoco_plus, Multimodal-Fatima 等）全 404
- 解决：从 lichengunc/refer GitHub 仓库下原版 pickle + COCO 2014 instances JSON
  ```bash
  wget https://web.archive.org/web/2024/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip
  wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
  ```
- 写新的 `RefCOCOPickleTaskDataset` 类适配

### 4. OOM at batch=16 + no_gradient_checkpointing（cross_entropy 阶段）
- 现象：forward 跑完了，cross_entropy 想分配 10.37 GB 给 fp32 logits 时 OOM
- 原因：accelerate 的 `convert_to_fp32` 把 bf16 logits 升精度，
  `[16, 979, 152064] × 4B = 9.5GB` 单个 tensor
- 结合 forward 累积的 ~78GB activations → 超 95GB 上限
- 修复：降到 **batch=8 grad_accum=4**（保持 effective batch=32）
- 此时 logits 仅 4.77GB，活动内存 ~50GB，安全

### 5. 数据移动到 Drive 路径混乱
- `wget && unzip` 时 cwd 在 `refcoco_plus_train/`，所以解压到 `refcoco_plus_train/refcoco+/`
- 后续 `mv` 命令 cwd 自动重置到 `/content`，找不到 `refcoco+/...`（相对路径）
- 修复：所有 `mv` 命令用绝对路径

### 6. silent fallback 必须改成 loud
- v1 的 `_try_load_hf_dataset` 默默 fallback to val，多么危险的设计
- v2 修：load 时强制 print "loaded split=X (n=N)"，如果非 train 还会大字号 warning
- 原则：**任何"找不到首选"的 fallback 都应该 print，不能静默**

---

## 📂 文件说明

| 文件 | 改动 vs v1 |
|---|---|
| `setup.sh` | 同 v1 |
| `01_prepare_data.py` | +RefCOCO/g (jxu124) + RefCOCO+ (手动 wget UNC pickle) |
| `_common2.py` | +`RefCOCOPickleTaskDataset`, `RefCOCOTaskDataset` 改 `bbox_format` 参数, `+TextVQATaskDataset` |
| `03_train_stage2.py` | +argparse 6 个新 task 参数, +`--no_gradient_checkpointing`, default LoRA r=64 |

---

## 🛠️ 启动命令（实际用的）

```bash
# 0. setup（同 v1）
bash stage2-v2/setup.sh

# 1. 下数据
python stage2-v2/01_prepare_data.py --only_phase1plus

# 2. 手动下 RefCOCO+ UNC 原版（pickle + COCO instances）
mkdir -p /content/drive/MyDrive/qwenvl3/data/stage2/refcoco_plus_train
cd /content/drive/MyDrive/qwenvl3/data/stage2/refcoco_plus_train
wget https://web.archive.org/web/2024/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip
unzip refcoco+.zip
mv refcoco+/refs\(unc\).p refs.p
mv refcoco+/instances.json instances.json
rm -rf refcoco+ refcoco+.zip

# 3. 烟雾测试
python stage2-v2/03_train_stage2.py --smoke_test \
    --stage1_ckpt /content/drive/MyDrive/qwenvl3/stage1_ckpt_v3/checkpoint-11500 \
    --processor_dir /content/drive/MyDrive/qwenvl3/stage1_init \
    --stage2_data_root /content/drive/MyDrive/qwenvl3/data/stage2 \
    --output_dir /content/drive/MyDrive/qwenvl3/stage2_v2_smoke \
    --batch_size 16 --grad_accum 2 --report_to none

# 4. 正式训练（Blackwell 102GB）
python stage2-v2/03_train_stage2.py \
    --stage1_ckpt /content/drive/MyDrive/qwenvl3/stage1_ckpt_v3/checkpoint-11500 \
    --processor_dir /content/drive/MyDrive/qwenvl3/stage1_init \
    --stage2_data_root /content/drive/MyDrive/qwenvl3/data/stage2 \
    --output_dir /content/drive/MyDrive/qwenvl3/stage2_v2_ckpt \
    --batch_size 8 --grad_accum 4 \
    --num_workers 8 \
    --save_steps 400 \
    --no_gradient_checkpointing \
    --run_name stage2-v2-phase1plus
```

⚠️ 注意：v2 训练用 `batch=8 grad_accum=4`（不是烟雾测试的 16/2），是因为 batch=16
+ no_gradient_checkpointing 在 forward 后 cross_entropy 阶段会 OOM（坑 #4）。

---

## 📈 训练监控

- wandb run: `stage2-v2-phase1plus`
- run id: `lodxfdjy` (示例)
- 健康曲线：loss 1.72 → 0.85 在前 1000 步，grad_norm 0.3-0.7 稳定

---

## ✅ 衡量"v2 训练成功"的标准

| KPI | 目标 | 解读 |
|---|---|---|
| RefCOCO val Acc@0.5 ≥ 40% | 比 v1 (20%) 翻倍 | grounding 真有质变 |
| RefCOCO testB Acc@0.5 ≥ 25% | 比 v1 (11%) 翻倍 | 复杂物体也能定位 |
| TextVQA acc ≥ 35% | 从 0 到有 OCR 能力 | 新维度 |
| POPE F1 ≥ 78% | 不退化 | 多任务平衡 |
| NoCaps rep_rate < 5% | 不退化 | Stage 1 痛点保持解决 |
| Stage 1 regression rep_rate < 20% | 不退化 | 没遗忘 caption 能力 |

---

## 🔬 这一阶段的方法论收获

1. **永远不要相信 silent fallback**：每个 try/except 失败都要 print，否则会发生
   "训练 22h 才发现一直在用错数据" 这种灾难
2. **数据格式异质性**：jxu124 / lmms-lab / UNC pickle 三个 RefCOCO 源字段全不同。
   写 dataset class 时要支持多源
3. **大显存的优化方式**：关 gradient_checkpointing 对 90GB+ 显存的卡更友好；
   batch_size 别盲目 doubling，注意 logits fp32 内存爆炸
4. **教育复现 vs SOTA**：我们用 2% 的参数量 + 0.02% 的训练数据，做 SOTA 30-40% 的
   能力水平。**目标不是数字漂亮，是把每个齿轮看一遍**
