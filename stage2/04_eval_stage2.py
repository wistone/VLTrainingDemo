"""Stage 2 训练后 OOD 综合评测。

跟 02_baseline_eval.py 的差异：
  - 02_baseline_eval：训练前 baseline，从训练集中抽样（in-distribution，不严谨）
  - 04_eval_stage2：训练后正式评测，全部用 OOD 公开 benchmark（严谨可对比）

覆盖 4 个评测任务：
  1. RefCOCO val/testA/testB    grounding，IoU@0.5/IoU@0.7/mean IoU         ⭐ 主指标（天然 held-out）
  2. POPE                       幻觉测试，Yes/No 二分类，F1 / Acc / Yes-ratio  ⭐ 诚实度
  3. VQAv2 val 子集             通用 VQA，标准 VQA accuracy                  ⭐ 通用能力
  4. Stage 1 holdout 回归       caption 长度 + token 重复率                  ⭐ 防灾难性遗忘

== 模型加载流程 ==
  Stage 2 ckpt 目录里只有：
    - adapter_model.safetensors  (PEFT LoRA)
    - adapter_config.json
    - multi_modal_projector.safetensors  (callback 单独存的)
    - tokenizer files (有的话)

  完整 base model.safetensors 在 Stage 1 ckpt 里。所以加载顺序：
    1. 从 Stage 1 ckpt 加载 base LlavaForConditionalGeneration
    2. install_custom_projector(init_dir=stage2_ckpt)  ← 用 Stage 2 训好的 projector 替换
    3. PeftModel.from_pretrained(model, stage2_ckpt)   ← 套上 LoRA adapter
    4. merge_and_unload() 合并 LoRA 到 base，推理快 1.5×

== 用法 ==

  全跑：
    python stage2/04_eval_stage2.py \\
        --stage2_ckpt /content/drive/MyDrive/qwenvl3/stage2_ckpt \\
        --stage1_ckpt /content/drive/MyDrive/qwenvl3/stage1_ckpt_v3 \\
        --eval_data_root /content/drive/MyDrive/qwenvl3/data/eval \\
        --stage2_data_root /content/drive/MyDrive/qwenvl3/data/stage2 \\
        --stage1_data_root /content/drive/MyDrive/qwenvl3/data/llava-pretrain

  快速 sanity（每任务 100 题，约 30 min）:
    python stage2/04_eval_stage2.py ... --n_refcoco 100 --n_pope 200 --n_vqav2 100

  只跑 grounding (天然 held-out 最严谨):
    python stage2/04_eval_stage2.py ... --skip pope vqav2 stage1_regression
"""
import argparse
import io
import json
import re
import sys
from pathlib import Path

import torch
from PIL import Image


# ============================================================================
# 路径设置 & helper imports
# ============================================================================

# stage1/_common.py: install_custom_projector
sys.path.insert(0, str(Path(__file__).parent.parent / "stage1"))
from _common import install_custom_projector  # noqa: E402

# stage2/_common2.py: ChatFormatter（用其 image token expansion 和 chat 模板逻辑）
sys.path.insert(0, str(Path(__file__).parent))


# ============================================================================
# 通用 helpers
# ============================================================================

def compute_num_image_tokens(config):
    vc = config.vision_config
    n = (vc.image_size // vc.patch_size) ** 2
    if config.vision_feature_select_strategy == "default":
        n -= 1
    return n


def parse_bbox(text):
    """从生成文本中提取 bbox。返回归一化 (x1,y1,x2,y2) 或 None。
    支持: <box>(x,y),(x,y)</box> / [x,y,x,y] / (x,y,x,y) / 0-1000 整数坐标
    """
    patterns = [
        r"<box>\(([\d.]+),([\d.]+)\),\(([\d.]+),([\d.]+)\)</box>",
        r"\[([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\]",
        r"\(([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            try:
                x1, y1, x2, y2 = map(float, m.groups())
                if max(x1, y1, x2, y2) > 1.5:
                    if max(x1, y1, x2, y2) <= 1000:
                        x1, y1, x2, y2 = x1/1000, y1/1000, x2/1000, y2/1000
                # 防止 x2 < x1 / y2 < y1
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                return (x1, y1, x2, y2)
            except Exception:
                pass
    return None


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    ab = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


def normalize_text(s):
    return re.sub(r"[^\w\s]", " ", str(s).lower()).strip()


def detect_yes_no(text):
    """从模型自由生成中提取 Yes/No 答案。返回 'yes'/'no'/'unknown'。"""
    t = normalize_text(text)
    if not t:
        return "unknown"
    # 取第一句
    first = re.split(r"[.!?\n]", t)[0]
    words = first.split()
    if not words:
        return "unknown"
    # 首词强匹配
    if words[0] in ("yes", "yeah", "yep", "y"):
        return "yes"
    if words[0] in ("no", "nope", "n"):
        return "no"
    # 短语兜底
    if "yes" in words[:3]:
        return "yes"
    if any(w in words[:5] for w in ("no", "not", "isn", "doesn", "aren", "cannot")):
        return "no"
    return "unknown"


def vqa_acc(pred, gt_answers):
    """VQAv2 标准 accuracy = min(matches / 3, 1.0)。

    匹配规则：normalize 后完全相等，或 GT 完整包含在 pred 内（应对 free-form 长答案）。
    """
    pred_norm = normalize_text(pred)
    if not pred_norm:
        return 0.0
    matches = 0
    for ans in gt_answers:
        ans_norm = normalize_text(ans)
        if not ans_norm:
            continue
        if ans_norm == pred_norm:
            matches += 1
        elif f" {ans_norm} " in f" {pred_norm} ":
            matches += 0.5  # 部分匹配权重减半
    return min(matches / 3.0, 1.0)


def detect_repetition(text, min_run=4):
    """检测连续相同词重复（>=min_run 次）。"""
    words = text.lower().split()
    if not words:
        return False, 0
    max_run = cur = 1
    for i in range(1, len(words)):
        if words[i] == words[i-1]:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 1
    return max_run >= min_run, max_run


# ============================================================================
# 模型加载（base + projector + LoRA adapter，最后 merge）
# ============================================================================

def load_stage2_model(stage2_ckpt, stage1_ckpt, processor_dir=None,
                      merge_lora=True):
    """加载训完的 Stage 2 模型用于推理。

    返回 (model, tokenizer, image_processor)。
    """
    from peft import PeftModel
    from transformers import (
        AutoImageProcessor, AutoTokenizer, LlavaForConditionalGeneration,
    )

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
        print(f"[device] CUDA + bf16")
    else:
        device, dtype = "cpu", torch.float32
        print(f"[device] CPU + fp32（慢；建议 GPU runtime）")

    stage2_ckpt = Path(stage2_ckpt)
    stage1_ckpt = Path(stage1_ckpt)

    # 1. base model 来自 Stage 1（adapter_config 里的 base_model_name_or_path 在 Drive 上路径不可靠，直接用 stage1_ckpt）
    print(f"[load] base model from {stage1_ckpt}")
    model = LlavaForConditionalGeneration.from_pretrained(
        str(stage1_ckpt), torch_dtype=dtype,
    )

    # 2. 替换 projector 为 ProjectorWithNorm，并从 stage2_ckpt 加载训完的权重
    print(f"[load] custom projector from {stage2_ckpt}")
    install_custom_projector(model, init_dir=str(stage2_ckpt), dtype=dtype)

    # 3. 套 LoRA adapter
    print(f"[load] LoRA adapter from {stage2_ckpt}")
    model = PeftModel.from_pretrained(model, str(stage2_ckpt))

    # 4. Merge LoRA 到 base 里，推理更快（一次性、不可逆）
    if merge_lora:
        print(f"[load] merging LoRA into base...")
        model = model.merge_and_unload()

    model = model.to(device).eval()

    # 5. tokenizer + image_processor — 优先 processor_dir，fallback stage2 / stage1
    proc_candidates = [processor_dir, stage2_ckpt, stage1_ckpt]
    proc_dir = None
    for c in proc_candidates:
        if c and (Path(c) / "tokenizer_config.json").exists():
            proc_dir = c
            break
    if proc_dir is None:
        raise FileNotFoundError(
            f"找不到 tokenizer。试过 {proc_candidates}，"
            f"都没有 tokenizer_config.json"
        )
    print(f"[load] tokenizer + image_processor from {proc_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(proc_dir))
    image_processor = AutoImageProcessor.from_pretrained(str(proc_dir))

    return model, tokenizer, image_processor


# ============================================================================
# 推理（chat template — Stage 2 训过的格式）
# ============================================================================

class ChatPromptBuilder:
    """构造 Qwen2.5 chat 推理 prompt 并展开 <image> token。

    单 turn 推理，输入：
        <|im_start|>user
        <image>×729 + question<|im_end|>
        <|im_start|>assistant\n
    （模型从 \\n 后开始生成，遇到 <|im_end|> 停止）
    """
    def __init__(self, tokenizer, num_image_tokens):
        self.tokenizer = tokenizer
        self.num_image_tokens = num_image_tokens
        self.image_token_id = tokenizer.convert_tokens_to_ids("<image>")
        self.user_prefix = tokenizer("<|im_start|>user\n", add_special_tokens=False).input_ids
        self.end_marker = tokenizer("<|im_end|>\n", add_special_tokens=False).input_ids
        self.asst_prefix = tokenizer("<|im_start|>assistant\n", add_special_tokens=False).input_ids
        self.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    def build(self, question_with_image_placeholder):
        """question 文本里应包含一个 <image> 占位符。"""
        text_ids = self.tokenizer(
            question_with_image_placeholder, add_special_tokens=False,
        ).input_ids
        # 展开 <image> token
        expanded = []
        for tok in text_ids:
            if tok == self.image_token_id:
                expanded.extend([self.image_token_id] * self.num_image_tokens)
            else:
                expanded.append(tok)
        prompt_ids = self.user_prefix + expanded + self.end_marker + self.asst_prefix
        return prompt_ids


@torch.inference_mode()
def chat_generate(model, image_processor, image, prompt_builder, question,
                  max_new_tokens=80):
    """用 chat template 生成回答。

    question: str。会自动加 <image>\\n 前缀。
    """
    pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(
        model.device, dtype=model.dtype,
    )
    full_q = f"<image>\n{question}" if question else "<image>"
    prompt_ids = prompt_builder.build(full_q)
    input_ids = torch.tensor([prompt_ids]).to(model.device)

    eos_id = prompt_builder.tokenizer.eos_token_id
    stop_ids = [prompt_builder.im_end_id]
    if eos_id and eos_id != prompt_builder.im_end_id:
        stop_ids.append(eos_id)

    out = model.generate(
        input_ids=input_ids,
        pixel_values=pixel_values,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=stop_ids,
        pad_token_id=prompt_builder.tokenizer.pad_token_id or eos_id,
    )
    gen_ids = out[0][input_ids.size(1):]
    return prompt_builder.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


# ============================================================================
# Task 1: RefCOCO val/testA/testB（天然 held-out，IoU@0.5）
# ============================================================================

def eval_refcoco_split(model, image_processor, prompt_builder,
                       data_root, split_name, n_samples, out_path):
    """RefCOCO 在指定 split 上做 grounding 评测。

    问题模板（与训练一致）：
        Provide the bounding box coordinates of <ref expression>.

    指标：Acc@0.5 / Acc@0.7 / mean IoU / parse_rate
    """
    from datasets import load_dataset

    rc_dir = Path(data_root) / "refcoco"
    if not rc_dir.exists():
        print(f"[skip] RefCOCO ({split_name}): {rc_dir} 不存在")
        return None

    print(f"\n[task] RefCOCO {split_name}  (n_target={n_samples})")
    try:
        ds = load_dataset(str(rc_dir), split=split_name, trust_remote_code=True)
    except Exception as e:
        print(f"  [skip] split={split_name} 加载失败: {e}")
        return None
    print(f"  数据加载成功: {len(ds)} 条，字段 {list(ds.features.keys())[:8]}")

    n = min(n_samples, len(ds))
    results = []
    ious = []
    parseable = 0

    for i in range(n):
        s = ds[i]
        # 取图
        img_field = s.get("image")
        if isinstance(img_field, dict) and "bytes" in img_field:
            image = Image.open(io.BytesIO(img_field["bytes"])).convert("RGB")
        elif hasattr(img_field, "convert"):
            image = img_field.convert("RGB")
        else:
            continue
        iw, ih = image.size

        # 取 ref expression（lmms-lab/RefCOCO 用 'answer' 字段存 ref）
        ref = None
        for key in ["answer", "sentences", "sentence", "ref",
                    "referring_expression", "caption"]:
            v = s.get(key)
            if isinstance(v, list) and v:
                v = v[0]
                if isinstance(v, dict):
                    v = v.get("sent") or v.get("raw") or v.get("text")
            if isinstance(v, str) and v.strip():
                ref = v.strip()
                break
        if not ref:
            continue

        # 取 GT bbox（COCO 的 [x, y, w, h] 像素坐标）
        bbox = s.get("bbox") or s.get("box")
        if not bbox or len(bbox) != 4:
            continue
        if max(bbox) > 1.5:
            x, y, w, h = bbox
            gt_box = (x/iw, y/ih, (x+w)/iw, (y+h)/ih)
        else:
            gt_box = tuple(bbox)

        # 推理
        question = f"Provide the bounding box coordinates of {ref}."
        gen = chat_generate(model, image_processor, image, prompt_builder,
                            question, max_new_tokens=40)
        pred_box = parse_bbox(gen)
        sample_iou = 0.0
        if pred_box is not None:
            parseable += 1
            sample_iou = iou(pred_box, gt_box)
        ious.append(sample_iou)
        results.append({
            "idx": i+1, "ref": ref,
            "gt_bbox": [round(c, 4) for c in gt_box],
            "pred_bbox": [round(c, 4) for c in pred_box] if pred_box else None,
            "iou": round(sample_iou, 4), "generated": gen,
        })
        if i < 3 or (i+1) % 200 == 0 or i == n-1:
            print(f"  [{i+1}/{n}] {ref[:40]!r}  IoU={sample_iou:.3f}  gen={gen[:60]!r}")

    if not ious:
        print(f"  [skip] 0 个有效样本")
        return None

    n_eval = len(ious)
    acc_05 = sum(1 for x in ious if x >= 0.5) / n_eval
    acc_07 = sum(1 for x in ious if x >= 0.7) / n_eval
    mean_iou = sum(ious) / n_eval
    parse_rate = parseable / n_eval

    print(f"  [{split_name}]  Acc@0.5={acc_05:.2%}  Acc@0.7={acc_07:.2%}  "
          f"mIoU={mean_iou:.3f}  parse_rate={parse_rate:.2%}")

    summary = {
        "task": f"refcoco_{split_name}",
        "n_evaluated": n_eval,
        "metrics": {
            "acc@0.5": acc_05, "acc@0.7": acc_07,
            "mean_iou": mean_iou, "parse_rate": parse_rate,
        },
        "samples": results,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    return summary


# ============================================================================
# Task 2: POPE（幻觉测试）
# ============================================================================

def eval_pope(model, image_processor, prompt_builder,
              eval_root, n_samples, out_path):
    """POPE — 是非题幻觉测试。

    问题示例: "Is there a dog in the image?" → Yes / No
    指标: Accuracy / F1 / Yes-ratio
    Yes-ratio 偏离 50% 太多说明模型有 yes-bias 或 no-bias。
    """
    from datasets import load_dataset

    pope_dir = Path(eval_root) / "pope"
    if not pope_dir.exists():
        print(f"[skip] POPE: {pope_dir} 不存在")
        return None

    print(f"\n[task] POPE  (n_target={n_samples})")
    # POPE 通常只有 test split
    ds = None
    for split in ["test", "validation", "train"]:
        try:
            ds = load_dataset(str(pope_dir), split=split, trust_remote_code=True)
            break
        except Exception:
            continue
    if ds is None:
        try:
            ds_dict = load_dataset(str(pope_dir), trust_remote_code=True)
            ds = ds_dict[list(ds_dict.keys())[0]]
        except Exception as e:
            print(f"  [skip] POPE 加载失败: {e}")
            return None
    print(f"  数据加载: {len(ds)} 条，字段 {list(ds.features.keys())[:10]}")

    n = min(n_samples, len(ds))
    tp = fp = tn = fn = 0
    yes_count = unknown_count = 0
    results = []
    cat_stats = {}  # category 维度的细分（POPE 有 random/popular/adversarial）

    for i in range(n):
        s = ds[i]
        img_field = s.get("image")
        if isinstance(img_field, dict) and "bytes" in img_field:
            image = Image.open(io.BytesIO(img_field["bytes"])).convert("RGB")
        elif hasattr(img_field, "convert"):
            image = img_field.convert("RGB")
        else:
            continue
        question = s.get("question") or s.get("text") or ""
        gt_label = (s.get("answer") or s.get("label") or "").strip().lower()
        if gt_label not in ("yes", "no"):
            continue
        category = s.get("category") or s.get("subset") or "all"

        gen = chat_generate(model, image_processor, image, prompt_builder,
                            question, max_new_tokens=10)
        pred_label = detect_yes_no(gen)

        cs = cat_stats.setdefault(category, {"tp": 0, "fp": 0, "tn": 0, "fn": 0,
                                             "yes": 0, "n": 0, "unk": 0})
        cs["n"] += 1
        if pred_label == "yes":
            yes_count += 1
            cs["yes"] += 1
            if gt_label == "yes":
                tp += 1; cs["tp"] += 1
            else:
                fp += 1; cs["fp"] += 1
        elif pred_label == "no":
            if gt_label == "no":
                tn += 1; cs["tn"] += 1
            else:
                fn += 1; cs["fn"] += 1
        else:
            unknown_count += 1
            cs["unk"] += 1

        results.append({
            "idx": i+1, "question": question, "category": category,
            "gt": gt_label, "pred": pred_label, "generated": gen,
        })
        if i < 3 or (i+1) % 500 == 0 or i == n-1:
            print(f"  [{i+1}/{n}]  {category} Q={question[:50]!r}  GT={gt_label} → PRED={pred_label}")

    total = tp + fp + tn + fn
    if total == 0:
        return None
    acc = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    yes_ratio = yes_count / (total + unknown_count)

    # 各 category 拆分
    cat_metrics = {}
    for cat, cs in cat_stats.items():
        ct = cs["tp"] + cs["fp"] + cs["tn"] + cs["fn"]
        if ct == 0:
            continue
        ca = (cs["tp"] + cs["tn"]) / ct
        cp = cs["tp"] / (cs["tp"] + cs["fp"]) if (cs["tp"] + cs["fp"]) else 0
        cr = cs["tp"] / (cs["tp"] + cs["fn"]) if (cs["tp"] + cs["fn"]) else 0
        cf1 = 2 * cp * cr / (cp + cr) if (cp + cr) else 0
        cat_metrics[cat] = {
            "acc": ca, "f1": cf1, "yes_ratio": cs["yes"] / cs["n"],
            "n": cs["n"], "unknown": cs["unk"],
        }

    print(f"\n  [POPE] Acc={acc:.2%}  F1={f1:.3f}  Yes-ratio={yes_ratio:.2%}  "
          f"unknown={unknown_count}/{n}")
    for cat, m in cat_metrics.items():
        print(f"    [{cat:12s}] Acc={m['acc']:.2%} F1={m['f1']:.3f} "
              f"Yes={m['yes_ratio']:.2%}  (n={m['n']})")

    summary = {
        "task": "pope",
        "n_evaluated": total,
        "metrics": {
            "accuracy": acc, "precision": precision, "recall": recall,
            "f1": f1, "yes_ratio": yes_ratio,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "unknown": unknown_count,
        },
        "by_category": cat_metrics,
        "samples": results,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


# ============================================================================
# Task 3: VQAv2 子集
# ============================================================================

def eval_vqav2(model, image_processor, prompt_builder,
               eval_root, n_samples, out_path):
    """VQAv2 val 子集 — 标准 VQA accuracy（多答案投票）。"""
    from datasets import load_dataset

    vqa_dir = Path(eval_root) / "vqav2"
    if not vqa_dir.exists():
        print(f"[skip] VQAv2: {vqa_dir} 不存在")
        return None

    print(f"\n[task] VQAv2  (n_target={n_samples})")
    ds = None
    for split in ["validation", "val", "test", "train"]:
        try:
            ds = load_dataset(str(vqa_dir), split=split, trust_remote_code=True)
            break
        except Exception:
            continue
    if ds is None:
        print(f"  [skip] VQAv2 加载失败")
        return None
    print(f"  数据加载: {len(ds)} 条，字段 {list(ds.features.keys())[:10]}")

    n = min(n_samples, len(ds))
    accs = []
    results = []
    for i in range(n):
        s = ds[i]
        img_field = s.get("image")
        if isinstance(img_field, dict) and "bytes" in img_field:
            image = Image.open(io.BytesIO(img_field["bytes"])).convert("RGB")
        elif hasattr(img_field, "convert"):
            image = img_field.convert("RGB")
        else:
            continue
        question = s.get("question") or ""

        # GT answers 可能是 list[str] 或 list[{answer: str}]
        answers = s.get("answers") or s.get("answer") or []
        if isinstance(answers, list):
            gt_answers = []
            for a in answers:
                if isinstance(a, str):
                    gt_answers.append(a)
                elif isinstance(a, dict):
                    gt_answers.append(a.get("answer") or a.get("text") or "")
        else:
            gt_answers = [str(answers)]
        gt_answers = [a for a in gt_answers if a]
        if not gt_answers:
            continue

        gen = chat_generate(model, image_processor, image, prompt_builder,
                            question, max_new_tokens=20)
        a = vqa_acc(gen, gt_answers)
        accs.append(a)
        results.append({
            "idx": i+1, "question": question,
            "gt_answers": gt_answers[:5], "generated": gen,
            "vqa_acc": round(a, 3),
        })
        if i < 3 or (i+1) % 200 == 0 or i == n-1:
            print(f"  [{i+1}/{n}] Q={question[:50]!r}")
            print(f"           GT={gt_answers[:3]}  GEN={gen[:40]!r}  acc={a:.2f}")

    if not accs:
        return None
    avg = sum(accs) / len(accs)
    print(f"  [VQAv2]  avg accuracy = {avg:.2%}  (n={len(accs)})")

    summary = {
        "task": "vqav2",
        "n_evaluated": len(accs),
        "metrics": {"accuracy": avg},
        "samples": results,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


# ============================================================================
# Task 4: Stage 1 holdout 回归测试（防灾难性遗忘）
# ============================================================================

def eval_stage1_regression(model, image_processor, prompt_builder,
                           holdout_json, image_root_or_zip,
                           out_path):
    """在 Stage 1 那 20 张 holdout 图上跑 caption，看 Stage 2 有没有把 caption 能力训坏。

    用 chat template + "Describe this image briefly." 让模型生成。
    主要看：avg_length、repetition_rate（token 死循环率），跟 Stage 1 的 11500 ckpt 对比。
    """
    holdout_path = Path(holdout_json)
    if not holdout_path.exists():
        print(f"[skip] Stage 1 holdout: {holdout_path} 不存在")
        return None

    # 选择 image loader：目录 / zip 两种模式
    image_root = Path(image_root_or_zip)
    if image_root.is_file() and image_root.suffix == ".zip":
        import zipfile
        zf = zipfile.ZipFile(image_root)
        def loader(rel):
            with zf.open(rel) as f:
                return Image.open(io.BytesIO(f.read())).convert("RGB")
    else:
        def loader(rel):
            return Image.open(image_root / rel).convert("RGB")

    print(f"\n[task] Stage 1 holdout 回归")
    with open(holdout_path) as f:
        holdout = json.load(f)
    print(f"  载入 {len(holdout)} 条 holdout")

    results = []
    lengths = []
    rep_count = 0
    for i, s in enumerate(holdout):
        try:
            image = loader(s["image"])
        except Exception as e:
            print(f"  [{i+1}] skip: {e}")
            continue
        gt = s["conversations"][1]["value"]
        gen = chat_generate(model, image_processor, image, prompt_builder,
                            "Describe this image briefly.", max_new_tokens=80)
        words = gen.split()
        lengths.append(len(words))
        is_rep, max_run = detect_repetition(gen)
        if is_rep:
            rep_count += 1
        results.append({
            "idx": i+1, "image": s["image"],
            "gt": gt, "generated": gen,
            "gen_length": len(words), "max_run": max_run, "repetition": is_rep,
        })
        print(f"  [{i+1}/{len(holdout)}] {s['image']}")
        print(f"    GT:  {gt[:80]}")
        print(f"    GEN: {gen[:120]}")

    if not lengths:
        return None
    avg_len = sum(lengths) / len(lengths)
    rep_rate = rep_count / len(results)
    print(f"\n  [stage1 regression] avg_len={avg_len:.1f}  rep_rate={rep_rate:.2%}")

    summary = {
        "task": "stage1_regression",
        "n_evaluated": len(results),
        "metrics": {
            "avg_gen_length": avg_len,
            "repetition_rate": rep_rate,
        },
        "samples": results,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


# ============================================================================
# Main
# ============================================================================

def parse_args():
    ap = argparse.ArgumentParser()
    # 必需路径
    ap.add_argument("--stage2_ckpt", required=True,
                    help="Stage 2 训完 ckpt 目录（adapter + projector）")
    ap.add_argument("--stage1_ckpt", required=True,
                    help="Stage 1 base ckpt（提供完整 model.safetensors）")
    ap.add_argument("--processor_dir", default=None,
                    help="tokenizer 目录；不指定则按 stage2 → stage1 顺序找")
    ap.add_argument("--eval_data_root", required=True,
                    help="OOD 评测数据根目录（含 pope/、vqav2/）")
    ap.add_argument("--stage2_data_root", default=None,
                    help="Stage 2 训练数据根（含 refcoco/）")
    ap.add_argument("--stage1_data_root", default=None,
                    help="Stage 1 数据根；含 holdout_20.json + 图像")
    ap.add_argument("--stage1_images_zip", default=None,
                    help="Stage 1 图像 zip（如果 stage1_data_root 没解压图）")
    ap.add_argument("--out_dir", default="/content/drive/MyDrive/qwenvl3/eval_stage2")

    # 各任务样本数
    ap.add_argument("--n_refcoco", type=int, default=1000,
                    help="每个 RefCOCO split 评测样本数")
    ap.add_argument("--n_pope", type=int, default=3000)
    ap.add_argument("--n_vqav2", type=int, default=1000)

    # 跳过任务
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["refcoco", "pope", "vqav2", "stage1_regression"],
                    help="跳过指定评测")

    # 其他
    ap.add_argument("--no_merge_lora", action="store_true",
                    help="不 merge LoRA 到 base（推理慢一点，但占显存少）")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir) / Path(args.stage2_ckpt).name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[out_dir] {out_dir}\n")

    # ---- 加载模型 ----
    model, tokenizer, image_processor = load_stage2_model(
        args.stage2_ckpt, args.stage1_ckpt, args.processor_dir,
        merge_lora=not args.no_merge_lora,
    )
    num_image_tokens = compute_num_image_tokens(model.config)
    print(f"[model] num_image_tokens = {num_image_tokens}")
    print(f"[model] dtype = {model.dtype}, device = {model.device}\n")

    prompt_builder = ChatPromptBuilder(tokenizer, num_image_tokens)

    all_metrics = {}

    # ---- Task 1: RefCOCO val/testA/testB ----
    if "refcoco" not in args.skip and args.stage2_data_root:
        for split_name in ["val", "testA", "testB"]:
            r = eval_refcoco_split(
                model, image_processor, prompt_builder,
                args.stage2_data_root, split_name, args.n_refcoco,
                out_dir / f"refcoco_{split_name}.json",
            )
            if r:
                all_metrics[f"refcoco_{split_name}"] = r["metrics"]

    # ---- Task 2: POPE ----
    if "pope" not in args.skip:
        r = eval_pope(
            model, image_processor, prompt_builder,
            args.eval_data_root, args.n_pope, out_dir / "pope.json",
        )
        if r:
            all_metrics["pope"] = r["metrics"]

    # ---- Task 3: VQAv2 ----
    if "vqav2" not in args.skip:
        r = eval_vqav2(
            model, image_processor, prompt_builder,
            args.eval_data_root, args.n_vqav2, out_dir / "vqav2.json",
        )
        if r:
            all_metrics["vqav2"] = r["metrics"]

    # ---- Task 4: Stage 1 regression ----
    if "stage1_regression" not in args.skip:
        # 找 holdout_20.json
        candidates = []
        if args.stage1_data_root:
            candidates.append(Path(args.stage1_data_root) / "holdout_20.json")
        for c in candidates:
            if c.exists():
                holdout_json = c
                break
        else:
            holdout_json = None
            print(f"[skip] Stage 1 regression: 没找到 holdout_20.json")

        if holdout_json:
            # 选图像源
            if args.stage1_images_zip:
                img_src = args.stage1_images_zip
            elif args.stage1_data_root:
                img_src = args.stage1_data_root
            else:
                img_src = None
                print(f"[skip] Stage 1 regression: 没指定图像源")

            if img_src:
                r = eval_stage1_regression(
                    model, image_processor, prompt_builder,
                    holdout_json, img_src,
                    out_dir / "stage1_regression.json",
                )
                if r:
                    all_metrics["stage1_regression"] = r["metrics"]

    # ---- 总结 ----
    summary = {
        "stage2_ckpt": str(args.stage2_ckpt),
        "stage1_ckpt": str(args.stage1_ckpt),
        "results": all_metrics,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("=== Stage 2 OOD 评测完成 ===")
    print(f"详细结果: {out_dir}/")
    print(f"\n各任务指标:")
    for task, metrics in all_metrics.items():
        print(f"\n  [{task}]")
        for k, v in metrics.items():
            if isinstance(v, float):
                if "ratio" in k or "acc" in k or "rate" in k or "@" in k:
                    print(f"    {k}: {v:.2%}")
                else:
                    print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")

    print(f"\n业界参考（你应该期待的数字）:")
    print(f"  RefCOCO val Acc@0.5:  ~40-55%   (LLaVA-1.5-7B ~30%, Qwen-VL-7B ~88%)")
    print(f"  POPE F1:              ~70-80%   (LLaVA-1.5-7B ~86%)")
    print(f"  VQAv2 acc:            ~55-65%   (LLaVA-1.5-7B 78.5%)")
    print(f"  stage1 rep_rate:      <15%      (Stage 1 ckpt-11500 ≈ 10%)")


if __name__ == "__main__":
    main()
