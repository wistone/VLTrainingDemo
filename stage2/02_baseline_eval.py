"""Stage 2 训练前 baseline 评估 — 覆盖所有任务类型，支持两种 prompt 模式。

目的：训练前后对比，量化 Stage 2 的提升。

5 个 dataset 中 COCO 是图源（不单独评估），剩 4 个任务：
  1. LLaVA-Instruct VQA  — 多轮 VQA + 推理
  2. TextVQA              — OCR 类问答
  3. RefCOCO              — Grounding（bbox 输出）
  4. ShareGPT4V           — 长 caption

每任务 30–50 样本，结果存 json，留作训练后 A/B 对比。

== 两种 prompt 模式 ==

  --prompt_mode caption_only (默认)
    只喂 <image> × 729，不带 question 文本。让模型按 Stage 1 训练方式生成 caption，
    再跟各任务 GT 做关键词/子串/长度对比。**适合 Stage 1 baseline**——因为 Stage 1
    模型从没见过 question 文本，with_question 模式会全部生成空字符串。
    对 RefCOCO 这种必须看到 bbox 格式才能评的任务，结果仍然是 0（合理）。
    输出文件后缀: _caption

  --prompt_mode with_question
    喂 <image> × 729 + question 文本，期望模型按 Q&A 回答。**Stage 2 训完后用**——
    模型学会 chat format 后能针对性回答而不是 caption。
    输出文件后缀: _chat

== 用法 ==

  Stage 1 baseline (caption-only, 默认):
    python stage2/02_baseline_eval.py \\
        --ckpt_dir /content/drive/MyDrive/qwenvl3/stage1_ckpt_v3/checkpoint-XXXX \\
        --processor_dir /content/drive/MyDrive/qwenvl3/stage1_init \\
        --stage2_data_root /content/drive/MyDrive/qwenvl3/data/stage2 \\
        --n_per_task 40

  Stage 2 后正式 eval (with_question):
    python stage2/02_baseline_eval.py ... --prompt_mode with_question
"""
import argparse
import io
import json
import random
import re
import sys
import zipfile
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    LlavaForConditionalGeneration,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "stage1"))
from _common import install_custom_projector  # noqa: E402

IGNORE_INDEX = -100


def compute_num_image_tokens(config):
    vc = config.vision_config
    n = (vc.image_size // vc.patch_size) ** 2
    if config.vision_feature_select_strategy == "default":
        n -= 1
    return n


# ============================================================================
# 模型加载
# ============================================================================

def load_model(ckpt_dir, processor_dir):
    if torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
        print(f"[device] CUDA, bf16")
    else:
        device, dtype = "cpu", torch.float32
        print(f"[device] CPU, fp32（慢；建议 GPU runtime）")

    print(f"[load] model from {ckpt_dir}")
    model = LlavaForConditionalGeneration.from_pretrained(ckpt_dir, torch_dtype=dtype)
    install_custom_projector(model, init_dir=ckpt_dir, dtype=dtype)
    model = model.to(device).eval()

    proc_dir = processor_dir or ckpt_dir
    print(f"[load] tokenizer + image_processor from {proc_dir}")
    tokenizer = AutoTokenizer.from_pretrained(proc_dir)
    image_processor = AutoImageProcessor.from_pretrained(proc_dir)
    return model, tokenizer, image_processor


# ============================================================================
# 图像加载工具
# ============================================================================

class CocoZipLoader:
    """从 COCO train2017.zip 直读图。"""
    def __init__(self, zip_path):
        self.zip = zipfile.ZipFile(zip_path)
        self.names = set(self.zip.namelist())

    def open(self, image_filename):
        full_path = f"train2017/{image_filename}"
        if full_path not in self.names:
            raise FileNotFoundError(f"{full_path} not in COCO zip")
        with self.zip.open(full_path) as f:
            return Image.open(io.BytesIO(f.read())).convert("RGB")

    def close(self):
        try:
            self.zip.close()
        except Exception:
            pass


# ============================================================================
# 推理（极简模板：N×<image> + question + 生成）
# ============================================================================

@torch.inference_mode()
def generate_answer(model, tokenizer, image_processor, image, question,
                    num_image_tokens, max_new_tokens=80):
    pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(
        model.device, dtype=model.dtype
    )
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    prompt_ids = [image_token_id] * num_image_tokens
    if question:
        prompt_ids = prompt_ids + tokenizer(
            question, add_special_tokens=False
        ).input_ids
    input_ids = torch.tensor([prompt_ids]).to(model.device)
    out = model.generate(
        input_ids=input_ids,
        pixel_values=pixel_values,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
    )
    return tokenizer.decode(
        out[0][input_ids.size(1):], skip_special_tokens=True
    ).strip()


def prompt_for_mode(question, prompt_mode):
    """根据 prompt_mode 决定喂给模型的文本。

    caption_only: 只喂 <image> × N，让模型按 Stage 1 训练方式生成 caption。
                  适合 Stage 1 baseline，因为模型只见过这种格式。
    with_question: 喂 <image> × N + question，期望模型按 Q&A 方式回答。
                   适合 Stage 2 训完后用，模型已学会 chat format。
    """
    if prompt_mode == "caption_only":
        return ""
    return question


# ============================================================================
# 通用 metric helpers
# ============================================================================

def normalize(s):
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower())


def keyword_match_rate(gt, gen, min_kw_len=4):
    gt_words = set(w for w in normalize(gt).split() if len(w) >= min_kw_len)
    if not gt_words:
        return None
    gen_words = set(normalize(gen).split())
    return len(gt_words & gen_words) / len(gt_words)


def substring_match(gt, gen):
    return normalize(gt).strip() in normalize(gen)


def detect_repetition(text, min_run=4):
    """检测连续重复词（>=min_run 次）"""
    words = text.lower().split()
    max_run = 1
    cur_run = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 1
    return max_run >= min_run, max_run


def parse_bbox(text):
    """从生成文本中提取 bbox。
    支持几种常见格式：
      <box>(x1,y1),(x2,y2)</box>
      [x1,y1,x2,y2]
      (x1,y1,x2,y2)
      [0.123, 0.456, 0.789, 0.901]
    返回 (x1, y1, x2, y2) 元组（归一化坐标 0-1），或 None。
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
                # 归一化到 0-1（如果 > 1 则按 1000 归一化，常见格式）
                if max(x1, y1, x2, y2) > 1.5:
                    if max(x1, y1, x2, y2) <= 1000:
                        x1, y1, x2, y2 = x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000
                    else:
                        # pixel coords，无法归一化，返回原值
                        pass
                return (x1, y1, x2, y2)
            except Exception:
                pass
    return None


def iou(a, b):
    """两个 bbox 的 IoU，输入为 (x1, y1, x2, y2)。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    aa, ab = max(0, ax2 - ax1) * max(0, ay2 - ay1), max(0, bx2 - bx1) * max(0, by2 - by1)
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


# ============================================================================
# 任务 1：LLaVA-Instruct VQA
# ============================================================================

def eval_llava_instruct(model, tokenizer, image_processor, data_root, n_samples,
                        num_image_tokens, out_path, coco_loader, image_dir,
                        prompt_mode="caption_only"):
    json_path = Path(data_root) / "llava_instruct" / "llava_instruct_150k.json"
    if not json_path.exists():
        print(f"[skip] LLaVA-Instruct: {json_path} 不存在")
        return None
    if coco_loader is None:
        print(f"[skip] LLaVA-Instruct: COCO zip 不可用")
        return None

    print(f"\n[task] LLaVA-Instruct VQA (n={n_samples})")
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    with open(json_path) as f:
        all_data = json.load(f)
    random.seed(42)
    samples = random.sample(all_data, n_samples)

    results = []
    rates = []
    for i, s in enumerate(samples):
        try:
            image = coco_loader.open(s["image"])
        except FileNotFoundError as e:
            print(f"  [{i+1}/{n_samples}] skip (no image)")
            continue
        img_save_path = image_dir / f"{i+1:03d}_{s['image']}"
        image.save(img_save_path)
        question = s["conversations"][0]["value"].replace("<image>", "").strip()
        gt_answer = s["conversations"][1]["value"]
        gen = generate_answer(model, tokenizer, image_processor, image,
                              prompt_for_mode(question, prompt_mode),
                              num_image_tokens)
        rate = keyword_match_rate(gt_answer, gen)
        if rate is not None:
            rates.append(rate)
        results.append({
            "idx": i + 1, "image": s["image"],
            "image_saved_path": str(img_save_path),
            "question": question[:200], "gt_answer": gt_answer,
            "generated": gen, "keyword_match_rate": rate,
        })
        if i < 3 or i == n_samples - 1:
            print(f"  [{i+1}/{n_samples}] {s['image']}")
            print(f"    Q:   {question[:80]}")
            print(f"    GT:  {gt_answer[:80]}")
            print(f"    GEN: {gen[:80]}")
            print(f"    kw match: {rate:.2f}" if rate is not None else "    (no kw)")

    avg = sum(rates) / len(rates) if rates else 0.0
    print(f"  [task=llava_instruct] avg keyword match: {avg:.3f}  (n={len(rates)})")
    summary = {
        "task": "llava_instruct",
        "n_total": len(samples), "n_evaluated": len(rates),
        "metrics": {"avg_keyword_match_rate": avg},
        "samples": results,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


# ============================================================================
# 任务 2：TextVQA
# ============================================================================

def eval_textvqa(model, tokenizer, image_processor, data_root, n_samples,
                  num_image_tokens, out_path, image_dir,
                  prompt_mode="caption_only"):
    """TextVQA — 用 datasets 库读 parquet。如果 repo 字段不一致会自动跳过。"""
    tv_dir = Path(data_root) / "textvqa"
    if not tv_dir.exists() or not any(tv_dir.iterdir()):
        print(f"\n[skip] TextVQA: 数据未下载 ({tv_dir})")
        return None

    print(f"\n[task] TextVQA (n={n_samples})")
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    try:
        from datasets import load_dataset
        ds = load_dataset(str(tv_dir), split="validation", trust_remote_code=True)
    except Exception:
        try:
            ds = load_dataset(str(tv_dir), split="train", trust_remote_code=True)
        except Exception as e:
            print(f"  [skip] TextVQA: 加载失败 ({e})")
            return None

    print(f"  数据加载成功，总数 {len(ds)}, 字段 {list(ds.features.keys())[:8]}")
    random.seed(42)
    indices = random.sample(range(len(ds)), min(n_samples, len(ds)))

    results = []
    matches = []
    for i, idx in enumerate(indices):
        s = ds[idx]
        # 字段宽容匹配
        try:
            image = s.get("image")
            if isinstance(image, dict) and "bytes" in image:
                image = Image.open(io.BytesIO(image["bytes"])).convert("RGB")
            elif hasattr(image, "convert"):
                image = image.convert("RGB")
            else:
                continue
            question = s.get("question") or s.get("query") or ""
            answers = s.get("answers") or s.get("answer") or []
            if isinstance(answers, list) and answers:
                gt_answer = answers[0] if isinstance(answers[0], str) else str(answers[0])
            else:
                gt_answer = str(answers)
        except Exception as e:
            continue

        img_save_path = image_dir / f"{i+1:03d}.jpg"
        image.save(img_save_path)
        gen = generate_answer(model, tokenizer, image_processor, image,
                              prompt_for_mode(question, prompt_mode),
                              num_image_tokens,
                              max_new_tokens=120 if prompt_mode == "caption_only" else 40)
        match = substring_match(gt_answer, gen)
        matches.append(match)
        results.append({
            "idx": i + 1, "image_saved_path": str(img_save_path),
            "question": question, "gt_answer": gt_answer,
            "generated": gen, "substring_match": match,
        })
        if i < 3 or i == len(indices) - 1:
            print(f"  [{i+1}/{len(indices)}]  Q: {question[:60]}")
            print(f"    GT: {gt_answer}  GEN: {gen[:60]}  match: {match}")

    rate = sum(matches) / len(matches) if matches else 0.0
    print(f"  [task=textvqa] substring match rate: {rate:.3f}  (n={len(matches)})")
    summary = {
        "task": "textvqa",
        "n_total": len(indices), "n_evaluated": len(matches),
        "metrics": {"substring_match_rate": rate},
        "samples": results,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


# ============================================================================
# 任务 3：RefCOCO 定位
# ============================================================================

def eval_refcoco(model, tokenizer, image_processor, data_root, n_samples,
                 num_image_tokens, out_path, coco_loader, image_dir,
                 prompt_mode="caption_only"):
    """RefCOCO grounding — 输入 ref expression，期望模型输出 bbox。

    Stage 1 没见过 bbox 格式，预期：
      - bbox 格式合规率 < 5%
      - mean IoU ≈ 0
    """
    rc_dir = Path(data_root) / "refcoco"
    if not rc_dir.exists() or not any(rc_dir.iterdir()):
        print(f"\n[skip] RefCOCO: 数据未下载 ({rc_dir})")
        return None
    if coco_loader is None:
        print(f"[skip] RefCOCO: COCO zip 不可用")
        return None

    print(f"\n[task] RefCOCO grounding (n={n_samples})")
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    try:
        from datasets import load_dataset
        ds = None
        for split in ["validation", "val", "test", "train"]:
            try:
                ds = load_dataset(str(rc_dir), split=split, trust_remote_code=True)
                break
            except Exception:
                continue
        if ds is None:
            print(f"  [skip] RefCOCO: 找不到合适的 split")
            return None
    except Exception as e:
        print(f"  [skip] RefCOCO: load 失败 {e}")
        return None

    print(f"  数据加载，总数 {len(ds)}, 字段 {list(ds.features.keys())[:10]}")
    random.seed(42)
    indices = random.sample(range(len(ds)), min(n_samples, len(ds)))

    results = []
    parseable = 0
    ious = []
    for i, idx in enumerate(indices):
        s = ds[idx]
        # 字段尝试：image (str/dict/PIL), sentences/sentence/question, bbox
        try:
            img = s.get("image")
            if isinstance(img, dict) and "bytes" in img:
                image = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
            elif hasattr(img, "convert"):
                image = img.convert("RGB")
            elif isinstance(img, str):
                # 假设是 COCO 文件名
                image = coco_loader.open(img)
            else:
                continue

            # ref expression
            ref = (s.get("sentences") or s.get("sentence") or
                   s.get("question") or s.get("ref"))
            if isinstance(ref, list) and ref:
                ref = ref[0] if isinstance(ref[0], str) else (ref[0].get("sent") or ref[0].get("raw") or str(ref[0]))
            ref = str(ref)

            # ground truth bbox
            bbox = s.get("bbox") or s.get("box") or s.get("answer")
            if not bbox or len(bbox) != 4:
                continue
            # 假设 bbox 是 [x, y, w, h] (COCO 格式)；有些是 [x1, y1, x2, y2]
            # 我们试着归一化（用图像尺寸）
            iw, ih = image.size
            if max(bbox) > 1.5:  # 像素坐标
                # 假设是 xywh
                x, y, w, h = bbox
                gt_box_norm = (x / iw, y / ih, (x + w) / iw, (y + h) / ih)
            else:
                gt_box_norm = tuple(bbox)
        except Exception:
            continue

        img_save_path = image_dir / f"{i+1:03d}.jpg"
        image.save(img_save_path)
        question = f"Where is {ref}? Answer with a bounding box."
        gen = generate_answer(model, tokenizer, image_processor, image,
                              prompt_for_mode(question, prompt_mode),
                              num_image_tokens,
                              max_new_tokens=120 if prompt_mode == "caption_only" else 40)
        pred_box = parse_bbox(gen)
        sample_iou = 0.0
        if pred_box is not None:
            parseable += 1
            sample_iou = iou(pred_box, gt_box_norm)
        ious.append(sample_iou)

        results.append({
            "idx": i + 1, "image_saved_path": str(img_save_path),
            "ref": ref, "gt_bbox": gt_box_norm,
            "generated": gen, "pred_bbox": pred_box, "iou": sample_iou,
        })
        if i < 3 or i == len(indices) - 1:
            print(f"  [{i+1}/{len(indices)}] ref={ref[:50]!r}")
            print(f"    GT bbox: {gt_box_norm}  GEN: {gen[:80]}  IoU: {sample_iou:.2f}")

    parse_rate = parseable / len(indices) if indices else 0
    mean_iou = sum(ious) / len(ious) if ious else 0
    print(f"  [task=refcoco] bbox 解析率: {parse_rate:.2%}  mean IoU: {mean_iou:.3f}")
    summary = {
        "task": "refcoco",
        "n_total": len(indices), "n_evaluated": len(ious),
        "metrics": {"bbox_parse_rate": parse_rate, "mean_iou": mean_iou},
        "samples": results,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    return summary


# ============================================================================
# 任务 4：ShareGPT4V 长 caption
# ============================================================================

def eval_sharegpt4v(model, tokenizer, image_processor, data_root, n_samples,
                    num_image_tokens, out_path, coco_loader, image_dir,
                    prompt_mode="caption_only"):
    sg_dir = Path(data_root) / "sharegpt4v"
    if not sg_dir.exists() or not any(sg_dir.iterdir()):
        print(f"\n[skip] ShareGPT4V: 数据未下载 ({sg_dir})")
        return None
    if coco_loader is None:
        print(f"  [warn] COCO zip 不可用，会跳过非 COCO 图样本")

    # ShareGPT4V json 格式：list of {id, image, conversations}
    json_files = sorted(sg_dir.rglob("*.json"))
    if not json_files:
        print(f"\n[skip] ShareGPT4V: 找不到 json")
        return None

    print(f"\n[task] ShareGPT4V long caption (n={n_samples})")
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    print(f"  尝试解析 {json_files[0].name}")

    try:
        with open(json_files[0]) as f:
            data = json.load(f)
        if not isinstance(data, list) or not data or "conversations" not in data[0]:
            print(f"  [skip] ShareGPT4V json 格式异常")
            return None
    except Exception as e:
        print(f"  [skip] {e}")
        return None

    print(f"  数据 {len(data)} 条；只取 image 在 COCO 里的样本")
    # 过滤：image 路径含 'coco' 或 'train2017'
    coco_samples = [s for s in data if "coco" in s.get("image", "").lower()
                    or "train2017" in s.get("image", "")]
    if not coco_samples:
        # 退而求其次：取所有
        coco_samples = data
        print(f"  [warn] 没有 COCO 样本，用全部 (可能有 image 找不到)")

    random.seed(42)
    samples = random.sample(coco_samples, min(n_samples, len(coco_samples)))

    results = []
    lengths = []
    rates = []
    repetition_count = 0
    for i, s in enumerate(samples):
        try:
            img_path = s["image"]
            # 提取 COCO 文件名
            fn = Path(img_path).name
            image = coco_loader.open(fn)
        except Exception as e:
            continue

        img_save_path = image_dir / f"{i+1:03d}_{fn}"
        image.save(img_save_path)
        question = "Describe this image in detail."
        gt_caption = s["conversations"][1]["value"]
        gen = generate_answer(model, tokenizer, image_processor, image,
                              prompt_for_mode(question, prompt_mode),
                              num_image_tokens, max_new_tokens=120)

        word_count = len(gen.split())
        lengths.append(word_count)
        rate = keyword_match_rate(gt_caption, gen)
        if rate is not None:
            rates.append(rate)
        is_rep, max_run = detect_repetition(gen)
        if is_rep:
            repetition_count += 1

        results.append({
            "idx": i + 1, "image": img_path,
            "image_saved_path": str(img_save_path),
            "gt_caption_preview": gt_caption[:200],
            "gt_length_words": len(gt_caption.split()),
            "generated": gen, "gen_length_words": word_count,
            "keyword_match_rate": rate, "repetition": is_rep, "max_run": max_run,
        })
        if i < 3 or i == len(samples) - 1:
            print(f"  [{i+1}/{len(samples)}] {fn}  len: gt={len(gt_caption.split())} gen={word_count}")
            print(f"    GEN: {gen[:120]}")

    avg_len = sum(lengths) / len(lengths) if lengths else 0
    avg_rate = sum(rates) / len(rates) if rates else 0
    rep_rate = repetition_count / len(results) if results else 0
    print(f"  [task=sharegpt4v] avg gen length: {avg_len:.1f} words")
    print(f"                    avg keyword match: {avg_rate:.3f}")
    print(f"                    repetition rate: {rep_rate:.2%}")
    summary = {
        "task": "sharegpt4v",
        "n_total": len(samples), "n_evaluated": len(results),
        "metrics": {
            "avg_gen_length_words": avg_len,
            "avg_keyword_match_rate": avg_rate,
            "repetition_rate": rep_rate,
        },
        "samples": results,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    return summary


def resolve_checkpoint(ckpt_arg):
    """检查 ckpt_dir 是否真的可用（model.safetensors 存在且 > 1GB）。
    如果不可用且路径形如 .../checkpoint-NNNN，自动 fallback 到同目录最新的可用 checkpoint。
    避免 Drive 同步滞后问题。
    """
    ckpt_path = Path(ckpt_arg)
    sft = ckpt_path / "model.safetensors"

    if sft.exists() and sft.stat().st_size > 1e9:
        return str(ckpt_path)

    # 不可用：找同目录最新可用 checkpoint
    if ckpt_path.parent.name and ckpt_path.name.startswith("checkpoint-"):
        all_ckpts = sorted(
            (p for p in ckpt_path.parent.glob("checkpoint-*") if p.is_dir()),
            key=lambda p: int(p.name.split("-")[1]),
            reverse=True,
        )
        for c in all_ckpts:
            sft_c = c / "model.safetensors"
            if sft_c.exists() and sft_c.stat().st_size > 1e9:
                print(f"[warn] {ckpt_path.name} 不可用（Drive 同步滞后？），")
                print(f"       自动 fallback 到 {c.name}")
                return str(c)

    raise FileNotFoundError(
        f"找不到可用 checkpoint。{ckpt_path} 下无完整 model.safetensors，"
        f"且同目录其他 checkpoint 也不可用。"
    )


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--processor_dir", default=None)
    ap.add_argument("--stage2_data_root", required=True)
    ap.add_argument("--out_dir", default="/content/drive/MyDrive/qwenvl3/eval_baseline")
    ap.add_argument("--n_per_task", type=int, default=40)
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["llava_instruct", "textvqa", "refcoco", "sharegpt4v"],
                    help="跳过指定任务")
    ap.add_argument("--no_auto_fallback", action="store_true",
                    help="禁用 ckpt-NNNN 不可用时 fallback 到上一个可用 checkpoint")
    ap.add_argument("--prompt_mode", default="caption_only",
                    choices=["caption_only", "with_question"],
                    help="caption_only (默认 / 适合 Stage 1 baseline): "
                         "只喂图，让模型按 Stage 1 训练方式生成 caption，再跟各任务 GT 做匹配。"
                         "with_question (Stage 2 后用): 喂图 + 完整问题文本，模型应按 Q&A 回答。")
    args = ap.parse_args()

    # 自动 resolve checkpoint
    if args.no_auto_fallback:
        ckpt_dir = args.ckpt_dir
    else:
        ckpt_dir = resolve_checkpoint(args.ckpt_dir)

    # 按 checkpoint 名建子目录,避免不同 ckpt 互相覆盖
    out_dir = Path(args.out_dir) / Path(ckpt_dir).name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[out_dir] {out_dir}")

    model, tokenizer, image_processor = load_model(ckpt_dir, args.processor_dir)
    num_image_tokens = compute_num_image_tokens(model.config)
    print(f"num_image_tokens = {num_image_tokens}\n")

    coco_zip = Path(args.stage2_data_root) / "coco" / "train2017.zip"
    coco_loader = None
    if coco_zip.exists() and coco_zip.stat().st_size > 1e9:
        coco_loader = CocoZipLoader(coco_zip)
        print(f"[init] CocoZipLoader OK ({coco_zip.stat().st_size / 1e9:.1f}GB)")
    else:
        print(f"[warn] COCO zip 不可用，依赖 COCO 的任务会被跳过")

    all_results = {}

    images_root = out_dir / "images"
    mode = args.prompt_mode
    # 文件名后缀，避免不同模式相互覆盖
    suffix = "_caption" if mode == "caption_only" else "_chat"
    print(f"[mode] {mode}  (输出文件后缀: {suffix})")

    if "llava_instruct" not in args.skip:
        all_results["llava_instruct"] = eval_llava_instruct(
            model, tokenizer, image_processor, args.stage2_data_root,
            args.n_per_task, num_image_tokens,
            out_dir / f"baseline_llava_vqa{suffix}.json", coco_loader,
            images_root / "llava_instruct",
            prompt_mode=mode,
        )

    if "textvqa" not in args.skip:
        all_results["textvqa"] = eval_textvqa(
            model, tokenizer, image_processor, args.stage2_data_root,
            args.n_per_task, num_image_tokens,
            out_dir / f"baseline_textvqa{suffix}.json",
            images_root / "textvqa",
            prompt_mode=mode,
        )

    if "refcoco" not in args.skip:
        all_results["refcoco"] = eval_refcoco(
            model, tokenizer, image_processor, args.stage2_data_root,
            args.n_per_task, num_image_tokens,
            out_dir / f"baseline_refcoco{suffix}.json", coco_loader,
            images_root / "refcoco",
            prompt_mode=mode,
        )

    if "sharegpt4v" not in args.skip:
        all_results["sharegpt4v"] = eval_sharegpt4v(
            model, tokenizer, image_processor, args.stage2_data_root,
            args.n_per_task, num_image_tokens,
            out_dir / f"baseline_sharegpt4v{suffix}.json", coco_loader,
            images_root / "sharegpt4v",
            prompt_mode=mode,
        )

    if coco_loader:
        coco_loader.close()

    # 总结
    summary = {
        "ckpt_dir": str(ckpt_dir),
        "n_per_task": args.n_per_task,
        "prompt_mode": mode,
        "tasks": {},
    }
    for task, res in all_results.items():
        if res:
            summary["tasks"][task] = res["metrics"]
        else:
            summary["tasks"][task] = "skipped/failed"
    with open(out_dir / f"baseline_summary{suffix}.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("=== Baseline 评估完成 ===")
    print(f"详细结果: {out_dir}/")
    print(f"\n各任务指标:")
    for task, metrics in summary["tasks"].items():
        print(f"  {task}: {metrics}")
    print(f"\n训完 Stage 2 后用同一 n_per_task=40 跑 04_eval_stage2.py 做 A/B 对比")


if __name__ == "__main__":
    main()
