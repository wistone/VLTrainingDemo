"""Rex-Omni RefCOCO 对比评测 — 跟我们 Stage 2 模型在同样的 RefCOCO val/testA/testB 上较量。

Rex-Omni 是 IDEA Research 的 grounding 专用 3B 模型 (https://github.com/IDEA-Research/Rex-Omni)，
只支持 detection / pointing / keypoint / OCR / GUI 类视觉感知任务，**不支持 free-form VQA 或 caption**。
所以这个脚本只跑 RefCOCO，POPE/VQAv2/NoCaps 留给 04_eval_stage2.py 评我们的模型。

输出 JSON 跟 04_eval_stage2.py 的 refcoco_*.json 完全同结构 → 后续可直接 diff 或生成对比 HTML。

== 安装 (在新 Colab session 跑，避免 torch 冲突) ==

  # 1. 挂 Drive
  from google.colab import drive
  drive.mount('/content/drive')

  # 2. 装 Rex-Omni
  !git clone https://github.com/IDEA-Research/Rex-Omni.git /content/Rex-Omni
  %cd /content/Rex-Omni
  !pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128
  !pip install -r requirements.txt
  !pip install -v -e .

  # 3. 拉我们的代码
  !git clone https://github.com/wistone/VLTrainingDemo.git /content/QwenVL3

== 用法 ==

  默认 1000 sample × 3 split (~3.5h on L4):
    python /content/QwenVL3/stage2/06_eval_rex_omni.py \\
        --stage2_data_root /content/drive/MyDrive/qwenvl3/data/stage2 \\
        --out_dir /content/drive/MyDrive/qwenvl3/eval_external/rex_omni

  快速 sanity (~25 min on L4):
    python /content/QwenVL3/stage2/06_eval_rex_omni.py \\
        --stage2_data_root /content/drive/MyDrive/qwenvl3/data/stage2 \\
        --out_dir /content/drive/MyDrive/qwenvl3/eval_external/rex_omni_sanity \\
        --n_refcoco 100

  只跑 val (省一半时间):
    python /content/QwenVL3/stage2/06_eval_rex_omni.py ... --skip_splits testA testB
"""
import argparse
import io
import json
from pathlib import Path

from PIL import Image


# ============================================================================
# 通用 helpers (跟 04_eval_stage2.py 同公式，确保数字可比)
# ============================================================================

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


# ============================================================================
# Rex-Omni 加载 + 推理
# ============================================================================

def load_rex_omni(model_path):
    """加载 Rex-Omni 模型。需要先 pip install Rex-Omni 包。"""
    try:
        from rex_omni import RexOmniWrapper
    except ImportError:
        raise ImportError(
            "rex_omni 包未安装。先在 Colab 跑：\n"
            "  !git clone https://github.com/IDEA-Research/Rex-Omni.git /content/Rex-Omni\n"
            "  %cd /content/Rex-Omni && pip install -r requirements.txt && pip install -v -e ."
        )

    print(f"[load] Rex-Omni from {model_path}")
    rex = RexOmniWrapper(
        model_path=model_path,
        backend="transformers",     # "vllm" 更快但 Colab 上不一定装得上
        max_tokens=512,             # bbox 输出短，不需要 4096
        temperature=0.0,
        top_p=0.05,
        top_k=1,
        repetition_penalty=1.05,
    )
    print(f"[load] Rex-Omni 加载完成")
    return rex


def predict_bbox_pixel(rex, image, ref):
    """用 detection task 让 Rex-Omni 把 ref expression 框出来。

    返回 (x0, y0, x1, y1) 像素坐标，或 None（推理失败 / 没框出来）。
    """
    try:
        results = rex.inference(
            images=image, task="detection", categories=[ref],
        )
    except Exception as e:
        return None, f"inference exception: {e}"

    if not results or not results[0].get("success", False):
        err = results[0].get("error", "unknown") if results else "empty result"
        return None, f"inference failed: {err}"

    preds = results[0].get("extracted_predictions", {})
    # Rex-Omni 可能把 ref expression 略微 normalize（如 lowercase）
    # 我们宽松匹配：先按原 key，再按 lowercase，再退到第一个非空 list
    boxes = preds.get(ref) or preds.get(ref.lower())
    if not boxes:
        for cat, b in preds.items():
            if b:
                boxes = b
                break
    if not boxes:
        return None, "no box predicted"

    # 取第一个（Rex-Omni 内部会按 confidence 排序）
    coords = boxes[0].get("coords")
    if not coords or len(coords) != 4:
        return None, f"malformed coords: {coords}"
    return tuple(float(c) for c in coords), None


# ============================================================================
# RefCOCO 评测（跟 04_eval_stage2.py 同样的样本采集逻辑，确保数字可比）
# ============================================================================

def eval_refcoco_split(rex, stage2_data_root, split_name, n_samples, out_path):
    from datasets import load_dataset

    rc_dir = Path(stage2_data_root) / "refcoco"
    if not rc_dir.exists():
        print(f"[skip] RefCOCO ({split_name}): {rc_dir} 不存在")
        return None

    print(f"\n[task] Rex-Omni RefCOCO {split_name}  (n_target={n_samples})")
    try:
        ds = load_dataset(str(rc_dir), split=split_name, trust_remote_code=True)
    except Exception as e:
        print(f"  [skip] split={split_name} 加载失败: {e}")
        return None
    print(f"  数据加载: {len(ds)} 条")

    n = min(n_samples, len(ds))
    results = []
    ious = []
    parseable = 0
    failures = {"image": 0, "ref": 0, "bbox": 0, "inference": 0}

    for i in range(n):
        s = ds[i]

        # 取图（同 04_eval_stage2 逻辑）
        img_field = s.get("image")
        if isinstance(img_field, dict) and "bytes" in img_field:
            image = Image.open(io.BytesIO(img_field["bytes"])).convert("RGB")
        elif hasattr(img_field, "convert"):
            image = img_field.convert("RGB")
        else:
            failures["image"] += 1
            continue
        iw, ih = image.size

        # 取 ref（同 04_eval_stage2 逻辑）
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
            failures["ref"] += 1
            continue

        # 取 GT bbox（同 04_eval_stage2 逻辑）
        bbox = s.get("bbox") or s.get("box")
        if not bbox or len(bbox) != 4:
            failures["bbox"] += 1
            continue
        if max(bbox) > 1.5:
            x, y, w, h = bbox
            gt_box_norm = (x/iw, y/ih, (x+w)/iw, (y+h)/ih)
            gt_box_pixel = (x, y, x+w, y+h)
        else:
            gt_box_norm = tuple(bbox)
            gt_box_pixel = (bbox[0]*iw, bbox[1]*ih, bbox[2]*iw, bbox[3]*ih)

        # 推理 — Rex-Omni 返回像素坐标
        pred_pixel, err = predict_bbox_pixel(rex, image, ref)
        sample_iou = 0.0
        pred_norm = None
        if pred_pixel is not None:
            parseable += 1
            pred_norm = (pred_pixel[0]/iw, pred_pixel[1]/ih,
                         pred_pixel[2]/iw, pred_pixel[3]/ih)
            sample_iou = iou(pred_norm, gt_box_norm)
        else:
            failures["inference"] += 1
        ious.append(sample_iou)
        results.append({
            "idx": i+1, "ref": ref,
            "gt_bbox": [round(c, 4) for c in gt_box_norm],
            "pred_bbox": [round(c, 4) for c in pred_norm] if pred_norm else None,
            "iou": round(sample_iou, 4),
            "error": err,
        })
        if i < 3 or (i+1) % 100 == 0 or i == n-1:
            print(f"  [{i+1}/{n}] ref={ref[:40]!r}  IoU={sample_iou:.3f}"
                  f"{f'  err={err}' if err else ''}")

    if not ious:
        print(f"  [skip] 0 个有效样本，failures={failures}")
        return None

    n_eval = len(ious)
    acc_05 = sum(1 for x in ious if x >= 0.5) / n_eval
    acc_07 = sum(1 for x in ious if x >= 0.7) / n_eval
    mean_iou = sum(ious) / n_eval
    parse_rate = parseable / n_eval

    print(f"\n  [{split_name}]  Acc@0.5={acc_05:.2%}  Acc@0.7={acc_07:.2%}  "
          f"mIoU={mean_iou:.3f}  parse_rate={parse_rate:.2%}")
    print(f"             failures: {failures}")

    summary = {
        "task": f"refcoco_{split_name}",
        "model": "rex_omni",
        "n_evaluated": n_eval,
        "metrics": {
            "acc@0.5": acc_05, "acc@0.7": acc_07,
            "mean_iou": mean_iou, "parse_rate": parse_rate,
        },
        "failures": failures,
        "samples": results,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    return summary


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2_data_root", required=True,
                    help="包含 refcoco/ 的目录（跟我们 04_eval_stage2.py 用同源数据）")
    ap.add_argument("--out_dir", default="/content/drive/MyDrive/qwenvl3/eval_external/rex_omni")
    ap.add_argument("--n_refcoco", type=int, default=1000,
                    help="每个 split 的样本数（默认 1000，sanity 可用 100）")
    ap.add_argument("--skip_splits", nargs="*", default=[],
                    choices=["val", "testA", "testB"],
                    help="跳过指定 split（如 --skip_splits testA testB 只跑 val）")
    ap.add_argument("--model_path", default="IDEA-Research/Rex-Omni",
                    help="HF model id 或本地路径。可用 IDEA-Research/Rex-Omni-AWQ 量化版省显存")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[out_dir] {out_dir}\n")

    rex = load_rex_omni(args.model_path)

    all_metrics = {}
    splits = [s for s in ["val", "testA", "testB"] if s not in args.skip_splits]
    for split in splits:
        r = eval_refcoco_split(
            rex, args.stage2_data_root, split, args.n_refcoco,
            out_dir / f"refcoco_{split}.json",
        )
        if r:
            all_metrics[f"refcoco_{split}"] = r["metrics"]

    summary = {
        "model": "rex_omni",
        "model_path": args.model_path,
        "n_per_split": args.n_refcoco,
        "results": all_metrics,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("=== Rex-Omni RefCOCO 评测完成 ===")
    print(f"详细结果: {out_dir}/")
    print()
    for task, metrics in all_metrics.items():
        print(f"  {task}:")
        print(f"    Acc@0.5    = {metrics['acc@0.5']:.2%}")
        print(f"    Acc@0.7    = {metrics['acc@0.7']:.2%}")
        print(f"    mean IoU   = {metrics['mean_iou']:.3f}")
        print(f"    parse_rate = {metrics['parse_rate']:.2%}")

    print()
    print("📌 Rex-Omni 不支持 POPE / VQAv2 / NoCaps（perception 专用模型，不做 free-form QA/caption）")
    print("📌 跟你的 Stage 2 模型对比：用同样 split + 同样 n_refcoco 跑 04_eval_stage2.py 即可")


if __name__ == "__main__":
    main()
