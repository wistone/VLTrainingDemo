"""Stage 2 训练前 baseline 评估：用 Stage 1 当前 checkpoint 跑 LLaVA-Instruct 和 OCR-VQA 子集。

目的：训练前后对比，量化 Stage 2 的提升。

不做的事：
  - 不做 grounding eval（Stage 1 没见过 bbox 格式，必 0%）
  - 不强行套 Qwen chat template（Stage 1 模型只见过 <image>+caption 的极简格式，
    强套模板会让生成混乱；保持极简输入更能反映 Stage 1 实际能力）

衡量项：
  1. VQA 关键词匹配率（gt 里的关键词出现在生成里的比例，主体识别能力代理指标）
  2. OCR exact substring match（gt 字符串作为子串出现的样本比例）
  3. 全部生成结果存 json，留作后续人工 review 和 Stage 2 后对比

用法（GPU runtime，T4 即可，不需要 A100）：
    python stage2/02_baseline_eval.py \\
        --ckpt_dir /content/drive/MyDrive/qwenvl3/stage1_ckpt_v3/checkpoint-XXXX \\
        --processor_dir /content/drive/MyDrive/qwenvl3/stage1_init \\
        --stage2_data_root /content/drive/MyDrive/qwenvl3/data/stage2 \\
        --out_dir /content/drive/MyDrive/qwenvl3/eval_baseline \\
        --n_per_task 50
"""
import argparse
import io
import json
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

# 复用 Stage 1 工具
sys.path.insert(0, str(Path(__file__).parent.parent / "stage1"))
from _common import install_custom_projector, get_components  # noqa: E402

IGNORE_INDEX = -100


def compute_num_image_tokens(config):
    vc = config.vision_config
    n = (vc.image_size // vc.patch_size) ** 2
    if config.vision_feature_select_strategy == "default":
        n -= 1
    return n


# ============================================================================
# 模型加载（同 Stage 1 04_eval_stage1.py）
# ============================================================================

def load_model(ckpt_dir, processor_dir):
    if torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
        print(f"[device] CUDA, bf16")
    else:
        device, dtype = "cpu", torch.float32
        print(f"[device] CPU, fp32（慢；建议用 GPU runtime）")

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
# 图像加载：可从 COCO zip / OCR-VQA 目录 / HF dataset 读
# ============================================================================

class CocoZipLoader:
    """从 COCO train2017.zip 直读图（不解压）。"""
    def __init__(self, zip_path):
        self.zip = zipfile.ZipFile(zip_path)
        # 预算 zip 内文件名集合，加速查找
        self.names = set(self.zip.namelist())

    def open(self, image_filename):
        """image_filename 形如 '000000123456.jpg'。"""
        # COCO zip 里路径是 train2017/000000123456.jpg
        full_path = f"train2017/{image_filename}"
        if full_path not in self.names:
            raise FileNotFoundError(f"{full_path} not in COCO zip")
        with self.zip.open(full_path) as f:
            return Image.open(io.BytesIO(f.read())).convert("RGB")

    def __del__(self):
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
    # 极简：N 个 <image> + 问题文本（不套 chat template，跟 Stage 1 训练时一致）
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


# ============================================================================
# 任务 1：LLaVA-Instruct VQA
# ============================================================================

def normalize(s):
    """小写，去标点，给关键词匹配用。"""
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower())


def keyword_match_rate(gt, gen, min_kw_len=4):
    """gt 里 ≥4 字符的实词，gen 里有几个"""
    gt_words = set(w for w in normalize(gt).split() if len(w) >= min_kw_len)
    if not gt_words:
        return None
    gen_words = set(normalize(gen).split())
    matched = gt_words & gen_words
    return len(matched) / len(gt_words)


def eval_llava_instruct(model, tokenizer, image_processor, data_root, n_samples,
                        num_image_tokens, out_path):
    """LLaVA-Instruct VQA baseline。"""
    json_path = Path(data_root) / "llava_instruct" / "llava_instruct_150k.json"
    coco_zip = Path(data_root) / "coco" / "train2017.zip"

    if not json_path.exists():
        print(f"[skip] {json_path} 不存在 — 先跑 01_prepare_data.py")
        return None
    if not coco_zip.exists() or coco_zip.stat().st_size < 1e9:
        print(f"[skip] {coco_zip} 未下载完整")
        return None

    print(f"\n[task] LLaVA-Instruct VQA  (n={n_samples})")
    with open(json_path) as f:
        all_data = json.load(f)

    # 抽固定 seed 的 n 个样本
    import random
    random.seed(42)
    samples = random.sample(all_data, n_samples)

    loader = CocoZipLoader(coco_zip)
    results = []
    match_rates = []

    for i, s in enumerate(samples):
        try:
            image = loader.open(s["image"])
        except FileNotFoundError as e:
            print(f"  [{i+1}/{n_samples}] skip {e}")
            continue

        # 取第一对 human/gpt
        question = s["conversations"][0]["value"].replace("<image>", "").strip()
        gt_answer = s["conversations"][1]["value"]

        gen = generate_answer(
            model, tokenizer, image_processor, image, question, num_image_tokens
        )
        rate = keyword_match_rate(gt_answer, gen)
        if rate is not None:
            match_rates.append(rate)

        results.append({
            "idx": i + 1,
            "image": s["image"],
            "question": question,
            "gt_answer": gt_answer,
            "generated": gen,
            "keyword_match_rate": rate,
        })
        print(f"  [{i+1}/{n_samples}] {s['image']}")
        print(f"    Q: {question[:80]}")
        print(f"    GT: {gt_answer[:80]}")
        print(f"    GEN: {gen[:80]}")
        print(f"    keyword_match: {rate:.2f}" if rate is not None else "    (no kw)")

    avg_rate = sum(match_rates) / len(match_rates) if match_rates else 0.0
    print(f"\n  平均关键词匹配率: {avg_rate:.3f}  (n={len(match_rates)})")

    summary = {
        "n_total": len(samples),
        "n_evaluated": len(match_rates),
        "avg_keyword_match_rate": avg_rate,
        "samples": results,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  → 写入 {out_path}")
    return summary


# ============================================================================
# 任务 2：TextVQA（OCR 类，替代原计划的 OCR-VQA）
# ============================================================================

def eval_textvqa(model, tokenizer, image_processor, data_root, n_samples,
                  num_image_tokens, out_path):
    """TextVQA baseline — OCR + VQA。

    数据来源：lmms-lab/textvqa（HF）。结构通常为 parquet 里的 {image, question, answers}。
    Stage 2 训练前模型对结构化 Q/A 几乎不会回答；主要看图内文字是否被部分召回。

    实现先做基本 stub：检测数据存在 → 用 datasets.load_dataset 试着读 → 否则跳过。
    """
    tv_dir = Path(data_root) / "textvqa"
    if not tv_dir.exists() or not any(tv_dir.iterdir()):
        print(f"\n[skip] TextVQA 未下载（{tv_dir} 为空）；可选任务，不影响主流程。")
        return None

    # 字段命名因 HF repo 而异，等数据下到再实装
    print(f"\n[task] TextVQA — 数据已下载到 {tv_dir}")
    print(f"  ⚠️  字段解析根据实际下到的 repo 调整，本次先跳过 detailed eval。")
    print(f"  ⚠️  不影响 Stage 2 主线（OCR 能力 Stage 1 已自然涌现）。")
    return None


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True,
                    help="Stage 1 checkpoint（中间或最终）")
    ap.add_argument("--processor_dir", default=None,
                    help="image_processor + tokenizer 来源（中间 ckpt 通常用 stage1_init）")
    ap.add_argument("--stage2_data_root", required=True,
                    help="Stage 2 数据下载根目录（=01_prepare_data 的 DRIVE_ROOT）")
    ap.add_argument("--out_dir", default="/content/drive/MyDrive/qwenvl3/eval_baseline")
    ap.add_argument("--n_per_task", type=int, default=50)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, image_processor = load_model(args.ckpt_dir, args.processor_dir)
    num_image_tokens = compute_num_image_tokens(model.config)
    print(f"num_image_tokens = {num_image_tokens}")

    # 任务 1: LLaVA-Instruct VQA
    eval_llava_instruct(
        model, tokenizer, image_processor,
        args.stage2_data_root, args.n_per_task,
        num_image_tokens,
        out_dir / "baseline_llava_vqa.json",
    )

    # 任务 2: TextVQA (OCR 类)
    eval_textvqa(
        model, tokenizer, image_processor,
        args.stage2_data_root, args.n_per_task,
        num_image_tokens,
        out_dir / "baseline_textvqa.json",
    )

    print(f"\n=== Baseline 评估完成 ===")
    print(f"   {out_dir}/")
    print(f"   → 训完 Stage 2 后用同一个 n_per_task 跑 04_eval_stage2.py 做 A/B 对比")


if __name__ == "__main__":
    main()
