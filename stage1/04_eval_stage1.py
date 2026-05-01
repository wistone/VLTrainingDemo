"""Stage 1 评估：caption 质量 + image-token 利用率 ablation。

两项评估：
1. Caption 质量（人工 review）：在 20 张 held-out 图上生成 caption，存为 json
2. Image-token ablation（自动指标）：同一批样本，分别"带图"和"不带图"做 forward，
   loss 差应 ≥ 1.0 ——证明 projector 真的把视觉信息注入进了 LLM。
   如果差距很小，说明视觉对齐没成功。

用法：
    python stage1/04_eval_stage1.py \\
        --ckpt_dir /content/drive/MyDrive/qwenvl3/stage1_ckpt \\
        --data_root /content/data/llava-pretrain \\
        --out_dir /content/drive/MyDrive/qwenvl3/eval_stage1
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoTokenizer,
    LlavaForConditionalGeneration,
    SiglipImageProcessor,
)

IGNORE_INDEX = -100


def load_model(ckpt_dir):
    print(f"加载 checkpoint: {ckpt_dir}")
    model = LlavaForConditionalGeneration.from_pretrained(
        ckpt_dir, torch_dtype=torch.bfloat16
    ).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    image_processor = SiglipImageProcessor.from_pretrained(ckpt_dir)
    return model, tokenizer, image_processor


@torch.inference_mode()
def generate_caption(model, tokenizer, image_processor, image_path, max_new_tokens=64):
    image = Image.open(image_path).convert("RGB")
    pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(
        model.device, dtype=model.dtype
    )

    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    input_ids = torch.tensor([[image_token_id]]).to(model.device)

    out = model.generate(
        input_ids=input_ids,
        pixel_values=pixel_values,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
    )
    return tokenizer.decode(out[0][input_ids.size(1):], skip_special_tokens=True)


def caption_quality(model, tokenizer, image_processor, holdout_data, image_root, out_path):
    print(f"\n[Caption 质量] 在 {len(holdout_data)} 张 held-out 图上生成 caption")
    results = []
    for i, s in enumerate(holdout_data):
        img_path = Path(image_root) / s["image"]
        cap = generate_caption(model, tokenizer, image_processor, img_path)
        gt = s["conversations"][1]["value"]
        results.append({
            "image": s["image"],
            "generated": cap,
            "ground_truth": gt,
        })
        print(f"  [{i+1}/{len(holdout_data)}] {s['image']}")
        print(f"    生成: {cap}")
        print(f"    GT:   {gt[:100]}...")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  → 写入 {out_path}（请人工 review，目标 20 张里 ≥15 张能抓到主体）")


@torch.inference_mode()
def compute_loss_with_without_image(model, tokenizer, image_processor, holdout_data, image_root, n=10):
    """对 n 个样本计算 (有图 loss) vs (无图 loss)，期望差 ≥ 1.0。

    无图模式：把 pixel_values 替换成全黑图（或全零）。这样输入仍然有 <image> token 占位，
    但视觉特征是无信息的。如果 projector 真的在传递信息，loss 应该会显著上升。
    """
    print(f"\n[Image-token Ablation] {n} 个样本 ×（有图 vs 全黑图）")
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    eos_id = tokenizer.eos_token_id

    losses_with, losses_without = [], []
    for i, s in enumerate(holdout_data[:n]):
        img_path = Path(image_root) / s["image"]
        image = Image.open(img_path).convert("RGB")
        caption = s["conversations"][1]["value"].strip()

        # 构造 input
        prompt_ids = [image_token_id]
        target_ids = tokenizer(caption, add_special_tokens=False).input_ids + [eos_id]
        input_ids = torch.tensor([prompt_ids + target_ids]).to(model.device)
        labels = torch.tensor([[IGNORE_INDEX] * len(prompt_ids) + target_ids]).to(model.device)
        attention_mask = torch.ones_like(input_ids)

        # 有图
        pixel_values_real = image_processor(image, return_tensors="pt").pixel_values.to(
            model.device, dtype=model.dtype
        )
        out_real = model(
            input_ids=input_ids, labels=labels, attention_mask=attention_mask,
            pixel_values=pixel_values_real,
        )
        # 无图（黑图）
        black = Image.new("RGB", image.size, (0, 0, 0))
        pixel_values_black = image_processor(black, return_tensors="pt").pixel_values.to(
            model.device, dtype=model.dtype
        )
        out_black = model(
            input_ids=input_ids, labels=labels, attention_mask=attention_mask,
            pixel_values=pixel_values_black,
        )

        l_with = out_real.loss.item()
        l_without = out_black.loss.item()
        losses_with.append(l_with)
        losses_without.append(l_without)
        print(f"  [{i+1}/{n}] {s['image']}: with_img={l_with:.3f}  black_img={l_without:.3f}  Δ={l_without-l_with:+.3f}")

    avg_with = sum(losses_with) / len(losses_with)
    avg_without = sum(losses_without) / len(losses_without)
    delta = avg_without - avg_with

    print(f"\n  平均 loss: 真实图={avg_with:.3f}  黑图={avg_without:.3f}  Δ={delta:+.3f}")
    if delta >= 1.0:
        verdict = "✅ PASS（视觉信息被有效利用）"
    elif delta >= 0.3:
        verdict = "⚠️ WARN（视觉信号弱，建议继续训）"
    else:
        verdict = "❌ FAIL（视觉对齐失败，projector 可能没在工作）"
    print(f"  判定: {verdict}")

    return {
        "avg_loss_with_image": avg_with,
        "avg_loss_black_image": avg_without,
        "delta": delta,
        "verdict": verdict,
        "per_sample": list(zip(losses_with, losses_without)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--data_root", required=True, help="包含 images/ 和 holdout_20.json")
    ap.add_argument("--out_dir", default="/content/drive/MyDrive/qwenvl3/eval_stage1")
    ap.add_argument("--ablation_n", type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    holdout_path = Path(args.data_root) / "holdout_20.json"
    if not holdout_path.exists():
        raise FileNotFoundError(f"找不到 {holdout_path}，请先跑 01_prepare_data.py")
    with open(holdout_path) as f:
        holdout = json.load(f)
    image_root = Path(args.data_root) / "images"

    model, tokenizer, image_processor = load_model(args.ckpt_dir)

    # 1. caption 质量
    caption_quality(
        model, tokenizer, image_processor, holdout, image_root,
        out_dir / "captions.json",
    )

    # 2. ablation
    result = compute_loss_with_without_image(
        model, tokenizer, image_processor, holdout, image_root,
        n=args.ablation_n,
    )
    with open(out_dir / "ablation.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  → 写入 {out_dir / 'ablation.json'}")

    print("\n=== Stage 1 评估完成 ===")
    print(f"  人工 review: {out_dir / 'captions.json'}")
    print(f"  自动指标:    {out_dir / 'ablation.json'}")


if __name__ == "__main__":
    main()
