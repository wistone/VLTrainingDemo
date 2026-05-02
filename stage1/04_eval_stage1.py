"""Stage 1 评估：caption 质量 + image-token 利用率 ablation。

两项评估：
1. Caption 质量（人工 review）：在 20 张 held-out 图上生成 caption，存为 json
2. Image-token ablation（自动指标）：同一批样本，分别"带图"和"不带图"做 forward，
   loss 差应 ≥ 1.0 ——证明 projector 真的把视觉信息注入进了 LLM。
   如果差距很小，说明视觉对齐没成功。

可用于：
- 训练完成后的 final eval
- 训练过程中（不会影响训练 session）的中间 checkpoint 提前 eval

用法（final）：
    python stage1/04_eval_stage1.py \\
        --ckpt_dir /content/drive/MyDrive/qwenvl3/stage1_ckpt_v3 \\
        --data_root /content/data/llava-pretrain

用法（中间 checkpoint）：
    python stage1/04_eval_stage1.py \\
        --ckpt_dir /content/drive/MyDrive/qwenvl3/stage1_ckpt_v3/checkpoint-2500 \\
        --processor_dir /content/drive/MyDrive/qwenvl3/stage1_init \\
        --data_root /content/data/llava-pretrain
"""
import argparse
import io
import json
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


class ImageLoader:
    """统一的图像加载接口：可从目录或 zip 加载。

    用法：
      ImageLoader(image_root="/content/data/llava-pretrain")  # 从解压目录
      ImageLoader(images_zip="/content/drive/.../images.zip")  # 从 zip 直接读
    """
    def __init__(self, image_root=None, images_zip=None):
        if (image_root is None) == (images_zip is None):
            raise ValueError("image_root 和 images_zip 必须二选一")
        self.image_root = Path(image_root) if image_root else None
        self.zip = zipfile.ZipFile(images_zip) if images_zip else None

    def open(self, rel_path):
        """rel_path 形如 '00188/001883900.jpg'。返回 PIL.Image (RGB)。"""
        if self.image_root:
            return Image.open(self.image_root / rel_path).convert("RGB")
        with self.zip.open(rel_path) as f:
            data = f.read()
        return Image.open(io.BytesIO(data)).convert("RGB")

    def __del__(self):
        if self.zip is not None:
            try:
                self.zip.close()
            except Exception:
                pass

# 让脚本无论从哪里启动都能 import 同目录下的 _common
sys.path.insert(0, str(Path(__file__).parent))
from _common import install_custom_projector, get_components  # noqa: E402

IGNORE_INDEX = -100


def compute_num_image_tokens(config):
    vc = config.vision_config
    n = (vc.image_size // vc.patch_size) ** 2
    if config.vision_feature_select_strategy == "default":
        n -= 1
    return n


def load_model(ckpt_dir, processor_dir=None):
    """加载 model + tokenizer + image_processor。

    processor_dir：为 None 时从 ckpt_dir 加载（适合 final ckpt）；
                   指定时从该目录加载（适合中间 checkpoint，里面没有 image processor）。
    """
    # 自动选 device：优先 CUDA，没有就用 CPU（CPU 慢但能跑）
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
        print(f"[device] CUDA 可用，使用 GPU + bf16")
    else:
        device = "cpu"
        dtype = torch.float32  # CPU 上 bf16 慢且不全支持，用 fp32 更稳
        print(f"[device] 无 CUDA，使用 CPU + fp32（慢；20 张图大约 15–30 min）")

    print(f"加载 checkpoint: {ckpt_dir}")
    model = LlavaForConditionalGeneration.from_pretrained(
        ckpt_dir, torch_dtype=dtype
    )

    # 替换 projector 为 ProjectorWithNorm 并从 ckpt 加载权重（包括 LayerNorm）
    print("装载 ProjectorWithNorm...")
    install_custom_projector(model, init_dir=ckpt_dir, dtype=dtype)

    model = model.to(device).eval()

    # 中间 checkpoint 既不存 tokenizer 也不存 image_processor
    # （HF Trainer 默认行为：未传 processing_class 时 intermediate checkpoint 只有 model + 训练状态）
    # 所以两者都从 processor_dir（通常是 stage1_init）加载
    proc_dir = processor_dir or ckpt_dir
    print(f"加载 tokenizer + image_processor from: {proc_dir}")
    tokenizer = AutoTokenizer.from_pretrained(proc_dir)
    image_processor = AutoImageProcessor.from_pretrained(proc_dir)

    return model, tokenizer, image_processor


@torch.inference_mode()
def generate_caption(model, tokenizer, image_processor, image, num_image_tokens, max_new_tokens=64):
    """image 可以是 PIL.Image 也可以是文件路径。"""
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(
        model.device, dtype=model.dtype
    )

    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    input_ids = torch.tensor([[image_token_id] * num_image_tokens]).to(model.device)

    out = model.generate(
        input_ids=input_ids,
        pixel_values=pixel_values,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
    )
    return tokenizer.decode(out[0][input_ids.size(1):], skip_special_tokens=True)


def caption_quality(model, tokenizer, image_processor, holdout_data, image_loader, num_image_tokens, out_path):
    print(f"\n[Caption 质量] 在 {len(holdout_data)} 张 held-out 图上生成 caption")
    results = []
    for i, s in enumerate(holdout_data):
        image = image_loader.open(s["image"])
        cap = generate_caption(model, tokenizer, image_processor, image, num_image_tokens)
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
def compute_loss_with_without_image(model, tokenizer, image_processor, holdout_data, image_loader,
                                    num_image_tokens, n=10):
    """对 n 个样本计算 (有图 loss) vs (无图 loss)，期望差 ≥ 1.0。

    无图模式：把 pixel_values 替换成全黑图（或全零）。这样输入仍然有 <image> token 占位，
    但视觉特征是无信息的。如果 projector 真的在传递信息，loss 应该会显著上升。
    """
    print(f"\n[Image-token Ablation] {n} 个样本 ×（有图 vs 全黑图）")
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    eos_id = tokenizer.eos_token_id

    losses_with, losses_without = [], []
    for i, s in enumerate(holdout_data[:n]):
        image = image_loader.open(s["image"])
        caption = s["conversations"][1]["value"].strip()

        # 构造 input：N 个 <image> 占位符 + caption
        prompt_ids = [image_token_id] * num_image_tokens
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
    ap.add_argument("--ckpt_dir", required=True,
                    help="模型 checkpoint 目录（可以是 final ckpt 或 checkpoint-NNNN）")
    ap.add_argument("--processor_dir", default=None,
                    help="image processor 目录（中间 checkpoint 通常没存 processor，"
                         "需要指定到 stage1_init）")
    ap.add_argument("--holdout_json", required=True,
                    help="holdout_20.json 路径（如 /content/drive/MyDrive/qwenvl3/data/llava-pretrain/holdout_20.json）")
    ap.add_argument("--image_root", default=None,
                    help="图像解压根目录（含 00xxx 子目录）。与 --images_zip 二选一")
    ap.add_argument("--images_zip", default=None,
                    help="images.zip 路径（直接从 zip 读，省去解压 25GB）。与 --image_root 二选一")
    ap.add_argument("--out_dir", default="/content/drive/MyDrive/qwenvl3/eval_stage1")
    ap.add_argument("--ablation_n", type=int, default=10)
    args = ap.parse_args()

    if (args.image_root is None) == (args.images_zip is None):
        raise ValueError("必须二选一指定 --image_root 或 --images_zip")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    holdout_path = Path(args.holdout_json)
    if not holdout_path.exists():
        raise FileNotFoundError(f"找不到 {holdout_path}")
    with open(holdout_path) as f:
        holdout = json.load(f)
    print(f"载入 holdout: {len(holdout)} 张")

    image_loader = ImageLoader(image_root=args.image_root, images_zip=args.images_zip)
    print(f"图像源: {args.image_root or args.images_zip}")

    model, tokenizer, image_processor = load_model(args.ckpt_dir, args.processor_dir)
    num_image_tokens = compute_num_image_tokens(model.config)
    print(f"num_image_tokens = {num_image_tokens}")

    # 1. caption 质量
    caption_quality(
        model, tokenizer, image_processor, holdout, image_loader,
        num_image_tokens,
        out_dir / "captions.json",
    )

    # 2. ablation
    result = compute_loss_with_without_image(
        model, tokenizer, image_processor, holdout, image_loader,
        num_image_tokens,
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
