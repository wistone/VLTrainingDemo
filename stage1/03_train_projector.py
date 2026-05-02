"""Stage 1 主训练：只训 multi_modal_projector，冻结 vision_tower + language_model。

核心训练逻辑：
- 数据：LLaVA-Pretrain-558K caption 数据
- 输入格式：[<image>] + caption_tokens
- 标签：<image> 部分 mask 为 -100，caption 部分计算 next-token loss
- 优化器：AdamW(lr=1e-3, projector only)，cosine schedule，3% warmup
- 精度：bf16 + gradient checkpointing
- Checkpoint：每 500 步存到 Drive，最多保留 2 个

Colab session 断开后重跑此脚本会自动从最近 checkpoint 续训。

烟雾测试模式（先验证流程）：
    python stage1/03_train_projector.py --smoke_test \\
        --model_init_dir /content/drive/MyDrive/qwenvl3/stage1_init \\
        --data_root /content/data/llava-pretrain \\
        --output_dir /content/drive/MyDrive/qwenvl3/stage1_smoke

正式训练：
    python stage1/03_train_projector.py \\
        --model_init_dir /content/drive/MyDrive/qwenvl3/stage1_init \\
        --data_root /content/data/llava-pretrain \\
        --output_dir /content/drive/MyDrive/qwenvl3/stage1_ckpt \\
        --batch_size 32 --num_epochs 1
"""
import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    LlavaForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

# 让脚本无论从哪里启动都能 import 同目录下的 _common
sys.path.insert(0, str(Path(__file__).parent))
from _common import install_custom_projector  # noqa: E402


def print_gpu_mem_and_recommend(batch_size, num_image_tokens):
    """打印 GPU 显存并校准 batch_size 是否安全。"""
    if not torch.cuda.is_available():
        print("[GPU] no CUDA")
        return
    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / 1e9
    name = props.name
    print(f"[GPU] {name}, {total_gb:.1f}GB")
    # logits fp32 占用估算（vocab=152K for Qwen2.5）
    seq_estimate = num_image_tokens + 64  # 假设 caption ~64 token
    logits_gb = batch_size * seq_estimate * 152064 * 4 / 1e9
    print(f"[mem] 估算 logits fp32 张量: {logits_gb:.2f}GB（batch={batch_size}, seq~{seq_estimate}, vocab=152K）")
    if total_gb < 50 and batch_size > 8:
        print(f"[warn] 40GB 卡上 batch>{8} 容易 OOM。建议 --batch_size 8 --grad_accum 4")

IGNORE_INDEX = -100


def compute_num_image_tokens(config):
    """LlavaForConditionalGeneration 要求 input_ids 里 <image> 数量 == 视觉 feature 数量。
    feature 数量由 vision_config 决定。"""
    vc = config.vision_config
    n_patches = (vc.image_size // vc.patch_size) ** 2
    if config.vision_feature_select_strategy == "default":
        n_patches -= 1  # 砍 CLS（CLIP 风格）
    return n_patches


class LlavaPretrainDataset(Dataset):
    """LLaVA-Pretrain-558K caption 数据集。

    每条样本：
        input_ids = [<image>] * num_image_tokens + tokenize(caption) + [eos]
        labels    = [-100]    * num_image_tokens + tokenize(caption) + [eos]
        pixel_values = SigLIP2 预处理的 384x384 图像

    关键：input_ids 里 <image> token 的数量必须等于视觉 feature 数量（每个 <image>
    占位符 1:1 替换为一个视觉 feature 向量）。SigLIP2-SO400M-patch14-384 + "full"
    策略下，每张图产生 27*27=729 个视觉 feature。
    """

    def __init__(self, json_path, image_root, tokenizer, image_processor, num_image_tokens,
                 max_len=1024, limit=None):
        with open(json_path) as f:
            self.data = json.load(f)
        if limit is not None:
            self.data = self.data[:limit]
        self.image_root = Path(image_root)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.num_image_tokens = num_image_tokens
        self.max_len = max_len
        self.image_token_id = tokenizer.convert_tokens_to_ids("<image>")
        self.eos_id = tokenizer.eos_token_id

        # max_len 必须能装下所有 image tokens + 至少 32 个 caption token
        assert max_len >= num_image_tokens + 32, \
            f"max_len={max_len} 太小，至少要 {num_image_tokens + 32}（{num_image_tokens} 视觉 token + caption）"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s = self.data[idx]
        # 图像
        img = Image.open(self.image_root / s["image"]).convert("RGB")
        pixel_values = self.image_processor(img, return_tensors="pt").pixel_values[0]

        # 文本：Stage 1 极简——N 个 <image> 占位符 + caption + eos
        caption = s["conversations"][1]["value"].strip()

        prompt_ids = [self.image_token_id] * self.num_image_tokens
        target_ids = self.tokenizer(caption, add_special_tokens=False).input_ids + [self.eos_id]

        input_ids = prompt_ids + target_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + target_ids

        # 截断时永远不能砍到 image token 段（否则 image features 数量对不上）
        if len(input_ids) > self.max_len:
            # 只截 caption 尾部
            input_ids = input_ids[: self.max_len]
            labels = labels[: self.max_len]
            # 保证最后一个 token 是 eos（如果被截了，强制设回去）
            if input_ids[-1] != self.eos_id:
                input_ids[-1] = self.eos_id
                labels[-1] = self.eos_id

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pixel_values": pixel_values,
        }


@dataclass
class Collator:
    pad_token_id: int

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(x["input_ids"].size(0) for x in batch)
        input_ids, labels, attention_mask = [], [], []
        for x in batch:
            pad = max_len - x["input_ids"].size(0)
            input_ids.append(
                torch.cat([x["input_ids"], torch.full((pad,), self.pad_token_id, dtype=torch.long)])
            )
            labels.append(
                torch.cat([x["labels"], torch.full((pad,), IGNORE_INDEX, dtype=torch.long)])
            )
            attention_mask.append(
                torch.cat(
                    [
                        torch.ones(x["input_ids"].size(0), dtype=torch.long),
                        torch.zeros(pad, dtype=torch.long),
                    ]
                )
            )
        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attention_mask),
            "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        }


def freeze_except_projector(model):
    """冻结所有参数，除了 multi_modal_projector。"""
    n_train, n_total = 0, 0
    for name, param in model.named_parameters():
        n_total += param.numel()
        if "multi_modal_projector" in name:
            param.requires_grad = True
            n_train += param.numel()
        else:
            param.requires_grad = False
    print(f"可训练参数: {n_train/1e6:.2f}M / 总参数: {n_total/1e9:.3f}B")
    print(f"  比例: {n_train/n_total*100:.3f}%")


def find_latest_checkpoint(output_dir: Path):
    if not output_dir.exists():
        return None
    ckpts = sorted(
        [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[1]),
    )
    return ckpts[-1] if ckpts else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_init_dir", required=True)
    ap.add_argument("--data_root", required=True, help="包含 images/ 和 blip_laion_cc_sbu_558k.json")
    ap.add_argument("--output_dir", required=True)
    # 默认 batch=8 / grad_accum=4 → 等效 batch 32，且能在 40GB A100 上跑
    # 80GB 卡可改为 --batch_size 32 --grad_accum 1 提速
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--num_epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--report_to", default="wandb", choices=["wandb", "none", "tensorboard"])
    ap.add_argument("--smoke_test", action="store_true", help="只用 100 条数据跑 50 步，验证流程")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    print(f"加载模型 from {args.model_init_dir}")
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_init_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",  # 比 eager 省 ~50% attention 激活
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_init_dir)
    image_processor = AutoImageProcessor.from_pretrained(args.model_init_dir)

    # 关键：HF 用 config 里默认的 LlavaMultiModalProjector 构造 projector，
    # 会忽略 safetensors 里 multi_modal_projector.norm.* 权重。必须替换并手动加载。
    print("\n替换 projector 为 ProjectorWithNorm（带 LayerNorm）...")
    install_custom_projector(model, init_dir=args.model_init_dir, dtype=torch.bfloat16)

    # 冻结
    freeze_except_projector(model)
    model.gradient_checkpointing_enable()
    # 让冻结的 LLM 也走 ckpting，节省 activation 显存
    if hasattr(model.language_model, "gradient_checkpointing_enable"):
        model.language_model.gradient_checkpointing_enable()

    # 计算视觉 token 数（必须和 model 实际产生的 feature 数对上）
    num_image_tokens = compute_num_image_tokens(model.config)
    vc = model.config.vision_config
    print(f"\nnum_image_tokens = {num_image_tokens}  "
          f"(image_size={vc.image_size}, patch={vc.patch_size}, "
          f"strategy={model.config.vision_feature_select_strategy})")

    print()
    print_gpu_mem_and_recommend(args.batch_size, num_image_tokens)

    # 数据集
    # LLaVA-Pretrain 的 zip 解压后 00xxx 子目录直接在 data_root 下，没有 images/ 这层
    print(f"\n加载数据 from {args.data_root}")
    json_path = Path(args.data_root) / "blip_laion_cc_sbu_558k.json"
    image_root = Path(args.data_root)
    limit = 100 if args.smoke_test else None
    dataset = LlavaPretrainDataset(
        json_path, image_root, tokenizer, image_processor,
        num_image_tokens=num_image_tokens,
        max_len=args.max_len, limit=limit,
    )
    print(f"数据集大小: {len(dataset)}")

    collator = Collator(pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)

    # 烟雾测试用更小步数
    max_steps = 50 if args.smoke_test else -1

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        save_strategy="steps" if not args.smoke_test else "no",
        report_to=args.report_to,
        run_name=f"stage1-projector{'-smoke' if args.smoke_test else ''}",
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        optim="adamw_torch",
    )

    # 把 tokenizer 传给 Trainer → 每个中间 checkpoint 也会存 tokenizer 文件，
    # 方便从 ckpt 直接 resume 或做 eval（无需额外指定 processor_dir）。
    # HF transformers ≥4.46 用 processing_class；旧版用 tokenizer。
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    try:
        trainer = Trainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        # 旧版 transformers fallback
        trainer = Trainer(**trainer_kwargs, tokenizer=tokenizer)

    # 续训
    last_ckpt = find_latest_checkpoint(output_dir)
    if last_ckpt and not args.smoke_test:
        print(f"从 checkpoint 续训: {last_ckpt}")
        trainer.train(resume_from_checkpoint=str(last_ckpt))
    else:
        trainer.train()

    if not args.smoke_test:
        print(f"\n保存最终 checkpoint 到 {output_dir}")
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        image_processor.save_pretrained(str(output_dir))

    print("Done.")


if __name__ == "__main__":
    main()
