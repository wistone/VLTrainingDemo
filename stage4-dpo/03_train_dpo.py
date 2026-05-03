"""Stage 4 DPO 训练 — 在 Stage 2-v2 final ckpt 上做偏好对齐。

继承 HF Trainer 但自定义 compute_loss 实现 DPO，不依赖 TRL。
原因：TRL 的多模态 DPO 在不同版本兼容性不稳，自己实现 ~50 行透明可控。

Reference model 用 PEFT 的 disable_adapter trick：训练时 active 模型 = base + LoRA，
context manager 内 = base only（即 reference model）。共用 base weights 省显存。

== 数据流 ==
  RLAIF-V parquet
       ↓ DPOPreferenceDataset
  {chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels, pixel_values}
       ↓ DPOCollator (padding)
  batch
       ↓ compute_loss (4 forwards: active+ref × chosen+rejected)
  DPO loss + metrics

== 用法 ==

  烟雾测试（10 min, 50 steps）:
    python stage4-dpo/03_train_dpo.py --smoke_test \\
        --stage2_ckpt /content/drive/MyDrive/qwenvl3/stage2_v2_ckpt \\
        --stage1_ckpt /content/drive/MyDrive/qwenvl3/stage1_ckpt_v3 \\
        --processor_dir /content/drive/MyDrive/qwenvl3/stage1_init \\
        --dpo_data_dir /content/drive/MyDrive/qwenvl3/data/dpo/rlaif_v \\
        --output_dir /content/drive/MyDrive/qwenvl3/stage4_dpo_smoke \\
        --report_to none

  正式训练（~1-2h on Blackwell）:
    python stage4-dpo/03_train_dpo.py \\
        --stage2_ckpt /content/drive/MyDrive/qwenvl3/stage2_v2_ckpt \\
        --stage1_ckpt /content/drive/MyDrive/qwenvl3/stage1_ckpt_v3 \\
        --processor_dir /content/drive/MyDrive/qwenvl3/stage1_init \\
        --dpo_data_dir /content/drive/MyDrive/qwenvl3/data/dpo/rlaif_v \\
        --output_dir /content/drive/MyDrive/qwenvl3/stage4_dpo_ckpt
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def _ensure_torchao_compat():
    """同 stage2/03_train_stage2.py 的处理：卸掉 Colab 预装的 torchao 0.10。"""
    import importlib.util
    if importlib.util.find_spec("torchao") is None:
        return
    try:
        import torchao
        from packaging import version
        if version.parse(getattr(torchao, "__version__", "0.0.0")) >= version.parse("0.16.0"):
            return
    except Exception:
        pass
    print("[setup] 检测到旧版 torchao（与 PEFT 不兼容），卸载中...")
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"],
        check=False, capture_output=True,
    )
    for mod in list(sys.modules):
        if mod.startswith("torchao"):
            del sys.modules[mod]


_ensure_torchao_compat()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from peft import PeftModel  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from transformers import (  # noqa: E402
    AutoImageProcessor,
    AutoTokenizer,
    LlavaForConditionalGeneration,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# stage1/_common.py: install_custom_projector
sys.path.insert(0, str(Path(__file__).parent.parent / "stage1"))
from _common import get_components, install_custom_projector  # noqa: E402

# stage4-dpo/_common_dpo.py
sys.path.insert(0, str(Path(__file__).parent))
from _common_dpo import (  # noqa: E402
    DPOChatBuilder,
    DPOCollator,
    DPOPreferenceDataset,
    compute_response_logp,
    dpo_loss,
)


# ============================================================================
# Helpers — Stage 2 ckpt resolve / num_image_tokens
# ============================================================================

def compute_num_image_tokens(config):
    vc = config.vision_config
    n = (vc.image_size // vc.patch_size) ** 2
    if config.vision_feature_select_strategy == "default":
        n -= 1
    return n


def resolve_stage2_ckpt(p: str) -> str:
    """同 04_eval_stage2.py 的 resolve 逻辑：顶层不可用时 fallback 到 checkpoint-NNNN。"""
    path = Path(p)
    def is_complete(d: Path) -> bool:
        return ((d / "adapter_model.safetensors").exists()
                and (d / "multi_modal_projector.safetensors").exists())

    if is_complete(path):
        return str(path)
    if path.exists() and path.is_dir():
        candidates = sorted(
            (c for c in path.glob("checkpoint-*") if c.is_dir() and is_complete(c)),
            key=lambda c: int(c.name.split("-")[1]),
            reverse=True,
        )
        if candidates:
            print(f"[warn] {path} 顶层不完整，fallback 到 {candidates[0].name}")
            return str(candidates[0])
    raise FileNotFoundError(f"找不到可用 Stage 2 ckpt: {path}")


def resolve_stage1_ckpt(p: str) -> str:
    path = Path(p)
    def is_complete(d: Path) -> bool:
        sft = d / "model.safetensors"
        return sft.exists() and sft.stat().st_size > 1e9

    if is_complete(path):
        return str(path)
    if path.exists() and path.is_dir():
        candidates = sorted(
            (c for c in path.glob("checkpoint-*") if c.is_dir() and is_complete(c)),
            key=lambda c: int(c.name.split("-")[1]),
            reverse=True,
        )
        if candidates:
            print(f"[warn] {path} 顶层 model.safetensors 缺失，fallback 到 {candidates[0].name}")
            return str(candidates[0])
    raise FileNotFoundError(f"找不到可用 Stage 1 ckpt: {path}")


# ============================================================================
# 模型加载（Stage 2-v2 base + projector + LoRA → 继续训）
# ============================================================================

def load_stage2_v2_model(stage1_ckpt, stage2_ckpt, dtype):
    """加载 Stage 2-v2 模型用于 DPO 继续训：base + projector + LoRA r=64。

    - base 权重从 stage1_ckpt 加载
    - projector 从 stage2_ckpt 装 ProjectorWithNorm + 加载训好的 weights
    - LoRA adapter 从 stage2_ckpt 装载 (PeftModel.from_pretrained)
    - **不 merge_and_unload**：DPO 要保留 LoRA 才能用 disable_adapter() 当 reference
    - 解冻 LoRA + projector 让它们继续训
    """
    print(f"[load] base from {stage1_ckpt}")
    model = LlavaForConditionalGeneration.from_pretrained(
        stage1_ckpt, torch_dtype=dtype, attn_implementation="sdpa",
    )

    print(f"[load] custom projector from {stage2_ckpt}")
    install_custom_projector(model, init_dir=stage2_ckpt, dtype=dtype)

    # 冻结 vision tower
    _, vt_module, proj_module = get_components(model)
    for p in vt_module.parameters():
        p.requires_grad = False
    print(f"[model] vision_tower 冻结 ({sum(p.numel() for p in vt_module.parameters())/1e6:.1f}M)")

    # 加 LoRA adapter (Stage 2-v2 训好的)
    print(f"[load] LoRA adapter from {stage2_ckpt}")
    model = PeftModel.from_pretrained(model, stage2_ckpt, is_trainable=True)

    # 解冻 projector (DPO 中也微调 projector)
    for name, param in model.named_parameters():
        if "multi_modal_projector" in name:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[model] 可训练参数: {trainable/1e6:.1f}M / 总: {total/1e9:.2f}B "
          f"(比例: {trainable/total*100:.2f}%)")

    return model, proj_module


# ============================================================================
# DPO Trainer (HF Trainer subclass with custom DPO loss)
# ============================================================================

class DPOTrainerCustom(Trainer):
    """HF Trainer 子类，覆盖 compute_loss 实现 DPO。

    Reference model = PEFT disable_adapter (base only)，跟 active 共用 base weights。
    """
    def __init__(self, *args, beta=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 1. Active forward (with LoRA)
        chosen_logp = compute_response_logp(
            model,
            input_ids=inputs["chosen_input_ids"],
            attention_mask=inputs["chosen_attention_mask"],
            pixel_values=inputs["pixel_values"],
            labels=inputs["chosen_labels"],
        )
        rejected_logp = compute_response_logp(
            model,
            input_ids=inputs["rejected_input_ids"],
            attention_mask=inputs["rejected_attention_mask"],
            pixel_values=inputs["pixel_values"],
            labels=inputs["rejected_labels"],
        )

        # 2. Reference forward (no grad, disable adapter = base only)
        with torch.no_grad():
            with model.disable_adapter():
                ref_chosen_logp = compute_response_logp(
                    model,
                    input_ids=inputs["chosen_input_ids"],
                    attention_mask=inputs["chosen_attention_mask"],
                    pixel_values=inputs["pixel_values"],
                    labels=inputs["chosen_labels"],
                )
                ref_rejected_logp = compute_response_logp(
                    model,
                    input_ids=inputs["rejected_input_ids"],
                    attention_mask=inputs["rejected_attention_mask"],
                    pixel_values=inputs["pixel_values"],
                    labels=inputs["rejected_labels"],
                )

        # 3. DPO loss
        loss, metrics = dpo_loss(
            chosen_logp, rejected_logp,
            ref_chosen_logp, ref_rejected_logp,
            beta=self.beta,
        )

        # 4. Log metrics 到 trainer 状态（自动 wandb / log）
        if self.state.is_world_process_zero:
            for k, v in metrics.items():
                self.log({k: v.item() if isinstance(v, torch.Tensor) else float(v)})

        return (loss, None) if return_outputs else loss


# ============================================================================
# Projector 单独保存 callback（同 stage2-v2）
# ============================================================================

class ProjectorSaverCallback(TrainerCallback):
    def __init__(self, projector_module):
        self.projector = projector_module

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        target = ckpt_dir / "multi_modal_projector.safetensors"
        sd = {k: v.detach().cpu().contiguous() for k, v in self.projector.state_dict().items()}
        save_file(sd, target)


# ============================================================================
# Main
# ============================================================================

def parse_args():
    ap = argparse.ArgumentParser()
    # 路径
    ap.add_argument("--stage2_ckpt", required=True, help="Stage 2-v2 final ckpt")
    ap.add_argument("--stage1_ckpt", required=True, help="Stage 1 base (含 model.safetensors)")
    ap.add_argument("--processor_dir", required=True, help="tokenizer + image_processor 目录")
    ap.add_argument("--dpo_data_dir", required=True,
                    help="RLAIF-V (或类似) HF dataset 目录")
    ap.add_argument("--output_dir", required=True)

    # 数据
    ap.add_argument("--n_dpo_samples", type=int, default=0,
                    help="DPO 样本上限；0 = 全部")
    ap.add_argument("--max_len", type=int, default=1500)

    # DPO 超参
    ap.add_argument("--beta", type=float, default=0.1,
                    help="DPO KL 约束强度（0.05-0.5 常见）")
    ap.add_argument("--lr", type=float, default=1e-6,
                    help="DPO 必须很小，1e-7 ~ 1e-5")
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--num_epochs", type=int, default=1)

    # 训练超参
    ap.add_argument("--batch_size", type=int, default=4,
                    help="DPO 内存翻倍（chosen + rejected），保守一点")
    ap.add_argument("--grad_accum", type=int, default=8,
                    help="effective batch = batch × grad_accum，目标 32")

    # Logging / saving
    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--logging_steps", type=int, default=5)
    ap.add_argument("--save_total_limit", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--report_to", default="wandb",
                    choices=["wandb", "none", "tensorboard"])
    ap.add_argument("--run_name", default="stage4-dpo")

    # Modes
    ap.add_argument("--smoke_test", action="store_true",
                    help="只用 200 个 pair + 50 步快速验证")
    ap.add_argument("--no_gradient_checkpointing", action="store_true")

    return ap.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 解析 ckpt 路径
    args.stage1_ckpt = resolve_stage1_ckpt(args.stage1_ckpt)
    args.stage2_ckpt = resolve_stage2_ckpt(args.stage2_ckpt)
    print(f"[ckpt] stage1 = {args.stage1_ckpt}")
    print(f"[ckpt] stage2 = {args.stage2_ckpt}")

    # tokenizer / image_processor
    print(f"[load] tokenizer + image_processor from {args.processor_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.processor_dir)
    image_processor = AutoImageProcessor.from_pretrained(args.processor_dir)

    # 拿 num_image_tokens
    print(f"[load] config 用以确定 num_image_tokens")
    tmp_model = LlavaForConditionalGeneration.from_pretrained(
        args.stage1_ckpt, torch_dtype=torch.bfloat16,
    )
    num_image_tokens = compute_num_image_tokens(tmp_model.config)
    del tmp_model
    print(f"[model] num_image_tokens = {num_image_tokens}")

    # GPU 检查
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"[GPU] {props.name}, {props.total_memory / 1e9:.1f}GB")
    else:
        print("[GPU] no CUDA")

    # 模型 (active model with LoRA, can disable_adapter for reference)
    model, proj_module = load_stage2_v2_model(
        args.stage1_ckpt, args.stage2_ckpt, dtype=torch.bfloat16,
    )

    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("[mem] gradient_checkpointing: ON")
    else:
        print("[mem] gradient_checkpointing: OFF (Blackwell 大显存可关掉换速度)")

    # 数据
    chat_builder = DPOChatBuilder(tokenizer, num_image_tokens)
    limit = 200 if args.smoke_test else (args.n_dpo_samples if args.n_dpo_samples > 0 else None)
    train_dataset = DPOPreferenceDataset(
        hf_dataset_dir=args.dpo_data_dir,
        chat_builder=chat_builder,
        image_processor=image_processor,
        max_len=args.max_len,
        limit=limit,
    )
    print(f"[data] DPO 训练 pair 数: {len(train_dataset)}")

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    collator = DPOCollator(pad_token_id=pad_id)

    # TrainingArguments
    max_steps = 50 if args.smoke_test else -1
    save_strategy = "no" if args.smoke_test else "steps"

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
        gradient_checkpointing=not args.no_gradient_checkpointing,
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False} if not args.no_gradient_checkpointing else None
        ),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_strategy=save_strategy,
        report_to=args.report_to,
        run_name=f"{args.run_name}{'-smoke' if args.smoke_test else ''}",
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        optim="adamw_torch",
    )

    # Trainer
    callbacks = []
    if not args.smoke_test:
        callbacks.append(ProjectorSaverCallback(proj_module))

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        callbacks=callbacks,
        beta=args.beta,
    )
    try:
        trainer = DPOTrainerCustom(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = DPOTrainerCustom(**trainer_kwargs, tokenizer=tokenizer)

    # 训练
    print(f"\n=== 启动 DPO 训练 ===")
    print(f"  beta = {args.beta}")
    print(f"  LR = {args.lr}")
    print(f"  effective batch = {args.batch_size} × {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"  total samples = {len(train_dataset)}")
    print(f"  total iters ≈ {len(train_dataset) // (args.batch_size * args.grad_accum)}")
    print()
    trainer.train()

    # 保存
    if not args.smoke_test:
        print(f"\n[save] 最终 LoRA adapter + projector → {output_dir}")
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        image_processor.save_pretrained(str(output_dir))
        sd = {k: v.detach().cpu().contiguous() for k, v in proj_module.state_dict().items()}
        save_file(sd, output_dir / "multi_modal_projector.safetensors")

    print("Done.")


if __name__ == "__main__":
    main()
