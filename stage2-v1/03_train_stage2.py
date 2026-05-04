"""Stage 2 多任务预训练 — LoRA on LLM + projector 全参，ViT 冻结。

继续在 Stage 1 训好的 projector 上学习：
  - 多任务格式（VQA / grounding / 长 caption）
  - Qwen2.5 chat template（user/assistant 角色）
  - bbox 输出格式 <box>(x,y),(x,y)</box>

可训练参数：
  - LoRA on LLM 的 q/k/v/o + gate/up/down (~10M)
  - multi_modal_projector（含 LayerNorm, ~5M）
  - 共 ~15M (vs Stage 1 的 5M)

加载 Stage 1 ckpt → 装 ProjectorWithNorm → 应用 LoRA → 训练。

== 用法 ==

  烟雾测试（10 分钟内，验证 pipeline 通畅）:
    python stage2-v1/03_train_stage2.py --smoke_test \\
        --stage1_ckpt /content/drive/MyDrive/qwenvl3/stage1_ckpt_v3/checkpoint-XXXX \\
        --processor_dir /content/drive/MyDrive/qwenvl3/stage1_init \\
        --stage2_data_root /content/drive/MyDrive/qwenvl3/data/stage2 \\
        --output_dir /content/drive/MyDrive/qwenvl3/stage2_smoke \\
        --report_to none

  正式训练（~6h on 40GB A100, batch 4 + grad_accum 8 = effective 32）:
    python stage2-v1/03_train_stage2.py \\
        --stage1_ckpt /content/drive/MyDrive/qwenvl3/stage1_ckpt_v3/checkpoint-XXXX \\
        --processor_dir /content/drive/MyDrive/qwenvl3/stage1_init \\
        --stage2_data_root /content/drive/MyDrive/qwenvl3/data/stage2 \\
        --output_dir /content/drive/MyDrive/qwenvl3/stage2_ckpt
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def _ensure_torchao_compat():
    """PEFT 的 dispatch_torchao 检查 torchao 版本，0.16 以下硬 ImportError 不 fallback。
    Colab 预装 torchao 0.10.0，会中断 get_peft_model。我们没用 torchao，卸了让 PEFT 走标准 Linear 路径。

    必须在 import peft 之前调用（确保 PEFT 内部检查时 torchao 已不可见）。
    """
    import importlib.util
    if importlib.util.find_spec("torchao") is None:
        return
    try:
        import torchao
        from packaging import version
        if version.parse(getattr(torchao, "__version__", "0.0.0")) >= version.parse("0.16.0"):
            return  # 版本够新，留着
    except Exception:
        pass
    print("[setup] 检测到旧版 torchao（与 PEFT 不兼容），卸载中...")
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"],
        check=False, capture_output=True,
    )
    # 清掉已 import 的 torchao 模块（如果有）
    for mod in list(sys.modules):
        if mod.startswith("torchao"):
            del sys.modules[mod]


_ensure_torchao_compat()

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from transformers import (  # noqa: E402
    AutoImageProcessor,
    AutoTokenizer,
    LlavaForConditionalGeneration,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# stage1/_common.py: ProjectorWithNorm + install_custom_projector + get_components
sys.path.insert(0, str(Path(__file__).parent.parent / "stage1"))
from _common import get_components, install_custom_projector  # noqa: E402

# stage2-v1/_common2.py: 数据 + chat 格式 + LoRA helpers
sys.path.insert(0, str(Path(__file__).parent))
from _common2 import (  # noqa: E402
    ChatFormatter,
    CocoZipLoader,
    LlavaInstructTaskDataset,
    MultitaskCollator,
    MultitaskTrainingDataset,
    RefCOCOTaskDataset,
    ShareGPT4VTaskDataset,
    find_lm_lora_targets,
)


def compute_num_image_tokens(config):
    vc = config.vision_config
    n = (vc.image_size // vc.patch_size) ** 2
    if config.vision_feature_select_strategy == "default":
        n -= 1
    return n


# ============================================================================
# Projector 单独保存（PEFT save 默认只存 adapter，不存 projector）
# ============================================================================

class ProjectorSaverCallback(TrainerCallback):
    """每次 Trainer 触发 save 时，额外把 projector 的 state_dict 单独存到该 checkpoint 目录。

    为什么要这么做：
      - PEFT 的 save_pretrained 只保存 LoRA adapter
      - Stage 2 的 projector 是被 unfreeze 训练的（modules_to_save 也可，但配合
        ProjectorWithNorm 自定义结构有兼容性问题），用 callback 自管最稳
      - 加载时：先 install_custom_projector 装结构，再从 multi_modal_projector.safetensors 加载权重
    """
    def __init__(self, projector_module):
        self.projector = projector_module

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        target = ckpt_dir / "multi_modal_projector.safetensors"
        # 把 weights 移到 CPU 再存，避免 cuda tensor 引发 safetensors 错误
        sd = {k: v.detach().cpu().contiguous() for k, v in self.projector.state_dict().items()}
        save_file(sd, target)


# ============================================================================
# 数据集装配
# ============================================================================

def build_task_datasets(args, coco_loader: CocoZipLoader):
    """按 args 中各任务的样本数，组装多任务 dataset 列表。

    返回 list of (task_name, dataset)。后续给 ConcatDataset / MultitaskTrainingDataset 用。
    """
    data_root = Path(args.stage2_data_root)
    task_dsets = []

    # ---- LLaVA-Instruct ----
    json_path = data_root / "llava_instruct" / "llava_instruct_150k.json"
    if json_path.exists() and args.n_llava_instruct > 0:
        ds = LlavaInstructTaskDataset(
            json_path, coco_loader, limit=args.n_llava_instruct,
        )
        task_dsets.append(("llava_instruct", ds))
        print(f"[task] llava_instruct: {len(ds)} 样本")
    else:
        print(f"[skip] llava_instruct (json 不存在 或 n=0)")

    # ---- RefCOCO ----
    rc_dir = data_root / "refcoco"
    if rc_dir.exists() and any(rc_dir.iterdir()) and args.n_refcoco > 0:
        try:
            from datasets import load_dataset
            for split in ["train", "validation", "val"]:
                try:
                    hf_ds = load_dataset(str(rc_dir), split=split, trust_remote_code=True)
                    break
                except Exception:
                    hf_ds = None
            if hf_ds is None:
                print("[skip] refcoco: 找不到 split")
            else:
                ds = RefCOCOTaskDataset(hf_ds, coco_loader=coco_loader,
                                        limit=args.n_refcoco)
                task_dsets.append(("refcoco", ds))
                print(f"[task] refcoco: {len(ds)} 样本")
        except Exception as e:
            print(f"[skip] refcoco load 失败: {e}")
    else:
        print(f"[skip] refcoco")

    # ---- ShareGPT4V ----
    # 显式偏好 sharegpt4v_instruct (GPT-4V 详细长 caption, 216 词)，
    # 而不是 sorted()[0] 默认拿到的 share-captioner（155 词，字母序在前）。
    # 改这一处后所有未来训练都会用更高质量的数据。
    SHAREGPT4V_PREFERENCE = [
        "sharegpt4v_instruct_gpt4-vision_cap100k.json",
        "share-captioner_coco_lcs_sam_1246k_1107.json",
    ]
    sg_dir = data_root / "sharegpt4v"
    if sg_dir.exists() and args.n_sharegpt4v > 0:
        json_files = sorted(sg_dir.rglob("*.json"))
        chosen = None
        for preferred_name in SHAREGPT4V_PREFERENCE:
            for f in json_files:
                if f.name == preferred_name:
                    chosen = f
                    break
            if chosen is not None:
                break
        if chosen is None and json_files:
            chosen = json_files[0]
            print(f"[warn] sharegpt4v: 优先列表里的文件都没找到，fallback 到 {chosen.name}")
        if chosen is not None:
            ds = ShareGPT4VTaskDataset(chosen, coco_loader, limit=args.n_sharegpt4v)
            if len(ds) > 0:
                task_dsets.append(("sharegpt4v", ds))
                print(f"[task] sharegpt4v: {len(ds)} 样本（COCO 子集，文件={chosen.name}）")
            else:
                print(f"[skip] sharegpt4v: {chosen.name} 过滤后 0 样本")
        else:
            print("[skip] sharegpt4v: 找不到 json")
    else:
        print(f"[skip] sharegpt4v")

    if not task_dsets:
        raise RuntimeError("一个任务都没成功加载，无法训练")
    return task_dsets


# ============================================================================
# 模型与 LoRA
# ============================================================================

def setup_model_for_stage2(args, num_image_tokens):
    """加载 Stage 1 ckpt → 装 ProjectorWithNorm → 应用 LoRA → 调整 requires_grad。"""
    print(f"[model] 加载 Stage 1 ckpt: {args.stage1_ckpt}")
    model = LlavaForConditionalGeneration.from_pretrained(
        args.stage1_ckpt,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    print("[model] 安装 ProjectorWithNorm + 加载 Stage 1 训练好的 projector 权重")
    install_custom_projector(model, init_dir=args.stage1_ckpt, dtype=torch.bfloat16)

    # 拿到内部组件用于精确 freeze / unfreeze
    _, vt_module, proj_module = get_components(model)

    # 冻结 ViT
    for p in vt_module.parameters():
        p.requires_grad = False
    print(f"[model] vision_tower 冻结 ({sum(p.numel() for p in vt_module.parameters())/1e6:.1f}M)")

    # 应用 LoRA 到 LLM
    lora_targets = find_lm_lora_targets(model)
    print(f"[model] 找到 {len(lora_targets)} 个 LoRA target Linear 层（仅 LLM 内）")
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=lora_targets,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    print(f"[model] LoRA 应用完成 (rank={args.lora_r}, alpha={args.lora_alpha})")

    # PEFT 默认会冻结所有非 LoRA 参数，包括我们的 projector
    # 必须手动 unfreeze projector
    for name, param in model.named_parameters():
        if "multi_modal_projector" in name:
            param.requires_grad = True

    # 统计可训练参数
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[model] 可训练参数: {trainable/1e6:.1f}M / 总: {total/1e9:.2f}B "
          f"(比例: {trainable/total*100:.2f}%)")

    return model, proj_module


# ============================================================================
# 显存预估
# ============================================================================

def print_gpu_estimate(args, num_image_tokens):
    if not torch.cuda.is_available():
        print("[GPU] no CUDA")
        return
    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / 1e9
    print(f"[GPU] {props.name}, {total_gb:.1f}GB")
    seq = num_image_tokens + 250
    logits_gb = args.batch_size * seq * 152064 * 4 / 1e9
    print(f"[mem] 估算 logits fp32: {logits_gb:.2f}GB"
          f"（batch={args.batch_size}, seq~{seq}, vocab=152K）")
    if total_gb < 50 and args.batch_size > 4:
        print(f"[warn] 40GB 卡上 batch>4 + 长序列可能 OOM。建议 --batch_size 4 --grad_accum 8")


# ============================================================================
# 找最近的 checkpoint 用于续训
# ============================================================================

def find_latest_ckpt(output_dir: Path):
    if not output_dir.exists():
        return None
    ckpts = sorted(
        (p for p in output_dir.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-")[1]),
    )
    return ckpts[-1] if ckpts else None


# ============================================================================
# Stage 1 checkpoint 自动 resolve（应对 save_total_limit 删旧 ckpt 的场景）
# ============================================================================

def resolve_stage1_ckpt(ckpt_arg: str) -> str:
    """检查 stage1_ckpt 是否真实存在且包含 model.safetensors。

    Stage 1 训练期间 save_total_limit=2 会持续删旧 checkpoint，用户脚本里写的
    `--stage1_ckpt .../checkpoint-XXXX` 经常过期。这个 helper 仿照 baseline_eval
    的做法：路径是 checkpoint-NNNN 形式时自动 fallback 到同目录下最新可用的。
    """
    ckpt_path = Path(ckpt_arg)
    sft = ckpt_path / "model.safetensors"

    if sft.exists() and sft.stat().st_size > 1e9:
        return str(ckpt_path)

    # 不可用：找同目录最新可用 checkpoint
    if ckpt_path.parent.exists() and ckpt_path.name.startswith("checkpoint-"):
        all_ckpts = sorted(
            (p for p in ckpt_path.parent.glob("checkpoint-*") if p.is_dir()),
            key=lambda p: int(p.name.split("-")[1]),
            reverse=True,
        )
        for c in all_ckpts:
            if (c / "model.safetensors").exists() \
               and (c / "model.safetensors").stat().st_size > 1e9:
                print(f"[warn] {ckpt_path.name} 不可用（已被 save_total_limit 删除？）")
                print(f"       自动 fallback 到 {c.name}")
                return str(c)

    raise FileNotFoundError(
        f"找不到可用 Stage 1 checkpoint。{ckpt_path} 下无 model.safetensors，"
        f"且同目录其他 checkpoint 也不可用。可用 ls 看一下 "
        f"{ckpt_path.parent} 现有哪些 checkpoint-*。"
    )


# ============================================================================
# Main
# ============================================================================

def parse_args():
    ap = argparse.ArgumentParser()
    # 路径
    ap.add_argument("--stage1_ckpt", required=True)
    ap.add_argument("--processor_dir", default=None,
                    help="tokenizer + image_processor 目录；未指定则从 stage1_ckpt 同目录找")
    ap.add_argument("--stage2_data_root", required=True)
    ap.add_argument("--output_dir", required=True)

    # 数据混合配比
    ap.add_argument("--n_llava_instruct", type=int, default=150_000)
    ap.add_argument("--n_refcoco",        type=int, default=50_000)
    ap.add_argument("--n_sharegpt4v",     type=int, default=100_000)

    # LoRA
    ap.add_argument("--lora_r",       type=int, default=16)
    ap.add_argument("--lora_alpha",   type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # 训练超参
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--num_epochs", type=int, default=1)
    ap.add_argument("--lr",         type=float, default=2e-4)  # LoRA 标准
    ap.add_argument("--projector_lr_mult", type=float, default=0.1,
                    help="projector 用 lr * 这个倍数（LoRA lr 太高会让 projector 跑飞）")
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--max_len",   type=int, default=1500)

    # Logging / saving
    ap.add_argument("--save_steps",    type=int, default=500)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_total_limit", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=2,
                    help="DataLoader worker 数。Colab T4 一般 vCPU 只有 2，A100 有 12。")
    ap.add_argument("--report_to",   default="wandb",
                    choices=["wandb", "none", "tensorboard"])

    # Modes
    ap.add_argument("--smoke_test", action="store_true",
                    help="只用 200 样本 + 50 步快速验证流程")
    return ap.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 自动 resolve Stage 1 ckpt（save_total_limit 可能已删除指定的 checkpoint-XXXX）
    args.stage1_ckpt = resolve_stage1_ckpt(args.stage1_ckpt)

    # ----- tokenizer / image_processor -----
    proc_dir = args.processor_dir or args.stage1_ckpt
    print(f"[load] tokenizer + image_processor from {proc_dir}")
    tokenizer = AutoTokenizer.from_pretrained(proc_dir)
    image_processor = AutoImageProcessor.from_pretrained(proc_dir)

    # ----- model + LoRA -----
    # 先加载 model 拿 num_image_tokens
    print(f"[load] 模型 config 用以确定 num_image_tokens")
    tmp_model = LlavaForConditionalGeneration.from_pretrained(
        args.stage1_ckpt, torch_dtype=torch.bfloat16
    )
    num_image_tokens = compute_num_image_tokens(tmp_model.config)
    del tmp_model
    print(f"[model] num_image_tokens = {num_image_tokens}")

    print_gpu_estimate(args, num_image_tokens)

    model, proj_module = setup_model_for_stage2(args, num_image_tokens)
    model.gradient_checkpointing_enable()

    # ----- 数据 -----
    coco_zip = Path(args.stage2_data_root) / "coco" / "train2017.zip"
    if not coco_zip.exists():
        raise FileNotFoundError(f"COCO zip 不存在: {coco_zip}")
    coco_loader = CocoZipLoader(coco_zip)
    print(f"[data] COCO zip: {coco_zip.stat().st_size / 1e9:.1f}GB OK")

    # 烟雾测试时大幅缩小数据
    if args.smoke_test:
        args.n_llava_instruct = min(args.n_llava_instruct, 100)
        args.n_refcoco = min(args.n_refcoco, 50)
        args.n_sharegpt4v = min(args.n_sharegpt4v, 50)

    task_dsets = build_task_datasets(args, coco_loader)
    chat_formatter = ChatFormatter(tokenizer, num_image_tokens)
    train_dataset = MultitaskTrainingDataset(
        task_dsets, chat_formatter, image_processor, max_len=args.max_len,
    )
    print(f"[data] 总训练样本数: {len(train_dataset)}")

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    collator = MultitaskCollator(pad_token_id=pad_id)

    # ----- TrainingArguments -----
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
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_strategy=save_strategy,
        report_to=args.report_to,
        run_name=f"stage2{'-smoke' if args.smoke_test else ''}",
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        optim="adamw_torch",
    )

    # ----- Trainer -----
    callbacks = []
    if not args.smoke_test:
        callbacks.append(ProjectorSaverCallback(proj_module))

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )
    try:
        trainer = Trainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = Trainer(**trainer_kwargs, tokenizer=tokenizer)

    # ----- 续训检查 -----
    last_ckpt = find_latest_ckpt(output_dir)
    if last_ckpt and not args.smoke_test:
        print(f"[resume] 从 {last_ckpt} 续训")
        trainer.train(resume_from_checkpoint=str(last_ckpt))
    else:
        trainer.train()

    # ----- 最终保存 -----
    if not args.smoke_test:
        print(f"[save] 最终 LoRA adapter + projector → {output_dir}")
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        image_processor.save_pretrained(str(output_dir))
        # 显式存一份 final projector
        sd = {k: v.detach().cpu().contiguous() for k, v in proj_module.state_dict().items()}
        save_file(sd, output_dir / "multi_modal_projector.safetensors")

    print("Done.")


if __name__ == "__main__":
    main()
