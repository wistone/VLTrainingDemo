"""Stage 2-v2 (Phase 1+) 多任务训练。

跟 stage2/03_train_stage2.py 区别：
  ✨ 新增 RefCOCO+ / RefCOCOg 加入训练 mix
  ✨ 新增 TextVQA 加入训练 mix（首次接入，给 OCR 专项能力）
  ✨ 默认 LoRA r=64 (从 r=16 升级)，alpha=128 (保持 alpha/r=2)
  ✨ 默认 LLaVA-Instruct 削到 100K (从 150K 降)，平衡总训练时长

== 数据混合（Phase 1+ 默认配比）==
  LLaVA-Instruct:   100K  ━━━━━━━━━━━━━━━━ 24%
  RefCOCO:           50K  ━━━━━━━━ 12%
  RefCOCO+:          50K  ━━━━━━━━ 12%   ← 新增
  RefCOCOg:          80K  ━━━━━━━━━━━━ 19%   ← 新增
  ShareGPT4V:       100K  ━━━━━━━━━━━━━━━━ 24%
  TextVQA:           28K  ━━━━ 7%   ← 新增 (OCR 专项)
  Total:            408K  (vs 现在 300K, +36%)

  Grounding 占比:  44%   General VQA: 31%   Long Caption: 24%

== 预期训练时长 ==
  A100 (5.8 s/iter): ~21h
  H100 (~2.5 s/iter): ~9h    ⭐ 推荐

== 用法 ==

  烟雾测试（10 分钟，验证 pipeline 通畅）:
    python stage2-v2/03_train_stage2.py --smoke_test \\
        --stage1_ckpt /content/drive/MyDrive/qwenvl3/stage1_ckpt_v3/checkpoint-11500 \\
        --processor_dir /content/drive/MyDrive/qwenvl3/stage1_init \\
        --stage2_data_root /content/drive/MyDrive/qwenvl3/data/stage2 \\
        --output_dir /content/drive/MyDrive/qwenvl3/stage2_v2_smoke \\
        --report_to none

  正式训练（默认 Phase 1+ 配比）:
    python stage2-v2/03_train_stage2.py \\
        --stage1_ckpt /content/drive/MyDrive/qwenvl3/stage1_ckpt_v3/checkpoint-11500 \\
        --processor_dir /content/drive/MyDrive/qwenvl3/stage1_init \\
        --stage2_data_root /content/drive/MyDrive/qwenvl3/data/stage2 \\
        --output_dir /content/drive/MyDrive/qwenvl3/stage2_v2_ckpt
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def _ensure_torchao_compat():
    """同 v1：卸掉 Colab 预装的 torchao 0.10 让 PEFT 走标准路径。"""
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

# stage1/_common.py: ProjectorWithNorm + helpers
sys.path.insert(0, str(Path(__file__).parent.parent / "stage1"))
from _common import get_components, install_custom_projector  # noqa: E402

# stage2-v2/_common2.py: 含 v1 的所有 + TextVQATaskDataset
sys.path.insert(0, str(Path(__file__).parent))
from _common2 import (  # noqa: E402
    ChatFormatter,
    CocoZipLoader,
    LlavaInstructTaskDataset,
    MultitaskCollator,
    MultitaskTrainingDataset,
    RefCOCOTaskDataset,
    ShareGPT4VTaskDataset,
    TextVQATaskDataset,    # ⭐ 新增
    find_lm_lora_targets,
)


def compute_num_image_tokens(config):
    vc = config.vision_config
    n = (vc.image_size // vc.patch_size) ** 2
    if config.vision_feature_select_strategy == "default":
        n -= 1
    return n


# ============================================================================
# Projector 单独保存 callback — 与 v1 一致
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
# 数据集装配 — Phase 1+ 包含 6 个 task
# ============================================================================

def _try_load_hf_dataset(local_dir: Path, label: str,
                         splits=("train", "validation", "val", "test"),
                         require_train: bool = True):
    """通用：尝试从 local_dir 加载某个 split，**显式打印用了哪个**。

    require_train=True 时，如果只能 fallback 到 val/test 会打印警告 ——
    防止 lmms-lab/RefCOCO 那种 silent fallback 到 val（你以为在训 train，
    实际在训 eval set，会污染未来评测）。
    """
    from datasets import load_dataset

    for split in splits:
        try:
            ds = load_dataset(str(local_dir), split=split, trust_remote_code=True)
            if split != "train" and require_train:
                print(f"  ⚠️  [{label}] 没找到 train split！实际加载了 split={split} "
                      f"(n={len(ds)})。这是 eval split，不应用于训练！")
                print(f"      建议：换有 train split 的数据源（如 jxu124/refcoco）。")
            else:
                print(f"  ✅ [{label}] 加载 split={split} (n={len(ds)})")
            return ds, split
        except Exception:
            continue

    # 都失败：尝试不指定 split 加载第一个
    try:
        ds_dict = load_dataset(str(local_dir), trust_remote_code=True)
        first = list(ds_dict.keys())[0]
        ds = ds_dict[first]
        if first != "train" and require_train:
            print(f"  ⚠️  [{label}] 只找到 split={first} (n={len(ds)})。"
                  f"这不是 train，谨慎使用。")
        else:
            print(f"  ✅ [{label}] 加载 split={first} (n={len(ds)})")
        return ds, first
    except Exception as e:
        print(f"  ❌ [{label}] 加载失败: {e}")
        return None, None


def build_task_datasets(args, coco_loader: CocoZipLoader):
    """组装 Phase 1+ 的 6 个 task dataset。"""
    data_root = Path(args.stage2_data_root)
    task_dsets = []

    # ---- 1. LLaVA-Instruct (100K, vs v1 150K 削过) ----
    json_path = data_root / "llava_instruct" / "llava_instruct_150k.json"
    if json_path.exists() and args.n_llava_instruct > 0:
        ds = LlavaInstructTaskDataset(json_path, coco_loader, limit=args.n_llava_instruct)
        task_dsets.append(("llava_instruct", ds))
        print(f"[task] llava_instruct: {len(ds)} 样本")
    else:
        print(f"[skip] llava_instruct (json 不存在 或 n=0)")

    # ---- 2. RefCOCO train (jxu124, 42K) ----
    rc_dir = data_root / "refcoco_train"   # 改用 _train 后缀目录（jxu124，含 train split）
    if rc_dir.exists() and any(rc_dir.iterdir()) and args.n_refcoco > 0:
        hf_ds, split = _try_load_hf_dataset(rc_dir, "refcoco")
        if hf_ds is None:
            print("[skip] refcoco: 加载失败")
        else:
            ds = RefCOCOTaskDataset(
                hf_ds, coco_loader=coco_loader,
                limit=args.n_refcoco, source_name="refcoco",
                bbox_format="xyxy",       # jxu124 用 xyxy 像素
                random_caption=True,      # 训练时随机选 caption (3× 数据增强)
            )
            task_dsets.append(("refcoco", ds))
            print(f"[task] refcoco: {len(ds)} 样本")
    else:
        print(f"[skip] refcoco (目录 {rc_dir} 不存在 或 n=0)")

    # ---- 3. ⭐ RefCOCO+ train (尝试 jxu124 备选) ----
    rcp_dir = data_root / "refcoco_plus_train"
    if rcp_dir.exists() and any(rcp_dir.iterdir()) and args.n_refcoco_plus > 0:
        hf_ds, split = _try_load_hf_dataset(rcp_dir, "refcoco_plus")
        if hf_ds is None:
            print("[skip] refcoco_plus: 加载失败")
        else:
            # bbox_format 看实际 repo 决定。jxu124 系列用 xyxy；如果是其他源，
            # 可能还得手动调整。这里默认 xyxy + fallback 检查（_extract_bbox 会
            # 检测是否归一化）。
            ds = RefCOCOTaskDataset(
                hf_ds, coco_loader=coco_loader,
                limit=args.n_refcoco_plus, source_name="refcoco_plus",
                bbox_format="xyxy",
                random_caption=True,
            )
            task_dsets.append(("refcoco_plus", ds))
            print(f"[task] refcoco_plus: {len(ds)} 样本  ⭐ Phase 1+ 新增")
    else:
        print(f"[skip] refcoco_plus (目录 {rcp_dir} 不存在 或 n=0)")
        print(f"        如果 RefCOCO+ HF 没有 train split repo，跳过即可，"
              f"Phase 1+ 用 RefCOCO + RefCOCOg 训也成立。")

    # ---- 4. ⭐ RefCOCOg train (jxu124, 42K) ----
    rcg_dir = data_root / "refcocog_train"
    if rcg_dir.exists() and any(rcg_dir.iterdir()) and args.n_refcocog > 0:
        hf_ds, split = _try_load_hf_dataset(rcg_dir, "refcocog")
        if hf_ds is None:
            print("[skip] refcocog: 加载失败")
        else:
            ds = RefCOCOTaskDataset(
                hf_ds, coco_loader=coco_loader,
                limit=args.n_refcocog, source_name="refcocog",
                bbox_format="xyxy",
                random_caption=True,
            )
            task_dsets.append(("refcocog", ds))
            print(f"[task] refcocog: {len(ds)} 样本  ⭐ Phase 1+ 新增")
    else:
        print(f"[skip] refcocog (目录 {rcg_dir} 不存在 或 n=0)")

    # ---- 5. ShareGPT4V (100K，跟 v1 一样) ----
    sg_dir = data_root / "sharegpt4v"
    if sg_dir.exists() and args.n_sharegpt4v > 0:
        json_files = sorted(sg_dir.rglob("*.json"))
        if json_files:
            ds = ShareGPT4VTaskDataset(json_files[0], coco_loader, limit=args.n_sharegpt4v)
            if len(ds) > 0:
                task_dsets.append(("sharegpt4v", ds))
                print(f"[task] sharegpt4v: {len(ds)} 样本（COCO 子集）")
            else:
                print("[skip] sharegpt4v: 过滤后 0 样本")
        else:
            print("[skip] sharegpt4v: 找不到 json")
    else:
        print(f"[skip] sharegpt4v")

    # ---- 6. ⭐ TextVQA (28K, 新增) ----
    tv_dir = data_root / "textvqa"
    if tv_dir.exists() and any(tv_dir.iterdir()) and args.n_textvqa > 0:
        hf_ds, split = _try_load_hf_dataset(tv_dir, "textvqa")
        if hf_ds is None:
            print("[skip] textvqa: 加载失败")
        else:
            ds = TextVQATaskDataset(
                hf_ds, limit=args.n_textvqa,
                min_consensus=args.textvqa_min_consensus,
            )
            task_dsets.append(("textvqa", ds))
            print(f"[task] textvqa: {len(ds)} 样本  ⭐ Phase 1+ 新增 (OCR 专项)")
    else:
        print(f"[skip] textvqa (目录不存在 或 n=0)")

    if not task_dsets:
        raise RuntimeError("一个任务都没成功加载，无法训练")

    # 打印总览
    print(f"\n[mix] 总数据组成:")
    total = sum(len(d) for _, d in task_dsets)
    for name, d in task_dsets:
        print(f"      {name:18s} {len(d):>7,d}  ({len(d)/total:.1%})")
    print(f"      {'TOTAL':18s} {total:>7,d}")

    return task_dsets


# ============================================================================
# 模型与 LoRA — 与 v1 一致（仅默认 r/alpha 不同，参数化）
# ============================================================================

def setup_model_for_stage2(args, num_image_tokens):
    print(f"[model] 加载 Stage 1 ckpt: {args.stage1_ckpt}")
    model = LlavaForConditionalGeneration.from_pretrained(
        args.stage1_ckpt,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    print("[model] 安装 ProjectorWithNorm + 加载 Stage 1 训练好的 projector 权重")
    install_custom_projector(model, init_dir=args.stage1_ckpt, dtype=torch.bfloat16)

    _, vt_module, proj_module = get_components(model)

    for p in vt_module.parameters():
        p.requires_grad = False
    print(f"[model] vision_tower 冻结 ({sum(p.numel() for p in vt_module.parameters())/1e6:.1f}M)")

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

    for name, param in model.named_parameters():
        if "multi_modal_projector" in name:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[model] 可训练参数: {trainable/1e6:.1f}M / 总: {total/1e9:.2f}B "
          f"(比例: {trainable/total*100:.2f}%)")

    return model, proj_module


# ============================================================================
# 显存预估 + ckpt resolve — 与 v1 一致
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


def find_latest_ckpt(output_dir: Path):
    if not output_dir.exists():
        return None
    ckpts = sorted(
        (p for p in output_dir.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-")[1]),
    )
    return ckpts[-1] if ckpts else None


def resolve_stage1_ckpt(ckpt_arg: str) -> str:
    ckpt_path = Path(ckpt_arg)
    sft = ckpt_path / "model.safetensors"
    if sft.exists() and sft.stat().st_size > 1e9:
        return str(ckpt_path)
    if ckpt_path.parent.exists() and ckpt_path.name.startswith("checkpoint-"):
        all_ckpts = sorted(
            (p for p in ckpt_path.parent.glob("checkpoint-*") if p.is_dir()),
            key=lambda p: int(p.name.split("-")[1]),
            reverse=True,
        )
        for c in all_ckpts:
            if (c / "model.safetensors").exists() \
               and (c / "model.safetensors").stat().st_size > 1e9:
                print(f"[warn] {ckpt_path.name} 不可用，自动 fallback 到 {c.name}")
                return str(c)
    raise FileNotFoundError(
        f"找不到可用 Stage 1 checkpoint。{ckpt_path} 下无完整 model.safetensors。"
    )


# ============================================================================
# Main
# ============================================================================

def parse_args():
    ap = argparse.ArgumentParser()
    # 路径
    ap.add_argument("--stage1_ckpt", required=True)
    ap.add_argument("--processor_dir", default=None)
    ap.add_argument("--stage2_data_root", required=True)
    ap.add_argument("--output_dir", required=True)

    # 数据混合配比（Phase 1+ 默认）
    ap.add_argument("--n_llava_instruct", type=int, default=100_000,
                    help="Phase 1+ 默认削到 100K（v1 是 150K）")
    ap.add_argument("--n_refcoco",        type=int, default=50_000)
    ap.add_argument("--n_refcoco_plus",   type=int, default=50_000,
                    help="⭐ Phase 1+ 新增 RefCOCO+")
    ap.add_argument("--n_refcocog",       type=int, default=80_000,
                    help="⭐ Phase 1+ 新增 RefCOCOg")
    ap.add_argument("--n_sharegpt4v",     type=int, default=100_000)
    ap.add_argument("--n_textvqa",        type=int, default=28_000,
                    help="⭐ Phase 1+ 新增 TextVQA (OCR)")
    ap.add_argument("--textvqa_min_consensus", type=int, default=3,
                    help="TextVQA 多数投票最少几人同意才接受（默认 3/10）")

    # LoRA (Phase 1+ 默认升级)
    ap.add_argument("--lora_r",       type=int, default=64,
                    help="Phase 1+ 默认 r=64（v1 是 16）")
    ap.add_argument("--lora_alpha",   type=int, default=128,
                    help="Phase 1+ 默认 alpha=128（保持 alpha/r=2）")
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # 训练超参
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--num_epochs", type=int, default=1)
    ap.add_argument("--lr",         type=float, default=2e-4)
    ap.add_argument("--projector_lr_mult", type=float, default=0.1)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--max_len",   type=int, default=1500)

    # Logging / saving
    ap.add_argument("--save_steps",    type=int, default=400,
                    help="Phase 1+ 默认 400（更密的 fallback 点，21h 训练更稳妥）")
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_total_limit", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--report_to",   default="wandb",
                    choices=["wandb", "none", "tensorboard"])
    ap.add_argument("--run_name", default="stage2-v2-phase1plus",
                    help="wandb run name")

    # Modes
    ap.add_argument("--smoke_test", action="store_true",
                    help="只用 200 样本 + 50 步快速验证流程")
    return ap.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args.stage1_ckpt = resolve_stage1_ckpt(args.stage1_ckpt)

    # tokenizer / image_processor
    proc_dir = args.processor_dir or args.stage1_ckpt
    print(f"[load] tokenizer + image_processor from {proc_dir}")
    tokenizer = AutoTokenizer.from_pretrained(proc_dir)
    image_processor = AutoImageProcessor.from_pretrained(proc_dir)

    # 拿 num_image_tokens
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

    # 数据
    coco_zip = Path(args.stage2_data_root) / "coco" / "train2017.zip"
    if not coco_zip.exists():
        raise FileNotFoundError(f"COCO zip 不存在: {coco_zip}")
    coco_loader = CocoZipLoader(coco_zip)
    print(f"[data] COCO zip: {coco_zip.stat().st_size / 1e9:.1f}GB OK")

    if args.smoke_test:
        # 烟雾测试时把所有 task 都缩到极小
        args.n_llava_instruct = min(args.n_llava_instruct, 60)
        args.n_refcoco = min(args.n_refcoco, 30)
        args.n_refcoco_plus = min(args.n_refcoco_plus, 30)
        args.n_refcocog = min(args.n_refcocog, 30)
        args.n_sharegpt4v = min(args.n_sharegpt4v, 30)
        args.n_textvqa = min(args.n_textvqa, 30)

    task_dsets = build_task_datasets(args, coco_loader)
    chat_formatter = ChatFormatter(tokenizer, num_image_tokens)
    train_dataset = MultitaskTrainingDataset(
        task_dsets, chat_formatter, image_processor, max_len=args.max_len,
    )
    print(f"[data] 总训练样本数: {len(train_dataset)}")

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    collator = MultitaskCollator(pad_token_id=pad_id)

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
        run_name=f"{args.run_name}{'-smoke' if args.smoke_test else ''}",
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        optim="adamw_torch",
    )

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

    last_ckpt = find_latest_ckpt(output_dir)
    if last_ckpt and not args.smoke_test:
        print(f"[resume] 从 {last_ckpt} 续训")
        trainer.train(resume_from_checkpoint=str(last_ckpt))
    else:
        trainer.train()

    if not args.smoke_test:
        print(f"[save] 最终 LoRA adapter + projector → {output_dir}")
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        image_processor.save_pretrained(str(output_dir))
        sd = {k: v.detach().cpu().contiguous() for k, v in proj_module.state_dict().items()}
        save_file(sd, output_dir / "multi_modal_projector.safetensors")

    print("Done.")


if __name__ == "__main__":
    main()
