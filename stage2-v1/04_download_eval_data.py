"""Stage 2 OOD 评测数据下载 — 跑完这个再跑 04_eval_stage2.py。

用 huggingface_hub.snapshot_download 缓存到 Drive 持久存储；断点续传。
建议在训练空闲时提前跑（不会占 GPU）。

== 下载内容 ==
  POPE       ~ 0.5GB   ⭐ 必下：幻觉测试，9K 是非题，自动评
  MME        ~ 0.6GB   ⭐ 必下：14 sub-task 综合评测，自动评
  NoCaps     ~ 0.7GB   ⭐ 推荐：长 caption OOD，4.5K 样本
  VQAv2 sub  ~ 3-5GB   ⭐⭐ 推荐：标准 VQA benchmark，但比较大
                           （--vqav2_subset val_lite 用 lite 版会小很多）

总量约 5-7GB，看具体启用项。RefCOCO 已经在 Stage 2 训练目录下，无需再下。

== 用法 ==

  默认全下：
    python stage2-v1/04_download_eval_data.py \\
        --eval_root /content/drive/MyDrive/qwenvl3/data/eval

  只下小的 (POPE + MME)，跳过 VQAv2 / NoCaps：
    python stage2-v1/04_download_eval_data.py \\
        --eval_root /content/drive/MyDrive/qwenvl3/data/eval \\
        --skip vqav2 nocaps

  Drive 空间紧张：
    python stage2-v1/04_download_eval_data.py ... --skip vqav2
"""
import argparse
import os
import shutil
from pathlib import Path


# 数据集映射：local_dir -> (HF repo, repo_type, ~size_MB, 描述)
DATASETS = {
    "pope":   ("lmms-lab/POPE",   "dataset",  500, "幻觉测试 (Yes/No, 9K)"),
    "mme":    ("lmms-lab/MME",    "dataset",  600, "综合 14 sub-task (~2400 题)"),
    "nocaps": ("lmms-lab/NoCaps", "dataset",  700, "OOD 长 caption (4.5K)"),
    "vqav2":  ("lmms-lab/VQAv2",  "dataset", 4500, "标准 VQA val（大，可 skip）"),
}


def fmt_size(bytes_):
    if bytes_ > 1e9:
        return f"{bytes_ / 1e9:.1f}GB"
    return f"{bytes_ / 1e6:.0f}MB"


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def download_one(name, repo_id, repo_type, target_dir: Path, hf_token=None):
    """snapshot_download 到 target_dir。如果已存在且非空，跳过。"""
    from huggingface_hub import snapshot_download

    target_dir.mkdir(parents=True, exist_ok=True)
    existing = dir_size(target_dir)
    if existing > 100 * 1024 * 1024:  # > 100MB 算已下过
        print(f"  [skip] {name} 已下载 ({fmt_size(existing)} in {target_dir})")
        return

    print(f"  [download] {name} <- {repo_id}")
    print(f"             目标: {target_dir}")
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,  # Drive 上 symlink 行为不可靠
            token=hf_token,
            resume_download=True,
        )
        size = dir_size(target_dir)
        print(f"  [done] {name}: {fmt_size(size)}")
    except Exception as e:
        print(f"  [fail] {name}: {e}")
        # 部分下载残留有时会让后续 load_dataset 失败，但不删除以便重试断点续传
        raise


def verify_loadable(name, target_dir: Path):
    """用 datasets.load_dataset 试加载，确认可用（数据完整性快速检查）。"""
    try:
        from datasets import load_dataset
    except ImportError:
        print(f"  [skip verify] datasets 包未安装")
        return

    print(f"  [verify] {name}: 试 load_dataset...")
    try:
        # 不锁定具体 split，让 datasets 自己探测
        ds = load_dataset(str(target_dir), trust_remote_code=True)
        # 取第一个可用 split 的 size
        if hasattr(ds, "keys"):
            split_name = list(ds.keys())[0]
            print(f"            ✅ split={split_name} n={len(ds[split_name])}")
        else:
            print(f"            ✅ n={len(ds)}")
    except Exception as e:
        print(f"            ⚠️  加载失败: {e}")
        print(f"            (不要紧，eval 脚本里有 fallback 逻辑)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", required=True,
                    help="评测数据根目录，建议放 Drive: "
                         "/content/drive/MyDrive/qwenvl3/data/eval")
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=list(DATASETS.keys()),
                    help="跳过指定数据集（如 --skip vqav2 nocaps 省空间）")
    ap.add_argument("--no_verify", action="store_true",
                    help="下完不做 load_dataset 验证（更快）")
    ap.add_argument("--hf_token", default=None,
                    help="HF token（如需访问私有 repo；一般不用）")
    args = ap.parse_args()

    eval_root = Path(args.eval_root)
    eval_root.mkdir(parents=True, exist_ok=True)

    print(f"=== Stage 2 OOD 评测数据下载 ===")
    print(f"目标根目录: {eval_root}")
    print(f"已有空间检查: {shutil.disk_usage(eval_root).free / 1e9:.1f}GB free\n")

    # 估算总大小
    plan = [(k, v) for k, v in DATASETS.items() if k not in args.skip]
    total_mb = sum(v[2] for _, v in plan)
    print(f"准备下载: {len(plan)} 个数据集，约 {total_mb / 1024:.1f}GB\n")

    for name, (repo_id, repo_type, size_mb, desc) in plan:
        target = eval_root / name
        print(f"[{name}]  ({size_mb}MB · {desc})")
        try:
            download_one(name, repo_id, repo_type, target, args.hf_token)
            if not args.no_verify:
                verify_loadable(name, target)
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            print(f"  继续下个数据集...\n")
            continue
        print()

    # 总结
    print("=" * 60)
    print("=== 下载完成 ===")
    for name in DATASETS:
        d = eval_root / name
        if d.exists():
            print(f"  {name}: {fmt_size(dir_size(d))}  ({d})")
        else:
            print(f"  {name}: (跳过 / 未下载)")

    print("\n下一步:")
    print(f"  python stage2-v1/04_eval_stage2.py \\")
    print(f"      --stage2_ckpt /content/drive/MyDrive/qwenvl3/stage2_ckpt \\")
    print(f"      --stage1_ckpt /content/drive/MyDrive/qwenvl3/stage1_ckpt_v3 \\")
    print(f"      --eval_data_root {eval_root} \\")
    print(f"      --stage2_data_root /content/drive/MyDrive/qwenvl3/data/stage2 \\")
    print(f"      --stage1_data_root /content/drive/MyDrive/qwenvl3/data/llava-pretrain \\")
    print(f"      --out_dir /content/drive/MyDrive/qwenvl3/eval_stage2")


if __name__ == "__main__":
    main()
