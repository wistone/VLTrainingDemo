"""Stage 2-v2 (Phase 1+) 数据下载 — 在 v1 基础上补 RefCOCO+ 和 RefCOCOg。

跟 stage2/01_prepare_data.py 共享相同 DRIVE_ROOT，所以已经下过的数据
（LLaVA-Instruct / COCO / RefCOCO / TextVQA / ShareGPT4V）会被跳过，
只下新增的 RefCOCO+ 和 RefCOCOg。

== 数据集（Phase 1+ 训练用）==
  ✅ LLaVA-Instruct-150K       (~50MB)    — 已有
  ✅ COCO train2017 zip        (~18GB)    — 已有
  ✅ RefCOCO                                — 已有
  ⭐ RefCOCO+                              — 新增 (50K, 禁位置词，更难的指代)
  ⭐ RefCOCOg                              — 新增 (80K, 长描述指代)
  ✅ ShareGPT4V                            — 已有
  ✅ TextVQA                               — 已有 (Phase 1+ 首次接入训练)

== 用法 ==

  补下 Phase 1+ 新增的两个 (RefCOCO+/g)：
    python stage2-v2/01_prepare_data.py --only_phase1plus

  全量（包括 v1 已有的，但已下过的会自动跳过）：
    python stage2-v2/01_prepare_data.py
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

DRIVE_ROOT = Path(os.environ.get(
    "STAGE2_DATA_ROOT",
    "/content/drive/MyDrive/qwenvl3/data/stage2"
))


def fmt_time(s):
    return f"{s:.0f}s" if s < 60 else f"{s/60:.1f}min"


def fmt_size(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


# ============================================================================
# HF snapshot 下载
# ============================================================================

def hf_download(repo_id, repo_type, target_dir, label, patterns=None):
    from huggingface_hub import snapshot_download

    target_dir.mkdir(parents=True, exist_ok=True)
    existing = sum(f.stat().st_size for f in target_dir.rglob("*") if f.is_file())
    if existing > 10 * 1024 * 1024:
        print(f"  [skip] {label} 已存在 ({fmt_size(existing)} in {target_dir})")
        return True

    print(f"  [download] {label} from HF: {repo_id}")
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=str(target_dir),
            allow_patterns=patterns,
            max_workers=4,
        )
        size = sum(f.stat().st_size for f in target_dir.rglob("*") if f.is_file())
        print(f"  [done] {label}: {fmt_size(size)} → {target_dir}")
        return True
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        return False


def url_download(url, target_path, label, min_size_gb=1):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        size = target_path.stat().st_size
        if size > min_size_gb * 1024 * 1024 * 1024:
            print(f"  [skip] {label} 已下载完整 ({fmt_size(size)})")
            return True
        else:
            print(f"  [resume] {label} 当前 {fmt_size(size)}，断点续传")

    print(f"  [download] {label} from {url}")
    t0 = time.time()
    try:
        subprocess.run(
            ["wget", "-c", "--progress=dot:giga", "-O", str(target_path), url],
            check=True,
        )
        size = target_path.stat().st_size
        print(f"  [done] {label}: {fmt_size(size)} (耗时 {fmt_time(time.time() - t0)})")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [FAIL] {label}: wget 失败 {e}")
        return False


# ============================================================================
# 各任务下载函数（v1 已有的就直接复用 v1 的逻辑/路径）
# ============================================================================

def task_llava_instruct():
    return hf_download(
        repo_id="liuhaotian/LLaVA-Instruct-150K",
        repo_type="dataset",
        patterns=["*.json"],
        target_dir=DRIVE_ROOT / "llava_instruct",
        label="LLaVA-Instruct-150K (json)",
    )


def task_coco_images():
    return url_download(
        url="http://images.cocodataset.org/zips/train2017.zip",
        target_path=DRIVE_ROOT / "coco" / "train2017.zip",
        label="COCO train2017 (~18GB)",
        min_size_gb=18,
    )


def task_textvqa():
    """TextVQA — Phase 1+ 首次接入训练 (之前 v1 下过但只用于 eval)。"""
    candidates = [
        ("lmms-lab/textvqa", "dataset"),
        ("facebook/textvqa", "dataset"),
    ]
    for repo_id, rtype in candidates:
        ok = hf_download(
            repo_id=repo_id,
            repo_type=rtype,
            target_dir=DRIVE_ROOT / "textvqa",
            label=f"TextVQA (尝试 {repo_id})",
        )
        if ok:
            return True
    return False


def task_refcoco():
    for repo_id in ["lmms-lab/RefCOCO", "jxu124/refcoco"]:
        ok = hf_download(
            repo_id=repo_id, repo_type="dataset",
            target_dir=DRIVE_ROOT / "refcoco",
            label=f"RefCOCO (尝试 {repo_id})",
        )
        if ok:
            return True
    return False


def task_refcoco_plus():
    """RefCOCO+ — 跟 RefCOCO 同源但禁位置词，更考验"理解物体属性"。

    HF repo 命名因 mirror 而异，按候选顺序尝试。
    """
    candidates = [
        "lmms-lab/RefCOCOplus",   # 最常见
        "lmms-lab/RefCOCO+",      # 带 + 号的写法
        "jxu124/refcoco-plus",
    ]
    for repo_id in candidates:
        ok = hf_download(
            repo_id=repo_id, repo_type="dataset",
            target_dir=DRIVE_ROOT / "refcoco_plus",
            label=f"RefCOCO+ (尝试 {repo_id})",
        )
        if ok:
            return True
    print(f"  [info] RefCOCO+ 全部候选都失败；可手动确认 HF 上的 repo 名后改本脚本。")
    return False


def task_refcocog():
    """RefCOCOg — 长 description 指代（"the woman in green dress holding a phone"）。"""
    candidates = [
        "lmms-lab/RefCOCOg",
        "jxu124/refcocog",
    ]
    for repo_id in candidates:
        ok = hf_download(
            repo_id=repo_id, repo_type="dataset",
            target_dir=DRIVE_ROOT / "refcocog",
            label=f"RefCOCOg (尝试 {repo_id})",
        )
        if ok:
            return True
    return False


def task_sharegpt4v():
    return hf_download(
        repo_id="Lin-Chen/ShareGPT4V",
        repo_type="dataset",
        patterns=[
            "sharegpt4v_instruct_gpt4-vision_cap100k.json",
            "share-captioner_coco_lcs_sam_1246k_1107.json",
        ],
        target_dir=DRIVE_ROOT / "sharegpt4v",
        label="ShareGPT4V (json subset)",
    )


# ============================================================================
# 验证：每个 task 抽 1 条验证字段
# ============================================================================

def verify_phase1plus():
    """验证 Phase 1+ 新增/复用的数据集都能被 datasets.load_dataset 正确加载。"""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [skip verify] datasets 未安装")
        return

    datasets_to_check = [
        ("RefCOCO+",  DRIVE_ROOT / "refcoco_plus"),
        ("RefCOCOg",  DRIVE_ROOT / "refcocog"),
        ("TextVQA",   DRIVE_ROOT / "textvqa"),
    ]
    for name, path in datasets_to_check:
        if not path.exists() or not any(path.iterdir()):
            print(f"  [verify] {name}: 目录不存在/为空，跳过")
            continue
        try:
            # 不锁 split，让 datasets 自己探测
            ds = None
            for split in ["train", "validation", "val", "test"]:
                try:
                    ds = load_dataset(str(path), split=split, trust_remote_code=True)
                    break
                except Exception:
                    continue
            if ds is None:
                ds_dict = load_dataset(str(path), trust_remote_code=True)
                split = list(ds_dict.keys())[0]
                ds = ds_dict[split]
            else:
                pass  # split 已找到
            n = len(ds)
            keys = list(ds.features.keys())[:8]
            print(f"  [verify] {name}: ✅ n={n}, fields={keys}")
            sample = ds[0]
            # 简单字段抽查
            if name in ("RefCOCO+", "RefCOCOg"):
                ref = sample.get("answer") or sample.get("sentences") or sample.get("ref")
                bbox = sample.get("bbox") or sample.get("box")
                print(f"             sample0: ref={str(ref)[:80]!r}, bbox={bbox}")
            elif name == "TextVQA":
                q = sample.get("question") or ""
                a = sample.get("answers") or []
                print(f"             sample0: Q={q[:80]!r}, "
                      f"answers[:3]={a[:3] if isinstance(a, list) else a}")
        except Exception as e:
            print(f"  [verify] {name}: ⚠️ 加载/解析失败: {e}")


# ============================================================================
# Main
# ============================================================================

ALL_TASKS = {
    "llava_instruct": (task_llava_instruct, "v1_existing"),
    "coco":           (task_coco_images,    "v1_existing"),
    "textvqa":        (task_textvqa,        "v1_existing"),
    "refcoco":        (task_refcoco,        "v1_existing"),
    "sharegpt4v":     (task_sharegpt4v,     "v1_existing"),
    "refcoco_plus":   (task_refcoco_plus,   "phase1plus_new"),  # ⭐
    "refcocog":       (task_refcocog,       "phase1plus_new"),  # ⭐
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only_phase1plus", action="store_true",
                    help="只下 Phase 1+ 新增的（RefCOCO+, RefCOCOg）")
    ap.add_argument("--skip", nargs="*", default=[], choices=list(ALL_TASKS.keys()),
                    help="跳过指定 task")
    args = ap.parse_args()

    if not Path("/content/drive/MyDrive").exists():
        print("[ERROR] Drive 未挂载。先在 Colab 跑：")
        print("  from google.colab import drive; drive.mount('/content/drive')")
        sys.exit(1)

    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"数据下载根目录: {DRIVE_ROOT}\n")

    results = {}
    for tid, (fn, kind) in ALL_TASKS.items():
        if tid in args.skip:
            print(f"=== [{tid}] 用户跳过 ===\n")
            continue
        if args.only_phase1plus and kind != "phase1plus_new":
            print(f"=== [{tid}] 不在 --only_phase1plus 范围 ===\n")
            continue
        marker = "⭐" if kind == "phase1plus_new" else " "
        print(f"=== [{tid}] {marker} ({kind}) ===")
        try:
            results[tid] = fn()
        except Exception as e:
            print(f"  [EXCEPTION] {e}")
            results[tid] = False
        print()

    print("\n=== 总结 ===")
    for tid, ok in results.items():
        marker = "✓" if ok else "✗"
        print(f"  {marker} {tid}")

    print("\n=== 验证 Phase 1+ 数据集字段 ===")
    verify_phase1plus()

    print("\n下一步：")
    print("  python stage2-v2/03_train_stage2.py \\")
    print("      --stage1_ckpt /content/drive/MyDrive/qwenvl3/stage1_ckpt_v3/checkpoint-11500 \\")
    print("      --processor_dir /content/drive/MyDrive/qwenvl3/stage1_init \\")
    print("      --stage2_data_root /content/drive/MyDrive/qwenvl3/data/stage2 \\")
    print("      --output_dir /content/drive/MyDrive/qwenvl3/stage2_v2_ckpt")


if __name__ == "__main__":
    main()
