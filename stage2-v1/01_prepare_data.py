"""Stage 2 多任务数据下载到 Drive（持久缓存）。

策略：所有原始数据（json + COCO 图 zip）下到 Drive，单 session 不解压、不复制到 /content。
训练/评估时按需从 Drive 直读 zip 内的图（zipfile 接口），节省 /content 空间。

数据集（按重要性排序，可独立启用/跳过）：
  ✅ LLaVA-Instruct-150K  json (~50MB)             — 多轮 VQA + 推理
  ✅ COCO train2017 图 zip (~18GB, 单文件)         — LLaVA-Instruct/RefCOCO 共用
  ⚠️ TextVQA (~7GB)                                — 图内文字 VQA（OCR 类，替代 OCR-VQA）
  ⚠️ RefCOCO/RefCOCO+/RefCOCOg annotations         — Grounding（可选；图用 COCO）
  ⚠️ ShareGPT4V json subset                        — 长 caption（可选）

注意：原 OCR-VQA-200K 主要在 Google Drive 上分发，HF 没有可靠 mirror，
我们用 TextVQA 替代——任务性质几乎相同（都是图内文字 + 问答）。

可选数据 HF repo id 不绝对稳定，脚本对每个数据 try/except；下不到不影响其他数据。
跑完后按 `[done]` 行核对实际成功项。

用法（CPU runtime 即可，能并行训练 session）：
    !python stage2-v1/01_prepare_data.py                   # 全量
    !python stage2-v1/01_prepare_data.py --only essential  # 只下 LLaVA-Instruct + COCO
    !python stage2-v1/01_prepare_data.py --skip coco       # 跳过大文件
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
import zipfile
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
# 工具：HuggingFace snapshot download
# ============================================================================

def hf_download(repo_id, repo_type, patterns, target_dir, label):
    """从 HF 下载 dataset/model 文件到 target_dir。"""
    from huggingface_hub import snapshot_download

    target_dir.mkdir(parents=True, exist_ok=True)

    # 简单的"已下载"检测：目录里有 >50MB 内容就跳过
    if target_dir.exists():
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
            allow_patterns=patterns,  # None = 全下
            max_workers=4,
        )
        size = sum(f.stat().st_size for f in target_dir.rglob("*") if f.is_file())
        print(f"  [done] {label}: {fmt_size(size)} → {target_dir}")
        return True
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        return False


# ============================================================================
# 工具：直接 URL 下载（COCO 用）
# ============================================================================

def url_download(url, target_path, label):
    """用 wget -c 下载，支持断点续传。"""
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists():
        size = target_path.stat().st_size
        # COCO train2017.zip 是 19,343,994,941 bytes (~18GB)
        # 完整文件应 > 18GB，否则视为不完整、续传
        if size > 18 * 1024 * 1024 * 1024:
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
# 数据集任务定义
# ============================================================================

def task_llava_instruct():
    """LLaVA-Instruct-150K: 多轮 VQA + 推理。

    HF repo: liuhaotian/LLaVA-Instruct-150K
    包含 llava_instruct_150k.json (158K samples) + complex_reasoning_77k.json + detail_23k.json
    """
    return hf_download(
        repo_id="liuhaotian/LLaVA-Instruct-150K",
        repo_type="dataset",
        patterns=["*.json"],
        target_dir=DRIVE_ROOT / "llava_instruct",
        label="LLaVA-Instruct-150K (json)",
    )


def task_coco_images():
    """COCO train2017 图像 zip — LLaVA-Instruct 和 RefCOCO 共用。"""
    return url_download(
        url="http://images.cocodataset.org/zips/train2017.zip",
        target_path=DRIVE_ROOT / "coco" / "train2017.zip",
        label="COCO train2017 (~18GB)",
    )


def task_textvqa():
    """TextVQA：图像内文字识别+问答。OCR 类任务的 HF 替代品。

    OCR-VQA-200K 原始数据在 Google Drive 上，HF 没有靠谱 mirror。
    用 TextVQA（性质几乎相同）替代——HF 上有 lmms-lab 维护的版本。
    """
    candidates = [
        ("lmms-lab/textvqa", "dataset"),
        ("facebook/textvqa", "dataset"),
    ]
    for repo_id, rtype in candidates:
        ok = hf_download(
            repo_id=repo_id,
            repo_type=rtype,
            patterns=None,
            target_dir=DRIVE_ROOT / "textvqa",
            label=f"TextVQA (尝试 {repo_id})",
        )
        if ok:
            return True
    print(f"  [info] TextVQA 全部候选都失败；这是可选数据，跳过不影响 Stage 2 主线。")
    print(f"  [info] Stage 1 模型已自然涌现 OCR 能力，没有这块也能学到 80% 效果。")
    return False


def task_refcoco():
    """RefCOCO/+/g grounding annotations（图用 COCO）。"""
    for repo_id in ["lmms-lab/RefCOCO", "jxu124/refcoco"]:
        ok = hf_download(
            repo_id=repo_id,
            repo_type="dataset",
            patterns=None,
            target_dir=DRIVE_ROOT / "refcoco",
            label=f"RefCOCO (尝试 {repo_id})",
        )
        if ok:
            return True
    return False


def task_sharegpt4v():
    """ShareGPT4V detailed captions (取 100K subset)."""
    return hf_download(
        repo_id="Lin-Chen/ShareGPT4V",
        repo_type="dataset",
        patterns=["sharegpt4v_instruct_gpt4-vision_cap100k.json", "share-captioner_coco_lcs_sam_1246k_1107.json"],
        target_dir=DRIVE_ROOT / "sharegpt4v",
        label="ShareGPT4V (json subset)",
    )


# ============================================================================
# 验证：抽 5 条 LLaVA-Instruct 样本，验证图能从 COCO zip 里读到
# ============================================================================

def verify_llava_coco_link():
    json_path = DRIVE_ROOT / "llava_instruct" / "llava_instruct_150k.json"
    coco_zip = DRIVE_ROOT / "coco" / "train2017.zip"

    if not json_path.exists():
        print("  [verify] 跳过 — LLaVA-Instruct json 未下载")
        return
    if not coco_zip.exists() or coco_zip.stat().st_size < 1e9:
        print("  [verify] 跳过 — COCO zip 未下载完整")
        return

    print(f"  [verify] 抽 5 条 LLaVA-Instruct 样本，验证图能从 COCO zip 读到")
    with open(json_path) as f:
        data = json.load(f)
    random.seed(0)
    samples = random.sample(data, 5)

    with zipfile.ZipFile(coco_zip) as zf:
        zip_names = set(zf.namelist())
        for s in samples:
            img = s["image"]  # 形如 "000000123456.jpg"
            # COCO zip 内路径是 train2017/000000123456.jpg
            zip_path = f"train2017/{img}"
            in_zip = zip_path in zip_names
            print(f"    {img}: {'✓ in zip' if in_zip else '✗ NOT FOUND'} | "
                  f"Q: {s['conversations'][0]['value'][:60]!r}")


# ============================================================================
# Main
# ============================================================================

ALL_TASKS = {
    "llava_instruct": (task_llava_instruct, "essential"),
    "coco":           (task_coco_images,    "essential"),
    "textvqa":        (task_textvqa,        "optional"),  # OCR 类（替代 OCR-VQA）
    "refcoco":        (task_refcoco,        "optional"),
    "sharegpt4v":     (task_sharegpt4v,     "optional"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all", choices=["all", "essential", "optional"],
                    help="只下哪类（essential = LLaVA-Instruct + COCO; optional = 其他）")
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
        if args.only != "all" and kind != args.only:
            print(f"=== [{tid}] 不在 --only={args.only} 范围 ===\n")
            continue
        print(f"=== [{tid}] {kind} ===")
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

    print("\n=== 验证 LLaVA-Instruct ↔ COCO 图链接 ===")
    verify_llava_coco_link()

    print("\n下一步：")
    print("  1. 跑 baseline eval: python stage2-v1/02_baseline_eval.py ...")
    print("  2. 等数据下完 + Stage 1 训完 → Stage 2 训练")


if __name__ == "__main__":
    main()
