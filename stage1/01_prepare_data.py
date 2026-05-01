"""下载并准备 LLaVA-Pretrain-558K 数据。

数据集源：https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain
- images.zip: ~25GB，约 558K 张图
- blip_laion_cc_sbu_558k.json: caption 标注

策略：Drive 缓存 zip + 本地 SSD 解压
- Drive 上保留 images.zip 和 json（持久，避免每次新 session 从 HF 重下 ~30 min）
- /content 上解压图像（训练时随机读 55 万小文件，必须在本地 SSD 上）
- 不要把解压后的 images/ 目录放 Drive，会让训练 I/O 慢 5–10 倍

每次新 session 流程（自动判定）：
  1. Drive 没有 zip → 从 HF 下载到 Drive（首次 30–40 min）
  2. Drive 有 zip → 复制到 /content（~5 min）
  3. /content 上解压（~5 min）

环境变量：
  DATA_ROOT       本地工作目录，默认 /content/data
  DRIVE_DATA_ROOT Drive 缓存目录，默认 /content/drive/MyDrive/qwenvl3/data

用法：
    python stage1/01_prepare_data.py
"""
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download
from PIL import Image

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/content/data"))
DRIVE_DATA_ROOT = Path(os.environ.get("DRIVE_DATA_ROOT", "/content/drive/MyDrive/qwenvl3/data"))

LOCAL_DIR = DATA_ROOT / "llava-pretrain"
DRIVE_DIR = DRIVE_DATA_ROOT / "llava-pretrain"

ZIP_NAME = "images.zip"
JSON_NAME = "blip_laion_cc_sbu_558k.json"


def fmt_time(s):
    return f"{s:.0f}s" if s < 60 else f"{s/60:.1f}min"


def download_to_drive():
    """首次下载到 Drive（持久缓存）。已存在则跳过。"""
    DRIVE_DIR.mkdir(parents=True, exist_ok=True)
    drive_zip = DRIVE_DIR / ZIP_NAME
    drive_json = DRIVE_DIR / JSON_NAME

    if drive_zip.exists() and drive_json.exists():
        size_gb = drive_zip.stat().st_size / 1e9
        print(f"[skip] Drive 已有缓存 ({drive_zip}, {size_gb:.1f}GB)")
        return

    if not Path("/content/drive/MyDrive").exists():
        print("[error] Google Drive 未挂载。先在 Colab cell 跑：")
        print("  from google.colab import drive; drive.mount('/content/drive')")
        sys.exit(1)

    print(f"[download] 首次下载 LLaVA-Pretrain → Drive ({DRIVE_DIR})")
    print(f"          预计 30–40 min（仅首次；以后从 Drive 缓存直接复制）")
    t0 = time.time()
    snapshot_download(
        repo_id="liuhaotian/LLaVA-Pretrain",
        repo_type="dataset",
        local_dir=str(DRIVE_DIR),
        allow_patterns=[ZIP_NAME, JSON_NAME],
        max_workers=4,
    )
    print(f"[download] 完成（耗时 {fmt_time(time.time()-t0)}）")


def sync_drive_to_local():
    """把 Drive 的 zip + json 复制到 /content。"""
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    drive_zip = DRIVE_DIR / ZIP_NAME
    drive_json = DRIVE_DIR / JSON_NAME
    local_zip = LOCAL_DIR / ZIP_NAME
    local_json = LOCAL_DIR / JSON_NAME

    if not drive_zip.exists():
        print(f"[error] {drive_zip} 不存在，先跑 download_to_drive")
        sys.exit(1)

    # JSON 小，直接复制
    if not local_json.exists():
        print(f"[sync] 复制 json → {local_json}")
        shutil.copy2(drive_json, local_json)

    # ZIP 大，先看看是否需要复制（已存在且大小一致就跳过）
    if local_zip.exists() and local_zip.stat().st_size == drive_zip.stat().st_size:
        print(f"[skip] {local_zip} 已是最新")
        return

    size_gb = drive_zip.stat().st_size / 1e9
    print(f"[sync] 复制 zip → {local_zip}（{size_gb:.1f}GB，~5 min）")
    t0 = time.time()
    shutil.copy2(drive_zip, local_zip)
    print(f"[sync] 完成（耗时 {fmt_time(time.time()-t0)}）")


def unzip_images():
    images_zip = LOCAL_DIR / ZIP_NAME
    images_dir = LOCAL_DIR / "images"

    if images_dir.exists() and len(list(images_dir.iterdir())) > 100:
        print(f"[skip] 图像已解压于 {images_dir}")
        return

    print(f"[unzip] 解压 {images_zip} → {LOCAL_DIR}/images/（~5 min）")
    t0 = time.time()
    subprocess.run(
        ["unzip", "-q", "-o", str(images_zip), "-d", str(LOCAL_DIR)],
        check=True,
    )
    print(f"[unzip] 完成（耗时 {fmt_time(time.time()-t0)}）")


def verify():
    json_path = LOCAL_DIR / JSON_NAME
    images_dir = LOCAL_DIR / "images"

    print(f"[verify] 加载 {json_path}")
    with open(json_path) as f:
        data = json.load(f)
    print(f"[verify] 总样本数: {len(data)}")

    print(f"[verify] 抽样 5 条校验：")
    random.seed(0)
    sampled = random.sample(data, 5)
    all_ok = True
    for s in sampled:
        img_path = images_dir / s["image"]
        if not img_path.exists():
            print(f"  MISSING: {s['image']}")
            all_ok = False
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            cap = s["conversations"][1]["value"][:80]
            print(f"  OK {s['image']} ({img.size}) | {cap}...")
        except Exception as e:
            print(f"  CORRUPT {s['image']}: {e}")
            all_ok = False

    if not all_ok:
        sys.exit(1)
    print("[verify] 全部通过")


def split_holdout():
    """切出 20 张 held-out 用于评估，写到 Drive（持久）和 /content（训练用）。"""
    json_path = LOCAL_DIR / JSON_NAME
    local_holdout = LOCAL_DIR / "holdout_20.json"
    drive_holdout = DRIVE_DIR / "holdout_20.json"

    if drive_holdout.exists():
        # 从 Drive 同步过来，保证多次 session 用的是同一批
        print(f"[skip] holdout 从 Drive 复用 ({drive_holdout})")
        shutil.copy2(drive_holdout, local_holdout)
        return

    with open(json_path) as f:
        data = json.load(f)
    random.seed(42)
    holdout = random.sample(data[-1000:], 20)

    with open(local_holdout, "w") as f:
        json.dump(holdout, f, indent=2)
    DRIVE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_holdout, drive_holdout)
    print(f"[holdout] 已写入 {local_holdout} 和 {drive_holdout}")


if __name__ == "__main__":
    print(f"DATA_ROOT       (本地，训练用):    {DATA_ROOT}")
    print(f"DRIVE_DATA_ROOT (Drive，持久缓存): {DRIVE_DATA_ROOT}")
    print()

    download_to_drive()
    sync_drive_to_local()
    unzip_images()
    verify()
    split_holdout()
    print("\n所有步骤完成。下一步：python stage1/02_assemble_model.py")
