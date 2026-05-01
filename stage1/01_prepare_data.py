"""下载并准备 LLaVA-Pretrain-558K 数据。

数据集源：https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain
- images.zip: ~25GB，约 558K 张图
- blip_laion_cc_sbu_558k.json: caption 标注

下载到 /content/data（Colab 本地 SSD），不下到 Drive（Drive I/O 慢）。
注意：/content 在 session 结束后会被清空，每次新 session 都要重跑此脚本。

用法：
    python stage1/01_prepare_data.py
"""
import json
import os
import random
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from PIL import Image

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/content/data"))
PRETRAIN_DIR = DATA_ROOT / "llava-pretrain"


def download():
    PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)
    json_path = PRETRAIN_DIR / "blip_laion_cc_sbu_558k.json"
    images_zip = PRETRAIN_DIR / "images.zip"

    if json_path.exists() and images_zip.exists():
        print(f"[skip] 文件已存在于 {PRETRAIN_DIR}")
        return

    print(f"[download] 下载 LLaVA-Pretrain → {PRETRAIN_DIR}")
    snapshot_download(
        repo_id="liuhaotian/LLaVA-Pretrain",
        repo_type="dataset",
        local_dir=str(PRETRAIN_DIR),
        allow_patterns=["images.zip", "blip_laion_cc_sbu_558k.json"],
        max_workers=4,
    )
    print("[download] 完成")


def unzip_images():
    images_zip = PRETRAIN_DIR / "images.zip"
    images_dir = PRETRAIN_DIR / "images"

    if images_dir.exists() and len(list(images_dir.iterdir())) > 100:
        print(f"[skip] 图像已解压于 {images_dir}")
        return

    print(f"[unzip] 解压 {images_zip}（这一步约 5–10 分钟）")
    subprocess.run(
        ["unzip", "-q", "-o", str(images_zip), "-d", str(PRETRAIN_DIR)],
        check=True,
    )
    print("[unzip] 完成")


def verify():
    json_path = PRETRAIN_DIR / "blip_laion_cc_sbu_558k.json"
    images_dir = PRETRAIN_DIR / "images"

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
    """切出 20 张 held-out 用于评估。从最后 100 条里随机选。"""
    json_path = PRETRAIN_DIR / "blip_laion_cc_sbu_558k.json"
    holdout_path = PRETRAIN_DIR / "holdout_20.json"
    if holdout_path.exists():
        print(f"[skip] holdout 已存在于 {holdout_path}")
        return
    with open(json_path) as f:
        data = json.load(f)
    random.seed(42)
    holdout = random.sample(data[-1000:], 20)
    with open(holdout_path, "w") as f:
        json.dump(holdout, f, indent=2)
    print(f"[holdout] 已写入 {holdout_path}")


if __name__ == "__main__":
    download()
    unzip_images()
    verify()
    split_holdout()
    print("\n所有步骤完成。下一步：python stage1/02_assemble_model.py")
