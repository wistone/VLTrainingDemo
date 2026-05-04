#!/bin/bash
# Stage 2 Colab 环境初始化
# 在 Stage 1 setup.sh 基础上加：peft（LoRA）、bitsandbytes（QLoRA 备用）、wget（COCO 下载）
# 用法: !bash stage2-v1/setup.sh
set -e

echo "=== GPU/CPU 检查 ==="
nvidia-smi 2>/dev/null | head -20 || echo "无 GPU runtime（CPU only — 适合用作 download 节点）"

echo
echo "=== 安装依赖 ==="
pip install -q --upgrade pip
pip install -q \
    "transformers>=4.49.0,<5.0" \
    "accelerate>=1.0.0" \
    "datasets>=3.0.0" \
    "torch>=2.4.0" \
    "torchvision" \
    "peft>=0.13.0" \
    "bitsandbytes>=0.43.0" \
    "huggingface_hub" \
    "wandb" \
    "sentencepiece" \
    "tiktoken" \
    "Pillow" \
    "matplotlib" \
    "numpy<2.0"

apt-get -qq install -y wget unzip > /dev/null 2>&1 || true

echo
echo "=== 挂载 Drive 检查 ==="
ls /content/drive/MyDrive 2>/dev/null && echo "Drive OK" || echo "WARN: Drive 未挂载，请先在 Colab 中挂载"

echo
echo "=== 验证关键 import ==="
python -c "
import torch, transformers, peft
print(f'torch: {torch.__version__}, cuda: {torch.cuda.is_available()}')
print(f'transformers: {transformers.__version__}')
print(f'peft: {peft.__version__}')
from transformers import LlavaForConditionalGeneration
from peft import LoraConfig, get_peft_model
print('Stage 2 deps OK')
"

echo
echo "=== 提示 ==="
echo "数据下载到 Drive:    /content/drive/MyDrive/qwenvl3/data/stage2/"
echo "首次启动数据下载:    !python /content/QwenVL3/stage2-v1/01_prepare_data.py"
echo "预期 Drive 占用:     LLaVA-Instruct (~50MB) + COCO 18GB + OCR-VQA ~3GB ≈ 22GB"
echo
echo "Setup complete."
