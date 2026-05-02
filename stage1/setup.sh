#!/bin/bash
# Stage 1 Colab 环境初始化
# 用法: !bash stage1/setup.sh
set -e

echo "=== GPU 检查 ==="
nvidia-smi | head -20

echo
echo "=== 安装依赖 ==="
pip install -q --upgrade pip
pip install -q \
    "transformers>=4.49.0,<5.0" \
    "accelerate>=1.0.0" \
    "datasets>=3.0.0" \
    "torch>=2.4.0" \
    "torchvision" \
    "ms-swift>=3.0.0" \
    "peft>=0.13.0" \
    "huggingface_hub" \
    "wandb" \
    "sentencepiece" \
    "tiktoken" \
    "Pillow"
# transformers >=4.49 required for SigLIP2 support (Siglip2VisionModel)

# Colab 预装的 torchao 0.10.0 跟 peft 的 dispatch_torchao 不兼容（peft 要求 >=0.16.0）。
# 我们没用 torchao（那是量化加速库，跟 LoRA 无关），卸掉它让 peft 走标准 Linear LoRA 路径。
pip uninstall -y -q torchao 2>/dev/null || true

echo
echo "=== 挂载 Google Drive（如未挂载，请在 Colab cell 中先执行 from google.colab import drive; drive.mount('/content/drive')）==="
ls /content/drive/MyDrive 2>/dev/null && echo "Drive OK" || echo "WARN: Drive 未挂载，请先在 Colab 中挂载"

echo
echo "=== 验证关键 import ==="
python -c "
import torch
print(f'torch: {torch.__version__}')
print(f'cuda available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'device: {torch.cuda.get_device_name(0)}')
    print(f'mem: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
from transformers import LlavaForConditionalGeneration, AutoModel, AutoImageProcessor, AutoModelForCausalLM
import transformers
print(f'transformers: {transformers.__version__}')
# 验证 SigLIP2 类存在
from transformers import Siglip2VisionModel
print('SigLIP2 support OK')
"

echo
echo "=== 提示 ==="
echo "1. 首次使用：huggingface-cli login --token <your_HF_token>"
echo "2. 首次使用：wandb login <your_wandb_token>"
echo "3. 创建工作目录："
echo "   mkdir -p /content/data /content/drive/MyDrive/qwenvl3"
mkdir -p /content/data /content/drive/MyDrive/qwenvl3 2>/dev/null || true

echo
echo "Setup complete."
