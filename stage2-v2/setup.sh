#!/bin/bash
# Stage 2-v2 (Phase 1+) Colab 环境初始化
# 用法: !bash stage2-v2/setup.sh
#
# 跟 Stage 1 的环境完全相同（transformers / peft / datasets / wandb 等），
# 唯一新加：huggingface_hub snapshot_download 用于补下 RefCOCO+ / RefCOCOg / TextVQA。
#
# 如果你已经跑过 stage1/setup.sh + stage2 训练过，可以跳过这个，直接进 01_prepare_data.py。
set -e

echo "=== GPU 检查 ==="
nvidia-smi | head -20

echo
echo "=== 安装依赖（与 Stage 1 一致 + 补充包）==="
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
    "Pillow" \
    "safetensors"

# 同步 stage1 的 torchao 兼容处理
pip uninstall -y -q torchao 2>/dev/null || true

echo
echo "=== 挂载 Drive ==="
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
import transformers, peft, datasets
print(f'transformers: {transformers.__version__}')
print(f'peft: {peft.__version__}')
print(f'datasets: {datasets.__version__}')
from transformers import Siglip2VisionModel
print('SigLIP2 support OK')
"

echo
echo "=== Phase 1+ 提示 ==="
echo "1. 数据下载（补下 RefCOCO+/g）："
echo "   python stage2-v2/01_prepare_data.py --only_phase1plus"
echo "2. 训练："
echo "   python stage2-v2/03_train_stage2.py ..."
echo
echo "Setup complete."
