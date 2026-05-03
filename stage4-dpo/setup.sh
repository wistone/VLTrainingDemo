#!/bin/bash
# Stage 4 DPO 环境初始化
# 用法: !bash stage4-dpo/setup.sh
#
# 跟 stage2/setup.sh 几乎一样，多装个 trl（参考用，我们的训练代码用的是自定义 DPO loss
# 不强依赖 trl，但 trl 的 reference 实现可以用来交叉验证）。
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
    "peft>=0.13.0" \
    "trl>=0.11.0" \
    "huggingface_hub" \
    "wandb" \
    "sentencepiece" \
    "tiktoken" \
    "Pillow" \
    "safetensors"

# 同 v1/v2：处理 torchao 跟 PEFT 不兼容
pip uninstall -y -q torchao 2>/dev/null || true

echo
echo "=== 挂载 Drive 检查 ==="
ls /content/drive/MyDrive 2>/dev/null && echo "Drive OK" || echo "WARN: Drive 未挂载"

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
try:
    import trl
    print(f'trl: {trl.__version__}')
except ImportError:
    print('trl: not installed (我们用自定义 DPO loss，可选)')
from transformers import Siglip2VisionModel
print('SigLIP2 support OK')
"

echo
echo "=== Stage 4 DPO 提示 ==="
echo "1. 数据下载（如果没下过）："
echo "   python stage4-dpo/01_prepare_dpo_data.py --dpo_data_root /content/drive/MyDrive/qwenvl3/data/dpo"
echo "2. 烟雾测试（10 min, 50 steps）："
echo "   python stage4-dpo/03_train_dpo.py --smoke_test ..."
echo "3. 正式训练（~1-2h）："
echo "   python stage4-dpo/03_train_dpo.py ..."
echo
echo "Setup complete."
