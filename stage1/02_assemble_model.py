"""组装 LLaVA-style VL 模型：Qwen2.5-1.5B-Instruct + SigLIP2-SO400M + 2-layer MLP projector。

核心步骤：
1. 加载 Qwen2.5-1.5B 文本模型 + tokenizer
2. 加载 SigLIP2-SO400M 视觉塔 + image processor（用 AutoModel/AutoImageProcessor 以兼容 v1/v2）
3. 给 tokenizer 添加 <image> 特殊 token，扩展 LLM embedding
4. 构造 LlavaConfig，把 text_config + vision_config 拼起来
5. 用 LlavaForConditionalGeneration(config) 构建空壳模型
6. 把文本模型权重拷进 model.language_model
7. 把视觉塔权重拷进 model.vision_tower
8. multi_modal_projector 保持随机初始化（这就是 Stage 1 要训的部分）
9. 保存到 Drive，供 03_train_projector.py 加载

只需跑一次。

用法：
    python stage1/02_assemble_model.py

切回 SigLIP v1（如需对比实验）：
    VISION_NAME=google/siglip-so400m-patch14-384 python stage1/02_assemble_model.py
"""
import os
from pathlib import Path

import sys

import torch
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlavaConfig,
    LlavaForConditionalGeneration,
)

# 让脚本无论从哪里启动都能 import 同目录下的 _common
sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402
    ProjectorWithNorm,
    get_components,
    install_custom_projector,
)

OUT_DIR = Path(os.environ.get("MODEL_INIT_DIR", "/content/drive/MyDrive/qwenvl3/stage1_init"))
LLM_NAME = os.environ.get("LLM_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
# SigLIP2 SO400M @ patch14-384：同尺寸同 token 数（729）作为 v1 的 drop-in 替换。
# v1 备选: google/siglip-so400m-patch14-384
VISION_NAME = os.environ.get("VISION_NAME", "google/siglip2-so400m-patch14-384")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {OUT_DIR}")

    # 1. 文本模型
    print(f"\n[1/5] 加载文本模型: {LLM_NAME}")
    text_tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
    text_model = AutoModelForCausalLM.from_pretrained(
        LLM_NAME,
        torch_dtype=torch.bfloat16,
    )

    # 2. 视觉塔（用 AutoModel 自动选 SiglipVisionModel 或 Siglip2VisionModel）
    print(f"\n[2/5] 加载视觉塔: {VISION_NAME}")
    image_processor = AutoImageProcessor.from_pretrained(VISION_NAME)
    # SigLIP/SigLIP2 完整模型是双塔（vision + text），LLaVA 只需要 vision 塔
    full_siglip = AutoModel.from_pretrained(VISION_NAME, torch_dtype=torch.bfloat16)
    vision_model = full_siglip.vision_model
    vision_config = full_siglip.config.vision_config
    print(f"  vision tower 类型: {type(vision_model).__name__}")
    print(f"  vision_config: hidden_size={vision_config.hidden_size}, patches={vision_config.image_size//vision_config.patch_size}^2")

    # 3. 添加 <image> token
    print("\n[3/5] 添加 <image> 特殊 token")
    num_added = text_tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<image>"]}
    )
    image_token_id = text_tokenizer.convert_tokens_to_ids("<image>")
    print(f"  添加了 {num_added} 个 token; <image> id = {image_token_id}")
    text_model.resize_token_embeddings(len(text_tokenizer))

    # 确保 pad_token 存在
    if text_tokenizer.pad_token is None:
        text_tokenizer.pad_token = text_tokenizer.eos_token
        print(f"  pad_token 设为 eos_token: {text_tokenizer.pad_token}")

    # 4. 构造 LlavaConfig
    print("\n[4/5] 构造 LlavaConfig")
    # 注意：SigLIP/SigLIP2 没有 CLS token（不像 CLIP），所以用 "full" 而不是 "default"
    # "default" 会无缘无故砍掉第一个 patch；"full" 保留全部 27*27=729 个
    llava_config = LlavaConfig(
        text_config=text_model.config,
        vision_config=vision_config,
        image_token_index=image_token_id,
        projector_hidden_act="gelu",
        vision_feature_layer=-2,
        vision_feature_select_strategy="full",
        ignore_index=-100,
    )
    # 同步 vocab size（前面 resize 过）
    llava_config.text_config.vocab_size = len(text_tokenizer)
    llava_config.pad_token_id = text_tokenizer.pad_token_id

    # 5. 构造 LlavaForConditionalGeneration 并迁移权重
    print("\n[5/5] 构造 LlavaForConditionalGeneration 并迁移权重")
    model = LlavaForConditionalGeneration(llava_config)
    model = model.to(torch.bfloat16)

    # 拿到内部组件（兼容新旧 API）
    lm_module, vt_module, _ = get_components(model)

    # 视觉塔权重
    missing, unexpected = vt_module.load_state_dict(
        vision_model.state_dict(), strict=False
    )
    print(f"  vision_tower: missing={len(missing)} unexpected={len(unexpected)}")

    # 文本模型权重
    missing, unexpected = lm_module.load_state_dict(
        text_model.state_dict(), strict=False
    )
    print(f"  language_model: missing={len(missing)} unexpected={len(unexpected)}")

    # ===== 关键修复：替换默认 projector 为带 LayerNorm 的版本 =====
    # Qwen2.5 + LLaVA 必须做这步，否则 projector 会被训练放大到 1000× norm
    install_custom_projector(model, init_dir=None, dtype=torch.bfloat16)
    print("  已替换 projector → ProjectorWithNorm（含 LayerNorm 输出）")

    # 重新拿一次 projector 的引用（替换之后变了）
    _, _, proj_module = get_components(model)
    proj_param = next(proj_module.parameters())
    print(f"  projector 初始权重 mean={proj_param.mean().item():.4f} std={proj_param.std().item():.4f}")

    # 保存
    print(f"\n保存到 {OUT_DIR}")
    model.save_pretrained(OUT_DIR, safe_serialization=True)
    text_tokenizer.save_pretrained(OUT_DIR)
    image_processor.save_pretrained(OUT_DIR)

    # Param 统计
    total = sum(p.numel() for p in model.parameters())
    proj = sum(p.numel() for p in proj_module.parameters())
    vision = sum(p.numel() for p in vt_module.parameters())
    lm = sum(p.numel() for p in lm_module.parameters())
    print(f"\n参数量统计:")
    print(f"  total:        {total/1e9:.3f}B")
    print(f"  vision_tower: {vision/1e6:.1f}M  (Stage 1 冻结)")
    print(f"  language:     {lm/1e9:.3f}B    (Stage 1 冻结)")
    print(f"  projector:    {proj/1e6:.2f}M   (Stage 1 唯一可训练)")

    print(f"\n下一步：python stage1/03_train_projector.py --model_init_dir {OUT_DIR} ...")


if __name__ == "__main__":
    main()
