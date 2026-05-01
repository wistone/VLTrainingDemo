"""组装 LLaVA-style VL 模型：Qwen2.5-1.5B-Instruct + SigLIP-SO400M + 2-layer MLP projector。

核心步骤：
1. 加载 Qwen2.5-1.5B 文本模型 + tokenizer
2. 加载 SigLIP-SO400M 视觉塔 + image processor
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
"""
import os
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlavaConfig,
    LlavaForConditionalGeneration,
    SiglipImageProcessor,
    SiglipVisionModel,
)

OUT_DIR = Path(os.environ.get("MODEL_INIT_DIR", "/content/drive/MyDrive/qwenvl3/stage1_init"))
LLM_NAME = os.environ.get("LLM_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
VISION_NAME = os.environ.get("VISION_NAME", "google/siglip-so400m-patch14-384")


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

    # 2. 视觉塔
    print(f"\n[2/5] 加载视觉塔: {VISION_NAME}")
    image_processor = SiglipImageProcessor.from_pretrained(VISION_NAME)
    vision_model = SiglipVisionModel.from_pretrained(
        VISION_NAME,
        torch_dtype=torch.bfloat16,
    )

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
    llava_config = LlavaConfig(
        text_config=text_model.config,
        vision_config=vision_model.config,
        image_token_index=image_token_id,
        projector_hidden_act="gelu",
        vision_feature_layer=-2,
        vision_feature_select_strategy="default",
        ignore_index=-100,
    )
    # 同步 vocab size（前面 resize 过）
    llava_config.text_config.vocab_size = len(text_tokenizer)
    llava_config.pad_token_id = text_tokenizer.pad_token_id

    # 5. 构造 LlavaForConditionalGeneration 并迁移权重
    print("\n[5/5] 构造 LlavaForConditionalGeneration 并迁移权重")
    model = LlavaForConditionalGeneration(llava_config)
    model = model.to(torch.bfloat16)

    # 视觉塔权重（注意 LLaVA 的 vision_tower 直接是 SiglipVisionModel）
    missing, unexpected = model.vision_tower.load_state_dict(
        vision_model.state_dict(), strict=False
    )
    print(f"  vision_tower: missing={len(missing)} unexpected={len(unexpected)}")

    # 文本模型权重
    missing, unexpected = model.language_model.load_state_dict(
        text_model.state_dict(), strict=False
    )
    print(f"  language_model: missing={len(missing)} unexpected={len(unexpected)}")

    # 检查 projector 是否随机初始化（Stage 1 训练目标）
    proj_param = next(model.multi_modal_projector.parameters())
    print(f"  projector 初始权重 mean={proj_param.mean().item():.4f} std={proj_param.std().item():.4f}（应该是随机小值）")

    # 保存
    print(f"\n保存到 {OUT_DIR}")
    model.save_pretrained(OUT_DIR, safe_serialization=True)
    text_tokenizer.save_pretrained(OUT_DIR)
    image_processor.save_pretrained(OUT_DIR)

    # Param 统计
    total = sum(p.numel() for p in model.parameters())
    proj = sum(p.numel() for p in model.multi_modal_projector.parameters())
    vision = sum(p.numel() for p in model.vision_tower.parameters())
    lm = sum(p.numel() for p in model.language_model.parameters())
    print(f"\n参数量统计:")
    print(f"  total:        {total/1e9:.3f}B")
    print(f"  vision_tower: {vision/1e6:.1f}M  (Stage 1 冻结)")
    print(f"  language:     {lm/1e9:.3f}B    (Stage 1 冻结)")
    print(f"  projector:    {proj/1e6:.2f}M   (Stage 1 唯一可训练)")

    print(f"\n下一步：python stage1/03_train_projector.py --model_init_dir {OUT_DIR} ...")


if __name__ == "__main__":
    main()
