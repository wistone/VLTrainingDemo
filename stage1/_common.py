"""Stage 1 共享工具：自定义 projector + transformers API 兼容层。

被 02_assemble_model.py 和 03_train_projector.py 共用，避免代码漂移。
"""
import glob

import torch.nn as nn


class ProjectorWithNorm(nn.Module):
    """LLaVA-style 2-layer MLP projector，输出加 LayerNorm。

    为什么需要 LayerNorm：
    Qwen2.5 的 token embedding norm ~0.78（远小于 LLaMA 的 ~5），
    默认 LlavaMultiModalProjector 输出会被训练放大到 800+ norm，
    淹没残差流，导致 loss 卡在 ~9.5 不下降。
    LayerNorm 强制输出归一化到 unit scale。

    属性命名（linear_1 / act / linear_2）刻意保持与官方
    LlavaMultiModalProjector 一致，确保 state_dict 可读。
    """
    def __init__(self, vision_hidden, text_hidden):
        super().__init__()
        self.linear_1 = nn.Linear(vision_hidden, text_hidden)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(text_hidden, text_hidden)
        self.norm = nn.LayerNorm(text_hidden)

    def forward(self, image_features):
        x = self.linear_1(image_features)
        x = self.act(x)
        x = self.linear_2(x)
        x = self.norm(x)
        return x


def get_inner(model):
    """兼容 transformers 新旧 API：组件可能在 model 或 model.model 下。"""
    if hasattr(model, "language_model"):
        return model
    return model.model


def get_components(model):
    inner = get_inner(model)
    return inner.language_model, inner.vision_tower, inner.multi_modal_projector


def set_projector(model, new_proj):
    inner = get_inner(model)
    inner.multi_modal_projector = new_proj


def install_custom_projector(model, init_dir=None, dtype=None):
    """把 model 的默认 projector 替换为 ProjectorWithNorm。

    若提供 init_dir，会从该目录的 safetensors 文件加载已存的 projector 权重
    （包括 LayerNorm 的 weight/bias）。这是必须的，因为 from_pretrained 用
    config 默认类构造 projector，会忽略 norm.* 权重。
    """
    vc = model.config.vision_config
    tc = model.config.text_config
    new_proj = ProjectorWithNorm(vc.hidden_size, tc.hidden_size)
    if dtype is not None:
        new_proj = new_proj.to(dtype)
    set_projector(model, new_proj)

    if init_dir is None:
        return new_proj

    # 延迟 import，避免本地无 safetensors 时模块不可加载
    from safetensors import safe_open  # noqa: PLC0415

    # 从 safetensors 抽出 multi_modal_projector.* 的权重
    proj_sd = {}
    for sf_file in glob.glob(f"{init_dir}/*.safetensors"):
        with safe_open(sf_file, framework="pt") as f:
            for key in f.keys():
                if "multi_modal_projector" in key:
                    stripped = key.split("multi_modal_projector.", 1)[1]
                    proj_sd[stripped] = f.get_tensor(key)

    if proj_sd:
        if dtype is not None:
            proj_sd = {k: v.to(dtype) for k, v in proj_sd.items()}
        missing, unexpected = new_proj.load_state_dict(proj_sd, strict=False)
        n_loaded = len(proj_sd) - len(unexpected)
        print(f"  projector 权重加载: {n_loaded} keys, missing={len(missing)}, unexpected={len(unexpected)}")
        for k in missing[:3]:
            print(f"    missing: {k}")
        for k in unexpected[:3]:
            print(f"    unexpected: {k}")
    else:
        print(f"  [warn] 在 {init_dir} 没找到 multi_modal_projector 权重，使用随机初始化")

    return new_proj
