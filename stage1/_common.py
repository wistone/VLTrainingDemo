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
    """返回真正持有 multi_modal_projector / vision_tower / language_model 的容器。

    新 API（transformers ≥4.50）：组件挂在 model.model（LlavaModel）下；顶层
    LlavaForConditionalGeneration 暴露的同名 property 不能简单地用 hasattr 判定，
    因为 setter 行为不可靠，赋值可能不生效。

    旧 API：组件直接挂在 model 下。

    判定逻辑：直接看 `multi_modal_projector` 这个真正的子模块在哪里。
    """
    if hasattr(model, "model") and isinstance(getattr(model.model, "multi_modal_projector", None), nn.Module):
        return model.model
    return model


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

    # 验证替换生效——直接检查 nn.Module._modules 里的对象 identity
    inner = get_inner(model)
    actual = inner._modules.get("multi_modal_projector")
    if actual is not new_proj:
        raise RuntimeError(
            f"Projector 替换失败！\n"
            f"  期望: ProjectorWithNorm (id={id(new_proj)})\n"
            f"  实际: type={type(actual).__name__}, id={id(actual) if actual else None}\n"
            f"  inner type: {type(inner).__name__}\n"
            f"  请检查 transformers 版本与 get_inner() 判定是否一致。"
        )
    print(f"  ✅ projector 替换成功: type={type(actual).__name__}, 挂载位置={type(inner).__name__}.multi_modal_projector")

    if init_dir is None:
        return new_proj

    # 延迟 import，避免本地无 safetensors 时模块不可加载
    from safetensors import safe_open  # noqa: PLC0415

    # 从 safetensors 抽出 multi_modal_projector.* 的权重
    #
    # 两种保存格式：
    #   1) Stage 1 训练完的整体 ckpt：projector 权重混在 model*.safetensors / model.safetensors 里，
    #      key 形如 'multi_modal_projector.linear_1.weight'，需按前缀过滤再剥前缀
    #   2) Stage 2 中间 checkpoint-NNNN/multi_modal_projector.safetensors（由
    #      ProjectorSaverCallback 单独保存）：key 已经是 'linear_1.weight' 形式（无前缀），
    #      整文件就是 projector，全部直接装载即可
    import os  # noqa: PLC0415
    proj_sd = {}
    for sf_file in glob.glob(f"{init_dir}/*.safetensors"):
        is_projector_only = os.path.basename(sf_file) == "multi_modal_projector.safetensors"
        with safe_open(sf_file, framework="pt") as f:
            for key in f.keys():
                if is_projector_only:
                    # 整文件就是 projector，无前缀，直接收
                    proj_sd[key] = f.get_tensor(key)
                elif "multi_modal_projector" in key:
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
