"""Stage 2 训练数据抽样可视化 — 3 个 task dataset 各 20 sample，生成自包含 HTML。

帮助理解：
  1. 每个 dataset 是什么格式的 image-text pair（原始 JSON / HF 字段）
  2. 各 task 解决的问题（VQA / grounding / 长 caption）
  3. 训练时如何统一成 Qwen2.5 chat 格式（user/assistant turn + <image>×729 + loss mask）

输出: 单个 HTML 文件，所有图都 base64 内嵌，下载后离线可看。

== 用法 ==

  python stage2/05_sample_training_data.py \\
      --stage2_data_root /content/drive/MyDrive/qwenvl3/data/stage2 \\
      --output /content/drive/MyDrive/qwenvl3/data_samples/stage2_training_data.html

  跑完后：Drive 网页里下载该 HTML，本地浏览器双击打开。

== 数据集 ==

  1. LLaVA-Instruct-150K  - 多轮 VQA + 推理；JSON 格式 {image, conversations}
  2. RefCOCO              - 视觉定位（ref expr → bbox）；HF dataset
  3. ShareGPT4V           - 长详细 caption；JSON 格式 {image, conversations}
"""
import argparse
import base64
import html as html_mod
import io
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from _common2 import CocoZipLoader  # noqa: E402

NUM_IMAGE_TOKENS = 729  # 仅作展示，实际 729 = (384/14)^2 - 1
THUMB_SIZE = 480


# ============================================================================
# 图像工具
# ============================================================================

def to_b64_jpeg(image: Image.Image, max_size=THUMB_SIZE, quality=85) -> str:
    img = image.copy()
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def draw_bbox_on_image(image: Image.Image, bbox_norm, color="#ef4444", width=5):
    """在图上画 bbox（红框）。bbox_norm 是 (x1, y1, x2, y2) 0-1 归一化坐标。"""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    iw, ih = img.size
    x1, y1, x2, y2 = bbox_norm
    draw.rectangle(
        [(x1 * iw, y1 * ih), (x2 * iw, y2 * ih)],
        outline=color, width=width,
    )
    return img


# ============================================================================
# 抽样：3 个 dataset 各 N 条
# ============================================================================

def sample_llava_instruct(json_path: Path, coco_loader: CocoZipLoader,
                          n: int, seed: int):
    """LLaVA-Instruct-150K — 多轮 VQA。"""
    print(f"[llava_instruct] reading {json_path}")
    with open(json_path) as f:
        data = json.load(f)
    print(f"  total samples: {len(data)}")

    rng = random.Random(seed)
    indices = rng.sample(range(len(data)), min(n * 3, len(data)))
    out = []
    for idx in indices:
        if len(out) >= n:
            break
        s = data[idx]
        try:
            image = coco_loader.open(s["image"])
        except FileNotFoundError:
            continue
        out.append({
            "image": image,
            "image_path": s["image"],
            "conversations": s["conversations"],
            "id": s.get("id", str(idx)),
            "n_turns": len(s["conversations"]),
        })
    print(f"  sampled: {len(out)}")
    return out


def sample_refcoco(rc_dir: Path, n: int, seed: int):
    """RefCOCO — 视觉定位 (lmms-lab/RefCOCO)。"""
    from datasets import load_dataset
    print(f"[refcoco] loading from {rc_dir}")
    ds = None
    for split in ["train", "validation", "val"]:
        try:
            ds = load_dataset(str(rc_dir), split=split, trust_remote_code=True)
            print(f"  using split={split}, total {len(ds)} samples")
            break
        except Exception:
            continue
    if ds is None:
        raise RuntimeError(f"RefCOCO 加载失败: {rc_dir}")

    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n * 3, len(ds)))
    out = []
    for i in indices:
        if len(out) >= n:
            break
        s = ds[i]
        # image
        img_field = s.get("image")
        if isinstance(img_field, dict) and "bytes" in img_field:
            image = Image.open(io.BytesIO(img_field["bytes"])).convert("RGB")
        elif hasattr(img_field, "convert"):
            image = img_field.convert("RGB")
        else:
            continue
        iw, ih = image.size

        # ref expression
        ref = None
        for key in ["answer", "sentences", "sentence", "ref", "caption"]:
            v = s.get(key)
            if isinstance(v, list) and v:
                v = v[0]
                if isinstance(v, dict):
                    v = v.get("sent") or v.get("raw") or v.get("text")
            if isinstance(v, str) and v.strip():
                ref = v.strip()
                break
        if not ref:
            continue

        # bbox (COCO xywh 像素 → 归一化 xyxy)
        bbox = s.get("bbox") or s.get("box")
        if not bbox or len(bbox) != 4:
            continue
        if max(bbox) > 1.5:
            x, y, w, h = bbox
            bbox_norm = (x / iw, y / ih, (x + w) / iw, (y + h) / ih)
        else:
            bbox_norm = tuple(bbox)

        out.append({
            "image": image,
            "ref": ref,
            "bbox_raw": list(bbox),
            "bbox_norm": bbox_norm,
            "image_size": (iw, ih),
        })
    print(f"  sampled: {len(out)}")
    return out


def sample_sharegpt4v(json_path: Path, coco_loader: CocoZipLoader,
                      n: int, seed: int):
    """ShareGPT4V — 长 caption。只取 COCO 子集。"""
    print(f"[sharegpt4v] reading {json_path}")
    with open(json_path) as f:
        data = json.load(f)
    coco_data = [s for s in data if "coco" in s.get("image", "").lower()]
    print(f"  total: {len(data)}, COCO subset: {len(coco_data)}")

    rng = random.Random(seed)
    samples = rng.sample(coco_data, min(n * 3, len(coco_data)))
    out = []
    for s in samples:
        if len(out) >= n:
            break
        try:
            fn = Path(s["image"]).name
            image = coco_loader.open(fn)
        except FileNotFoundError:
            continue
        gpt_words = sum(len(t["value"].split())
                        for t in s["conversations"] if t["from"] == "gpt")
        out.append({
            "image": image,
            "image_path": s["image"],
            "conversations": s["conversations"],
            "id": s.get("id", "?"),
            "gpt_word_count": gpt_words,
        })
    print(f"  sampled: {len(out)}")
    return out


# ============================================================================
# HTML 渲染
# ============================================================================

CSS = """
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'PingFang SC', 'Helvetica Neue', sans-serif;
  background: linear-gradient(135deg, #f5f7fa 0%, #ebeef3 100%);
  margin: 0; padding: 24px 16px; color: #1a202c; line-height: 1.55;
}
.container { max-width: 1280px; margin: 0 auto; }
h1 { font-size: 26px; margin: 0 0 6px; color: #0f172a; }
h2 {
  font-size: 22px; margin: 32px 0 12px;
  padding: 10px 14px; border-left: 5px solid;
  background: white; border-radius: 0 8px 8px 0;
}
h2.h-llava   { border-color: #3b82f6; color: #1e40af; }
h2.h-refcoco { border-color: #ef4444; color: #b91c1c; }
h2.h-sharegpt{ border-color: #10b981; color: #047857; }
.lead { color: #475569; font-size: 14px; max-width: 920px; margin-bottom: 18px; }

.intro {
  background: white; border-radius: 12px; padding: 22px 26px;
  box-shadow: 0 2px 8px rgba(15,23,42,0.05); margin-bottom: 24px;
}
.dataset-intro {
  background: white; border-radius: 12px; padding: 16px 20px;
  margin-bottom: 16px; box-shadow: 0 1px 4px rgba(15,23,42,0.04);
}
.dataset-intro h3 { margin: 0 0 6px; font-size: 16px; }
.dataset-intro .meta {
  display: flex; gap: 16px; font-size: 12px; color: #64748b; margin-top: 8px;
}
.dataset-intro .meta span {
  background: #f1f5f9; padding: 3px 9px; border-radius: 12px;
}

.grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(540px, 1fr));
  gap: 14px;
}
.card {
  background: white; border-radius: 10px; overflow: hidden;
  box-shadow: 0 2px 8px rgba(15,23,42,0.06);
  display: flex; flex-direction: column;
}
.card-img-wrap {
  position: relative; background: #1e293b;
  display: flex; justify-content: center; align-items: center;
  max-height: 360px;
}
.card-img-wrap img {
  width: 100%; max-height: 360px; object-fit: contain;
  display: block;
}
.card-tag {
  position: absolute; top: 6px; left: 6px;
  background: rgba(0,0,0,0.65); color: white; font-size: 11px;
  padding: 3px 8px; border-radius: 4px; font-family: monospace;
}
.card-body { padding: 12px 14px; }
.card-meta {
  font-size: 11px; color: #64748b; margin-bottom: 10px;
}

/* turn (one user / assistant) */
.turn {
  margin: 6px 0; padding: 8px 11px;
  border-radius: 6px; border-left: 3px solid;
  font-size: 13px;
}
.turn-user { background: #f8fafc; border-color: #94a3b8; }
.turn-asst {
  background: #ecfdf5; border-color: #10b981;
}
.turn .role {
  font-size: 10px; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.4px; margin-bottom: 4px;
  color: #475569;
}
.turn-asst .role { color: #047857; }
.turn .content {
  font-size: 13px; line-height: 1.5;
  word-wrap: break-word;
}
.turn .content code, .turn .content .img-token {
  font-family: 'SF Mono', Monaco, monospace;
  font-size: 11px; background: #fff7ed; color: #c2410c;
  padding: 1px 5px; border-radius: 3px;
}
.turn .content .bbox-token {
  font-family: 'SF Mono', Monaco, monospace;
  font-size: 12px; background: #fef2f2; color: #b91c1c;
  padding: 2px 6px; border-radius: 4px; font-weight: 600;
}

details.raw {
  margin-top: 8px; font-size: 11px;
}
details.raw summary {
  cursor: pointer; color: #6366f1; font-weight: 500;
}
details.raw pre {
  background: #1e293b; color: #cbd5e1;
  padding: 8px 10px; border-radius: 4px;
  overflow-x: auto; font-size: 11px; line-height: 1.4;
  margin: 6px 0 0;
}

.template-demo {
  background: #1e293b; color: #cbd5e1;
  padding: 14px 18px; border-radius: 8px;
  font-family: 'SF Mono', Monaco, monospace;
  font-size: 12px; line-height: 1.6;
  overflow-x: auto;
}
.template-demo .marker { color: #fbbf24; }
.template-demo .role { color: #60a5fa; font-weight: 700; }
.template-demo .image-block { color: #f87171; }
.template-demo .asst-content {
  background: rgba(16,185,129,0.15); color: #6ee7b7;
  padding: 1px 4px; border-radius: 3px;
}
.template-demo .label-loss {
  display: inline-block; background: #064e3b; color: #6ee7b7;
  padding: 1px 6px; border-radius: 8px; font-size: 10px;
  margin-left: 6px;
}
.template-demo .label-mask {
  display: inline-block; background: #475569; color: #cbd5e1;
  padding: 1px 6px; border-radius: 8px; font-size: 10px;
  margin-left: 6px;
}
"""


def html_escape(s: str) -> str:
    return html_mod.escape(s)


def render_image_token_placeholder(text: str) -> str:
    """把文本里的 <image> 替换为高亮的占位提示。"""
    s = html_escape(text)
    s = s.replace(
        "&lt;image&gt;",
        f'<span class="img-token">&lt;image&gt; ×{NUM_IMAGE_TOKENS}</span>',
    )
    return s


def render_conversation_html(conversations) -> str:
    out = []
    for turn in conversations:
        is_user = turn["from"] == "human"
        cls = "turn-user" if is_user else "turn-asst"
        role_label = "👤 USER (mask, no loss)" if is_user else "🤖 ASSISTANT (loss)"
        content = render_image_token_placeholder(turn["value"])
        # 长 caption 折行更友好：保留段落
        content = content.replace("\n", "<br>")
        out.append(
            f'<div class="turn {cls}">'
            f'<div class="role">{role_label}</div>'
            f'<div class="content">{content}</div>'
            f'</div>'
        )
    return "".join(out)


def card_llava(sample, idx) -> str:
    img_b64 = to_b64_jpeg(sample["image"])
    conv_html = render_conversation_html(sample["conversations"])
    return f"""
    <div class="card">
      <div class="card-img-wrap">
        <img src="data:image/jpeg;base64,{img_b64}" alt="{html_escape(sample['image_path'])}" />
        <div class="card-tag">#{idx} {html_escape(sample['image_path'])}</div>
      </div>
      <div class="card-body">
        <div class="card-meta">{sample['n_turns']} turns · LLaVA-Instruct multi-turn VQA</div>
        {conv_html}
      </div>
    </div>"""


def card_refcoco(sample, idx) -> str:
    boxed = draw_bbox_on_image(sample["image"], sample["bbox_norm"])
    img_b64 = to_b64_jpeg(boxed)
    ref = sample["ref"]
    bn = sample["bbox_norm"]
    bbox_str = f"&lt;box&gt;({bn[0]:.3f},{bn[1]:.3f}),({bn[2]:.3f},{bn[3]:.3f})&lt;/box&gt;"
    raw_block = (
        f"image_size: {sample['image_size']}\n"
        f"ref:        {ref!r}\n"
        f"bbox_raw:   {sample['bbox_raw']}    (COCO 像素 [x, y, w, h])\n"
        f"bbox_norm:  ({bn[0]:.3f}, {bn[1]:.3f}, {bn[2]:.3f}, {bn[3]:.3f})  (xyxy 0-1)"
    )
    return f"""
    <div class="card">
      <div class="card-img-wrap">
        <img src="data:image/jpeg;base64,{img_b64}" alt="refcoco sample {idx}" />
        <div class="card-tag">#{idx} bbox 红框为 GT</div>
      </div>
      <div class="card-body">
        <div class="card-meta">RefCOCO grounding · ref expression → bbox 输出</div>
        <div class="turn turn-user">
          <div class="role">👤 USER (mask)</div>
          <div class="content">
            <span class="img-token">&lt;image&gt; ×{NUM_IMAGE_TOKENS}</span>
            <br>Provide the bounding box coordinates of {html_escape(ref)}.
          </div>
        </div>
        <div class="turn turn-asst">
          <div class="role">🤖 ASSISTANT (loss)</div>
          <div class="content"><span class="bbox-token">{bbox_str}</span></div>
        </div>
        <details class="raw">
          <summary>HF 原始字段</summary>
          <pre>{html_escape(raw_block)}</pre>
        </details>
      </div>
    </div>"""


def card_sharegpt4v(sample, idx) -> str:
    img_b64 = to_b64_jpeg(sample["image"])
    conv_html = render_conversation_html(sample["conversations"])
    return f"""
    <div class="card">
      <div class="card-img-wrap">
        <img src="data:image/jpeg;base64,{img_b64}" alt="{html_escape(sample['image_path'])}" />
        <div class="card-tag">#{idx} {html_escape(Path(sample['image_path']).name)}</div>
      </div>
      <div class="card-body">
        <div class="card-meta">ShareGPT4V long caption · GPT-4V 标注的详细描述 ({sample['gpt_word_count']} 词)</div>
        {conv_html}
      </div>
    </div>"""


def render_template_demo() -> str:
    """展示 ChatFormatter 把单 turn 包装成 chat 格式的过程。"""
    return """
<div class="template-demo">
<span class="marker">&lt;|im_start|&gt;</span><span class="role">user</span>
<span class="image-block">&lt;image&gt; &lt;image&gt; &lt;image&gt; ... (×729 个 image_token_id)</span>
What is the man wearing?
<span class="marker">&lt;|im_end|&gt;</span> <span class="label-mask">labels = -100, no loss</span>
<span class="marker">&lt;|im_start|&gt;</span><span class="role">assistant</span> <span class="label-mask">prefix mask -100</span>
<span class="asst-content">The man is wearing a red jacket and blue jeans.</span> <span class="label-loss">这部分算 loss</span>
<span class="marker">&lt;|im_end|&gt;</span> <span class="label-loss">算 loss (教 EOS)</span>
</div>
"""


def render_html(llava_samples, refcoco_samples, sharegpt_samples, ckpt_info=""):
    llava_cards = "\n".join(card_llava(s, i+1) for i, s in enumerate(llava_samples))
    refcoco_cards = "\n".join(card_refcoco(s, i+1) for i, s in enumerate(refcoco_samples))
    sharegpt_cards = "\n".join(card_sharegpt4v(s, i+1) for i, s in enumerate(sharegpt_samples))
    template_demo = render_template_demo()

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>Stage 2 训练数据抽样可视化</title>
<style>{CSS}</style>
</head>
<body>
<div class="container">
  <h1>📚 Stage 2 训练数据抽样可视化</h1>
  <div class="lead">
    每个 dataset 抽 {len(llava_samples)} / {len(refcoco_samples)} / {len(sharegpt_samples)} 个样本。
    所有图片 base64 内嵌。{ckpt_info}
  </div>

  <div class="intro">
    <h2 style="border:none;background:transparent;padding:0;margin:0 0 10px;font-size:18px">
      🔍 数据如何被加工成训练 batch
    </h2>
    <p style="color:#475569;font-size:13px;margin:0 0 12px">
      所有 3 个 dataset 的 <code>{{from, value}}</code> 对话格式被
      <code>ChatFormatter</code> 统一包装成 Qwen2.5 chat template；
      文本里的 <code>&lt;image&gt;</code> 占位符展开成 729 个 <code>image_token_id</code>；
      只有 <span style="color:#047857;font-weight:600">assistant 的 content + &lt;|im_end|&gt;</span>
      算 loss（user / image / role marker / prefix 全部 mask 为 -100）。
    </p>
    {template_demo}
  </div>

  <h2 class="h-llava">📘 1. LLaVA-Instruct-150K · 多轮 VQA + 推理</h2>
  <div class="dataset-intro">
    <h3>解决的问题</h3>
    教模型回答关于图像的开放式问题、做简单推理、保持多轮对话连贯性。
    <div class="meta">
      <span>n_train = 150K</span>
      <span>图源: COCO train2017</span>
      <span>多轮: 是 (avg ~3 turns)</span>
      <span>答案长度: 中等 (10-50 词)</span>
    </div>
  </div>
  <div class="grid">
    {llava_cards}
  </div>

  <h2 class="h-refcoco">📕 2. RefCOCO · 视觉定位（grounding）</h2>
  <div class="dataset-intro">
    <h3>解决的问题</h3>
    给一段指代表达（"the man in red"），让模型输出该物体的 bounding box。
    教模型把视觉特征跟自然语言描述对齐到具体的图像区域，并学会
    <code>&lt;box&gt;(x1,y1),(x2,y2)&lt;/box&gt;</code> 输出格式。
    <div class="meta">
      <span>n_train ≈ 50K</span>
      <span>图源: 自带 (COCO val2014)</span>
      <span>多轮: 否 (单 Q+A)</span>
      <span>答案长度: 极短 (固定格式)</span>
    </div>
  </div>
  <div class="grid">
    {refcoco_cards}
  </div>

  <h2 class="h-sharegpt">📗 3. ShareGPT4V · 详细长 caption</h2>
  <div class="dataset-intro">
    <h3>解决的问题</h3>
    教模型生成段落级别的详细图像描述（GPT-4V 蒸馏来的高质量 caption），
    覆盖物体、属性、空间关系、场景氛围、文字内容等多维度信息。
    <div class="meta">
      <span>n_train ≈ 100K (filtered to COCO)</span>
      <span>图源: COCO train2017</span>
      <span>多轮: 否 (单 Q+长 A)</span>
      <span>答案长度: 长 (100-300 词)</span>
    </div>
  </div>
  <div class="grid">
    {sharegpt_cards}
  </div>
</div>
</body>
</html>
"""


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2_data_root", required=True,
                    help="Stage 2 数据根目录，含 llava_instruct/, refcoco/, sharegpt4v/, coco/")
    ap.add_argument("--output", required=True,
                    help="输出 HTML 路径，建议 Drive: /content/drive/MyDrive/qwenvl3/data_samples/stage2_training_data.html")
    ap.add_argument("--n_per_dataset", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_root = Path(args.stage2_data_root)

    # COCO image loader (LLaVA-Instruct + ShareGPT4V 都用)
    coco_zip = data_root / "coco" / "train2017.zip"
    if not coco_zip.exists():
        raise FileNotFoundError(f"COCO zip 不存在: {coco_zip}")
    print(f"[init] COCO zip: {coco_zip.stat().st_size / 1e9:.1f}GB")
    coco_loader = CocoZipLoader(coco_zip)

    # 1. LLaVA-Instruct
    llava_json = data_root / "llava_instruct" / "llava_instruct_150k.json"
    llava_samples = sample_llava_instruct(
        llava_json, coco_loader, args.n_per_dataset, args.seed,
    )

    # 2. RefCOCO
    rc_dir = data_root / "refcoco"
    refcoco_samples = sample_refcoco(rc_dir, args.n_per_dataset, args.seed)

    # 3. ShareGPT4V (json 文件名可能不固定，扫一下)
    sg_dir = data_root / "sharegpt4v"
    sg_jsons = sorted(sg_dir.rglob("*.json"))
    if not sg_jsons:
        raise FileNotFoundError(f"ShareGPT4V json 不存在: {sg_dir}")
    sharegpt_samples = sample_sharegpt4v(
        sg_jsons[0], coco_loader, args.n_per_dataset, args.seed,
    )

    # 渲染 HTML
    html = render_html(llava_samples, refcoco_samples, sharegpt_samples)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    size_mb = out_path.stat().st_size / 1e6
    print(f"\n=== 完成 ===")
    print(f"HTML 大小: {size_mb:.1f}MB")
    print(f"路径:      {out_path}")
    print(f"\n下一步: 在 Drive 网页里下载该文件，本地浏览器打开。")


if __name__ == "__main__":
    main()
