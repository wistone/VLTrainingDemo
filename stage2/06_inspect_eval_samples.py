"""Eval 结果抽样可视化 — 每个任务分层抽 10 个样本，渲染本地 HTML。

读取 04_eval_stage2.py 的 *.json 输出 + 重新加载对应 HF dataset 拿图，
为每个任务的 best 3 / random 4 / worst 3 case 生成可视化卡片。

帮助：
  - **直观感受**模型在每个任务上的表现
  - 看到 **best case 模型擅长什么** + **worst case 死在哪**
  - RefCOCO 卡片直接画 GT 框 (绿) + 预测框 (红)
  - 文本任务卡片对比 GT vs 模型生成

== 用法 ==

  python stage2/06_inspect_eval_samples.py \\
      --eval_out_dir /content/drive/MyDrive/qwenvl3/eval_stage2/stage2_ckpt \\
      --eval_data_root /content/drive/MyDrive/qwenvl3/data/eval \\
      --stage2_data_root /content/drive/MyDrive/qwenvl3/data/stage2

输出: <eval_out_dir>/<base>_step<N>_inspect_samples.html (单文件，base64 图)
     例如 stage2_ckpt_step8000_inspect_samples.html
"""
import argparse
import base64
import html as html_mod
import io
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


THUMB_SIZE = 480


# ============================================================================
# 图像处理
# ============================================================================

def to_b64_jpeg(image: Image.Image, max_size=THUMB_SIZE, quality=85) -> str:
    img = image.copy()
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def draw_bbox(image, bbox_norm, color, width=4, label=None):
    """在图上画 bbox。bbox_norm = (x1, y1, x2, y2) 归一化 0-1。"""
    if bbox_norm is None:
        return image
    img = image.copy()
    draw = ImageDraw.Draw(img)
    iw, ih = img.size
    x1, y1, x2, y2 = bbox_norm
    px = [(x1 * iw, y1 * ih), (x2 * iw, y2 * ih)]
    draw.rectangle(px, outline=color, width=width)
    # label
    if label:
        try:
            font = ImageFont.load_default()
            draw.text((px[0][0] + 4, px[0][1] + 4), label, fill=color, font=font)
        except Exception:
            pass
    return img


# ============================================================================
# 分层抽样：best 3 / random 4 / worst 3
# ============================================================================

def stratified_pick(samples, score_key, n_best=3, n_random=4, n_worst=3,
                    seed=42, higher_is_better=True):
    """按 score_key 排序后，返回 (best, random, worst) 三组样本列表。"""
    if not samples:
        return [], [], []
    sorted_s = sorted(samples,
                      key=lambda s: s.get(score_key, 0),
                      reverse=higher_is_better)
    n_total = len(sorted_s)
    if n_total <= (n_best + n_random + n_worst):
        # 数据不够分层，全返回
        return sorted_s[:n_best], [], sorted_s[-min(n_worst, n_total - n_best):] if n_total > n_best else []

    best = sorted_s[:n_best]
    worst = sorted_s[-n_worst:]
    middle_pool = sorted_s[n_best:-n_worst]
    rng = random.Random(seed)
    random_sel = rng.sample(middle_pool, min(n_random, len(middle_pool)))
    return best, random_sel, worst


# ============================================================================
# HF dataset 图片加载
# ============================================================================

def load_hf_dataset_image(local_dir, split, idx):
    """从 HF local dataset 加载第 idx 个样本的图。返回 PIL.Image 或 None。"""
    from datasets import load_dataset
    try:
        ds = load_dataset(str(local_dir), split=split, trust_remote_code=True)
    except Exception:
        try:
            ds_dict = load_dataset(str(local_dir), trust_remote_code=True)
            ds = ds_dict[list(ds_dict.keys())[0]]
        except Exception as e:
            print(f"  [warn] HF dataset 加载失败: {local_dir} - {e}")
            return None

    if idx >= len(ds):
        return None
    sample = ds[idx]
    img_field = sample.get("image")
    if isinstance(img_field, dict) and "bytes" in img_field:
        return Image.open(io.BytesIO(img_field["bytes"])).convert("RGB")
    if hasattr(img_field, "convert"):
        return img_field.convert("RGB")
    return None


def find_hf_split(local_dir):
    """探测 HF dataset 里实际存在的 split 名（return val/test/etc.）"""
    from datasets import load_dataset
    for split in ["validation", "val", "test", "train"]:
        try:
            ds = load_dataset(str(local_dir), split=split, trust_remote_code=True)
            return split
        except Exception:
            continue
    try:
        ds_dict = load_dataset(str(local_dir), trust_remote_code=True)
        return list(ds_dict.keys())[0]
    except Exception:
        return None


# ============================================================================
# CSS
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
.lead { color: #475569; font-size: 14px; margin-bottom: 24px; }
h2 {
  font-size: 22px; margin: 32px 0 4px;
  padding: 10px 14px; border-left: 5px solid;
  background: white; border-radius: 0 8px 8px 0;
}
h2.h-refcoco_val   { border-color: #ef4444; color: #b91c1c; }
h2.h-refcoco_testA { border-color: #f97316; color: #c2410c; }
h2.h-refcoco_testB { border-color: #eab308; color: #854d0e; }
h2.h-pope          { border-color: #06b6d4; color: #0e7490; }
h2.h-vqav2         { border-color: #3b82f6; color: #1e40af; }
h2.h-nocaps        { border-color: #10b981; color: #047857; }

.task-summary {
  background: white; border-radius: 8px; padding: 12px 16px;
  font-size: 13px; color: #475569; margin: 8px 0 14px;
  box-shadow: 0 1px 3px rgba(15,23,42,0.04);
}
.task-summary code {
  background: #f1f5f9; padding: 1px 6px; border-radius: 3px;
  font-family: 'SF Mono', Monaco, monospace;
}

.section-tier {
  display: flex; align-items: center; gap: 10px;
  margin: 14px 0 8px; font-size: 14px; font-weight: 600;
}
.tier-badge {
  display: inline-block; padding: 3px 10px; border-radius: 12px;
  font-size: 12px; font-weight: 600;
}
.tier-best   { background: #dcfce7; color: #15803d; }
.tier-random { background: #e0e7ff; color: #3730a3; }
.tier-worst  { background: #fee2e2; color: #b91c1c; }

.grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
  gap: 14px;
}
.card {
  background: white; border-radius: 10px; overflow: hidden;
  box-shadow: 0 2px 8px rgba(15,23,42,0.06);
  display: flex; flex-direction: column; border: 2px solid transparent;
}
.card.tier-best   { border-color: #86efac; }
.card.tier-random { border-color: #c7d2fe; }
.card.tier-worst  { border-color: #fca5a5; }

.card-img-wrap {
  position: relative; background: #1e293b;
  display: flex; justify-content: center; align-items: center;
  max-height: 360px;
}
.card-img-wrap img {
  width: 100%; max-height: 360px; object-fit: contain; display: block;
}
.card-tag {
  position: absolute; top: 6px; left: 6px;
  background: rgba(0,0,0,0.65); color: white; font-size: 11px;
  padding: 3px 8px; border-radius: 4px; font-family: monospace;
}
.card-score {
  position: absolute; top: 6px; right: 6px;
  background: rgba(255,255,255,0.92); font-size: 12px; font-weight: 700;
  padding: 3px 8px; border-radius: 4px;
}
.card-score.high { color: #15803d; }
.card-score.mid  { color: #ca8a04; }
.card-score.low  { color: #b91c1c; }

.card-body { padding: 12px 14px; }
.row {
  margin: 6px 0; padding: 6px 9px; border-radius: 5px;
  font-size: 12px; line-height: 1.45;
}
.row.gt   { background: #f0fdf4; border-left: 3px solid #22c55e; }
.row.gen  { background: #eff6ff; border-left: 3px solid #3b82f6; }
.row.q    { background: #fefce8; border-left: 3px solid #eab308; }
.row.meta { background: #f8fafc; border-left: 3px solid #94a3b8; color: #475569; }
.row .label {
  font-size: 10px; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.4px; margin-bottom: 3px; color: #475569;
}
.row.gt  .label { color: #15803d; }
.row.gen .label { color: #1e40af; }
.row.q   .label { color: #854d0e; }
.row .text { color: #1a202c; word-wrap: break-word; }

.bbox-ref {
  display: inline-block; background: #fef2f2; color: #b91c1c;
  padding: 1px 6px; border-radius: 3px; font-size: 11px;
  font-family: 'SF Mono', monospace;
}
.legend-box {
  display: inline-flex; align-items: center; gap: 4px; font-size: 11px;
  background: #f8fafc; padding: 2px 8px; border-radius: 12px; margin-right: 6px;
}
.legend-box::before {
  content: ''; display: inline-block; width: 14px; height: 14px;
  border: 3px solid; border-radius: 2px;
}
.legend-box.gt::before    { border-color: #10b981; }
.legend-box.pred::before  { border-color: #ef4444; }

.overall {
  background: white; border-radius: 12px; padding: 22px 26px;
  box-shadow: 0 2px 8px rgba(15,23,42,0.05); margin-bottom: 24px;
}
.overall h2 {
  background: transparent; padding: 0; margin: 0 0 12px;
  border-left-width: 0; border-bottom: 2px solid #8b5cf6;
  padding-bottom: 8px;
}
.overall h3 { font-size: 15px; margin: 12px 0 8px; color: #334155; }
.setup-summary {
  background: #f8fafc; padding: 10px 14px; border-radius: 6px;
  font-size: 12px; line-height: 1.6;
}
.setup-summary code {
  background: #fff; padding: 1px 5px; border-radius: 3px;
  font-family: 'SF Mono', Monaco, monospace; font-size: 11px;
}
.metrics-table {
  width: 100%; border-collapse: collapse; font-size: 12px;
  margin: 4px 0 8px;
}
.metrics-table th, .metrics-table td {
  padding: 6px 10px; text-align: left;
  border-bottom: 1px solid #e2e8f0;
}
.metrics-table th {
  background: #f1f5f9; color: #475569; font-weight: 600;
  font-size: 11px; text-transform: uppercase; letter-spacing: 0.4px;
}
.metrics-table tr:nth-child(even) { background: #fafbfc; }
.metrics-table code {
  background: #f1f5f9; padding: 1px 5px; border-radius: 3px;
  font-family: 'SF Mono', Monaco, monospace; font-size: 11px;
}
"""


def html_escape(s):
    return html_mod.escape(str(s))


# ============================================================================
# SOTA 参考值（2024-2025 主流 VL 模型）
# ============================================================================

SOTA_REFERENCES = {
    # task_id -> dict of {metric_key: {"ours_label": ..., "models": [...]}}
    "refcoco_val": {
        "acc@0.5": {
            "ours_label": "Acc@0.5",
            "models": [
                ("LLaVA-1.5-7B",       "30%",   "7B 全参，无专项 grounding 训练"),
                ("Shikra-7B",          "87%",   "7B 全参 + 专项 grounding"),
                ("Qwen-VL-7B",         "88%",   "7B 全参，含动态分辨率 + 大量 grounding 数据"),
                ("Qwen2.5-VL-72B",     "94%",   "72B 全参，SOTA"),
                ("GPT-4o",             "80%",   "通用模型"),
            ],
        },
    },
    "refcoco_testA": {
        "acc@0.5": {
            "ours_label": "Acc@0.5",
            "models": [
                ("LLaVA-1.5-7B",       "32%",   "—"),
                ("Qwen-VL-7B",         "92%",   "—"),
                ("Qwen2.5-VL-72B",     "94%",   "SOTA"),
            ],
        },
    },
    "refcoco_testB": {
        "acc@0.5": {
            "ours_label": "Acc@0.5",
            "models": [
                ("LLaVA-1.5-7B",       "28%",   "—"),
                ("Qwen-VL-7B",         "84%",   "—"),
                ("Qwen2.5-VL-72B",     "91%",   "SOTA"),
            ],
        },
    },
    "pope": {
        "f1": {
            "ours_label": "F1",
            "models": [
                ("InstructBLIP-7B",    "85%",   "—"),
                ("LLaVA-1.5-7B",       "86%",   "—"),
                ("Qwen2-VL-72B",       "87%",   "—"),
                ("InternVL2-26B",      "88%",   "—"),
                ("GPT-4V",             "88-92%", "—"),
            ],
        },
    },
    "vqav2": {
        "accuracy": {
            "ours_label": "Accuracy",
            "models": [
                ("LLaVA-1.5-7B",       "78.5%", "7B 全参 SFT"),
                ("Qwen-VL-7B",         "78.8%", "—"),
                ("InternVL-1.0",       "79%",   "—"),
                ("GPT-4o",             "82%",   "—"),
                ("Qwen2.5-VL-72B",     "84%",   "SOTA"),
            ],
        },
    },
    "nocaps": {
        # NoCaps 标准是 CIDEr 我们用 word_recall，无法直接比；备注用
        "avg_gen_length": {
            "ours_label": "avg_gen_length (词)",
            "models": [
                ("LLaVA-1.5-7B",       "~80",   "—"),
                ("(SOTA 评测一般用 CIDEr)", "—", "我们用 word_recall 自定义指标，不可直比"),
            ],
        },
    },
}


def fmt_metric_value(v, key):
    """根据 metric key 决定如何格式化数字。"""
    if isinstance(v, float):
        if any(x in key for x in ("ratio", "acc", "rate", "@", "f1", "precision", "recall")):
            return f"{v:.2%}"
        return f"{v:.3f}"
    return str(v)


def render_overall_summary(summary_metrics, eval_out_dir_name, ckpt_step=None):
    """生成顶部「整体结果 + SOTA 对比」section。"""
    # 1) 整体指标 table
    ours_rows = []
    for task_id, metrics in summary_metrics.items():
        if not isinstance(metrics, dict):
            continue
        for k, v in metrics.items():
            if isinstance(v, (int, float, str)):
                ours_rows.append(
                    f"<tr><td>{html_escape(task_id)}</td>"
                    f"<td><code>{html_escape(k)}</code></td>"
                    f"<td><b>{html_escape(fmt_metric_value(v, k))}</b></td></tr>"
                )

    ours_table = f"""
    <h3 style="margin-top:8px">📋 我们这次的全部指标</h3>
    <table class="metrics-table">
      <thead><tr><th>Task</th><th>Metric</th><th>Value</th></tr></thead>
      <tbody>{"".join(ours_rows)}</tbody>
    </table>"""

    # 2) SOTA 对比 table（只针对主指标）
    sota_rows = []
    for task_id, metric_specs in SOTA_REFERENCES.items():
        if task_id not in summary_metrics:
            continue
        ours_metrics = summary_metrics[task_id]
        if not isinstance(ours_metrics, dict):
            continue
        for metric_key, spec in metric_specs.items():
            if metric_key not in ours_metrics:
                continue
            ours_value = ours_metrics[metric_key]
            ours_str = fmt_metric_value(ours_value, metric_key)
            for model_name, sota_value, note in spec["models"]:
                sota_rows.append(
                    f"<tr>"
                    f"<td>{html_escape(task_id)}</td>"
                    f"<td>{html_escape(spec['ours_label'])}</td>"
                    f"<td><b>{html_escape(ours_str)}</b></td>"
                    f"<td>{html_escape(model_name)}</td>"
                    f"<td>{html_escape(sota_value)}</td>"
                    f"<td style='color:#64748b;font-size:11px'>{html_escape(note)}</td>"
                    f"</tr>"
                )

    sota_table = f"""
    <h3 style="margin-top:18px">🏆 vs 业界主流 VL 模型</h3>
    <p style="font-size:12px;color:#64748b;margin:6px 0">
      参考的都是 7B-72B 全参 finetune 模型；我们是 1.5B base + LoRA 18M-78M trainable，
      参数量小 4-50×，训练数据少 1000-10000×，差距主要来自这两项。
    </p>
    <table class="metrics-table">
      <thead><tr>
        <th>Task</th><th>Metric</th><th>我们</th><th>对照模型</th><th>对照值</th><th>备注</th>
      </tr></thead>
      <tbody>{"".join(sota_rows)}</tbody>
    </table>"""

    # 3) 我们的 setup 摘要
    step_str = f"step {ckpt_step}" if ckpt_step else "final"
    setup_html = f"""
    <div class="setup-summary">
      <b>本次评测对象</b>: <code>{html_escape(eval_out_dir_name)}</code> ({step_str})<br>
      <b>训练配置</b>: Qwen2.5-1.5B-Instruct + SigLIP2-SO400M + ProjectorWithNorm + LoRA<br>
      <b>训练数据</b>: 详见 <code>stage2/README.md</code> 或 <code>stage2-v2/README.md</code>
    </div>"""

    return f"""
    <div class="overall">
      <h2 style="border-color:#8b5cf6;color:#6d28d9">📊 整体结果 + SOTA 对比</h2>
      {setup_html}
      {ours_table}
      {sota_table}
    </div>"""


# ============================================================================
# 各 task 卡片渲染
# ============================================================================

def render_refcoco_card(s, image, tier):
    """RefCOCO: 显示图（GT 绿框 + 预测红框）+ ref + IoU"""
    iw, ih = image.size
    # 画 GT (绿) + 预测 (红)
    img_with_boxes = draw_bbox(image, s.get("gt_bbox"), color="#10b981", width=5, label="GT")
    img_with_boxes = draw_bbox(img_with_boxes, s.get("pred_bbox"),
                                color="#ef4444", width=4, label="pred")
    img_b64 = to_b64_jpeg(img_with_boxes)

    iou = s.get("iou", 0.0)
    score_class = "high" if iou >= 0.5 else ("mid" if iou >= 0.3 else "low")
    pred_box_str = (
        f"({s['pred_bbox'][0]:.2f},{s['pred_bbox'][1]:.2f},"
        f"{s['pred_bbox'][2]:.2f},{s['pred_bbox'][3]:.2f})"
        if s.get("pred_bbox") else "无解析"
    )
    gen = s.get("generated", "")[:200]
    return f"""
    <div class="card tier-{tier}">
      <div class="card-img-wrap">
        <img src="data:image/jpeg;base64,{img_b64}" />
        <div class="card-tag">#{s.get('idx')} · 图 {iw}×{ih}</div>
        <div class="card-score {score_class}">IoU {iou:.2f}</div>
      </div>
      <div class="card-body">
        <div class="row q">
          <div class="label">REFERRING EXPRESSION</div>
          <div class="text">{html_escape(s.get('ref', ''))}</div>
        </div>
        <div class="row gt">
          <div class="label">🟢 GT bbox (归一化)</div>
          <div class="text"><span class="bbox-ref">({s['gt_bbox'][0]:.2f},{s['gt_bbox'][1]:.2f},{s['gt_bbox'][2]:.2f},{s['gt_bbox'][3]:.2f})</span></div>
        </div>
        <div class="row gen">
          <div class="label">🔴 模型生成</div>
          <div class="text">原始输出: {html_escape(gen)}<br>解析 bbox: <span class="bbox-ref">{pred_box_str}</span></div>
        </div>
      </div>
    </div>"""


def render_pope_card(s, image, tier):
    """POPE: 显示图 + 问题 + GT (Yes/No) + 模型答案 + 是否正确"""
    img_b64 = to_b64_jpeg(image)
    gt = s.get("gt", "?")
    pred = s.get("pred", "?")
    correct = (gt == pred)
    score_class = "high" if correct else "low"
    score_text = "✓ 对" if correct else "✗ 错"
    return f"""
    <div class="card tier-{tier}">
      <div class="card-img-wrap">
        <img src="data:image/jpeg;base64,{img_b64}" />
        <div class="card-tag">#{s.get('idx')} · {html_escape(s.get('category', '?'))}</div>
        <div class="card-score {score_class}">{score_text}</div>
      </div>
      <div class="card-body">
        <div class="row q">
          <div class="label">QUESTION</div>
          <div class="text">{html_escape(s.get('question', ''))}</div>
        </div>
        <div class="row gt">
          <div class="label">🟢 GT</div>
          <div class="text"><b>{html_escape(gt)}</b></div>
        </div>
        <div class="row gen">
          <div class="label">🔵 模型答 → 解析为: <b>{html_escape(pred)}</b></div>
          <div class="text">原始输出: {html_escape(s.get('generated', '')[:120])}</div>
        </div>
      </div>
    </div>"""


def render_vqav2_card(s, image, tier):
    """VQAv2: 显示图 + 问题 + 10 GT 答案 + 模型答案"""
    img_b64 = to_b64_jpeg(image)
    acc = s.get("vqa_acc", 0.0)
    score_class = "high" if acc >= 0.5 else ("mid" if acc > 0 else "low")
    gt_answers = s.get("gt_answers", [])
    gt_str = " · ".join(html_escape(a) for a in gt_answers[:5])
    if len(gt_answers) > 5:
        gt_str += f" <span style='color:#94a3b8'>(+{len(gt_answers) - 5})</span>"
    return f"""
    <div class="card tier-{tier}">
      <div class="card-img-wrap">
        <img src="data:image/jpeg;base64,{img_b64}" />
        <div class="card-tag">#{s.get('idx')}</div>
        <div class="card-score {score_class}">acc {acc:.2f}</div>
      </div>
      <div class="card-body">
        <div class="row q">
          <div class="label">QUESTION</div>
          <div class="text">{html_escape(s.get('question', ''))}</div>
        </div>
        <div class="row gt">
          <div class="label">🟢 GT 答案 (10 标注员)</div>
          <div class="text">{gt_str}</div>
        </div>
        <div class="row gen">
          <div class="label">🔵 模型答</div>
          <div class="text">{html_escape(s.get('generated', '')[:200])}</div>
        </div>
      </div>
    </div>"""


def render_nocaps_card(s, image, tier):
    """NoCaps: 显示图 + 模型生成长 caption + 3 个 reference + word_recall"""
    img_b64 = to_b64_jpeg(image)
    recall = s.get("word_recall", 0.0)
    score_class = "high" if recall >= 0.35 else ("mid" if recall >= 0.20 else "low")
    refs = s.get("references", [])
    refs_str = "<br>".join(f"• {html_escape(r)}" for r in refs[:3])
    rep_badge = " 🔁" if s.get("repetition") else ""
    return f"""
    <div class="card tier-{tier}">
      <div class="card-img-wrap">
        <img src="data:image/jpeg;base64,{img_b64}" />
        <div class="card-tag">#{s.get('idx')} · {html_escape(s.get('domain', '?'))} · len={s.get('gen_length', 0)}</div>
        <div class="card-score {score_class}">recall {recall:.2f}{rep_badge}</div>
      </div>
      <div class="card-body">
        <div class="row gt">
          <div class="label">🟢 GT references (前 3 条 / 共 {s.get('n_references', '?')})</div>
          <div class="text">{refs_str}</div>
        </div>
        <div class="row gen">
          <div class="label">🔵 模型生成 ({s.get('gen_length', 0)} 词, max_run={s.get('max_run', 1)})</div>
          <div class="text">{html_escape(s.get('generated', ''))}</div>
        </div>
      </div>
    </div>"""


# ============================================================================
# 渲染一个 task 的整段 (含 best/random/worst 三栏)
# ============================================================================

def render_task_section(task_name, samples, dataset_dir, split, render_card_fn,
                        score_key, higher_is_better=True, label=None,
                        show_metrics_summary=None):
    """生成一个 task section（含 best/random/worst 3 栏）。"""
    label = label or task_name
    print(f"\n[render] {task_name}: 加载 {len(samples)} 个 eval 样本...")

    best, randm, worst = stratified_pick(
        samples, score_key=score_key,
        higher_is_better=higher_is_better,
    )

    # 加载图
    def get_image_for(s):
        idx = s.get("idx", 0) - 1   # idx 是 1-based
        return load_hf_dataset_image(dataset_dir, split, idx)

    def render_tier(tier_label, tier_samples, tier_class):
        if not tier_samples:
            return ""
        cards_html = []
        for s in tier_samples:
            try:
                img = get_image_for(s)
                if img is None:
                    print(f"  [warn] {task_name} idx={s.get('idx')} 图加载失败")
                    continue
                card = render_card_fn(s, img, tier_class)
                cards_html.append(card)
            except Exception as e:
                print(f"  [warn] {task_name} idx={s.get('idx')} 渲染失败: {e}")

        if not cards_html:
            return ""
        emoji = {"best": "🟢", "random": "🟡", "worst": "🔴"}[tier_class]
        tier_zh = {"best": "best 高分案例 (模型擅长)",
                   "random": "random 中等典型样本",
                   "worst": "worst 低分案例 (模型死在哪)"}[tier_class]
        return f"""
        <div class="section-tier">
          <span class="tier-badge tier-{tier_class}">{emoji} {tier_zh}</span>
          <span style="color:#64748b;font-size:13px">{len(tier_samples)} 个</span>
        </div>
        <div class="grid">
          {''.join(cards_html)}
        </div>"""

    metrics_html = ""
    if show_metrics_summary:
        items = []
        for k, v in show_metrics_summary.items():
            if isinstance(v, float):
                if "ratio" in k or "acc" in k or "rate" in k or "@" in k or "f1" in k:
                    items.append(f"<code>{k}={v:.2%}</code>")
                else:
                    items.append(f"<code>{k}={v:.3f}</code>")
            else:
                items.append(f"<code>{k}={v}</code>")
        metrics_html = f'<div class="task-summary"><b>整体指标</b>: {" · ".join(items)}</div>'

    return f"""
    <h2 class="h-{task_name}">{label}</h2>
    {metrics_html}
    {render_tier("Best 3", best, "best")}
    {render_tier("Random 4", randm, "random")}
    {render_tier("Worst 3", worst, "worst")}"""


# ============================================================================
# Main
# ============================================================================

TASK_CONFIG = {
    # task_id: (json_filename, score_key, higher_is_better, label, render_fn_name)
    "refcoco_val":    ("refcoco_val.json",   "iou", True, "📕 1. RefCOCO val · grounding (IoU 越高越好)", "refcoco"),
    "refcoco_testA":  ("refcoco_testA.json", "iou", True, "📕 2. RefCOCO testA · grounding (主要是人物)", "refcoco"),
    "refcoco_testB":  ("refcoco_testB.json", "iou", True, "📕 3. RefCOCO testB · grounding (主要是物体, 更难)", "refcoco"),
    "pope":           ("pope.json",          None,  None, "📗 4. POPE · 是非题幻觉测试", "pope"),
    "vqav2":          ("vqav2.json",         "vqa_acc", True, "📘 5. VQAv2 · 通用 VQA (acc 越高越好)", "vqav2"),
    "nocaps":         ("nocaps.json",        "word_recall", True, "📙 6. NoCaps · 长 caption (word_recall 越高越好)", "nocaps"),
}


def detect_step(eval_out_dir):
    """从 summary.json 找 model 路径，再去 model 目录扫 checkpoint-NNNN。"""
    summary = Path(eval_out_dir) / "summary.json"
    if not summary.exists():
        return None
    try:
        with open(summary) as f:
            data = json.load(f)
        model_path = data.get("stage2_ckpt") or data.get("ckpt_dir")
        if not model_path:
            return None
        p = Path(model_path)
        # 路径本身就是 checkpoint-NNNN？
        if p.name.startswith("checkpoint-"):
            return p.name.split("-")[1]
        # 不然扫子目录
        if p.exists():
            ckpts = sorted(
                (c for c in p.glob("checkpoint-*") if c.is_dir()),
                key=lambda c: int(c.name.split("-")[1]),
                reverse=True,
            )
            if ckpts:
                return ckpts[0].name.split("-")[1]
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_out_dir", required=True,
                    help="04_eval_stage2.py 的输出目录")
    ap.add_argument("--eval_data_root", required=True,
                    help="OOD eval 数据根目录（含 pope/, vqav2/, nocaps/）")
    ap.add_argument("--stage2_data_root", required=True,
                    help="Stage 2 数据根（含 refcoco/）")
    ap.add_argument("--output", default=None,
                    help="自定义输出路径；缺省自动 <out_dir>/<base>_step<N>_inspect_samples.html")
    ap.add_argument("--n_best", type=int, default=3)
    ap.add_argument("--n_random", type=int, default=4)
    ap.add_argument("--n_worst", type=int, default=3)
    args = ap.parse_args()

    eval_out_dir = Path(args.eval_out_dir)
    if not eval_out_dir.exists():
        raise FileNotFoundError(f"eval_out_dir 不存在: {eval_out_dir}")

    # 输出文件名
    if args.output:
        out_path = Path(args.output)
    else:
        base = eval_out_dir.name
        step = detect_step(eval_out_dir)
        if step:
            out_path = eval_out_dir / f"{base}_step{step}_inspect_samples.html"
        else:
            out_path = eval_out_dir / f"{base}_inspect_samples.html"
    print(f"[output] HTML 将写入: {out_path}\n")

    # 各 task 的数据集路径
    refcoco_dir = Path(args.stage2_data_root) / "refcoco"      # lmms-lab eval split
    pope_dir    = Path(args.eval_data_root)   / "pope"
    vqav2_dir   = Path(args.eval_data_root)   / "vqav2"
    nocaps_dir  = Path(args.eval_data_root)   / "nocaps"

    sections_html = []
    summary_metrics = {}

    for task_id, (json_name, score_key, higher_is_better, label, render_kind) in TASK_CONFIG.items():
        json_path = eval_out_dir / json_name
        if not json_path.exists():
            print(f"[skip] {task_id}: 缺 {json_name}")
            continue

        with open(json_path) as f:
            data = json.load(f)
        samples = data.get("samples", [])
        if not samples:
            print(f"[skip] {task_id}: 空 samples 列表")
            continue

        metrics = data.get("metrics", {})
        summary_metrics[task_id] = metrics

        # 各 task 加载 image 来源
        if task_id == "refcoco_val":
            split = "val"; ds_dir = refcoco_dir; render_fn = render_refcoco_card
        elif task_id == "refcoco_testA":
            split = "testA"; ds_dir = refcoco_dir; render_fn = render_refcoco_card
        elif task_id == "refcoco_testB":
            split = "testB"; ds_dir = refcoco_dir; render_fn = render_refcoco_card
        elif task_id == "pope":
            split = find_hf_split(pope_dir); ds_dir = pope_dir; render_fn = render_pope_card
        elif task_id == "vqav2":
            split = find_hf_split(vqav2_dir); ds_dir = vqav2_dir; render_fn = render_vqav2_card
        elif task_id == "nocaps":
            split = find_hf_split(nocaps_dir); ds_dir = nocaps_dir; render_fn = render_nocaps_card
        else:
            continue

        # POPE 没有 score_key，对 correct (gt==pred) 排序
        if task_id == "pope":
            for s in samples:
                s["__correct"] = 1 if (s.get("gt") == s.get("pred")) else 0
            score_key = "__correct"
            higher_is_better = True

        section = render_task_section(
            task_id, samples, ds_dir, split, render_fn,
            score_key=score_key, higher_is_better=higher_is_better, label=label,
            show_metrics_summary=metrics,
        )
        sections_html.append(section)

    if not sections_html:
        raise RuntimeError("没有生成任何 task section。检查 --eval_out_dir 是否含 *.json")

    # 顶部「整体结果 + SOTA 对比」
    ckpt_step = detect_step(eval_out_dir)
    overall_html = render_overall_summary(
        summary_metrics, eval_out_dir.name, ckpt_step=ckpt_step,
    )

    page_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>Eval Inspect Samples - {eval_out_dir.name}</title>
<meta name="google" content="notranslate">
<meta http-equiv="Content-Language" content="zh-CN">
<style>{CSS}</style>
</head>
<body>
<div class="container">
  <h1>🔍 Eval 抽样可视化 — {eval_out_dir.name}</h1>
  <div class="lead">
    上半部：整体指标 + 业界 SOTA 对比 ·
    下半部：每个任务分层抽 <b>3 best · 4 random · 3 worst</b> 渲染卡片。<br>
    RefCOCO 卡片含 <span class="legend-box gt">GT 框</span>
    <span class="legend-box pred">预测框</span> 叠加显示。
  </div>

  {overall_html}

  {''.join(sections_html)}
</div>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page_html, encoding="utf-8")
    size_mb = out_path.stat().st_size / 1e6
    print(f"\n=== 完成 ===")
    print(f"HTML 路径: {out_path}")
    print(f"HTML 大小: {size_mb:.1f}MB")
    print(f"\n下一步: 在 Drive 网页里下载该文件 → 本地浏览器双击打开")


if __name__ == "__main__":
    main()
