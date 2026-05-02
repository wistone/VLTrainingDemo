"""把两次 04_eval_stage1.py 的结果拼成本地 HTML 对比报告。

输入：两个 eval 输出目录（每个都含 captions.json 和 ablation.json）
输出：单个自包含 HTML 文件，包含：
  - 顶部摘要（caption 长度变化、ablation Δloss 变化）
  - 20 张 holdout 图，每张三栏：旧生成 / 新生成 / GT，带 Drive 缩略图

图像通过 Drive 的 thumbnail URL 渲染（用户登录的浏览器自动加载）。

用法：
  python stage1/05_compare_eval.py \\
      --old_dir /content/drive/MyDrive/qwenvl3/eval_stage1_at_4500 \\
      --new_dir /content/drive/MyDrive/qwenvl3/eval_stage1_at_11500 \\
      --old_label "step 4500 (26%)" \\
      --new_label "step 11500 (66%)" \\
      --out_html /content/drive/MyDrive/qwenvl3/stage1_eval_compare.html
"""
import argparse
import html
import json
from pathlib import Path

# Holdout 20 张图在 Drive `holdout_images/` 下的 file ID（一次性 lookup 后固化）
# 把 "00xxx/00xxxxxxxx.jpg" 这种 caption json 里的相对路径映射到 Drive thumbnail URL。
HOLDOUT_FILE_IDS = {
    "00188/001883900.jpg": "1LoaJZF7FjfpYrYWDGZciScV6Ti0ruq35",
    "00233/002336501.jpg": "1c0WsV24rmBZ75wvYYmoRTNbXo7j_4N8E",
    "00560/005605971.jpg": "1KvFkaY6llE9cBVUnTl9dRj0n1IQiyc-K",
    "00131/001312416.jpg": "10dOREIvJYJvEKQz-A1A2L50DDNhseXtm",
    "00061/000614067.jpg": "1uopfTNbnmZz1DbJDLRPATipliO93Fkwd",
    "00283/002830528.jpg": "17Wf6jfeo2JuTvlzKOJtPDHmRtCHZQ7yx",
    "00420/004208224.jpg": "1qo8a3UqbhZDShhWQbn1HjJAsp5t50Gl_",
    "00509/005091749.jpg": "1j5kRemByVE55X4Ab3FOkcKRILQRLBnHG",
    "00013/000134467.jpg": "11VojlpExwe6DXkP9tRLNQdenlZe2cE5U",
    "00183/001839237.jpg": "1-b3uizwbkwOuMfx5xkl0g1yyloGdG1R4",
    "00343/003439338.jpg": "1bP5nut3ooZfES6VT4CD-7BVChCG9no1a",
    "00075/000754812.jpg": "1W0MzqL_e8Nbtlq9XXDMDe-yyKOwAlPD5",
    "00339/003393398.jpg": "1LMLnONzAqI5j6ZEfk6glx-4msUkW1CGI",
    "00228/002286940.jpg": "1IwoWapWr7YGHOiVzwN6Nlzr1D_hfAbAh",
    "00158/001586094.jpg": "19pHh-Es3KrEJ92Y65XT9ZhbSf9H7J0Oj",
    "00288/002889383.jpg": "1eU5cIAdONLq4t2hWiHgMIWl1qF_HNUCg",
    "00118/001188493.jpg": "12dgDpUmRFF8krG5_rSyZYmUlvczEV8Te",
    "00278/002789130.jpg": "1wm6CeI3aqqR12FYKRbHbuvGjPK6szuN4",
    "00118/001185826.jpg": "12IBQr0hpt0zwv73TIlpnEh3m-r1gQxK-",
    "00093/000934915.jpg": "1bUw07MR455y7Y8G9KhOCJFfdqjeo1RgP",
}


def drive_thumb(file_id, w=600):
    return f"https://drive.google.com/thumbnail?id={file_id}&sz=w{w}"


def load_eval(eval_dir: Path):
    """读 captions.json + ablation.json，返回 (caps_by_image, ablation_dict)。"""
    caps_path = eval_dir / "captions.json"
    abl_path = eval_dir / "ablation.json"
    with open(caps_path) as f:
        caps_list = json.load(f)
    # captions.json 是 list；按 image 索引方便对比
    caps_by_image = {c["image"]: c for c in caps_list}

    ablation = None
    if abl_path.exists():
        with open(abl_path) as f:
            ablation = json.load(f)
    return caps_by_image, ablation


def caption_word_count(s: str) -> int:
    return len(s.split())


def build_html(old_caps, new_caps, old_abl, new_abl, old_label, new_label):
    # 计算摘要统计
    old_lens = [caption_word_count(c["generated"]) for c in old_caps.values()]
    new_lens = [caption_word_count(c["generated"]) for c in new_caps.values()]
    old_avg_len = sum(old_lens) / len(old_lens) if old_lens else 0
    new_avg_len = sum(new_lens) / len(new_lens) if new_lens else 0

    old_delta = old_abl["delta"] if old_abl else None
    new_delta = new_abl["delta"] if new_abl else None
    old_with = old_abl["avg_loss_with_image"] if old_abl else None
    new_with = new_abl["avg_loss_with_image"] if new_abl else None

    def fmt(v, fmt_spec=".3f"):
        return f"{v:{fmt_spec}}" if v is not None else "—"

    summary_html = f"""
    <div class="summary">
      <h2>📊 摘要对比</h2>
      <table class="summary-tbl">
        <thead><tr><th></th><th>{html.escape(old_label)}</th><th>{html.escape(new_label)}</th><th>变化</th></tr></thead>
        <tbody>
          <tr>
            <td>平均生成 caption 长度（词）</td>
            <td>{old_avg_len:.1f}</td>
            <td>{new_avg_len:.1f}</td>
            <td class="{'better' if new_avg_len > old_avg_len else 'neutral'}">
              {new_avg_len - old_avg_len:+.1f}
            </td>
          </tr>
          <tr>
            <td>带图 loss</td>
            <td>{fmt(old_with)}</td>
            <td>{fmt(new_with)}</td>
            <td class="{'better' if new_with and old_with and new_with < old_with else 'neutral'}">
              {fmt(new_with - old_with) if old_with and new_with else '—'}
            </td>
          </tr>
          <tr>
            <td>Image-token Δloss（视觉信号强度）</td>
            <td>{fmt(old_delta)}</td>
            <td>{fmt(new_delta)}</td>
            <td class="{'better' if new_delta and old_delta and new_delta > old_delta else 'neutral'}">
              {fmt(new_delta - old_delta, '+.3f') if old_delta and new_delta else '—'}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
    """

    # 每张图的对比卡片
    cards_html = []
    common_imgs = sorted(set(old_caps) & set(new_caps))
    for i, img_path in enumerate(common_imgs, 1):
        old = old_caps[img_path]
        new = new_caps[img_path]
        gt = old.get("ground_truth", "") or new.get("ground_truth", "")

        file_id = HOLDOUT_FILE_IDS.get(img_path)
        img_src = drive_thumb(file_id) if file_id else ""

        cards_html.append(f"""
        <div class="card">
          <div class="card-img-wrap">
            {'<img class="card-img" src="' + img_src + '" loading="lazy" alt="' + html.escape(img_path) + '">' if img_src else '<div class="no-img">No Drive ID</div>'}
            <div class="card-caption">#{i} · {html.escape(img_path)}</div>
          </div>
          <div class="card-body">
            <div class="row">
              <div class="label old">{html.escape(old_label)}</div>
              <div class="text">{html.escape(old["generated"]) or '<i>(empty)</i>'}</div>
              <div class="meta">{caption_word_count(old["generated"])} words</div>
            </div>
            <div class="row">
              <div class="label new">{html.escape(new_label)}</div>
              <div class="text">{html.escape(new["generated"]) or '<i>(empty)</i>'}</div>
              <div class="meta">{caption_word_count(new["generated"])} words</div>
            </div>
            <div class="row gt">
              <div class="label">Ground Truth</div>
              <div class="text">{html.escape(gt)}</div>
            </div>
          </div>
        </div>
        """)

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="UTF-8">
<title>Stage 1 Eval Compare: {html.escape(old_label)} → {html.escape(new_label)}</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'PingFang SC', 'Helvetica Neue', sans-serif;
    background: #f4f6f8; margin: 0; padding: 24px; color: #1a1a1a;
  }}
  .container {{ max-width: 1400px; margin: 0 auto; }}
  h1 {{ margin-top: 0; }}
  .summary {{
    background: white; padding: 18px 24px; border-radius: 10px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06); margin-bottom: 24px;
  }}
  .summary h2 {{ margin: 0 0 12px 0; font-size: 18px; }}
  .summary-tbl {{ border-collapse: collapse; width: 100%; }}
  .summary-tbl th, .summary-tbl td {{
    padding: 10px 12px; text-align: left; border-bottom: 1px solid #eee;
  }}
  .summary-tbl th {{ font-weight: 600; color: #6b7280; font-size: 13px; }}
  .better {{ color: #16a34a; font-weight: 600; }}
  .neutral {{ color: #6b7280; }}
  .grid {{
    display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
    gap: 16px;
  }}
  .card {{
    background: white; border-radius: 10px; overflow: hidden;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
  }}
  .card-img-wrap {{ position: relative; }}
  .card-img {{
    width: 100%; aspect-ratio: 4 / 3; object-fit: cover;
    background: #000; display: block;
  }}
  .no-img {{
    width: 100%; aspect-ratio: 4 / 3; background: #f3f4f6;
    display: flex; align-items: center; justify-content: center; color: #9ca3af;
  }}
  .card-caption {{
    position: absolute; bottom: 0; left: 0; right: 0;
    background: rgba(0,0,0,0.6); color: white; font-size: 11px;
    padding: 4px 10px;
  }}
  .card-body {{ padding: 12px 14px; }}
  .row {{ margin-bottom: 12px; }}
  .row .label {{
    font-size: 11px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.5px; color: #6b7280; margin-bottom: 4px;
  }}
  .row .label.old {{ color: #b45309; }}
  .row .label.new {{ color: #1d4ed8; }}
  .row .text {{ font-size: 13px; line-height: 1.45; }}
  .row .text i {{ color: #b91c1c; }}
  .row .meta {{ font-size: 11px; color: #9ca3af; margin-top: 2px; }}
  .row.gt .text {{
    background: #f9fafb; padding: 6px 10px; border-radius: 4px;
    color: #4b5563; font-style: italic; font-size: 12px;
  }}
</style>
</head><body>
<div class="container">
  <h1>Stage 1 训练前后对比</h1>
  <p style="color:#6b7280">{html.escape(old_label)} ⟶ {html.escape(new_label)}（同一批 holdout 20 张图）</p>
  {summary_html}
  <div class="grid">
    {''.join(cards_html)}
  </div>
</div>
</body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old_dir", required=True, help="老 eval 输出目录（含 captions.json + ablation.json）")
    ap.add_argument("--new_dir", required=True, help="新 eval 输出目录")
    ap.add_argument("--old_label", default="old")
    ap.add_argument("--new_label", default="new")
    ap.add_argument("--out_html", required=True)
    args = ap.parse_args()

    old_caps, old_abl = load_eval(Path(args.old_dir))
    new_caps, new_abl = load_eval(Path(args.new_dir))
    print(f"[load] old: {len(old_caps)} samples; new: {len(new_caps)} samples")
    common = set(old_caps) & set(new_caps)
    print(f"[load] common samples: {len(common)}")

    html_text = build_html(old_caps, new_caps, old_abl, new_abl,
                           args.old_label, args.new_label)
    out = Path(args.out_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_text, encoding="utf-8")
    print(f"[done] HTML 写入 {out}")
    print(f"[note] 用浏览器打开。Drive 缩略图需要登录 Drive 账号才能渲染。")


if __name__ == "__main__":
    main()
