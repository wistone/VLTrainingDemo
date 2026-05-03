"""Stage 4 DPO 偏好数据下载 + 格式标准化。

下载多模态偏好对数据集，转成统一格式存到 Drive：
  {prompt: str, chosen: str, rejected: str, image_id: int, image_path: str}

数据来源（按优先级 fallback）:
  1. zhiqings/LLaVA-RLHF-10K     — 跟 LLaVA 系列原生兼容 (推荐)
  2. MMInstruction/VLFeedback     — 80K 多 VLM 偏好对（备选大量数据用）
  3. zhiqings/LLaVA-RLHF-50K      — LLaVA-RLHF 大版本

== 用法 ==

  全部默认（仅下 LLaVA-RLHF 10K）:
    python stage4-dpo/01_prepare_dpo_data.py \\
        --dpo_data_root /content/drive/MyDrive/qwenvl3/data/dpo

  额外加 VLFeedback:
    python stage4-dpo/01_prepare_dpo_data.py \\
        --dpo_data_root /content/drive/MyDrive/qwenvl3/data/dpo \\
        --include_vlfeedback

== 输出 ==

  /content/drive/MyDrive/qwenvl3/data/dpo/
  ├── llava_rlhf/                       下载缓存
  │   └── ...
  ├── dpo_pairs.json                    标准化后的 DPO 对（直接给训练用）
  └── stats.json                        样本数 / 字段分布等统计
"""
import argparse
import io
import json
import os
import sys
from collections import Counter
from pathlib import Path


# ============================================================================
# 数据源候选
# ============================================================================

DPO_SOURCES = {
    "llava_rlhf_10k": {
        "candidates": [
            ("zhiqings/LLaVA-RLHF-10K",        "dataset"),
            ("zhiqings/LLaVA-RLHF",            "dataset"),    # alt name
            ("liuhaotian/LLaVA-Human-Preference-10K", "dataset"),  # alt source
        ],
        "approx_size_mb": 50,
        "description": "LLaVA-RLHF 10K 偏好对 (推荐)",
    },
    "vlfeedback": {
        "candidates": [
            ("MMInstruction/VLFeedback",    "dataset"),
        ],
        "approx_size_mb": 5000,
        "description": "VLFeedback ~80K 偏好对（多 VLM 来源）",
    },
}


# ============================================================================
# HF 下载
# ============================================================================

def fmt_size(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def hf_download(repo_id, repo_type, target_dir, label):
    from huggingface_hub import snapshot_download

    target_dir.mkdir(parents=True, exist_ok=True)
    existing = sum(f.stat().st_size for f in target_dir.rglob("*") if f.is_file())
    if existing > 1 * 1024 * 1024:
        print(f"  [skip] {label} 已下过 ({fmt_size(existing)})")
        return True

    print(f"  [download] {label}: {repo_id}")
    try:
        snapshot_download(
            repo_id=repo_id, repo_type=repo_type,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            max_workers=4,
        )
        size = sum(f.stat().st_size for f in target_dir.rglob("*") if f.is_file())
        print(f"  [done] {label}: {fmt_size(size)}")
        return True
    except Exception as e:
        print(f"  [FAIL] {label}: {str(e)[:200]}")
        return False


def download_with_fallback(source_key, target_dir):
    spec = DPO_SOURCES[source_key]
    for repo_id, repo_type in spec["candidates"]:
        ok = hf_download(repo_id, repo_type, target_dir, f"{source_key} (尝试 {repo_id})")
        if ok:
            return repo_id
    return None


# ============================================================================
# 格式标准化
# ============================================================================

def normalize_llava_rlhf(records):
    """LLaVA-RLHF-10K 字段适配。

    可能的格式（不同 mirror 字段名略有不同）：
      A: {image, conversations: [{from: human}], chosen: str, rejected: str}
      B: {image, prompt, chosen, rejected}
      C: {image, question, output_1, output_2, preference}
    """
    out = []
    field_stats = Counter()

    for r in records:
        if not isinstance(r, dict):
            continue
        field_stats.update(r.keys())

        # 取 prompt
        prompt = None
        if "prompt" in r and isinstance(r["prompt"], str):
            prompt = r["prompt"]
        elif "question" in r and isinstance(r["question"], str):
            prompt = r["question"]
        elif "conversations" in r and isinstance(r["conversations"], list):
            for turn in r["conversations"]:
                if isinstance(turn, dict) and turn.get("from") in ("human", "user"):
                    prompt = turn.get("value", "")
                    break

        # 取 chosen / rejected
        chosen, rejected = None, None
        if "chosen" in r and "rejected" in r:
            chosen = r["chosen"]
            rejected = r["rejected"]
        elif "output_1" in r and "output_2" in r and "preference" in r:
            # preference: 1 表示 output_1 是 chosen
            if r["preference"] == 1:
                chosen, rejected = r["output_1"], r["output_2"]
            else:
                chosen, rejected = r["output_2"], r["output_1"]
        elif "responses" in r and isinstance(r["responses"], list) and len(r["responses"]) >= 2:
            # 评分式：取最高分作 chosen，最低分作 rejected
            scored = [
                (resp.get("score", 0) if isinstance(resp, dict) else 0, resp)
                for resp in r["responses"]
            ]
            scored.sort(key=lambda x: x[0], reverse=True)
            best = scored[0][1]
            worst = scored[-1][1]
            chosen = best.get("text") if isinstance(best, dict) else str(best)
            rejected = worst.get("text") if isinstance(worst, dict) else str(worst)

        # image 字段（路径或 image_id）
        image_path = None
        image_id = None
        for k in ("image", "image_path", "img", "filename"):
            v = r.get(k)
            if isinstance(v, str) and v:
                image_path = v
                break
        for k in ("image_id", "id"):
            v = r.get(k)
            if isinstance(v, (int, str)):
                try:
                    image_id = int(v)
                except (TypeError, ValueError):
                    pass
                break

        # 必要字段都齐
        if not (prompt and chosen and rejected):
            continue
        if isinstance(chosen, str):
            chosen = chosen.strip()
        if isinstance(rejected, str):
            rejected = rejected.strip()
        if not chosen or not rejected:
            continue
        if chosen == rejected:
            continue   # 偶尔 chosen 跟 rejected 一模一样，没意义
        if not image_path and image_id is None:
            continue   # 没图

        out.append({
            "prompt": prompt.strip(),
            "chosen": chosen,
            "rejected": rejected,
            "image_path": image_path,
            "image_id": image_id,
        })

    return out, field_stats


def normalize_vlfeedback(records):
    """VLFeedback 字段。VLFeedback 通常字段:
      image, prompt, completions: [{model, response, ...}, ...] + preference labels
    """
    out = []
    field_stats = Counter()
    for r in records:
        if not isinstance(r, dict):
            continue
        field_stats.update(r.keys())

        prompt = r.get("prompt") or r.get("question") or ""
        completions = r.get("completions") or r.get("responses") or []
        image = r.get("image") or r.get("image_path")

        if not prompt or not isinstance(completions, list) or len(completions) < 2:
            continue
        # 按某种 quality 字段排序
        scored = []
        for c in completions:
            if not isinstance(c, dict):
                continue
            text = c.get("response") or c.get("text") or c.get("content")
            score = (c.get("annotations", {}).get("Helpfulness", {}).get("Rating", 0)
                     if isinstance(c.get("annotations"), dict) else c.get("score", 0))
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = 0
            if isinstance(text, str) and text.strip():
                scored.append((score, text.strip()))
        if len(scored) < 2:
            continue
        scored.sort(key=lambda x: x[0], reverse=True)
        chosen = scored[0][1]
        rejected = scored[-1][1]
        if chosen == rejected:
            continue

        out.append({
            "prompt": prompt.strip(),
            "chosen": chosen,
            "rejected": rejected,
            "image_path": image if isinstance(image, str) else None,
            "image_id": None,
        })
    return out, field_stats


# ============================================================================
# 加载（json / parquet 自动适配）
# ============================================================================

def load_dpo_records(local_dir):
    """从 local_dir 自动找 json/parquet/dataset 加载所有 records。返回 list[dict]。"""
    local_dir = Path(local_dir)

    # 1. 尝试 datasets.load_dataset
    try:
        from datasets import load_dataset
        for split in ("train", "validation", "test"):
            try:
                ds = load_dataset(str(local_dir), split=split, trust_remote_code=True)
                print(f"  [load] datasets.load_dataset(split={split}) → {len(ds)} 条")
                return [dict(ds[i]) for i in range(len(ds))]
            except Exception:
                continue
        try:
            ds_dict = load_dataset(str(local_dir), trust_remote_code=True)
            split = list(ds_dict.keys())[0]
            ds = ds_dict[split]
            print(f"  [load] datasets.load_dataset (default first split={split}) → {len(ds)} 条")
            return [dict(ds[i]) for i in range(len(ds))]
        except Exception as e:
            print(f"  [warn] datasets.load_dataset 失败: {str(e)[:160]}")
    except ImportError:
        pass

    # 2. 直接读 json files
    records = []
    for jf in sorted(local_dir.rglob("*.json")):
        if "config" in jf.name.lower() or "meta" in jf.name.lower():
            continue
        try:
            with open(jf) as f:
                data = json.load(f)
            if isinstance(data, list):
                records.extend(data)
                print(f"  [load] {jf.name} → +{len(data)} 条 (累计 {len(records)})")
            elif isinstance(data, dict) and "data" in data:
                records.extend(data["data"])
                print(f"  [load] {jf.name} → +{len(data['data'])} 条")
        except Exception as e:
            print(f"  [warn] 读 {jf.name} 失败: {str(e)[:120]}")

    # 3. 试 parquet
    if not records:
        try:
            import pyarrow.parquet as pq
            for pf in sorted(local_dir.rglob("*.parquet")):
                table = pq.read_table(pf)
                df = table.to_pandas()
                records.extend(df.to_dict(orient="records"))
                print(f"  [load] {pf.name} → +{len(df)} 条")
        except Exception as e:
            print(f"  [warn] 读 parquet 失败: {str(e)[:120]}")

    return records


# ============================================================================
# 验证 + 写出
# ============================================================================

def verify_pairs(pairs, n_show=5):
    """打印前 n 条，看格式 OK 否。"""
    print(f"\n  --- 前 {n_show} 条样本预览 ---")
    for i, p in enumerate(pairs[:n_show]):
        chosen_len = len(p["chosen"].split())
        rejected_len = len(p["rejected"].split())
        print(f"  [{i+1}] prompt: {p['prompt'][:80]!r}")
        print(f"      chosen ({chosen_len} 词): {p['chosen'][:120]!r}")
        print(f"      rejected ({rejected_len} 词): {p['rejected'][:120]!r}")
        print(f"      image: {p.get('image_path') or f'id={p.get(chr(34) + chr(105) + chr(109) + chr(97) + chr(103) + chr(101) + chr(95) + chr(105) + chr(100) + chr(34))}'}")
        print()


def compute_stats(pairs):
    chosen_lens = [len(p["chosen"].split()) for p in pairs]
    rejected_lens = [len(p["rejected"].split()) for p in pairs]
    return {
        "n_pairs": len(pairs),
        "chosen": {
            "avg_len": sum(chosen_lens) / max(1, len(chosen_lens)),
            "min_len": min(chosen_lens) if chosen_lens else 0,
            "max_len": max(chosen_lens) if chosen_lens else 0,
        },
        "rejected": {
            "avg_len": sum(rejected_lens) / max(1, len(rejected_lens)),
            "min_len": min(rejected_lens) if rejected_lens else 0,
            "max_len": max(rejected_lens) if rejected_lens else 0,
        },
        "n_with_image_path": sum(1 for p in pairs if p.get("image_path")),
        "n_with_image_id": sum(1 for p in pairs if p.get("image_id") is not None),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpo_data_root", required=True,
                    help="存到哪，建议 /content/drive/MyDrive/qwenvl3/data/dpo")
    ap.add_argument("--include_vlfeedback", action="store_true",
                    help="同时下 VLFeedback (~5GB，慢且未必需要)")
    ap.add_argument("--dry_run", action="store_true",
                    help="只检查 / 不真下数据")
    args = ap.parse_args()

    if not Path("/content/drive/MyDrive").exists() and not args.dry_run:
        print("[ERROR] Drive 未挂载。先在 Colab 跑：")
        print("  from google.colab import drive; drive.mount('/content/drive')")
        sys.exit(1)

    root = Path(args.dpo_data_root)
    root.mkdir(parents=True, exist_ok=True)
    print(f"[init] DPO 数据根目录: {root}\n")

    # 列要下的数据源
    sources_to_get = ["llava_rlhf_10k"]
    if args.include_vlfeedback:
        sources_to_get.append("vlfeedback")

    successful_sources = []
    for src in sources_to_get:
        spec = DPO_SOURCES[src]
        target = root / src
        print(f"=== {src} ({spec['approx_size_mb']}MB · {spec['description']}) ===")
        if args.dry_run:
            print(f"  [dry_run] would download to {target}")
            continue
        repo = download_with_fallback(src, target)
        if repo:
            successful_sources.append((src, repo, target))
        print()

    if args.dry_run:
        print("\n[dry_run] 完成。去掉 --dry_run 实际下载。")
        return

    # 加载 + 标准化
    print("\n=== 加载 + 标准化所有数据 ===")
    all_pairs = []
    all_field_stats = {}

    for src, repo, target in successful_sources:
        print(f"\n[normalize] {src}")
        records = load_dpo_records(target)
        if not records:
            print(f"  [skip] {src}: 0 records")
            continue
        if src.startswith("llava_rlhf"):
            pairs, field_stats = normalize_llava_rlhf(records)
        elif src == "vlfeedback":
            pairs, field_stats = normalize_vlfeedback(records)
        else:
            print(f"  [skip] {src}: 没有专用 normalizer")
            continue
        print(f"  → {len(pairs)} 个有效 pair (从 {len(records)} 条原始记录)")
        all_pairs.extend(pairs)
        all_field_stats[src] = dict(field_stats.most_common(20))

    if not all_pairs:
        print("\n[ERROR] 0 个有效 pair，下载或字段适配出问题")
        sys.exit(1)

    # 简单统计 + 预览
    stats = compute_stats(all_pairs)
    print(f"\n=== 总览 ===")
    print(f"  n_pairs:    {stats['n_pairs']}")
    print(f"  chosen:     avg {stats['chosen']['avg_len']:.1f} 词  "
          f"(min {stats['chosen']['min_len']}, max {stats['chosen']['max_len']})")
    print(f"  rejected:   avg {stats['rejected']['avg_len']:.1f} 词  "
          f"(min {stats['rejected']['min_len']}, max {stats['rejected']['max_len']})")
    print(f"  with_image_path: {stats['n_with_image_path']}/{stats['n_pairs']}")
    print(f"  with_image_id:   {stats['n_with_image_id']}/{stats['n_pairs']}")
    verify_pairs(all_pairs, n_show=3)

    # 写出
    out_pairs = root / "dpo_pairs.json"
    with open(out_pairs, "w") as f:
        json.dump(all_pairs, f, indent=2, ensure_ascii=False)
    print(f"\n[save] 标准化对存到: {out_pairs} ({fmt_size(out_pairs.stat().st_size)})")

    out_stats = root / "stats.json"
    with open(out_stats, "w") as f:
        json.dump({
            "stats": stats,
            "field_distributions": all_field_stats,
            "sources": [
                {"name": s, "repo": r, "local_dir": str(t)}
                for s, r, t in successful_sources
            ],
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"[save] 统计信息: {out_stats}")

    print(f"\n下一步：")
    print(f"  1. 检查 dpo_pairs.json 字段格式正确（特别是 image_path / image_id 字段）")
    print(f"  2. 等 stage4-dpo/03_train_dpo.py 写好后启动 DPO 训练")
    print(f"  3. DPO 训练命令的 --dpo_data 应该指向 {out_pairs}")


if __name__ == "__main__":
    main()
