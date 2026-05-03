"""Stage 2-v2 (Phase 1+) 共享工具。

跟 stage2/_common2.py 区别：
  ✨ 新增 TextVQATaskDataset    — OCR 类问答数据集（多数投票选答案 + 共识过滤）
  ✨ RefCOCO+/g 不需要新类       — 复用 RefCOCOTaskDataset，只是 HF 数据路径不同

设计要点（同 v1）：
- 用 Qwen2.5 chat template 包装多轮对话
- <image> 单 token 在 user turn 文本中作为占位，dataset 里展开成 729 个
- Loss 只算 assistant 部分（user / image / role marker 全部 mask 为 -100）
"""
import io
import json
import random
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

IGNORE_INDEX = -100
BBOX_PRECISION = 3


# ============================================================================
# 图像加载（COCO zip 直读）— 与 v1 一致
# ============================================================================

class CocoZipLoader:
    """从 COCO train2017.zip 直接读图，避免解压 30GB。

    多进程安全：每个 worker 进程通过 _get_zip() 懒加载自己的 ZipFile。
    """
    def __init__(self, zip_path):
        self.zip_path = str(zip_path)
        with zipfile.ZipFile(self.zip_path) as zf:
            self.names = set(zf.namelist())
        self._zip_cache = {}

    def _get_zip(self) -> zipfile.ZipFile:
        import os
        pid = os.getpid()
        if pid not in self._zip_cache:
            self._zip_cache[pid] = zipfile.ZipFile(self.zip_path)
        return self._zip_cache[pid]

    def open(self, image_filename) -> Image.Image:
        full_path = f"train2017/{image_filename}"
        if full_path not in self.names:
            raise FileNotFoundError(image_filename)
        with self._get_zip().open(full_path) as f:
            return Image.open(io.BytesIO(f.read())).convert("RGB")

    def has(self, image_filename) -> bool:
        return f"train2017/{image_filename}" in self.names

    def close(self):
        for zf in self._zip_cache.values():
            try:
                zf.close()
            except Exception:
                pass
        self._zip_cache.clear()


# ============================================================================
# Bbox 编码 — 与 v1 一致
# ============================================================================

def encode_bbox(box_norm: Tuple[float, float, float, float]) -> str:
    """归一化坐标 (x1, y1, x2, y2) → '<box>(x1,y1),(x2,y2)</box>'"""
    x1, y1, x2, y2 = box_norm
    return (f"<box>({x1:.{BBOX_PRECISION}f},{y1:.{BBOX_PRECISION}f}),"
            f"({x2:.{BBOX_PRECISION}f},{y2:.{BBOX_PRECISION}f})</box>")


# ============================================================================
# Chat template + image token expansion — 与 v1 一致
# ============================================================================

class ChatFormatter:
    """把 [{from, value}] 转成 Qwen2.5 chat template 的 input_ids + labels。"""

    def __init__(self, tokenizer, num_image_tokens: int):
        self.tokenizer = tokenizer
        self.num_image_tokens = num_image_tokens
        self.image_token_id = tokenizer.convert_tokens_to_ids("<image>")
        self._user_prefix = tokenizer("<|im_start|>user\n", add_special_tokens=False).input_ids
        self._asst_prefix = tokenizer("<|im_start|>assistant\n", add_special_tokens=False).input_ids
        self._end_marker = tokenizer("<|im_end|>\n", add_special_tokens=False).input_ids

    def _expand_image_tokens(self, ids: List[int]) -> List[int]:
        out = []
        for tok in ids:
            if tok == self.image_token_id:
                out.extend([self.image_token_id] * self.num_image_tokens)
            else:
                out.append(tok)
        return out

    def format(self, conversations: List[Dict[str, str]]) -> Tuple[List[int], List[int]]:
        input_ids: List[int] = []
        labels: List[int] = []
        for turn in conversations:
            text_ids = self.tokenizer(
                turn["value"], add_special_tokens=False
            ).input_ids
            text_ids = self._expand_image_tokens(text_ids)

            if turn["from"] == "human":
                turn_ids = self._user_prefix + text_ids + self._end_marker
                turn_lbl = [IGNORE_INDEX] * len(turn_ids)
            else:
                turn_ids = self._asst_prefix + text_ids + self._end_marker
                turn_lbl = (
                    [IGNORE_INDEX] * len(self._asst_prefix)
                    + text_ids
                    + self._end_marker
                )
            input_ids.extend(turn_ids)
            labels.extend(turn_lbl)
        return input_ids, labels


# ============================================================================
# 任务 dataset — 与 v1 一致：LLaVA-Instruct, RefCOCO, ShareGPT4V
# ============================================================================

class LlavaInstructTaskDataset(Dataset):
    """LLaVA-Instruct-150K — 多轮 VQA。"""
    def __init__(self, json_path, coco_loader: CocoZipLoader, limit=None):
        with open(json_path) as f:
            self.data = json.load(f)
        if limit:
            self.data = self.data[:limit]
        self.coco_loader = coco_loader

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s = self.data[idx]
        try:
            image = self.coco_loader.open(s["image"])
        except FileNotFoundError:
            return self.__getitem__((idx + 1) % len(self.data))
        return {
            "image": image,
            "conversations": s["conversations"],
            "task": "llava_instruct",
        }


class RefCOCOTaskDataset(Dataset):
    """RefCOCO/RefCOCO+/RefCOCOg grounding — (ref, bbox) → bbox 输出。

    支持两种数据源（自动适配，但建议显式传 bbox_format）:

    1) **jxu124 系列**（用于训练，有 train split）
       字段: image_id (int), bbox (xyxy 像素), captions (list[str]),
             sentences (list[dict]), file_name
       图片: 不 bundle，要从 COCO zip 通过 image_id 查找
       bbox_format: "xyxy"
       captions: 多个指代表达，训练时随机取一个 (random_caption=True)

    2) **lmms-lab 系列**（用于 eval，只有 val/test split）
       字段: image (PIL/bytes), bbox (xywh 像素), answer (单个 ref str)
       图片: bundle 在 image 字段
       bbox_format: "xywh"
       captions: 单个 (answer)
    """
    def __init__(self, hf_dataset, coco_loader: Optional[CocoZipLoader] = None,
                 limit=None, source_name="refcoco",
                 bbox_format: str = "xywh",  # "xywh" (lmms-lab) or "xyxy" (jxu124)
                 random_caption: bool = True):
        self.ds = hf_dataset
        self.indices = list(range(len(hf_dataset)))[:limit] if limit else list(range(len(hf_dataset)))
        self.coco_loader = coco_loader
        self.source_name = source_name
        assert bbox_format in ("xywh", "xyxy"), f"bbox_format must be xywh|xyxy, got {bbox_format}"
        self.bbox_format = bbox_format
        self.random_caption = random_caption

        if len(hf_dataset) > 0:
            keys = list(hf_dataset[0].keys())
            print(f"  [{source_name}] HF dataset 字段: {keys[:10]}")
            print(f"  [{source_name}] bbox_format={bbox_format}, random_caption={random_caption}")

    def __len__(self):
        return len(self.indices)

    def _extract_image_and_size(self, s) -> Optional[Tuple[Image.Image, Tuple[int, int]]]:
        # ---- 1) lmms-lab: image 字段含 bytes 或 PIL ----
        img_field = s.get("image")
        if isinstance(img_field, dict) and "bytes" in img_field:
            try:
                img = Image.open(io.BytesIO(img_field["bytes"])).convert("RGB")
                return img, img.size
            except Exception:
                pass
        if hasattr(img_field, "convert"):
            img = img_field.convert("RGB")
            return img, img.size

        # ---- 2) jxu124: 从 COCO train2017.zip 查图 ----
        if self.coco_loader is None:
            return None

        # 优先用 image_id (整数)
        image_id = s.get("image_id")
        if image_id is None:
            # fallback: 从 file_name / image_path 抠数字
            import re  # noqa: PLC0415
            for key in ["image_path", "file_name"]:
                v = s.get(key, "")
                if not isinstance(v, str):
                    continue
                # COCO 文件名格式: ...000000581857.jpg 或 COCO_train2014_000000581857_16.jpg
                m = re.search(r"(\d{6,})", v)
                if m:
                    image_id = int(m.group(1))
                    break
        if image_id is None:
            return None

        # train2017.zip 里的文件名格式: train2017/000000XXXXXX.jpg
        # COCO 2014 / 2017 共享 image_id 数字，但部分 2014 图不在 train2017 里
        fn = f"{int(image_id):012d}.jpg"
        try:
            img = self.coco_loader.open(fn)
            return img, img.size
        except FileNotFoundError:
            return None

    def _extract_ref(self, s) -> Optional[str]:
        # ---- 1) jxu124: captions (list[str]) ----
        captions = s.get("captions")
        if isinstance(captions, list) and captions:
            valid = [c for c in captions if isinstance(c, str) and c.strip()]
            if valid:
                return random.choice(valid).strip() if self.random_caption else valid[0].strip()

        # ---- 2) jxu124: sentences (list[dict]) ----
        sentences = s.get("sentences")
        if isinstance(sentences, list) and sentences:
            sents = []
            for x in sentences:
                if isinstance(x, dict):
                    sent = x.get("sent") or x.get("raw") or x.get("text")
                    if isinstance(sent, str) and sent.strip():
                        sents.append(sent.strip())
                elif isinstance(x, str) and x.strip():
                    sents.append(x.strip())
            if sents:
                return random.choice(sents) if self.random_caption else sents[0]

        # ---- 3) lmms-lab: 单 ref 字段 ----
        for key in ["answer", "sentence", "ref", "referring_expression", "caption"]:
            v = s.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
            elif isinstance(v, list) and v:
                first = v[0]
                if isinstance(first, str) and first.strip():
                    return first.strip()

        return None

    def _extract_bbox(self, s, im_size) -> Optional[Tuple[float, float, float, float]]:
        bbox = s.get("bbox") or s.get("box")
        if not bbox or len(bbox) != 4:
            return None
        iw, ih = im_size
        # 已经归一化 0-1?
        if max(bbox) <= 1.5:
            return tuple(bbox)
        # 像素坐标，按 bbox_format 解析
        if self.bbox_format == "xyxy":
            x1, y1, x2, y2 = bbox
            return (x1 / iw, y1 / ih, x2 / iw, y2 / ih)
        else:  # xywh (COCO 默认)
            x, y, w, h = bbox
            return (x / iw, y / ih, (x + w) / iw, (y + h) / ih)

    def __getitem__(self, idx):
        last_failure = "unknown"
        for tries in range(30):  # jxu124 部分图不在 train2017 里，多试几次
            i = self.indices[(idx + tries) % len(self.indices)]
            s = self.ds[i]
            img_pair = self._extract_image_and_size(s)
            if img_pair is None:
                last_failure = f"image (image_id={s.get('image_id')}; 可能不在 train2017.zip 里)"
                continue
            ref = self._extract_ref(s)
            if ref is None:
                last_failure = f"ref (sample keys: {list(s.keys())[:6]})"
                continue
            image, im_size = img_pair
            bbox = self._extract_bbox(s, im_size)
            if bbox is None:
                last_failure = f"bbox (raw bbox: {s.get('bbox')}, format={self.bbox_format})"
                continue

            # 健全性检查：归一化坐标合法 (0 ≤ x1 < x2 ≤ 1, 0 ≤ y1 < y2 ≤ 1)
            x1, y1, x2, y2 = bbox
            if not (0 <= x1 < x2 <= 1.01 and 0 <= y1 < y2 <= 1.01):
                last_failure = f"bbox 异常: {bbox} (im_size={im_size}, format={self.bbox_format})"
                continue
            # 防止小数误差越界
            bbox = (max(0, x1), max(0, y1), min(1, x2), min(1, y2))

            conversations = [
                {"from": "human", "value": f"<image>\nProvide the bounding box coordinates of {ref}."},
                {"from": "gpt",   "value": encode_bbox(bbox)},
            ]
            return {
                "image": image,
                "conversations": conversations,
                "task": self.source_name,
                "bbox": bbox,
            }
        raise RuntimeError(
            f"{self.source_name}: idx={idx} 连续 30 个样本解析失败。"
            f"最后失败原因: {last_failure}"
        )


class ShareGPT4VTaskDataset(Dataset):
    """ShareGPT4V — 长 caption。"""
    def __init__(self, json_path, coco_loader: CocoZipLoader, limit=None):
        with open(json_path) as f:
            data = json.load(f)
        self.data = [
            s for s in data
            if "coco" in s.get("image", "").lower()
            and coco_loader.has(Path(s["image"]).name)
        ]
        if limit:
            self.data = self.data[:limit]
        self.coco_loader = coco_loader

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s = self.data[idx]
        try:
            image = self.coco_loader.open(Path(s["image"]).name)
        except FileNotFoundError:
            return self.__getitem__((idx + 1) % len(self.data))
        return {
            "image": image,
            "conversations": s["conversations"],
            "task": "sharegpt4v",
        }


# ============================================================================
# 🆕 TextVQA — Phase 1+ 新增数据集
# ============================================================================

class TextVQATaskDataset(Dataset):
    """TextVQA — OCR 类问答（图中含文字，需要读懂文字回答问题）。

    HF lmms-lab/textvqa 字段：
      image     - PIL/bytes
      question  - str
      answers   - list[str]，10 个标注员答案

    设计：
      1. 多数投票：从 10 个答案中选投票最多的作为 GT
      2. 共识过滤：要求至少 min_consensus 人同意（默认 3）才接受这条样本，
         否则跳过 —— 答案分歧太大说明问题本身有歧义，不利于训练
      3. 训练格式：单 turn user-assistant 对话
            user:      <image>\nWhat is the price on the sign?
            assistant: $5.99
    """
    def __init__(self, hf_dataset, limit=None, min_consensus=3):
        self.ds = hf_dataset
        self.indices = list(range(len(hf_dataset)))[:limit] if limit else list(range(len(hf_dataset)))
        self.min_consensus = min_consensus
        if len(hf_dataset) > 0:
            keys = list(hf_dataset[0].keys())
            print(f"  [textvqa] HF dataset 字段: {keys[:8]}")
            print(f"  [textvqa] 采样时实时过滤 < {min_consensus} 人共识的样本")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        last_failure = "unknown"
        for tries in range(30):  # textvqa 过滤率比 refcoco 高，多试几次
            i = self.indices[(idx + tries) % len(self.indices)]
            s = self.ds[i]

            # 取图
            img_field = s.get("image")
            if isinstance(img_field, dict) and "bytes" in img_field:
                try:
                    image = Image.open(io.BytesIO(img_field["bytes"])).convert("RGB")
                except Exception:
                    last_failure = "image bytes decode 失败"
                    continue
            elif hasattr(img_field, "convert"):
                image = img_field.convert("RGB")
            else:
                last_failure = f"image 字段不识别: {type(img_field)}"
                continue

            # 取 question
            question = (s.get("question") or s.get("query") or "").strip()
            if not question:
                last_failure = "question 为空"
                continue

            # 取 answers + 多数投票
            answers = s.get("answers") or s.get("answer") or []
            if isinstance(answers, str):
                answers = [answers]
            normalized = [a.lower().strip() for a in answers
                          if isinstance(a, str) and a.strip()]
            if not normalized:
                last_failure = "answers 全为空"
                continue
            top_answer, top_count = Counter(normalized).most_common(1)[0]
            if top_count < self.min_consensus:
                last_failure = f"共识不足 ({top_count} < {self.min_consensus})"
                continue

            return {
                "image": image,
                "conversations": [
                    {"from": "human", "value": f"<image>\n{question}"},
                    {"from": "gpt",   "value": top_answer},
                ],
                "task": "textvqa",
            }
        raise RuntimeError(
            f"TextVQA: idx={idx} 连续 30 个样本失败。最后原因: {last_failure}。"
            f"可能是 dataset 字段命名跟预期不符，或 min_consensus 设得太严。"
        )


# ============================================================================
# Multi-task 包装 — 与 v1 一致
# ============================================================================

class MultitaskTrainingDataset(Dataset):
    """ConcatDataset 风格：把多个 task dataset 拼起来，按 idx 路由。"""
    def __init__(self, task_datasets: List[Tuple[str, Dataset]],
                 chat_formatter: ChatFormatter,
                 image_processor,
                 max_len: int = 1500):
        self.task_datasets = task_datasets
        self.chat_formatter = chat_formatter
        self.image_processor = image_processor
        self.max_len = max_len
        self.cumlen = [0]
        for _, d in task_datasets:
            self.cumlen.append(self.cumlen[-1] + len(d))

    def __len__(self):
        return self.cumlen[-1]

    def _route(self, idx):
        for i, end in enumerate(self.cumlen[1:]):
            if idx < end:
                return i, idx - self.cumlen[i]
        raise IndexError(idx)

    def __getitem__(self, idx):
        task_idx, local_idx = self._route(idx)
        task_name, ds = self.task_datasets[task_idx]
        sample = ds[local_idx]

        pixel_values = self.image_processor(
            sample["image"], return_tensors="pt"
        ).pixel_values[0]

        input_ids, labels = self.chat_formatter.format(sample["conversations"])

        if len(input_ids) > self.max_len:
            input_ids = input_ids[: self.max_len]
            labels = labels[: self.max_len]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pixel_values": pixel_values,
        }


class MultitaskCollator:
    """对一个 batch 做 padding，构建 attention_mask。"""
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(b["input_ids"].size(0) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            n_pad = max_len - b["input_ids"].size(0)
            input_ids.append(torch.cat([
                b["input_ids"],
                torch.full((n_pad,), self.pad_token_id, dtype=torch.long)
            ]))
            labels.append(torch.cat([
                b["labels"],
                torch.full((n_pad,), IGNORE_INDEX, dtype=torch.long)
            ]))
            attn.append(torch.cat([
                torch.ones(b["input_ids"].size(0), dtype=torch.long),
                torch.zeros(n_pad, dtype=torch.long),
            ]))
        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attn),
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        }


# ============================================================================
# LoRA target modules 解析 — 与 v1 一致
# ============================================================================

LM_LORA_SUFFIXES = {
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
}


def find_lm_lora_targets(model) -> List[str]:
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if "language_model" not in name:
            continue
        leaf = name.split(".")[-1]
        if leaf in LM_LORA_SUFFIXES:
            targets.append(name)
    return targets
