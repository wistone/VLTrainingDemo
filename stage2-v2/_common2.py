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

    Phase 1+ 重要变化：**这同一个类同时被 RefCOCO / RefCOCO+ / RefCOCOg 复用**，
    只是初始化时传入的 hf_dataset 来自不同 HF repo。
    """
    def __init__(self, hf_dataset, coco_loader: Optional[CocoZipLoader] = None,
                 limit=None, source_name="refcoco"):
        self.ds = hf_dataset
        self.indices = list(range(len(hf_dataset)))[:limit] if limit else list(range(len(hf_dataset)))
        self.coco_loader = coco_loader
        self.source_name = source_name  # 'refcoco' / 'refcoco_plus' / 'refcocog'

        if len(hf_dataset) > 0:
            keys = list(hf_dataset[0].keys())
            print(f"  [{source_name}] HF dataset 字段: {keys[:8]}")

    def __len__(self):
        return len(self.indices)

    def _extract_image_and_size(self, s) -> Optional[Tuple[Image.Image, Tuple[int, int]]]:
        img_field = s.get("image")
        if isinstance(img_field, dict) and "bytes" in img_field:
            img = Image.open(io.BytesIO(img_field["bytes"])).convert("RGB")
        elif hasattr(img_field, "convert"):
            img = img_field.convert("RGB")
        elif isinstance(img_field, str) and self.coco_loader:
            try:
                img = self.coco_loader.open(img_field)
            except FileNotFoundError:
                return None
        else:
            return None
        return img, img.size

    def _extract_ref(self, s) -> Optional[str]:
        for key in ["answer", "sentences", "sentence", "ref",
                    "referring_expression", "caption"]:
            v = s.get(key)
            if isinstance(v, list) and v:
                v = v[0]
                if isinstance(v, dict):
                    v = v.get("sent") or v.get("raw") or v.get("text")
                if isinstance(v, str) and v.strip():
                    return v.strip()
            elif isinstance(v, str) and v.strip():
                return v.strip()
        return None

    def _extract_bbox(self, s, im_size) -> Optional[Tuple[float, float, float, float]]:
        bbox = s.get("bbox") or s.get("box") or s.get("answer")
        if not bbox or len(bbox) != 4:
            return None
        iw, ih = im_size
        if max(bbox) > 1.5:
            x, y, w, h = bbox
            return (x / iw, y / ih, (x + w) / iw, (y + h) / ih)
        return tuple(bbox)

    def __getitem__(self, idx):
        last_failure = "unknown"
        for tries in range(20):
            i = self.indices[(idx + tries) % len(self.indices)]
            s = self.ds[i]
            img_pair = self._extract_image_and_size(s)
            if img_pair is None:
                last_failure = f"image (sample keys: {list(s.keys())})"
                continue
            ref = self._extract_ref(s)
            if ref is None:
                last_failure = f"ref (sample keys: {list(s.keys())})"
                continue
            image, im_size = img_pair
            bbox = self._extract_bbox(s, im_size)
            if bbox is None:
                last_failure = f"bbox (raw bbox: {s.get('bbox')})"
                continue

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
            f"{self.source_name}: idx={idx} 连续 20 个样本解析失败。"
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
