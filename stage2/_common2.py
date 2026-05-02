"""Stage 2 共享工具：chat template、多任务 dataset、bbox 编码、LoRA target 解析。

被 03_train_stage2.py 和 04_eval_stage2.py 共用。

设计要点：
- 用 Qwen2.5 chat template 包装多轮对话
- <image> 单 token 在 user turn 文本中作为占位，dataset 里展开成 729 个
- Loss 只算 assistant 部分（user / image / role marker 全部 mask 为 -100）
- 4 种任务统一成 [{from, value}] 格式，由 ChatFormatter 转 input_ids/labels
"""
import io
import json
import random
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

IGNORE_INDEX = -100
BBOX_PRECISION = 3   # bbox 坐标小数位


# ============================================================================
# 图像加载（COCO zip 直读）
# ============================================================================

class CocoZipLoader:
    """从 COCO train2017.zip 直接读图，避免解压 30GB。"""
    def __init__(self, zip_path):
        self.zip = zipfile.ZipFile(zip_path)
        self.names = set(self.zip.namelist())

    def open(self, image_filename) -> Image.Image:
        full_path = f"train2017/{image_filename}"
        if full_path not in self.names:
            raise FileNotFoundError(image_filename)
        with self.zip.open(full_path) as f:
            return Image.open(io.BytesIO(f.read())).convert("RGB")

    def has(self, image_filename) -> bool:
        return f"train2017/{image_filename}" in self.names

    def close(self):
        try:
            self.zip.close()
        except Exception:
            pass


# ============================================================================
# Bbox 编码
# ============================================================================

def encode_bbox(box_norm: Tuple[float, float, float, float]) -> str:
    """归一化坐标 (x1, y1, x2, y2) → '<box>(x1,y1),(x2,y2)</box>'

    LLaVA / Qwen-VL 一代的 grounding 表示法。坐标 0–1 区间，3 位小数。
    """
    x1, y1, x2, y2 = box_norm
    return (f"<box>({x1:.{BBOX_PRECISION}f},{y1:.{BBOX_PRECISION}f}),"
            f"({x2:.{BBOX_PRECISION}f},{y2:.{BBOX_PRECISION}f})</box>")


# ============================================================================
# Chat template + image token expansion
# ============================================================================

class ChatFormatter:
    """把 [{from: 'human'|'gpt', value: str}, ...] 格式的对话转成 input_ids + labels。

    Qwen2.5 chat template 格式：
        <|im_start|>user
        <text with <image>><|im_end|>
        <|im_start|>assistant
        <answer><|im_end|>

    特殊处理：
    1. 文本中的 <image> 占位符在 tokenize 后展开成 num_image_tokens 个 image_token_id
    2. Loss masking：assistant turn 的 content + <|im_end|> 算 loss；
       prefix `<|im_start|>assistant\\n` 和所有 user/system turn 都 mask 为 -100
    """

    def __init__(self, tokenizer, num_image_tokens: int):
        self.tokenizer = tokenizer
        self.num_image_tokens = num_image_tokens
        self.image_token_id = tokenizer.convert_tokens_to_ids("<image>")
        # 预 tokenize role markers
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
        """返回 (input_ids, labels)。"""
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
            else:  # gpt / assistant
                turn_ids = self._asst_prefix + text_ids + self._end_marker
                # mask prefix；text + end 算 loss
                turn_lbl = (
                    [IGNORE_INDEX] * len(self._asst_prefix)
                    + text_ids
                    + self._end_marker
                )

            input_ids.extend(turn_ids)
            labels.extend(turn_lbl)

        return input_ids, labels


# ============================================================================
# 任务 dataset 实现（统一返回 {image, conversations}）
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
            # 找不到图：随机换一个
            return self.__getitem__((idx + 1) % len(self.data))
        # 保留原始 conversations 格式（已含 <image> 占位）
        return {
            "image": image,
            "conversations": s["conversations"],
            "task": "llava_instruct",
        }


class RefCOCOTaskDataset(Dataset):
    """RefCOCO grounding — 把 (ref, bbox) 转成 (Q="Where is X?", A=<box>...)。

    适配 lmms-lab/RefCOCO 字段：
      image   - PIL/bytes
      answer  - 真正的 referring expression（"the man in red"），不是 question！
                lmms-lab 把任务设计成"对图中圈出的区域写描述"，所以 question 是
                通用 prompt，answer 才是描述/ref。
      bbox    - 目标 bounding box [x, y, w, h] (COCO) 或 [x1,y1,x2,y2]
    """
    def __init__(self, hf_dataset, coco_loader: Optional[CocoZipLoader] = None,
                 limit=None):
        self.ds = hf_dataset
        self.indices = list(range(len(hf_dataset)))[:limit] if limit else list(range(len(hf_dataset)))
        self.coco_loader = coco_loader

        # 启动时打印字段，便于以后字段命名又变化时快速诊断
        if len(hf_dataset) > 0:
            keys = list(hf_dataset[0].keys())
            print(f"  [refcoco] HF dataset 字段: {keys}")

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
        # 字段命名因 HF repo 而异，做宽松匹配
        # lmms-lab/RefCOCO 用 'answer' 字段存 ref expression（而不是 question——question 是通用 prompt）
        # 其他 RefCOCO mirrors 可能用 sentences/sentence/ref 等
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
        if max(bbox) > 1.5:  # 像素坐标，假设 [x, y, w, h]（COCO）
            x, y, w, h = bbox
            return (x / iw, y / ih, (x + w) / iw, (y + h) / ih)
        return tuple(bbox)

    def __getitem__(self, idx):
        last_failure = "unknown"
        for tries in range(20):  # 字段缺失时换样本
            i = self.indices[(idx + tries) % len(self.indices)]
            s = self.ds[i]
            img_pair = self._extract_image_and_size(s)
            if img_pair is None:
                last_failure = f"image (sample keys: {list(s.keys())})"
                continue
            ref = self._extract_ref(s)
            if ref is None:
                last_failure = f"ref (sample keys: {list(s.keys())}, " \
                               f"answer preview: {str(s.get('answer'))[:80]})"
                continue
            image, im_size = img_pair
            bbox = self._extract_bbox(s, im_size)
            if bbox is None:
                last_failure = f"bbox (raw bbox field: {s.get('bbox')})"
                continue

            conversations = [
                {"from": "human", "value": f"<image>\nProvide the bounding box coordinates of {ref}."},
                {"from": "gpt",   "value": encode_bbox(bbox)},
            ]
            return {
                "image": image,
                "conversations": conversations,
                "task": "refcoco",
                "bbox": bbox,  # 用于后续 eval
            }
        raise RuntimeError(
            f"RefCOCO: idx={idx} 连续 20 个样本解析失败。"
            f"最后失败原因: {last_failure}"
        )


class ShareGPT4VTaskDataset(Dataset):
    """ShareGPT4V — 长 caption。"""
    def __init__(self, json_path, coco_loader: CocoZipLoader, limit=None):
        with open(json_path) as f:
            data = json.load(f)
        # 只取 image 在 COCO train2017 里的样本（其他图源没下载）
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
# Multi-task 包装：统一 chat 格式 + 图像处理
# ============================================================================

class MultitaskTrainingDataset(Dataset):
    """把任务 dataset 包装成 Trainer 可消费的 {input_ids, labels, pixel_values}。

    输入：list of (task_name, task_dataset)。所有任务 ConcatDataset 拼起来，
    Trainer 自带 shuffle 实现混合采样。每条样本格式经 ChatFormatter 处理后输出。
    """
    def __init__(self, task_datasets: List[Tuple[str, Dataset]],
                 chat_formatter: ChatFormatter,
                 image_processor,
                 max_len: int = 1500):
        self.task_datasets = task_datasets
        self.chat_formatter = chat_formatter
        self.image_processor = image_processor
        self.max_len = max_len

        # 计算每个 task 在 ConcatDataset 中的索引区间
        self.cumlen = [0]
        for _, d in task_datasets:
            self.cumlen.append(self.cumlen[-1] + len(d))

    def __len__(self):
        return self.cumlen[-1]

    def _route(self, idx):
        # 二分查找所属 task
        for i, end in enumerate(self.cumlen[1:]):
            if idx < end:
                return i, idx - self.cumlen[i]
        raise IndexError(idx)

    def __getitem__(self, idx):
        task_idx, local_idx = self._route(idx)
        task_name, ds = self.task_datasets[task_idx]
        sample = ds[local_idx]

        # 图像处理
        pixel_values = self.image_processor(
            sample["image"], return_tensors="pt"
        ).pixel_values[0]

        # 文本处理
        input_ids, labels = self.chat_formatter.format(sample["conversations"])

        # 截断（保护：永远不能从中间砍掉 image tokens）
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
# LoRA target modules 解析（只对 LLM，不对 vision_tower）
# ============================================================================

LM_LORA_SUFFIXES = {
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
}


def find_lm_lora_targets(model) -> List[str]:
    """枚举 model.named_modules() 中 LLM 内部的 q/k/v/o + gate/up/down Linear 层。

    用全模块名（如 'model.language_model.layers.0.self_attn.q_proj'）作为 LoRA target。
    保证不会误伤 vision_tower 同名层。
    """
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
