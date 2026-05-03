"""Stage 4 DPO 共享工具：偏好数据集 + chat 模板 + collator + DPO loss helper。

设计要点：
- **直接从 RLAIF-V parquet 加载**（图片 bytes bundled 在 dataset 里），不依赖
  01_prepare_dpo_data.py 的 dpo_pairs.json — 那个 json 主要给统计用
- 输入格式跟 stage2 ChatFormatter 兼容（Qwen2.5 chat template + <image>×729 展开）
- 同一 prompt 的 chosen/rejected **共享 pixel_values**（图片只 forward 一次的潜在优化）
- Loss masking：prompt 部分 labels=-100，response 部分 = token id（chosen / rejected 各一份）

DPO loss 在 03_train_dpo.py 里，本文件只负责数据 pipeline。
"""
import io
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

IGNORE_INDEX = -100


# ============================================================================
# Chat prompt 构造（跟 stage2-v2 ChatPromptBuilder 一致）
# ============================================================================

class DPOChatBuilder:
    """构造 Qwen2.5 chat 格式的 prompt。

    DPO 数据格式：
        prompt:   <|im_start|>user\\n<image>×729 + question<|im_end|>\\n<|im_start|>assistant\\n
        chosen:   {chosen_text}<|im_end|>
        rejected: {rejected_text}<|im_end|>

    最终 input_ids = prompt + chosen (or rejected)
    labels      = [-100] × len(prompt) + chosen_ids (or rejected_ids)
    """
    def __init__(self, tokenizer, num_image_tokens: int):
        self.tokenizer = tokenizer
        self.num_image_tokens = num_image_tokens
        self.image_token_id = tokenizer.convert_tokens_to_ids("<image>")
        self.user_prefix = tokenizer("<|im_start|>user\n", add_special_tokens=False).input_ids
        self.end_marker = tokenizer("<|im_end|>\n", add_special_tokens=False).input_ids
        self.asst_prefix = tokenizer("<|im_start|>assistant\n", add_special_tokens=False).input_ids
        self.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    def _expand_image(self, ids: List[int]) -> List[int]:
        out = []
        for tok in ids:
            if tok == self.image_token_id:
                out.extend([self.image_token_id] * self.num_image_tokens)
            else:
                out.append(tok)
        return out

    def build_prompt_ids(self, question: str) -> List[int]:
        """构造到 assistant\\n 之前（不含 response）"""
        text = f"<image>\n{question}"
        text_ids = self.tokenizer(text, add_special_tokens=False).input_ids
        text_ids = self._expand_image(text_ids)
        return self.user_prefix + text_ids + self.end_marker + self.asst_prefix

    def build_response_ids(self, response: str) -> List[int]:
        """response + <|im_end|>\\n （DPO 算 logp 时只看这一段）"""
        resp_ids = self.tokenizer(response, add_special_tokens=False).input_ids
        return resp_ids + self.end_marker

    def build_full_pair(self, question: str, response: str) -> Tuple[List[int], List[int]]:
        """返回 (full_input_ids, labels) for chosen 或 rejected 单边。

        labels: prompt 部分 -100，response 部分跟 input_ids 一致。
        """
        prompt_ids = self.build_prompt_ids(question)
        response_ids = self.build_response_ids(response)
        full_ids = prompt_ids + response_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids
        return full_ids, labels


# ============================================================================
# 偏好数据集
# ============================================================================

class DPOPreferenceDataset(Dataset):
    """RLAIF-V (或类似) 偏好对 dataset，输出 chosen + rejected 双侧 tokenized 数据。

    每条样本输出（给 collator）:
        chosen_input_ids   : list[int]
        chosen_labels      : list[int]   (prompt 部分 -100)
        rejected_input_ids : list[int]
        rejected_labels    : list[int]
        pixel_values       : tensor [3, H, W]
        prompt_length      : int          (用来在 logp 计算时跳过 prompt 部分)
    """
    def __init__(self, hf_dataset_dir: str,
                 chat_builder: DPOChatBuilder,
                 image_processor,
                 max_len: int = 1500,
                 limit: Optional[int] = None,
                 split: str = "train"):
        from datasets import load_dataset

        print(f"[dpo_data] 加载 {hf_dataset_dir} (split={split})...")
        ds = None
        for try_split in [split, "train", "validation", "val"]:
            try:
                ds = load_dataset(str(hf_dataset_dir), split=try_split, trust_remote_code=True)
                print(f"  ✅ loaded split={try_split}, n={len(ds)}")
                break
            except Exception:
                continue
        if ds is None:
            try:
                ds_dict = load_dataset(str(hf_dataset_dir), trust_remote_code=True)
                first = list(ds_dict.keys())[0]
                ds = ds_dict[first]
                print(f"  ⚠️  fallback first split={first}, n={len(ds)}")
            except Exception as e:
                raise RuntimeError(f"无法加载 {hf_dataset_dir}: {e}")

        self.ds = ds
        self.indices = list(range(len(ds)))[:limit] if limit else list(range(len(ds)))
        self.chat_builder = chat_builder
        self.image_processor = image_processor
        self.max_len = max_len

        # 打印 sample 0 字段供调试
        if len(ds) > 0:
            keys = list(ds[0].keys())
            print(f"  fields: {keys[:10]}")

    def __len__(self):
        return len(self.indices)

    def _extract_image(self, sample) -> Optional[Image.Image]:
        img_field = sample.get("image")
        if isinstance(img_field, dict) and "bytes" in img_field:
            try:
                return Image.open(io.BytesIO(img_field["bytes"])).convert("RGB")
            except Exception:
                return None
        if hasattr(img_field, "convert"):
            return img_field.convert("RGB")
        return None

    def _extract_fields(self, sample):
        question = (sample.get("question") or sample.get("prompt") or "").strip()
        chosen = (sample.get("chosen") or "").strip()
        rejected = (sample.get("rejected") or "").strip()
        return question, chosen, rejected

    def __getitem__(self, idx):
        for tries in range(20):
            i = self.indices[(idx + tries) % len(self.indices)]
            sample = self.ds[i]

            image = self._extract_image(sample)
            if image is None:
                continue
            question, chosen, rejected = self._extract_fields(sample)
            if not (question and chosen and rejected):
                continue
            if chosen == rejected:
                continue

            # 图像处理
            pixel_values = self.image_processor(
                image, return_tensors="pt"
            ).pixel_values[0]

            # prompt + chosen
            chosen_ids, chosen_labels = self.chat_builder.build_full_pair(question, chosen)
            # prompt + rejected
            rejected_ids, rejected_labels = self.chat_builder.build_full_pair(question, rejected)

            # max_len 截断（保护：如果 prompt 都比 max_len 大，跳过这条样本）
            prompt_len = len(self.chat_builder.build_prompt_ids(question))
            if prompt_len >= self.max_len - 10:
                continue   # prompt 太长，无空间放 response

            if len(chosen_ids) > self.max_len:
                chosen_ids = chosen_ids[:self.max_len]
                chosen_labels = chosen_labels[:self.max_len]
            if len(rejected_ids) > self.max_len:
                rejected_ids = rejected_ids[:self.max_len]
                rejected_labels = rejected_labels[:self.max_len]

            return {
                "chosen_input_ids":   torch.tensor(chosen_ids, dtype=torch.long),
                "chosen_labels":      torch.tensor(chosen_labels, dtype=torch.long),
                "rejected_input_ids": torch.tensor(rejected_ids, dtype=torch.long),
                "rejected_labels":    torch.tensor(rejected_labels, dtype=torch.long),
                "pixel_values":       pixel_values,
                "prompt_length":      prompt_len,
            }

        raise RuntimeError(f"DPOPreferenceDataset: idx={idx} 连续 20 个样本失败")


# ============================================================================
# Collator: pad chosen 和 rejected 各自到 batch 最大长度
# ============================================================================

class DPOCollator:
    """对一个 batch 做 padding，分别处理 chosen 和 rejected 的 input_ids/labels。"""
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # 分别找 chosen 和 rejected 的最大长度
        max_chosen = max(b["chosen_input_ids"].size(0) for b in batch)
        max_rejected = max(b["rejected_input_ids"].size(0) for b in batch)

        def pad_seq(ids, target_len, pad_value):
            cur_len = ids.size(0)
            if cur_len >= target_len:
                return ids
            n_pad = target_len - cur_len
            return torch.cat([ids, torch.full((n_pad,), pad_value, dtype=ids.dtype)])

        def make_attn(ids_list, target_len):
            attn = []
            for b in ids_list:
                cur_len = b.size(0)
                m = torch.cat([
                    torch.ones(cur_len, dtype=torch.long),
                    torch.zeros(target_len - cur_len, dtype=torch.long),
                ])
                attn.append(m)
            return torch.stack(attn)

        chosen_ids = torch.stack([
            pad_seq(b["chosen_input_ids"], max_chosen, self.pad_token_id) for b in batch
        ])
        chosen_labels = torch.stack([
            pad_seq(b["chosen_labels"], max_chosen, IGNORE_INDEX) for b in batch
        ])
        chosen_attn = make_attn([b["chosen_input_ids"] for b in batch], max_chosen)

        rejected_ids = torch.stack([
            pad_seq(b["rejected_input_ids"], max_rejected, self.pad_token_id) for b in batch
        ])
        rejected_labels = torch.stack([
            pad_seq(b["rejected_labels"], max_rejected, IGNORE_INDEX) for b in batch
        ])
        rejected_attn = make_attn([b["rejected_input_ids"] for b in batch], max_rejected)

        return {
            "chosen_input_ids":   chosen_ids,
            "chosen_labels":      chosen_labels,
            "chosen_attention_mask": chosen_attn,
            "rejected_input_ids": rejected_ids,
            "rejected_labels":    rejected_labels,
            "rejected_attention_mask": rejected_attn,
            "pixel_values":       torch.stack([b["pixel_values"] for b in batch]),
        }


# ============================================================================
# DPO logp / loss helpers
# ============================================================================

def compute_response_logp(model, input_ids, attention_mask, pixel_values, labels) -> torch.Tensor:
    """计算 sum log p(response | image, prompt) for 每个 batch sample。

    labels: -100 处不算（即 prompt 部分），其他位置 = token_id。
    返回: shape [B] 的 tensor。
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
    )
    logits = outputs.logits  # [B, T, V]

    # next-token shift
    shift_logits = logits[..., :-1, :].contiguous()  # [B, T-1, V]
    shift_labels = labels[..., 1:].contiguous()       # [B, T-1]

    # log probs
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

    # mask
    mask = (shift_labels != IGNORE_INDEX)            # [B, T-1]
    safe_labels = shift_labels.masked_fill(~mask, 0) # 用 0 替换 -100 不会爆 gather

    # gather log_p at correct token positions
    gathered = log_probs.gather(2, safe_labels.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
    gathered = gathered.masked_fill(~mask, 0.0)

    # sum over response tokens
    sum_logp = gathered.sum(dim=-1)  # [B]
    return sum_logp


def dpo_loss(
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Direct Preference Optimization loss.

    Loss = -E[ log σ( β × ((π_θ(c) - π_ref(c)) - (π_θ(r) - π_ref(r))) ) ]

    Returns:
        loss: scalar
        metrics: dict with chosen/rejected reward, accuracy, etc.
    """
    # Active model log-probs ratio vs reference
    chosen_ratio = chosen_logp - ref_chosen_logp
    rejected_ratio = rejected_logp - ref_rejected_logp

    logits = beta * (chosen_ratio - rejected_ratio)
    # σ(x) → -log σ(x) is the DPO loss
    loss = -torch.nn.functional.logsigmoid(logits).mean()

    # 监控指标
    metrics = {
        "rewards/chosen": (beta * chosen_ratio).mean().detach(),
        "rewards/rejected": (beta * rejected_ratio).mean().detach(),
        "rewards/accuracies": (chosen_ratio > rejected_ratio).float().mean().detach(),
        "rewards/margins": (chosen_ratio - rejected_ratio).mean().detach(),
        "logits/chosen": chosen_logp.mean().detach(),
        "logits/rejected": rejected_logp.mean().detach(),
        "logits/ref_chosen": ref_chosen_logp.mean().detach(),
        "logits/ref_rejected": ref_rejected_logp.mean().detach(),
    }
    return loss, metrics
