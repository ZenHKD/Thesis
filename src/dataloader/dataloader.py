"""
SpatialVLM Dataset & DataLoader
================================

PyTorch Dataset for the NVIDIA Warehouse Spatial Intelligence dataset.
Each sample yields all tensors needed by SpatialVLM.forward() / loss.

Splits: train (499K), val (1.9K), test (19K)

Usage:
    from src.dataloader.dataloader import SpatialVLMDataset, get_dataloader

    dataset = SpatialVLMDataset("train", processor=pipeline.processor)
    loader  = get_dataloader(dataset, batch_size=1, shuffle=True)

    for batch in loader:
        out = pipeline(
            batch["pixel_values"], batch["image_grid_thw"],
            batch["depth_maps"],   batch["input_ids"],
            rle_list=batch["rle_list"][0],
            mask_token_positions=batch["mask_positions"][0],
        )
"""

import os
import re
import sys
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from model.pipeline import SYSTEM_PROMPT, find_mask_positions

# Paths
ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "nvidia_warehouse_dataset")

_SPLIT_CONFIG = {
    "train":        {"json": "train.json",        "dir": "train"},
    "val":          {"json": "val.json",           "dir": "val"},
    "test":         {"json": "test.json",          "dir": "test"},
    "train_sample": {"json": "train_sample/train_sample.json", "dir": "train_sample"},
}


# Answer formatting
def format_answer(category: str, normalized_answer) -> str:
    """Build the structured target string for training.

    Format: CATEGORY: <cat> | ANSWER: <value>

    Answer formatting per architecture.md:
        mcq        -> "5"    (quoted integer)
        left_right -> "left" (quoted string)
        distance   -> 5.2    (float, unquoted)
        count      -> 3      (integer, unquoted)
    """
    raw = str(normalized_answer)
    if category in ("mcq", "left_right"):
        formatted = f'"{raw}"'
    else:
        formatted = raw

    return f"CATEGORY: {category} | ANSWER: {formatted}"


# Dataset
class SpatialVLMDataset(Dataset):
    """PyTorch Dataset for SpatialVLM training/evaluation.

    Each __getitem__ returns a dict with:
        pixel_values   : [num_patches, 1536]  -- from Qwen processor
        image_grid_thw : [1, 3]               -- patch grid (t, h, w)
        depth_map      : [H, W]               -- raw depth (float32)
        input_ids      : [T]                  -- full sequence (prompt + answer)
        labels         : [T]                  -- masked prompt, active answer
        attention_mask : [T]                  -- 1s everywhere (no padding yet)
        mask_positions : list[int]            -- <mask> token positions
        rle_list       : list[dict]           -- RLE masks for each <mask>
        category       : str                  -- task category
        answer         : str                  -- formatted answer string
        image_name     : str                  -- filename for debugging

    Args:
        split:      "train", "val", "test", or "train_sample"
        processor:  Qwen AutoProcessor (for tokenization + image processing)
        max_samples: limit number of samples (None = use all)
    """

    def __init__(
        self,
        split: str,
        processor,
        max_samples: int | None = None,
    ):
        assert split in _SPLIT_CONFIG, f"Unknown split: {split}. Use: {list(_SPLIT_CONFIG.keys())}"
        cfg = _SPLIT_CONFIG[split]

        self.split = split
        self.processor = processor
        self.tokenizer = processor.tokenizer

        # Paths
        self.json_path = os.path.join(ROOT, cfg["json"])
        self.image_dir = os.path.join(ROOT, cfg["dir"], "images")
        self.depth_dir = os.path.join(ROOT, cfg["dir"], "depths")

        # Load annotations
        with open(self.json_path, "r") as f:
            self.data = json.load(f)

        if max_samples is not None:
            self.data = self.data[:max_samples]

        # Pre-compute the assistant token boundary for label masking
        # We'll use a dummy to find the token offset
        self._assistant_marker = "<|im_start|>assistant\n"

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        entry = self.data[idx]

        # 1. Load image
        image_name = entry["image"]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")

        # 2. Load depth map (8-bit grayscale, range 0-255)
        depth_path = os.path.join(self.depth_dir, image_name.replace(".png", "_depth.png"))
        depth_np = np.array(Image.open(depth_path), dtype=np.float32)
        depth_map = torch.from_numpy(depth_np)  # [H, W]

        # 3. Parse question
        question_raw = entry["conversations"][0]["value"]
        question = question_raw.replace("<image>\n", "").replace("<image>", "").strip()

        # 4. Build target answer string
        category = entry["category"]
        target_text = format_answer(
            category, entry["normalized_answer"]
        )

        # 5. Build chat messages WITH answer (for teacher forcing)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": question},
            ]},
            {"role": "assistant", "content": target_text},
        ]

        # Tokenize: no generation prompt (answer is included)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        # Strip <think>...</think> tags (Qwen 3.5 adds them even with enable_thinking=False)
        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt", padding=False
        )

        input_ids      = inputs["input_ids"].squeeze(0)        # [T]
        attention_mask = inputs["attention_mask"].squeeze(0)   # [T]
        pixel_values   = inputs["pixel_values"]                # [num_patches, 1536]
        image_grid_thw = inputs["image_grid_thw"]              # [1, 3]

        # 6. Find <mask> positions
        mask_positions = find_mask_positions(input_ids.unsqueeze(0), self.tokenizer)
        rle_list = entry["rle"]

        # Truncate to minimum if mismatch
        n = min(len(mask_positions), len(rle_list))
        mask_positions = mask_positions[:n]
        rle_list = rle_list[:n]

        # 7. Build labels (mask prompt, train only on answer)
        labels = self._build_labels(input_ids, text)

        # 8. Format answer for metadata
        raw_answer = str(entry["normalized_answer"])
        if category in ("mcq", "left_right"):
            answer_str = f'"{raw_answer}"'
        else:
            answer_str = raw_answer

        return {
            "pixel_values":   pixel_values,
            "image_grid_thw": image_grid_thw,
            "depth_map":      depth_map,
            "input_ids":      input_ids,
            "labels":         labels,
            "attention_mask": attention_mask,
            "mask_positions": mask_positions,
            "rle_list":       rle_list,
            "category":       category,
            "answer":         answer_str,
            "image_name":     image_name,
        }

    def _build_labels(self, input_ids: torch.Tensor, full_text: str) -> torch.Tensor:
        """Mask prompt tokens with -100, keep only answer tokens as labels.

        Searches for the last occurrence of '<|im_start|>assistant\n' token
        sequence in input_ids to find where the answer begins.
        """
        labels = input_ids.clone()

        # Tokenize the assistant marker to get its token ID(s)
        marker_ids = self.tokenizer.encode(
            self._assistant_marker, add_special_tokens=False
        )

        # Search for the last occurrence of marker_ids in input_ids
        ids_list = input_ids.tolist()
        marker_len = len(marker_ids)
        answer_start = -1

        for i in range(len(ids_list) - marker_len, -1, -1):
            if ids_list[i:i + marker_len] == marker_ids:
                answer_start = i + marker_len
                break

        if answer_start == -1:
            # Fallback: can't find marker, mask everything
            labels[:] = -100
            return labels

        # Mask everything up to (and including) the assistant marker
        labels[:answer_start] = -100

        return labels


# Collate function
def collate_fn(batch: list[dict]) -> dict:
    """Custom collate for SpatialVLMDataset.

    Since all images are FullHD (1920x1080), pixel_values and image_grid_thw
    have consistent shapes. input_ids/labels may vary slightly -- we pad them.
    """
    # Pad input_ids and labels to max length in batch
    max_len = max(d["input_ids"].shape[0] for d in batch)
    pad_id = 0  # Qwen pad token

    input_ids_padded = []
    labels_padded = []
    attention_masks = []

    for d in batch:
        T = d["input_ids"].shape[0]
        pad_len = max_len - T

        input_ids_padded.append(
            torch.cat([d["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)])
        )
        labels_padded.append(
            torch.cat([d["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        )
        attention_masks.append(
            torch.cat([d["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )

    return {
        "pixel_values":   torch.cat([d["pixel_values"] for d in batch], dim=0),
        "image_grid_thw": torch.cat([d["image_grid_thw"] for d in batch], dim=0),
        "depth_maps":     torch.stack([d["depth_map"] for d in batch]),
        "input_ids":      torch.stack(input_ids_padded),
        "labels":         torch.stack(labels_padded),
        "attention_mask": torch.stack(attention_masks),
        # Lists (not tensorizable) - one per sample
        "mask_positions": [d["mask_positions"] for d in batch],
        "rle_list":       [d["rle_list"] for d in batch],
        "categories":     [d["category"] for d in batch],
        "answers":        [d["answer"] for d in batch],
        "image_names":    [d["image_name"] for d in batch],
    }


# DataLoader factory
def get_dataloader(
    dataset: SpatialVLMDataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader with custom collation."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
    )