"""
SpatialVLM Micro — Batched DataLoader
======================================

PyTorch Dataset for the NVIDIA Warehouse dataset, adapted for the Micro
architecture with batch_size > 1 support and Number Head fields.

Key changes from v1 dataloader:
    1. format_answer(): distance/count -> "category | NUM" (Number Head)
    2. __getitem__() adds: is_numeric, target_num fields
    3. collate_fn(): supports batch_size > 1 with padding
    4. No more `assert batch_size == 1`

Splits: train (499K), val (1.9K), test (19K)

Usage:
    from src.dataloader.dataloader_new import SpatialVLMDataset, get_dataloader

    dataset = SpatialVLMDataset("train", processor=pipeline.processor)
    loader  = get_dataloader(dataset, batch_size=8, shuffle=True)

    for batch in loader:
        out = pipeline(
            pixel_values=batch["pixel_values"],
            image_grid_thw=batch["image_grid_thw"],
            depth_maps=batch["depth_maps"],
            input_ids=batch["input_ids"],
            rle_list=batch["rle_list"],
            mask_token_positions=batch["mask_positions"],
            decoded_masks=batch["decoded_masks"],
            num_token_positions=batch["num_token_positions"],
        )
        loss = criterion(
            out["logits"], batch["labels"],
            out["num_pred"], batch["target_num"], batch["is_numeric"],
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
import pycocotools.mask as mask_utils

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from model_micro.pipeline import SYSTEM_PROMPT, find_mask_positions

# Paths
ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "nvidia_warehouse_dataset")

_SPLIT_CONFIG = {
    "train":        {"json": "train.json",        "dir": "train"},
    "val":          {"json": "val.json",           "dir": "val"},
    "test":         {"json": "test.json",          "dir": "test"},
    "train_sample": {"json": "train_sample/train_sample.json", "dir": "train_sample"},
}


# ===========================================================================
# Answer formatting (Micro: NUM for numeric tasks)
# ===========================================================================

def format_answer(category: str, normalized_answer) -> str:
    """Build the structured target string for training.

    Format: <category> | <value>

    Micro architecture changes:
        distance -> "distance | NUM"    (Number Head predicts the value)
        count    -> "count | NUM"       (Number Head predicts the value)
        mcq      -> 'mcq | "5"'          (LM Head, quoted integer)
        left_right-> 'left_right | "left"' (LM Head, quoted string)
    """
    if category in ("distance", "count"):
        return f"{category} | NUM"
    else:
        raw = str(normalized_answer)
        formatted = f'"{raw}"'
        return f"{category} | {formatted}"


# ===========================================================================
# Dataset
# ===========================================================================

class SpatialVLMDataset(Dataset):
    """PyTorch Dataset for SpatialVLM Micro training/evaluation.

    Each __getitem__ returns a dict with:
        pixel_values     : [num_patches, 1536]
        image_grid_thw   : [1, 3]
        depth_map        : [H, W]
        input_ids        : [T]
        labels           : [T]
        attention_mask   : [T]
        mask_positions   : list[int]        — <mask> token positions
        rle_list         : list[dict]       — RLE masks per <mask>
        decoded_masks    : list[dict]       — pre-decoded {binary, soft2d}
        category         : str              — task category
        answer           : str              — formatted answer string
        image_name       : str              — filename for debugging
        is_numeric       : bool             — True for distance/count
        target_num       : float            — ground truth number (0.0 if not numeric)
        num_token_pos    : int              — position of NUM in input_ids (-1 if not numeric)
    """

    def __init__(
        self,
        split: str,
        processor,
        max_samples: int | None = None,
        target_size: tuple[int, int] | None = None,
    ):
        assert split in _SPLIT_CONFIG, f"Unknown split: {split}. Use: {list(_SPLIT_CONFIG.keys())}"
        cfg = _SPLIT_CONFIG[split]

        self.split = split
        self.processor = processor
        self.tokenizer = processor.tokenizer

        self.json_path = os.path.join(ROOT, cfg["json"])
        self.image_dir = os.path.join(ROOT, cfg["dir"], "images")
        self.depth_dir = os.path.join(ROOT, cfg["dir"], "depths")

        with open(self.json_path, "r") as f:
            self.data = json.load(f)

        if max_samples is not None:
            self.data = self.data[:max_samples]

        self.target_size = target_size
        self._assistant_marker = "<|im_start|>assistant\n"

        # Cache NUM token ID for position finding
        self._num_token_str = "NUM"

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        try:
            return self._load_sample(idx)
        except (OSError, Exception) as e:
            print(f"  [!] Skipping sample {idx} ({self.data[idx].get('image', '?')}): {e}")
            return self._load_sample((idx + 1) % len(self.data))

    def _load_sample(self, idx: int) -> dict:
        entry = self.data[idx]

        # 1. Load image
        image_name = entry["image"]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        if self.target_size:
            image = image.resize(self.target_size, Image.LANCZOS)

        # 2. Load depth map
        depth_path = os.path.join(self.depth_dir, image_name.replace(".png", "_depth.png"))
        depth_pil = Image.open(depth_path)
        if self.target_size:
            depth_pil = depth_pil.resize(self.target_size, Image.BILINEAR)
        depth_np = np.array(depth_pil, dtype=np.float32)
        depth_map = torch.from_numpy(depth_np)

        # 3. Parse question
        question_raw = entry["conversations"][0]["value"]
        question = question_raw.replace("<image>\n", "").replace("<image>", "").strip()

        # 4. Build target answer string (Micro: NUM for numeric)
        category = entry["category"]
        target_text = format_answer(category, entry["normalized_answer"])

        # 5. Determine numeric fields
        is_numeric = category in ("distance", "count")
        target_num = float(entry["normalized_answer"]) if is_numeric else 0.0

        # 6. Build chat messages WITH answer
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": question},
            ]},
            {"role": "assistant", "content": target_text},
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt", padding=False
        )

        input_ids      = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values   = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]

        # 7. Find <mask> positions
        mask_positions = find_mask_positions(input_ids.unsqueeze(0), self.tokenizer)
        rle_list = entry["rle"]

        n = min(len(mask_positions), len(rle_list))
        mask_positions = mask_positions[:n]
        rle_list = rle_list[:n]

        # 8. Build labels
        labels = self._build_labels(input_ids, text, category)

        # 9. Find NUM token position (for Number Head)
        num_token_pos = -1
        if is_numeric:
            num_token_pos = self._find_num_token_pos(input_ids)

        # 10. Format answer for metadata
        raw_answer = str(entry["normalized_answer"])
        if category in ("mcq", "left_right"):
            answer_str = f'"{raw_answer}"'
        elif is_numeric:
            answer_str = f"NUM={raw_answer}"
        else:
            answer_str = raw_answer

        # 11. Pre-decode RLE masks + soft masks
        _, h_p, w_p = [int(x) for x in image_grid_thw[0].tolist()]
        h_vis, w_vis = h_p // 2, w_p // 2
        decoded_masks = []
        for rle_entry in rle_list:
            binary = mask_utils.decode(rle_entry).astype(np.float32)
            if self.target_size:
                binary = np.array(
                    Image.fromarray(binary).resize(self.target_size, Image.NEAREST)
                )
            binary = binary.astype(bool)
            t = torch.from_numpy(binary.astype(np.float32))
            coverage = torch.nn.functional.adaptive_avg_pool2d(
                t.unsqueeze(0).unsqueeze(0), (h_vis, w_vis)
            ).squeeze()
            soft2d = torch.sigmoid(50.0 * (coverage - 0.3))
            decoded_masks.append({'binary': binary, 'soft2d': soft2d})

        return {
            "pixel_values":   pixel_values,
            "image_grid_thw": image_grid_thw,
            "depth_map":      depth_map,
            "input_ids":      input_ids,
            "labels":         labels,
            "attention_mask": attention_mask,
            "mask_positions": mask_positions,
            "rle_list":       rle_list,
            "decoded_masks":  decoded_masks,
            "category":       category,
            "answer":         answer_str,
            "image_name":     image_name,
            "is_numeric":     is_numeric,
            "target_num":     target_num,
            "num_token_pos":  num_token_pos,
        }

    def _build_labels(self, input_ids: torch.Tensor, full_text: str,
                       category: str) -> torch.Tensor:
        """Build labels: mask prompt, keep entire answer active."""
        labels = input_ids.clone()

        marker_ids = self.tokenizer.encode(
            self._assistant_marker, add_special_tokens=False
        )

        ids_list = input_ids.tolist()
        marker_len = len(marker_ids)
        answer_start = -1

        for i in range(len(ids_list) - marker_len, -1, -1):
            if ids_list[i:i + marker_len] == marker_ids:
                answer_start = i + marker_len
                break

        if answer_start == -1:
            labels[:] = -100
            return labels

        labels[:answer_start] = -100
        return labels

    def _find_num_token_pos(self, input_ids: torch.Tensor) -> int:
        """Find position of NUM token in input_ids.

        BPE is context-dependent: in 'distance | NUM', the tokenizer produces
        ' NUM' (with leading space, old_id=15473), NOT bare 'NUM' (old_id=16968).

        We encode '| NUM' and take the last token to get the correct ID.
        """
        ids_list = input_ids.tolist()

        # Get the actual token ID as it appears in context "| NUM"
        ctx_ids = self.tokenizer.encode("| NUM", add_special_tokens=False)
        num_id = ctx_ids[-1]  # ' NUM' (with space) = old_id 15473

        # Search from end (NUM is in the answer portion)
        for i in range(len(ids_list) - 1, -1, -1):
            if ids_list[i] == num_id:
                return i

        return -1


# ===========================================================================
# Collate function (batch_size > 1 supported!)
# ===========================================================================

def collate_fn(batch: list[dict]) -> dict:
    """Collate for SpatialVLM Micro with variable-length padding.

    Handles:
        - Variable text lengths: pad input_ids/labels to max in batch
        - Variable mask counts: nest as list-of-lists
        - Fixed-size tensors: stack pixel_values, depth_maps, image_grid_thw
        - Numeric fields: stack is_numeric and target_num as tensors
    """
    B = len(batch)

    # --- Pad variable-length text sequences ---
    max_text_len = max(d["input_ids"].shape[0] for d in batch)
    pad_token_id = 0  # Qwen pad token

    input_ids     = torch.full((B, max_text_len), pad_token_id, dtype=torch.long)
    labels        = torch.full((B, max_text_len), -100, dtype=torch.long)
    attention_mask = torch.zeros(B, max_text_len, dtype=torch.long)

    for i, d in enumerate(batch):
        L = d["input_ids"].shape[0]
        input_ids[i, :L]      = d["input_ids"]
        labels[i, :L]         = d["labels"]
        attention_mask[i, :L] = d["attention_mask"]

    # --- Stack fixed-size tensors ---
    # pixel_values might have variable patches (different resolution)
    # For same-resolution training, they should be stackable
    pixel_values = torch.cat([d["pixel_values"] for d in batch], dim=0)
    image_grid_thw = torch.cat([d["image_grid_thw"] for d in batch], dim=0)

    # Depth maps (same resolution if target_size is set)
    depth_maps = torch.stack([d["depth_map"] for d in batch])

    # --- Variable-length mask data (list-of-lists) ---
    rle_list       = [d["rle_list"] for d in batch]
    mask_positions = [d["mask_positions"] for d in batch]
    decoded_masks  = [d["decoded_masks"] for d in batch]

    # --- Numeric fields ---
    is_numeric = torch.tensor([d["is_numeric"] for d in batch], dtype=torch.bool)
    target_num = torch.tensor([d["target_num"] for d in batch], dtype=torch.float32)
    num_token_positions = [d["num_token_pos"] for d in batch]

    # --- Metadata ---
    categories  = [d["category"] for d in batch]
    answers     = [d["answer"] for d in batch]
    image_names = [d["image_name"] for d in batch]

    return {
        "pixel_values":        pixel_values,
        "image_grid_thw":      image_grid_thw,
        "depth_maps":          depth_maps,
        "input_ids":           input_ids,
        "labels":              labels,
        "attention_mask":      attention_mask,
        "rle_list":            rle_list,
        "mask_positions":      mask_positions,
        "decoded_masks":       decoded_masks,
        "is_numeric":          is_numeric,
        "target_num":          target_num,
        "num_token_positions": num_token_positions,
        "categories":          categories,
        "answers":             answers,
        "image_names":         image_names,
    }


# ===========================================================================
# DataLoader factory
# ===========================================================================

def get_dataloader(
    dataset: SpatialVLMDataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader with batched collation."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
