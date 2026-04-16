"""
SpatialVLM Micro — Batched DataLoader
======================================

PyTorch Dataset for the NVIDIA Warehouse dataset, adapted for the Micro
architecture with batch_size > 1 support and Number Head fields.

Key changes from v1 dataloader:
    1. format_answer(): distance/count -> "category | <|num|>" (Number Head)
    2. Chain-of-thought: GPT reasoning wrapped in <think>...</think>
    3. No chat template: question tokenized directly, image processed separately
    4. collate_fn(): supports batch_size > 1 with padding
    5. Separate tokenization: question & answer tokenized independently
       then concatenated to guarantee exact label boundaries (BPE-safe)
    6. RTI 3→3: <mask> (3 tokens) replaced by [mask_rgb, mask_depth, space]
       — NO sequence length change, no trimming needed

Splits: train (499K), val (1.9K), test (19K)

Usage:
    from src.dataloader.dataloader import SpatialVLMDataset, get_dataloader

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
            out["logits_per_step"], batch["labels"],
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
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
import pycocotools.mask as mask_utils
import random



sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from model_micro.pipeline import find_mask_positions, NUM_TOKEN_ID

# Paths
ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "nvidia_warehouse_dataset")
MODEL_MICRO_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "model_micro", "qwen3.5-micro")

_SPLIT_CONFIG = {
    "train":        {"json": "train.json",        "dir": "train"},
    "val":          {"json": "val.json",           "dir": "val"},
    "test":         {"json": "test.json",          "dir": "test"},
    "train_sample": {"json": "train_sample/train_sample.json", "dir": "train_sample"},
}


# ===========================================================================
# Answer formatting (Micro: <|num|> for numeric tasks)
# ===========================================================================

def format_answer(category: str, normalized_answer) -> str:
    """Build the structured target string for training.

    Format: <category> | <value>

    Micro architecture changes:
        distance -> "distance | <|num|>"    (Number Head predicts the value)
        count    -> "count | <|num|>"       (Number Head predicts the value)
        mcq      -> 'mcq | "5"'          (LM Head, quoted integer)
        left_right-> 'left_right | "left"' (LM Head, quoted string)
    """
    if category in ("distance", "count"):
        return f"{category} | <|num|>"
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
        pixel_values_rgb: [3, H_orig, W_orig] — raw image tensor for RTI
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
        num_token_pos    : int              — position of <|num|> in input_ids (-1 if not numeric)
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
        self.tokenizer = processor.tokenizer  # single tokenizer (full vocab, no remapping)

        self.json_path = os.path.join(ROOT, cfg["json"])
        self.image_dir = os.path.join(ROOT, cfg["dir"], "images")
        self.depth_dir = os.path.join(ROOT, cfg["dir"], "depths")

        with open(self.json_path, "r") as f:
            self.data = json.load(f)

        if max_samples is not None:
            self.data = self.data[:max_samples]

        self.target_size = target_size

        # Data augmentation (RGB only, no geometric transforms)
        # Only active for training splits — val/test stay deterministic
        self.augment = split in ("train", "train_sample")

        # Cache <|num|> token ID
        self.num_token_id = NUM_TOKEN_ID

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        try:
            return self._load_sample(idx)
        except (OSError, Exception) as e:
            print(f"  [!] Skipping sample {idx} ({self.data[idx].get('image', '?')}): {e}")
            return self._load_sample((idx + 1) % len(self.data))

    def _augment_rgb(self, image: Image.Image) -> Image.Image:
        """Mild appearance-only augmentation for RGB images.

        Safe transforms (no geometry changes):
            - Brightness jitter (±15%)
            - Contrast jitter (±15%)
            - Saturation jitter (±15%)
            - Gaussian blur (σ=0.1-1.0, 20% chance)

        NOT applied: flip, rotate, crop, resize, affine
        (these would break distance, left_right, and mask alignment)
        """
        from PIL import ImageEnhance

        # Brightness: randomly adjust ±15%
        if random.random() < 0.5:
            factor = random.uniform(0.85, 1.15)
            image = ImageEnhance.Brightness(image).enhance(factor)

        # Contrast: randomly adjust ±15%
        if random.random() < 0.5:
            factor = random.uniform(0.85, 1.15)
            image = ImageEnhance.Contrast(image).enhance(factor)

        # Saturation: randomly adjust ±15%
        if random.random() < 0.5:
            factor = random.uniform(0.85, 1.15)
            image = ImageEnhance.Color(image).enhance(factor)

        # Gaussian blur: mild, 20% chance
        if random.random() < 0.2:
            radius = random.uniform(0.1, 1.0)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))

        return image

    def _load_sample(self, idx: int) -> dict:
        entry = self.data[idx]

        # 1. Load image
        image_name = entry["image"]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        if self.target_size:
            image = image.resize(self.target_size, Image.LANCZOS)

        # Apply augmentation (RGB only, training splits only)
        # Safe transforms that preserve geometry (no flip/crop/rotate/resize)
        if self.augment:
            image = self._augment_rgb(image)

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

        # 4. Get GPT reasoning from dataset (chain-of-thought)
        gpt_reasoning = entry["conversations"][1]["value"]

        # 5. Build target answer string (Micro: <|num|> for numeric)
        category = entry["category"]
        target_text = format_answer(category, entry["normalized_answer"])

        # 6. Build full answer with <think> chain-of-thought
        full_answer = f"<think>{gpt_reasoning}</think>{target_text}<|im_end|>\n"

        # 7. Determine numeric fields
        is_numeric = category in ("distance", "count")
        target_num = float(entry["normalized_answer"]) if is_numeric else 0.0

        # 8. Mask replacement for Object Ref Grounding
        question = re.sub(r'(<mask.*?>)', r'<|object_ref_start|>\1<|object_ref_end|>', question)

        # 9. Process image separately (pixel_values + grid only)
        image_inputs = self.processor.image_processor(
            images=image, return_tensors="pt"
        )
        pixel_values   = image_inputs["pixel_values"]
        image_grid_thw = image_inputs["image_grid_thw"]

        sys_str = (
            "<|im_start|>system\n"
            "You are an expert AI assistant for warehouse spatial reasoning. "
            "Analyze the image and the specific object regions carefully. "
            "First, output your step-by-step reasoning inside <think></think> tags. "
            "Finally, output your exact answer strictly using the format: 'category | value'.<|im_end|>\n"
        )
        
        # 10. Calculate visual tokens correctly for inline injection
        h_p, w_p = image_grid_thw[0, 1].item(), image_grid_thw[0, 2].item()
        h_vis, w_vis = h_p // 2, w_p // 2
        num_visual_tokens = int(h_vis * w_vis)
        
        vision_str = "Picture 1: <|vision_start|>" + "<|image_pad|>" * num_visual_tokens + "<|vision_end|>\n"
        user_str = f"<|im_start|>user\n{vision_str}{question}<|im_end|>\n"
        eval_prompt = f"<|im_start|>assistant\n"
        
        q_ids = self.tokenizer.encode(sys_str + user_str + eval_prompt, add_special_tokens=False)

        # 11. For numeric categories, encode answer WITHOUT "<|num|>" and manually append <|num|> token
        # (BPE would encode "<|num|>" as regular tokens — we need the special <|num|> token ID)
        if is_numeric:
            # full_answer = "<think>...reasoning...</think>category | <|num|><|im_end|>\n"
            # Encode everything up to "<|num|>", then append <|num|> special token
            answer_text = full_answer.rsplit("<|num|>", 1)[0]  # e.g. "<think>...</think>distance | "
            a_tail = full_answer.rsplit("<|num|>", 1)[1]
            a_ids = self.tokenizer.encode(answer_text, add_special_tokens=False) + [self.num_token_id] + self.tokenizer.encode(a_tail, add_special_tokens=False)
        else:
            a_ids = self.tokenizer.encode(full_answer, add_special_tokens=False)

        all_ids    = q_ids + a_ids
        input_ids  = torch.tensor(all_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        # 12. Labels: question + separator = -100, answer (+ EOS) = active
        answer_start = len(q_ids)
        labels = input_ids.clone()
        labels[:answer_start] = -100

        # 13. Find <mask> positions in the token sequence
        mask_positions = find_mask_positions(input_ids.unsqueeze(0), self.tokenizer)
        rle_list = entry["rle"]

        n = min(len(mask_positions), len(rle_list))
        mask_positions = mask_positions[:n]
        rle_list = rle_list[:n]

        # 14. Find <|num|> token position (for Number Head)
        num_token_pos = -1
        if is_numeric:
            num_token_pos = self._find_num_token_pos(input_ids)

        # 15. Format answer for metadata
        raw_answer = str(entry["normalized_answer"])
        if category in ("mcq", "left_right"):
            answer_str = f'"{raw_answer}"'
        elif is_numeric:
            answer_str = f"<|num|>={raw_answer}"
        else:
            answer_str = raw_answer

        # 16. Pre-decode RLE masks + soft masks
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

            # Relative Coverage Normalization (must match rti.py)
            coverage = coverage / (coverage.max() + 1e-8)

            soft2d = torch.sigmoid(50.0 * (coverage - 0.3))
            decoded_masks.append({'binary': binary, 'soft2d': soft2d})

        # 17. Raw RGB for RTI (0-1 float)
        pixel_values_rgb = TF.to_tensor(image)  # [3, H_orig, W_orig]

        return {
            "pixel_values":   pixel_values,
            "pixel_values_rgb": pixel_values_rgb,
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

    def _find_num_token_pos(self, input_ids: torch.Tensor) -> int:
        """Find position of <|num|> token in input_ids.

        <|num|> token has a fixed ID in the vocab (appended by prune.py).
        """
        ids_list = input_ids.tolist()

        # Search from end (<|num|> is in the answer portion)
        for i in range(len(ids_list) - 1, -1, -1):
            if ids_list[i] == self.num_token_id:
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
    pad_token_id = 248044  # Qwen pad token (<|endoftext|>)

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
    pixel_values_rgb = torch.stack([d["pixel_values_rgb"] for d in batch], dim=0)
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
        "pixel_values_rgb":    pixel_values_rgb,
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
