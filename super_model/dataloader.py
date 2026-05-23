"""
SpatialVLM Super — Batched DataLoader (v2: Dual-Image + Simplified Output)
==========================================================================

PyTorch Dataset for the NVIDIA Warehouse dataset, adapted for the Super
architecture with 4 dedicated heads and batch_size > 1 support.

Key changes from v1:
    1. Dual-image input: [RGB, Depth] batched through ViT as Picture 1 & Picture 2
       - RGB  → visual_rgb_tokens [160, 1024] — scene appearance
       - Depth → visual_dep_tokens [160, 1024] — global depth context
       This provides the global depth features that v1 was missing (distance ~35% acc).
    2. Simplified LM output: LM head outputs ONLY the special token directly
       No more "category | <|token|>" format — just <|token|><|im_end|>
       The category is implied by which token is generated.
    3. DPT-based RTI: ViT intermediate features replace U-Net
       RTI now receives ViT intermediates instead of raw images.

Splits: train (499K), val (1.9K), test (19K)

Usage:
    from super_model.dataloader import SpatialVLMDataset, get_dataloader

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
            mcq_token_positions=batch["mcq_token_positions"],
            lr_token_positions=batch["lr_token_positions"],
            dist_token_positions=batch["dist_token_positions"],
            count_token_positions=batch["count_token_positions"],
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


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from super_model.pipeline import find_mask_positions, MCQ_TOKEN_ID, LR_TOKEN_ID, DIST_TOKEN_ID, COUNT_TOKEN_ID

# Paths
ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "nvidia_warehouse_dataset")
SUPER_MODEL_DIR = os.path.join(os.path.dirname(__file__), "qwen3.5-super")

_SPLIT_CONFIG = {
    "train":          {"json": "train.json",          "dir": "train"},
    "train_balanced": {"json": "train_balanced.json", "dir": "train"},
    "val":            {"json": "val.json",            "dir": "val"},
    "test":           {"json": "test.json",           "dir": "test"},
    "train_sample":   {"json": "train_sample/train_sample.json", "dir": "train_sample"},
}


# ===========================================================================
# Answer formatting (Super v2: direct token output, no category prefix)
# ===========================================================================

def format_answer(category: str, normalized_answer) -> str:
    """Build the structured target string for training.

    Super v2: LM head outputs only the special token directly.
        distance   → "<|dist|>"
        count      → "<|count|>"
        mcq        → "<|mcq|>"
        left_right → "<|lr|>"
    
    No more "category | <|token|>" format. The category is inferred
    from which special token the model generates.
    """
    if category == "distance":
        return "<|dist|>"
    elif category == "count":
        return "<|count|>"
    elif category == "mcq":
        return "<|mcq|>"
    elif category == "left_right":
        return "<|lr|>"
    else:
        raw = str(normalized_answer)
        return f'"{raw}"'


# ===========================================================================
# Dataset
# ===========================================================================

class SpatialVLMDataset(Dataset):
    """PyTorch Dataset for SpatialVLM Super training/evaluation.

    Each __getitem__ returns a dict with:
        pixel_values      : [num_patches_total, 1536] — RGB + Depth patches concatenated
        image_grid_thw    : [2, 3] — grid for [RGB, Depth] images
        depth_map         : [H, W]
        input_ids         : [T]
        labels            : [T]
        attention_mask    : [T]
        mask_positions    : list[int]        — <mask> token positions
        rle_list          : list[dict]       — RLE masks per <mask>
        decoded_masks     : list[dict]       — pre-decoded {binary, soft2d}
        category          : str              — task category
        answer            : str              — formatted answer string
        image_name        : str              — filename for debugging
        is_numeric        : bool             — True for distance/count
        is_categorical    : bool             — True for mcq/left_right
        target_num        : float            — ground truth number (0.0 if not numeric)
        target_cat_index  : int              — target mask index (-1 if not categorical)
        mcq_token_pos     : int              — position of <|mcq|> (-1 if not mcq)
        lr_token_pos      : int              — position of <|lr|> (-1 if not left_right)
        dist_token_pos    : int              — position of <|dist|> (-1 if not distance)
        count_token_pos   : int              — position of <|count|> (-1 if not count)
    """

    def __init__(
        self,
        split: str,
        processor,
        target_size: tuple[int, int] | None = None,
        max_samples: int | None = None,
        augment: bool = False,
        answer_weight: float = 1.0,
    ):
        assert split in _SPLIT_CONFIG, f"Unknown split: {split}. Use: {list(_SPLIT_CONFIG.keys())}"
        cfg = _SPLIT_CONFIG[split]

        self.split = split
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.target_size = target_size

        # Data augmentation (RGB only, no geometric transforms)
        # Only active for training splits — val/test stay deterministic
        self.augment = augment or (split in ("train", "train_sample"))

        self.json_path = os.path.join(ROOT, cfg["json"])
        self.image_dir = os.path.join(ROOT, cfg["dir"], "images")
        self.depth_dir = os.path.join(ROOT, cfg["dir"], "depths")

        with open(self.json_path, "r") as f:
            self.data = json.load(f)

        if max_samples is not None:
            self.data = self.data[:max_samples]

        # Cache special token IDs
        self.mcq_token_id   = MCQ_TOKEN_ID
        self.lr_token_id    = LR_TOKEN_ID
        self.dist_token_id  = DIST_TOKEN_ID
        self.count_token_id = COUNT_TOKEN_ID

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

        # 1. Load RGB image
        image_name = entry["image"]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        if self.target_size:
            image = image.resize(self.target_size, Image.LANCZOS)

        # Apply augmentation (RGB only, training splits only)
        if self.augment:
            image = self._augment_rgb(image)

        # 2. Load depth map (as raw tensor AND as RGB-converted PIL for ViT)
        depth_path = os.path.join(self.depth_dir, image_name.replace(".png", "_depth.png"))
        depth_pil = Image.open(depth_path)
        if self.target_size:
            depth_pil = depth_pil.resize(self.target_size, Image.BILINEAR)
        depth_np = np.array(depth_pil, dtype=np.float32)
        depth_map = torch.from_numpy(depth_np)

        # Convert depth to 3-channel "RGB" image for ViT processing
        # Normalize to 0-255 range and replicate across 3 channels
        depth_for_vit = depth_np.copy()
        if depth_for_vit.max() > 0:
            depth_for_vit = (depth_for_vit / depth_for_vit.max() * 255).astype(np.uint8)
        else:
            depth_for_vit = depth_for_vit.astype(np.uint8)
        depth_rgb = Image.fromarray(np.stack([depth_for_vit]*3, axis=-1))

        # 3. Parse question
        question_raw = entry["conversations"][0]["value"]
        question = question_raw.replace("<image>\n", "").replace("<image>", "").strip()

        # 4. Determine task type fields
        if self.split == "test":
            category = "unknown"
            is_numeric = False
            is_categorical = False
            target_num = 0.0
            target_cat_index = -1
            ans_text = ""
        else:
            category = entry["category"]
            is_numeric = category in ("distance", "count")
            is_categorical = category in ("mcq", "left_right")
            target_num = float(entry["normalized_answer"]) if is_numeric else 0.0
            
            # For classification heads: target index = which mask is the answer
            if is_categorical:
                raw_answer = str(entry["normalized_answer"]).strip().strip('"').strip("'")
                if category == "mcq":
                    target_cat_index = int(raw_answer)  # Region index (0-12)
                elif category == "left_right":
                    target_cat_index = 0 if raw_answer == "left" else 1
                else:
                    target_cat_index = -1
            else:
                target_cat_index = -1

            # 5. Build answer string (direct token output, no category prefix)
            if category == "distance":
                ans_text = ""
            elif category == "count":
                ans_text = ""
            elif category == "mcq":
                ans_text = ""
            elif category == "left_right":
                ans_text = ""
            else:
                raw = str(entry["normalized_answer"])
                ans_text = f'"{raw}"'

        tail_str = "<|im_end|>\n"

        # 6. Mask replacement for Object Ref Grounding
        mask_idx = [0]
        def replace_mask(match):
            i = mask_idx[0]
            mask_idx[0] += 1
            return f"[Region {i}]: <|object_ref_start|>{match.group(1)}<|object_ref_end|>"
        question = re.sub(r'(<mask.*?>)', replace_mask, question)

        # 7. Process BOTH images through image_processor (batch of 2)
        image_inputs = self.processor.image_processor(
            images=[image, depth_rgb], return_tensors="pt"
        )
        pixel_values   = image_inputs["pixel_values"]       # [num_patches_total, 1536]
        image_grid_thw = image_inputs["image_grid_thw"]      # [2, 3]

        # 8. Calculate visual tokens for EACH image (for inline injection)
        # RGB tokens
        h_p_rgb, w_p_rgb = image_grid_thw[0, 1].item(), image_grid_thw[0, 2].item()
        h_vis_rgb, w_vis_rgb = h_p_rgb // 2, w_p_rgb // 2
        num_visual_rgb_tokens = int(h_vis_rgb * w_vis_rgb)

        # Depth tokens  
        h_p_dep, w_p_dep = image_grid_thw[1, 1].item(), image_grid_thw[1, 2].item()
        h_vis_dep, w_vis_dep = h_p_dep // 2, w_p_dep // 2
        num_visual_dep_tokens = int(h_vis_dep * w_vis_dep)
        
        # Build dual-image vision string
        vision_str_1 = "Picture 1 (RGB): <|vision_start|>" + "<|image_pad|>" * num_visual_rgb_tokens + "<|vision_end|>\n"
        vision_str_2 = "Picture 2 (Depth): <|vision_start|>" + "<|image_pad|>" * num_visual_dep_tokens + "<|vision_end|>\n"
        user_str = f"<|im_start|>user\n{vision_str_1}{vision_str_2}{question}<|im_end|>\n"
        eval_prompt = f"<|im_start|>assistant\n"
        
        q_ids = self.tokenizer.encode(user_str + eval_prompt, add_special_tokens=False)

        # 9. Encode answer tokens
        ans_ids = self.tokenizer.encode(ans_text, add_special_tokens=False) if ans_text else []
        tail_ids = self.tokenizer.encode(tail_str, add_special_tokens=False)

        # Map each category to its dedicated special token (direct, no prefix)
        if category == "distance":
            target_ids = [self.dist_token_id] + tail_ids
        elif category == "count":
            target_ids = [self.count_token_id] + tail_ids
        elif category == "mcq":
            target_ids = [self.mcq_token_id] + tail_ids
        elif category == "left_right":
            target_ids = [self.lr_token_id] + tail_ids
        else:
            target_ids = ans_ids + tail_ids

        a_ids = target_ids
        all_ids    = q_ids + a_ids
        input_ids  = torch.tensor(all_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)


        # 10. Labels: question = -100, answer = active
        answer_start = len(q_ids)
        labels = input_ids.clone()
        labels[:answer_start] = -100

        # 11. Find <mask> positions in the token sequence
        mask_positions = find_mask_positions(input_ids.unsqueeze(0), self.tokenizer)
        rle_list = entry["rle"]

        n = min(len(mask_positions), len(rle_list))
        mask_positions = mask_positions[:n]
        rle_list = rle_list[:n]

        # 12. Find special token positions (one per head)
        mcq_token_pos   = -1
        lr_token_pos    = -1
        dist_token_pos  = -1
        count_token_pos = -1

        if category == "mcq":
            mcq_token_pos = self._find_token_pos(input_ids, self.mcq_token_id)
        elif category == "left_right":
            lr_token_pos = self._find_token_pos(input_ids, self.lr_token_id)
        elif category == "distance":
            dist_token_pos = self._find_token_pos(input_ids, self.dist_token_id)
        elif category == "count":
            count_token_pos = self._find_token_pos(input_ids, self.count_token_id)

        # 13. Format answer for metadata
        if self.split == "test":
            answer_str = ""
        else:
            raw_answer = str(entry["normalized_answer"])
            if category == "mcq":
                answer_str = f"<|mcq|>={raw_answer}"
            elif category == "left_right":
                answer_str = f"<|lr|>={raw_answer}"
            elif category == "distance":
                answer_str = f"<|dist|>={raw_answer}"
            elif category == "count":
                answer_str = f"<|count|>={raw_answer}"
            else:
                answer_str = raw_answer

        # 14. Pre-decode RLE masks + soft masks (use RGB grid for mask resolution)
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
            "_question":      question,
            "_id":            entry["id"],
            "is_numeric":     is_numeric,
            "is_categorical": is_categorical,
            "target_num":     target_num,
            "target_cat_index": target_cat_index,
            "mcq_token_pos":   mcq_token_pos,
            "lr_token_pos":    lr_token_pos,
            "dist_token_pos":  dist_token_pos,
            "count_token_pos": count_token_pos,
        }

    def _find_token_pos(self, input_ids: torch.Tensor, token_id: int) -> int:
        """Find position of a special token in input_ids.

        Searches from end since special tokens are in the answer portion.
        """
        ids_list = input_ids.tolist()

        # Search from end
        for i in range(len(ids_list) - 1, -1, -1):
            if ids_list[i] == token_id:
                return i

        return -1


# ===========================================================================
# Collate function (batch_size > 1 supported!)
# ===========================================================================

def collate_fn(batch: list[dict]) -> dict:
    """Collate for SpatialVLM Super with variable-length padding.

    Handles:
        - Variable text lengths: pad input_ids/labels to max in batch
        - Variable mask counts: nest as list-of-lists
        - Fixed-size tensors: stack pixel_values, depth_maps, image_grid_thw
        - Numeric fields: stack is_numeric and target_num as tensors
        - 4 separate token position lists (one per head)
        - Dual-image: each sample has 2 entries in image_grid_thw
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
    # pixel_values: each sample has patches for 2 images (RGB + Depth)
    pixel_values = torch.cat([d["pixel_values"] for d in batch], dim=0)
    # image_grid_thw: each sample has [2, 3], stack to [B*2, 3]
    image_grid_thw = torch.cat([d["image_grid_thw"] for d in batch], dim=0)

    # Depth maps (same resolution if target_size is set)
    depth_maps = torch.stack([d["depth_map"] for d in batch])

    # --- Variable-length mask data (list-of-lists) ---
    rle_list       = [d["rle_list"] for d in batch]
    mask_positions = [d["mask_positions"] for d in batch]
    decoded_masks  = [d["decoded_masks"] for d in batch]

    # --- Numeric fields ---
    is_numeric = torch.tensor([d["is_numeric"] for d in batch], dtype=torch.bool)
    is_categorical = torch.tensor([d["is_categorical"] for d in batch], dtype=torch.bool)
    target_num = torch.tensor([d["target_num"] for d in batch], dtype=torch.float32)
    target_cat_index = torch.tensor([d["target_cat_index"] for d in batch], dtype=torch.long)

    # --- 4 separate token position lists ---
    mcq_token_positions   = [d["mcq_token_pos"] for d in batch]
    lr_token_positions    = [d["lr_token_pos"] for d in batch]
    dist_token_positions  = [d["dist_token_pos"] for d in batch]
    count_token_positions = [d["count_token_pos"] for d in batch]

    # --- Metadata ---
    categories  = [d["category"] for d in batch]
    answers     = [d["answer"] for d in batch]
    image_names = [d["image_name"] for d in batch]

    return {
        "pixel_values":          pixel_values,
        "image_grid_thw":        image_grid_thw,
        "depth_maps":            depth_maps,
        "input_ids":             input_ids,
        "labels":                labels,

        "attention_mask":        attention_mask,
        "rle_list":              rle_list,
        "mask_positions":        mask_positions,
        "decoded_masks":         decoded_masks,
        "is_numeric":            is_numeric,
        "is_categorical":        is_categorical,
        "target_num":            target_num,
        "target_cat_index":      target_cat_index,
        "mcq_token_positions":   mcq_token_positions,
        "lr_token_positions":    lr_token_positions,
        "dist_token_positions":  dist_token_positions,
        "count_token_positions": count_token_positions,
        "categories":            categories,
        "answers":               answers,
        "image_names":           image_names,
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
