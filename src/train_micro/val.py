"""
SpatialVLM Micro — Validation
==============================

Runs the model on the val split and returns the average loss.
Called from train.py at the end of each epoch.

Usage (standalone):
    python src/train_micro/val.py
    python src/train_micro/val.py --resolution 320p --max-samples 100
"""
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import os
import sys
import argparse
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.dataloader.dataloader_new import SpatialVLMDataset, get_dataloader
from model_micro.pipeline import SpatialVLM, print_vram_usage
from model_micro.loss import SpatialLoss


@torch.no_grad()
def validate(pipeline, criterion, processor, resolution="320p",
             batch_size=4, num_workers=2, max_samples=None, split="val"):
    """Run validation and return average loss.

    Args:
        pipeline:    SpatialVLM model (already on GPU)
        criterion:   SpatialLoss
        processor:   Qwen processor
        resolution:  image resolution
        batch_size:  validation batch size
        num_workers: dataloader workers
        max_samples: limit number of samples (None = all)
        split:       dataset split

    Returns:
        dict with 'val_loss', 'val_ce', 'val_mse', 'n_samples'
    """
    target_size = {"1080p": None, "720p": (1280, 720),
                   "540p": (960, 540), "450p": (800, 450),
                   "320p": (512, 320)}[resolution]

    dataset = SpatialVLMDataset(split, processor=processor,
                                max_samples=max_samples, target_size=target_size)
    loader = get_dataloader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=False)

    dev = pipeline.device
    dtype = next(pipeline.parameters()).dtype
    pipeline.eval()

    total_loss_sum = 0.0
    total_ce_sum = 0.0
    total_sl1_sum = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="Validation", leave=False)
    for batch in pbar:
        pixel_values   = batch["pixel_values"].to(device=dev, dtype=dtype, non_blocking=True)
        pixel_values_rgb = batch["pixel_values_rgb"].to(device=dev, dtype=dtype, non_blocking=True)
        image_grid_thw = batch["image_grid_thw"].to(device=dev, non_blocking=True)
        depth_maps     = batch["depth_maps"].to(device=dev, dtype=dtype, non_blocking=True)
        input_ids      = batch["input_ids"].to(device=dev, non_blocking=True)
        labels         = batch["labels"].to(device=dev, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device=dev, non_blocking=True)

        try:
            output = pipeline(
                pixel_values=pixel_values,
                pixel_values_rgb=pixel_values_rgb,
                image_grid_thw=image_grid_thw,
                depth_maps=depth_maps,
                input_ids=input_ids,
                rle_list=batch["rle_list"],
                mask_token_positions=batch["mask_positions"],
                decoded_masks=batch["decoded_masks"],
                num_token_positions=batch.get("num_token_positions"),
                attention_mask=attention_mask,
            )
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                continue
            raise

        logits = output["logits"]
        num_pred = output["num_pred"]

        loss, components = criterion(
            logits, labels,
            num_pred, batch["target_num"].to(dev),
            batch["is_numeric"].to(dev),
            return_components=True,
        )
        total_loss_sum += loss.item()
        total_ce_sum += components['ce']
        total_sl1_sum += components['sl1']
        n_batches += 1
        pbar.set_postfix({
            "val_loss": f"{total_loss_sum / max(n_batches, 1):.4f}",
            "ce": f"{total_ce_sum / max(n_batches, 1):.4f}",
            "sl1": f"{total_sl1_sum / max(n_batches, 1):.4f}",
        })

    avg_loss = total_loss_sum / max(n_batches, 1)
    avg_ce = total_ce_sum / max(n_batches, 1)
    avg_sl1 = total_sl1_sum / max(n_batches, 1)
    return {
        "val_loss": avg_loss,
        "val_ce": avg_ce,
        "val_sl1": avg_sl1,
        "n_samples": len(dataset),
        "n_batches": n_batches,
    }

# =========================================================================
# Standalone
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="SpatialVLM Micro Validation")
    parser.add_argument("--device",      default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype",       default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--attn-impl",   default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--resolution",  default="320p",
                        choices=["1080p", "720p", "540p", "450p", "320p"])
    parser.add_argument("--batch-size",  type=int, default=8)
    parser.add_argument("--split",       default="val",
                        choices=["val", "train_sample"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--checkpoint",  type=str, default=None,
                        help="Path to checkpoint dir to load before validation")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    pipeline = SpatialVLM(dtype=dtype, device_map=args.device,
                          attn_implementation=args.attn_impl)
    print_vram_usage("after model load")

    # Load checkpoint if specified
    if args.checkpoint:
        ckpt_path = os.path.join(args.checkpoint, "checkpoint.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        pipeline.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"  Loaded checkpoint: {args.checkpoint}")

    criterion = SpatialLoss(alpha=0.1)

    results = validate(
        pipeline, criterion, pipeline.processor,
        resolution=args.resolution,
        batch_size=args.batch_size,
        split=args.split,
        max_samples=args.max_samples,
    )

    print(f"\n{'='*70}")
    print(f"  val_loss = {results['val_loss']:.6f}")
    print(f"  samples  = {results['n_samples']}")
    print(f"  batches  = {results['n_batches']}")
    print(f"{'='*70}")
    print_vram_usage("final")


if __name__ == "__main__":
    main()
