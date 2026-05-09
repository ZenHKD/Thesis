"""
SpatialVLM Micro — Validation (AMD ROCm)
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

from src.dataloader.dataloader import SpatialVLMDataset, get_dataloader
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
    total_cat_sum = 0.0
    total_cat_correct = 0
    total_cat_samples = 0
    total_num_correct = 0
    total_num_samples = 0
    n_batches = 0
    n_num_batches = 0   # batches with numeric samples
    n_cat_batches = 0   # batches with categorical samples

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
                cat_token_positions=batch.get("cat_token_positions"),
                attention_mask=attention_mask,
            )
        except RuntimeError as e:
            if "out of memory" in str(e):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise

        logits = output["logits"]
        num_pred = output["num_pred"]
        cat_logits = output.get("cat_logits", None)

        is_numeric = batch["is_numeric"].to(dev)
        is_categorical = batch.get("is_categorical", torch.zeros(1, dtype=torch.bool)).to(dev)
        target_cat = batch.get("target_cat_index", torch.zeros(1)).to(dev)

        loss, components = criterion(
            logits, labels,
            num_pred, batch["target_num"].to(dev),
            is_numeric,
            cat_logits=cat_logits,
            cat_targets=target_cat,
            is_categorical=is_categorical,
            return_components=True,
        )
        total_loss_sum += loss.item()
        total_ce_sum += components['ce']

        # Only count sl1/cat towards their respective averages
        # when the batch actually contains relevant samples
        loss_cat = components.get('cat', 0.0)
        sl1 = components['sl1']

        if is_numeric.any():
            total_sl1_sum += sl1
            n_num_batches += 1
            # Calculate numeric accuracy (within 10% relative error)
            for b in range(num_pred.shape[0]):
                if is_numeric[b]:
                    pred_val = num_pred[b].item()
                    gt_val = batch["target_num"][b].item()
                    if abs(gt_val) > 1e-6:
                        rel_err = abs(pred_val - gt_val) / abs(gt_val)
                        if rel_err <= 0.10:
                            total_num_correct += 1
                    else:
                        # For targets near zero, use absolute error < 0.5
                        if abs(pred_val - gt_val) < 0.5:
                            total_num_correct += 1
                    total_num_samples += 1

        if is_categorical.any() and loss_cat > 0:
            total_cat_sum += loss_cat
            n_cat_batches += 1
            # Calculate accuracy for this batch
            for b in range(len(cat_logits)):
                if is_categorical[b] and cat_logits[b] is not None:
                    pred_idx = torch.argmax(cat_logits[b]).item()
                    if pred_idx == target_cat[b].item():
                        total_cat_correct += 1
                    total_cat_samples += 1

        n_batches += 1

        avg_cat_disp = total_cat_sum / max(n_cat_batches, 1)
        avg_sl1_disp = total_sl1_sum / max(n_num_batches, 1)
        avg_cat_acc_disp = (total_cat_correct / max(total_cat_samples, 1)) * 100
        avg_num_acc_disp = (total_num_correct / max(total_num_samples, 1)) * 100
        pbar.set_postfix({
            "val_loss": f"{total_loss_sum / max(n_batches, 1):.4f}",
            "ce": f"{total_ce_sum / max(n_batches, 1):.4f}",
            "sl1": f"{avg_sl1_disp:.4f}",
            "cat": f"{avg_cat_disp:.4f}",
            "cat_acc": f"{avg_cat_acc_disp:.1f}%",
            "num_acc": f"{avg_num_acc_disp:.1f}%",
        })

    avg_loss = total_loss_sum / max(n_batches, 1)
    avg_ce = total_ce_sum / max(n_batches, 1)
    avg_sl1 = total_sl1_sum / max(n_num_batches, 1)
    avg_cat = total_cat_sum / max(n_cat_batches, 1)
    avg_cat_acc = (total_cat_correct / max(total_cat_samples, 1)) * 100
    avg_num_acc = (total_num_correct / max(total_num_samples, 1)) * 100
    return {
        "val_loss": avg_loss,
        "val_ce": avg_ce,
        "val_sl1": avg_sl1,
        "val_cat": avg_cat,
        "val_cat_acc": avg_cat_acc,
        "val_num_acc": avg_num_acc,
        "n_samples": len(dataset),
        "n_batches": n_batches,
        "n_num_batches": n_num_batches,
        "n_cat_batches": n_cat_batches,
    }

# =========================================================================
# Standalone
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="SpatialVLM Micro Validation")
    parser.add_argument("--device",      default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype",       default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--attn-impl",   default="sdpa",
                        choices=["sdpa", "eager"])
    parser.add_argument("--resolution",  default="320p",
                        choices=["1080p", "720p", "540p", "450p", "320p"])
    parser.add_argument("--batch-size",  type=int, default=4)
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
        model_path = os.path.join(args.checkpoint, "model.safetensors")
        if os.path.exists(model_path):
            from safetensors.torch import load_file
            model_state = load_file(model_path)
            pipeline.load_state_dict(model_state, strict=False)
        else:
            ckpt_path = os.path.join(args.checkpoint, "checkpoint.pt")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            pipeline.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"  Loaded checkpoint: {args.checkpoint}")

    criterion = SpatialLoss(alpha=1.0, gamma=1.0)

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
