"""
SpatialVLM Super — Validation
==============================

Runs the model on the val split and returns the average loss.
Called from train.py at the end of each epoch / at val_steps.

Usage (standalone):
    python src/train_super/val.py
    python src/train_super/val.py --resolution 320p --max-samples 100
"""
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import os
import sys
import argparse
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from super_model.dataloader import SpatialVLMDataset, get_dataloader
from super_model.pipeline import SpatialVLM, print_vram_usage
from super_model.loss import SpatialLoss


@torch.no_grad()
def validate(pipeline, criterion, processor, resolution="320p",
             batch_size=4, num_workers=2, max_samples=None, split="val"):
    """Run validation and return average loss + per-head metrics.

    Returns:
        dict with val_loss, val_ce, val_dist, val_count, val_mcq, val_lr,
        val_mcq_acc, val_lr_acc, val_dist_acc, val_count_acc, n_samples, ...
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
    total_ce_sum   = 0.0
    total_dist_sum = 0.0
    total_count_sum = 0.0
    total_mcq_sum  = 0.0
    total_lr_sum   = 0.0

    # Accuracy counters
    mcq_correct = mcq_total = 0
    lr_correct  = lr_total  = 0
    dist_correct = dist_total = 0
    count_correct = count_total = 0

    n_batches = 0
    n_dist_batches = 0
    n_count_batches = 0
    n_mcq_batches = 0
    n_lr_batches = 0

    pbar = tqdm(loader, desc="Validation", leave=False)
    for batch in pbar:
        pixel_values     = batch["pixel_values"].to(device=dev, dtype=dtype, non_blocking=True)
        image_grid_thw   = batch["image_grid_thw"].to(device=dev, non_blocking=True)
        depth_maps       = batch["depth_maps"].to(device=dev, dtype=dtype, non_blocking=True)
        input_ids        = batch["input_ids"].to(device=dev, non_blocking=True)
        labels           = batch["labels"].to(device=dev, non_blocking=True)
        attention_mask   = batch["attention_mask"].to(device=dev, non_blocking=True)

        categories = batch["categories"]
        B = len(categories)
        is_distance = torch.tensor([c == "distance" for c in categories], dtype=torch.bool, device=dev)
        is_count    = torch.tensor([c == "count" for c in categories], dtype=torch.bool, device=dev)
        is_mcq      = torch.tensor([c == "mcq" for c in categories], dtype=torch.bool, device=dev)
        is_lr       = torch.tensor([c == "left_right" for c in categories], dtype=torch.bool, device=dev)

        with torch.amp.autocast("cuda", dtype=dtype):
            try:
                output = pipeline(
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    depth_maps=depth_maps,
                    input_ids=input_ids,
                    rle_list=batch["rle_list"],
                    mask_token_positions=batch["mask_positions"],
                    decoded_masks=batch["decoded_masks"],
                    mcq_token_positions=batch.get("mcq_token_positions"),
                    lr_token_positions=batch.get("lr_token_positions"),
                    dist_token_positions=batch.get("dist_token_positions"),
                    count_token_positions=batch.get("count_token_positions"),
                    attention_mask=attention_mask,
                )
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    continue
                raise

            logits     = output["logits"]
            dist_pred  = output["dist_pred"]
            count_pred = output["count_pred"]
            mcq_logits = output.get("mcq_logits", None)
            lr_logits  = output.get("lr_logits", None)

            is_numeric = batch["is_numeric"].to(dev)
            is_categorical = batch.get("is_categorical", torch.zeros(B, dtype=torch.bool)).to(dev)
            target_cat = batch.get("target_cat_index", torch.zeros(B, dtype=torch.long)).to(dev)

            loss, components = criterion(
                logits, labels,
                dist_pred=dist_pred,
                dist_gt=batch["target_num"].to(dev),
                is_distance=is_distance,
                count_pred=count_pred,
                count_gt=batch["target_num"].to(dev),
                is_count=is_count,
                mcq_logits=mcq_logits,
                mcq_targets=target_cat,
                is_mcq=is_mcq,
                lr_logits=lr_logits,
                lr_targets=target_cat,
                is_lr=is_lr,
                return_components=True,
            )

        total_loss_sum += loss.item()
        total_ce_sum   += components['ce']
        n_batches += 1

        # Distance accuracy (within 10% relative error)
        if is_distance.any():
            total_dist_sum += components['dist']
            n_dist_batches += 1
            for b in range(B):
                if is_distance[b]:
                    pred_val = dist_pred[b].item()
                    gt_val = batch["target_num"][b].item()
                    if abs(gt_val) > 1e-6:
                        rel_err = abs(pred_val - gt_val) / abs(gt_val)
                        if rel_err <= 0.10:
                            dist_correct += 1
                    else:
                        if abs(pred_val - gt_val) < 0.5:
                            dist_correct += 1
                    dist_total += 1

        # Count accuracy (within 10% relative error)
        if is_count.any():
            total_count_sum += components['count']
            n_count_batches += 1
            for b in range(B):
                if is_count[b]:
                    pred_val = round(count_pred[b].item())
                    gt_val = batch["target_num"][b].item()
                    if abs(gt_val) > 1e-6:
                        rel_err = abs(pred_val - gt_val) / abs(gt_val)
                        if rel_err <= 0.10:
                            count_correct += 1
                    else:
                        if abs(pred_val - gt_val) < 0.5:
                            count_correct += 1
                    count_total += 1

        # MCQ accuracy
        if is_mcq.any() and mcq_logits:
            loss_mcq = components.get('mcq', 0.0)
            if loss_mcq > 0:
                total_mcq_sum += loss_mcq
                n_mcq_batches += 1
            for b in range(len(mcq_logits)):
                if is_mcq[b] and mcq_logits[b] is not None:
                    pred_idx = torch.argmax(mcq_logits[b]).item()
                    if pred_idx == target_cat[b].item():
                        mcq_correct += 1
                    mcq_total += 1

        # LeftRight accuracy
        if is_lr.any() and lr_logits:
            loss_lr = components.get('lr', 0.0)
            if loss_lr > 0:
                total_lr_sum += loss_lr
                n_lr_batches += 1
            for b in range(len(lr_logits)):
                if is_lr[b] and lr_logits[b] is not None:
                    pred_idx = torch.argmax(lr_logits[b]).item()
                    if pred_idx == target_cat[b].item():
                        lr_correct += 1
                    lr_total += 1

        # Progress bar
        avg_mcq_acc = (mcq_correct / max(mcq_total, 1)) * 100
        avg_lr_acc  = (lr_correct / max(lr_total, 1)) * 100
        avg_dist_acc = (dist_correct / max(dist_total, 1)) * 100
        avg_count_acc = (count_correct / max(count_total, 1)) * 100
        pbar.set_postfix({
            "loss": f"{total_loss_sum / max(n_batches, 1):.4f}",
            "mcq": f"{avg_mcq_acc:.0f}%",
            "lr": f"{avg_lr_acc:.0f}%",
            "dist": f"{avg_dist_acc:.0f}%",
            "cnt": f"{avg_count_acc:.0f}%",
        })

    avg_loss   = total_loss_sum / max(n_batches, 1)
    avg_ce     = total_ce_sum / max(n_batches, 1)
    avg_dist   = total_dist_sum / max(n_dist_batches, 1)
    avg_count  = total_count_sum / max(n_count_batches, 1)
    avg_mcq    = total_mcq_sum / max(n_mcq_batches, 1)
    avg_lr     = total_lr_sum / max(n_lr_batches, 1)
    avg_mcq_acc   = (mcq_correct / max(mcq_total, 1)) * 100
    avg_lr_acc    = (lr_correct / max(lr_total, 1)) * 100
    avg_dist_acc  = (dist_correct / max(dist_total, 1)) * 100
    avg_count_acc = (count_correct / max(count_total, 1)) * 100

    return {
        "val_loss": avg_loss,
        "val_ce": avg_ce,
        "val_dist": avg_dist,
        "val_count": avg_count,
        "val_mcq": avg_mcq,
        "val_lr": avg_lr,
        "val_mcq_acc": avg_mcq_acc,
        "val_lr_acc": avg_lr_acc,
        "val_dist_acc": avg_dist_acc,
        "val_count_acc": avg_count_acc,
        "n_samples": len(dataset),
        "n_batches": n_batches,
    }

# =========================================================================
# Standalone
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="SpatialVLM Super Validation")
    parser.add_argument("--device",      default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype",       default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--attn-impl",   default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])
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
    print("VALIDATION (Super)")
    print("=" * 70)

    pipeline = SpatialVLM(dtype=dtype, device_map=args.device,
                          attn_implementation=args.attn_impl)
    print_vram_usage("after model load")

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

    criterion = SpatialLoss()

    results = validate(
        pipeline, criterion, pipeline.processor,
        resolution=args.resolution,
        batch_size=args.batch_size,
        split=args.split,
        max_samples=args.max_samples,
    )

    print(f"\n{'='*70}")
    print(f"  val_loss      = {results['val_loss']:.6f}")
    print(f"  val_ce        = {results['val_ce']:.6f}")
    print(f"  val_dist      = {results['val_dist']:.6f}")
    print(f"  val_count     = {results['val_count']:.6f}")
    print(f"  val_mcq       = {results['val_mcq']:.6f}  (acc={results['val_mcq_acc']:.1f}%)")
    print(f"  val_lr        = {results['val_lr']:.6f}  (acc={results['val_lr_acc']:.1f}%)")
    print(f"  val_dist_acc  = {results['val_dist_acc']:.1f}%")
    print(f"  val_count_acc = {results['val_count_acc']:.1f}%")
    print(f"  samples       = {results['n_samples']}")
    print(f"{'='*70}")
    print_vram_usage("final")


if __name__ == "__main__":
    main()
