"""
SpatialVLM Micro Training — Full Fine-tuning from Scratch
==========================================================

All components trainable from epoch 1:
    - Vision Encoder (4 ViT blocks)       lr=5e-5
    - Token Embeddings + LM Head (tied)   lr=2e-5
    - Text Decoder (8 layers)             lr=2e-5
    - GSA (DFormerv2 Full_GSA x2)         lr=5e-5
    - RTI (Region Token Injector)         lr=5e-5
    - Number Head (xVal regression)       lr=5e-4

Loss:   L = L_CE + α · L_MSE
        CE on structured text targets (category | answer)
        MSE on Number Head output for distance + count samples

Usage:
    python src/train_micro/train.py
    python src/train_micro/train.py --epochs 5 --batch-size 4
    python src/train_micro/train.py --split train_sample --epochs 2
    python src/train_micro/train.py --resume checkpoints/micro/step_20000
"""

import os
import sys
import csv
import time
import math
import argparse
from collections import deque
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.dataloader.dataloader_new import SpatialVLMDataset, get_dataloader
from model_micro.pipeline import SpatialVLM, print_vram_usage
from model_micro.loss import SpatialLoss
from src.train_micro.val import validate


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints", "micro")


# =========================================================================
# Checkpoint Save / Load
# =========================================================================

def save_checkpoint(pipeline, optimizer, scheduler, step, epoch, loss, path):
    """Save full model checkpoint."""
    os.makedirs(path, exist_ok=True)

    torch.save({
        "step": step,
        "epoch": epoch,
        "loss": loss,
        # Full model state (all trainable -- no LoRA)
        "model_state_dict": {
            k: v for k, v in pipeline.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }, os.path.join(path, "checkpoint.pt"))
    print(f"  [*] Checkpoint saved: {path}")


def load_checkpoint(pipeline, optimizer, scheduler, path):
    """Load checkpoint and restore model + optimizer + scheduler."""
    ckpt_path = os.path.join(path, "checkpoint.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # Load model weights
    pipeline.load_state_dict(ckpt["model_state_dict"], strict=False)

    # Load optimizer + scheduler
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    print(f"  [*] Resumed from: {path} (step={ckpt['step']}, epoch={ckpt['epoch']})")
    return ckpt["step"], ckpt["epoch"]


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="SpatialVLM Micro Training")
    # Model
    parser.add_argument("--device",      default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype",       default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--attn-impl",   default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])
    # Training
    parser.add_argument("--split",       default="train", choices=["train", "train_sample"])
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--lr-vision",   type=float, default=5e-5)
    parser.add_argument("--lr-backbone", type=float, default=2e-5)
    parser.add_argument("--lr-embed",    type=float, default=2e-5)
    parser.add_argument("--lr-custom",   type=float, default=5e-5)
    parser.add_argument("--lr-numhead",  type=float, default=5e-4)
    parser.add_argument("--alpha",       type=float, default=1.0,
                        help="Weight for MSE loss (α in L = L_CE + α·L_MSE)")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size",  type=int,   default=8)
    parser.add_argument("--grad-accum",  type=int,   default=2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int,  default=500)
    parser.add_argument("--resolution",  default="450p",
                        choices=["1080p", "720p", "540p", "450p"])
    parser.add_argument("--grad-ckpt", action="store_true",
                        help="Enable gradient checkpointing (saves VRAM, slower)")
    # Validation
    parser.add_argument("--val-split",   default="val")
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument("--val-max-samples", type=int, default=None,
                        help="Limit val samples (None = full val set)")
    # Logging & Checkpointing
    parser.add_argument("--log-steps",   type=int,   default=100)
    parser.add_argument("--save-steps",  type=int,   default=20000)
    parser.add_argument("--resume",      type=str,   default=None)
    parser.add_argument("--num-workers", type=int,   default=4)
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    # Enable TF32 for Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    csv_path = os.path.join(CHECKPOINT_DIR, "training.csv")

    # ====================================================================
    # 1. LOAD MODEL
    # ====================================================================
    print("=" * 70)
    print("MICRO TRAINING: Full Fine-tuning from Scratch")
    print("=" * 70)

    pipeline = SpatialVLM(
        dtype=dtype,
        device_map=args.device,
        attn_implementation=args.attn_impl,
    )
    print_vram_usage("after model load")

    # ====================================================================
    # 2. CONFIGURE TRAINABLE PARAMETERS (5 groups)
    # ====================================================================
    print(f"\n{'='*70}")
    print("PARAMETER GROUPS (all trainable)")
    print("=" * 70)

    # All parameters trainable
    for param in pipeline.parameters():
        param.requires_grad = True

    # Build 5 optimizer param groups
    vision_params = list(pipeline.qwen.model.visual.parameters())
    embed_params  = list(pipeline.qwen.model.language_model.embed_tokens.parameters())
    decoder_params = list(pipeline.qwen.model.language_model.layers.parameters()) + \
                     list(pipeline.qwen.model.language_model.norm.parameters())
    gsa_rti_params = list(pipeline.gsa.parameters()) + \
                     list(pipeline.region_token_extractor.parameters())
    numhead_params = list(pipeline.num_head.parameters())

    # Deduplicate: LM head is tied to embed_tokens, already counted
    # Don't add lm_head params separately

    groups = [
        ("Vision Encoder",  vision_params,  args.lr_vision),
        ("Embeddings",      embed_params,   args.lr_embed),
        ("Decoder",         decoder_params, args.lr_backbone),
        ("GSA + RTI",       gsa_rti_params, args.lr_custom),
        ("Number Head",     numhead_params, args.lr_numhead),
    ]

    for name, params, lr in groups:
        n = sum(p.numel() for p in params)
        print(f"  {name:20s}: {n:>12,} ({n/1e6:.2f}M)  lr={lr}")

    total_trainable = sum(p.numel() for p in pipeline.parameters() if p.requires_grad)
    print(f"  {'─'*60}")
    print(f"  {'Total trainable':20s}: {total_trainable:>12,} ({total_trainable/1e6:.2f}M)")

    # ====================================================================
    # 3. LOAD DATA
    # ====================================================================
    print(f"\n{'='*70}")
    print("LOADING DATA")
    print("=" * 70)

    processor = pipeline.processor
    target_size = {"1080p": None, "720p": (1280, 720),
                   "540p": (960, 540), "450p": (800, 450)}[args.resolution]

    dataset = SpatialVLMDataset(
        args.split, processor=processor, target_size=target_size,
    )
    loader = get_dataloader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False,
    )

    total_samples = len(dataset)
    batches_per_epoch = math.ceil(total_samples / args.batch_size)
    steps_per_epoch = math.ceil(batches_per_epoch / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    effective_batch = args.batch_size * args.grad_accum

    print(f"  Split:             {args.split}")
    print(f"  Resolution:        {args.resolution}")
    print(f"  Samples:           {total_samples:,}")
    print(f"  Batch size:        {args.batch_size}")
    print(f"  Grad accumulation: {args.grad_accum}")
    print(f"  Effective batch:   {effective_batch}")
    print(f"  Steps/epoch:       {steps_per_epoch:,}")
    print(f"  Total steps:       {total_steps:,}")
    print(f"  Epochs:            {args.epochs}")

    # ====================================================================
    # 4. OPTIMIZER + SCHEDULER
    # ====================================================================
    param_groups = [
        {"params": vision_params,  "lr": args.lr_vision,   "name": "vision"},
        {"params": embed_params,   "lr": args.lr_embed,    "name": "embed"},
        {"params": decoder_params, "lr": args.lr_backbone,  "name": "decoder"},
        {"params": gsa_rti_params, "lr": args.lr_custom,    "name": "gsa_rti"},
        {"params": numhead_params, "lr": args.lr_numhead,   "name": "numhead"},
    ]

    optimizer = AdamW(param_groups, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max(total_steps - args.warmup_steps, 1), eta_min=1e-6,
    )
    criterion = SpatialLoss(alpha=args.alpha, remap_fn=pipeline.remap_to_new)
    dev = pipeline.device

    # ====================================================================
    # 5. RESUME (optional)
    # ====================================================================
    start_step = 0
    start_epoch = 0
    if args.resume:
        print(f"\n{'='*70}")
        print("RESUMING FROM CHECKPOINT")
        print("=" * 70)
        start_step, start_epoch = load_checkpoint(
            pipeline, optimizer, scheduler, args.resume,
        )

    # ====================================================================
    # 6. CSV LOG (with val_loss column)
    # ====================================================================
    csv_fields = [
        "step", "epoch", "avg_loss", "val_loss",
        "lr_vision", "lr_embed", "lr_decoder", "lr_custom", "lr_numhead",
        "grad_norm", "samples_per_sec",
    ]

    if args.resume and os.path.exists(csv_path):
        # Truncate CSV to resume point
        with open(csv_path, "r") as f:
            lines = f.readlines()
        header = lines[0]
        kept = [header]
        for line in lines[1:]:
            parts = line.strip().split(",")
            if parts and parts[0].isdigit() and int(parts[0]) <= start_step:
                kept.append(line)
        with open(csv_path, "w") as f:
            f.writelines(kept)
        print(f"  [*] CSV truncated to step {start_step} ({len(kept)-1} data rows)")
    elif not os.path.exists(csv_path) or args.resume is None:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_fields)

    # ====================================================================
    # 7. TRAINING LOOP
    # ====================================================================
    print(f"\n{'='*70}")
    print("TRAINING")
    print("=" * 70)
    print(f"  lr_vision={args.lr_vision}  lr_backbone={args.lr_backbone}  "
          f"lr_embed={args.lr_embed}  lr_custom={args.lr_custom}  lr_numhead={args.lr_numhead}")
    print(f"  warmup={args.warmup_steps}  max_grad_norm={args.max_grad_norm}  alpha={args.alpha}")
    print()

    pipeline.train()
    global_step = start_step
    micro_step = 0
    log_time = time.time()
    loss_window = deque(maxlen=100)

    all_trainable = [p for p in pipeline.parameters() if p.requires_grad]
    initial_lrs = [pg["lr"] for pg in optimizer.param_groups]

    for epoch in range(start_epoch, args.epochs):
        print(f"{'='*70}")
        print(f"  Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*70}")
        epoch_start_time = time.time()
        epoch_samples = 0
        total_loss_sum = 0.0

        pbar = tqdm(
            enumerate(loader),
            total=len(loader),
            desc=f"Epoch {epoch+1}/{args.epochs}",
            bar_format="{l_bar}{bar:30}{r_bar}",
            dynamic_ncols=True,
        )
        for batch_idx, batch in pbar:
            # Skip already-processed steps on resume
            samples_so_far = epoch * total_samples + batch_idx * args.batch_size
            if global_step > 0 and samples_so_far < start_step * effective_batch:
                continue

            # Move to device
            pixel_values   = batch["pixel_values"].to(device=dev, dtype=dtype)
            image_grid_thw = batch["image_grid_thw"].to(device=dev)
            depth_maps     = batch["depth_maps"].to(device=dev, dtype=dtype)
            input_ids      = batch["input_ids"].to(device=dev)
            labels         = batch["labels"].to(device=dev)

            # Forward
            try:
                output = pipeline(
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    depth_maps=depth_maps,
                    input_ids=input_ids,
                    rle_list=batch["rle_list"],
                    mask_token_positions=batch["mask_positions"],
                    decoded_masks=batch["decoded_masks"],
                    num_token_positions=batch.get("num_token_positions"),
                    use_gradient_checkpointing=args.grad_ckpt,
                    vision_requires_grad=True,
                )
            except RuntimeError as e:
                if "out of memory" in str(e):
                    tqdm.write(f"  [!] OOM at batch {batch_idx}, skipping")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad(set_to_none=True)
                    continue
                raise

            logits = output["logits"]
            num_pred = output["num_pred"]

            loss = criterion(
                logits, labels,
                num_pred, batch["target_num"].to(dev),
                batch["is_numeric"].to(dev),
            ) / args.grad_accum
            loss.backward()

            micro_step += 1
            epoch_samples += args.batch_size
            loss_val = loss.item() * args.grad_accum
            total_loss_sum += loss_val
            loss_window.append(loss_val)

            window_avg = sum(loss_window) / len(loss_window)
            pbar.set_postfix({
                "step": global_step,
                "loss": f"{window_avg:.4f}",
                "lr": f"{optimizer.param_groups[2]['lr']:.2e}",
            })

            del logits, output, loss, pixel_values, depth_maps, num_pred

            # Optimizer step every grad_accum micro-steps
            if micro_step % args.grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    all_trainable, args.max_grad_norm,
                ).item()

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

                # Warmup
                if global_step <= args.warmup_steps:
                    warmup_frac = global_step / args.warmup_steps
                    for pg, init_lr in zip(optimizer.param_groups, initial_lrs):
                        pg["lr"] = init_lr * warmup_frac
                else:
                    scheduler.step()

                window_avg = sum(loss_window) / len(loss_window)

                # Logging
                if global_step % args.log_steps == 0:
                    elapsed = time.time() - log_time
                    samples_sec = (args.grad_accum * args.batch_size * args.log_steps) / elapsed
                    current_epoch = (global_step * effective_batch) / total_samples

                    lr_v = optimizer.param_groups[0]["lr"]
                    lr_e = optimizer.param_groups[1]["lr"]
                    lr_d = optimizer.param_groups[2]["lr"]
                    lr_c = optimizer.param_groups[3]["lr"]
                    lr_n = optimizer.param_groups[4]["lr"]

                    tqdm.write(
                        f"  step={global_step:>7d}  "
                        f"epoch={current_epoch:.2f}  "
                        f"loss={window_avg:.4f}  "
                        f"lr_d={lr_d:.2e}  "
                        f"grad_norm={grad_norm:.3f}  "
                        f"samples/s={samples_sec:.1f}"
                    )

                    # Write to CSV (val_loss is empty at log steps, filled at epoch end)
                    with open(csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            global_step, f"{current_epoch:.4f}",
                            f"{window_avg:.6f}", "",  # val_loss empty
                            f"{lr_v:.8f}", f"{lr_e:.8f}", f"{lr_d:.8f}",
                            f"{lr_c:.8f}", f"{lr_n:.8f}",
                            f"{grad_norm:.6f}", f"{samples_sec:.2f}",
                        ])

                    log_time = time.time()

                # Mid-epoch checkpoint
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    ckpt_path = os.path.join(CHECKPOINT_DIR, f"step_{global_step}")
                    save_checkpoint(
                        pipeline, optimizer, scheduler,
                        global_step, epoch, window_avg, ckpt_path,
                    )

        # ==============================================================
        # END OF EPOCH — Validation + Checkpoint
        # ==============================================================
        epoch_elapsed = time.time() - epoch_start_time
        epoch_steps = epoch_samples // effective_batch
        avg_epoch_loss = total_loss_sum / max(epoch_samples // args.batch_size, 1)

        # Run validation
        print(f"\n  Running validation ({args.val_split})...")
        val_results = validate(
            pipeline, criterion, pipeline.processor,
            resolution=args.resolution,
            batch_size=args.val_batch_size,
            num_workers=args.num_workers,
            max_samples=args.val_max_samples,
            split=args.val_split,
        )
        val_loss = val_results["val_loss"]
        pipeline.train()  # Switch back to train mode

        print(f"\n  Epoch {epoch + 1} done: "
              f"train_loss={avg_epoch_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"steps={epoch_steps}  "
              f"time={epoch_elapsed/60:.1f}min")

        # Write epoch summary row to CSV (with val_loss filled)
        current_epoch_val = (global_step * effective_batch) / total_samples
        lr_v = optimizer.param_groups[0]["lr"]
        lr_e = optimizer.param_groups[1]["lr"]
        lr_d = optimizer.param_groups[2]["lr"]
        lr_c = optimizer.param_groups[3]["lr"]
        lr_n = optimizer.param_groups[4]["lr"]

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                global_step, f"{current_epoch_val:.4f}",
                f"{avg_epoch_loss:.6f}", f"{val_loss:.6f}",
                f"{lr_v:.8f}", f"{lr_e:.8f}", f"{lr_d:.8f}",
                f"{lr_c:.8f}", f"{lr_n:.8f}",
                "", "",  # grad_norm and samples/s not applicable
            ])

        # Save epoch checkpoint
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch + 1}")
        save_checkpoint(
            pipeline, optimizer, scheduler,
            global_step, epoch + 1, avg_epoch_loss, ckpt_path,
        )
        print_vram_usage(f"epoch {epoch + 1}")

    # ====================================================================
    # 8. FINAL SUMMARY
    # ====================================================================
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Total steps:  {global_step}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  CSV log:      {os.path.abspath(csv_path)}")
    print(f"  Checkpoints:  {os.path.abspath(CHECKPOINT_DIR)}")
    print_vram_usage("final")


if __name__ == "__main__":
    main()
