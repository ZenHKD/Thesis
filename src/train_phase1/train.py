"""
SpatialVLM Phase 1 Training -- GSA + RTI Warmup
=================================================

Phase 1: Qwen 3.5 frozen, only GSA (~16.9M) + RTI (~0.032M) trainable.
Loss:    L_lm (autoregressive CrossEntropy on structured target)
Goal:    Teach custom modules to produce geometry-aware representations.

Usage:
    python src/train_phase1/train.py
    python src/train_phase1/train.py --epochs 3 --lr 1e-4 --grad-accum 8
    python src/train_phase1/train.py --split train_sample --epochs 1  # quick test
    python src/train_phase1/train.py --resume checkpoints/phase1/step_10000
"""

import sys
import os
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


from src.dataloader.dataloader import SpatialVLMDataset, get_dataloader
from model.pipeline import SpatialVLM, print_vram_usage
from model.loss import SpatialVLMLoss

# Paths
ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints", "phase1")


def save_checkpoint(pipeline, optimizer, scheduler, step, epoch, loss, path):
    """Save GSA + RTI weights, optimizer, scheduler, and training state."""
    os.makedirs(path, exist_ok=True)
    torch.save({
        "step": step,
        "epoch": epoch,
        "loss": loss,
        "gsa_state_dict": pipeline.gsa.state_dict(),
        "rti_state_dict": pipeline.region_token_extractor.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }, os.path.join(path, "checkpoint.pt"))
    print(f"  [*] Checkpoint saved: {path}")


def load_checkpoint(pipeline, optimizer, scheduler, path):
    """Load GSA + RTI weights, optimizer, scheduler, and training state."""
    ckpt_path = os.path.join(path, "checkpoint.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    pipeline.gsa.load_state_dict(ckpt["gsa_state_dict"])
    pipeline.region_token_extractor.load_state_dict(ckpt["rti_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    print(f"  [*] Resumed from: {path} (step={ckpt['step']}, epoch={ckpt['epoch']})")
    return ckpt["step"], ckpt["epoch"]


def main():
    parser = argparse.ArgumentParser(description="SpatialVLM Phase 1 Training")
    # Model
    parser.add_argument("--device",     default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype",      default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--attn-impl",  default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])
    # Training
    parser.add_argument("--split",      default="train", choices=["train", "train_sample"])
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int,   default=1,
                        help="Micro-batch size per GPU step (must 1)")
    parser.add_argument("--grad-accum", type=int,   default=8,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--resolution",  default="1080p", choices=["1080p", "720p", "540p", "450p"],
                        help="Image resolution: 1080p (1920x1080), 720p (1280x720), 540p (960x540), 450p (800x450)")
    parser.add_argument("--no-grad-ckpt", action="store_true",
                        help="Disable gradient checkpointing (faster but uses more VRAM)")
    # Logging & Checkpointing
    parser.add_argument("--log-steps",  type=int,   default=100,
                        help="Log training metrics every N optimizer steps")
    parser.add_argument("--resume",     type=str,   default=None,
                        help="Path to checkpoint directory to resume from")
    parser.add_argument("--save-steps", type=int,   default=5000,
                        help="Save checkpoint every N optimizer steps (0 = epoch-only)")
    parser.add_argument("--num-workers", type=int,  default=4)
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    # Enable TF32 for Ampere GPUs (RTX 3060)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    csv_path = os.path.join(CHECKPOINT_DIR, "training.csv")

    # ====================================================================
    # 1. LOAD MODEL
    # ====================================================================
    print("=" * 70)
    print("PHASE 1 TRAINING: GSA + RTI Warmup")
    print("=" * 70)

    pipeline = SpatialVLM(
        dtype=dtype,
        device_map=args.device,
        attn_implementation=args.attn_impl,
    )
    print_vram_usage("after model load")

    # Freeze Qwen, keep GSA + RTI trainable
    for param in pipeline.qwen.parameters():
        param.requires_grad = False
    for param in pipeline.gsa.parameters():
        param.requires_grad = True
    for param in pipeline.region_token_extractor.parameters():
        param.requires_grad = True

    trainable_params = [p for p in pipeline.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_frozen = sum(p.numel() for p in pipeline.parameters() if not p.requires_grad)
    print(f"\n  Trainable: {n_trainable:,} ({n_trainable/1e6:.2f}M)")
    print(f"  Frozen:    {n_frozen:,} ({n_frozen/1e6:.2f}M)")

    # ====================================================================
    # 2. LOAD DATA
    # ====================================================================
    print(f"\n{'='*70}")
    print("LOADING DATA")
    print("=" * 70)

    processor = pipeline.processor  # reuse processor from pipeline (already loaded)
    target_size = {"1080p": None, "720p": (1280, 720), "540p": (960, 540), "450p": (800, 450)}[args.resolution]
    dataset = SpatialVLMDataset(args.split, processor=processor, target_size=target_size)
    loader = get_dataloader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )

    # RTI processes masks per-image -- batch_size must be 1.
    # Use --grad-accum for effective batching.
    assert args.batch_size == 1, (
        "batch_size > 1 not supported (RTI processes masks per-image). "
        "Use --grad-accum for effective batching."
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
    # 3. OPTIMIZER + SCHEDULER
    # ====================================================================
    optimizer = AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # Cosine decay with linear warmup
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - args.warmup_steps,
        eta_min=args.lr * 0.1,
    )

    criterion = SpatialVLMLoss()
    dev = pipeline.device

    # ====================================================================
    # 4. RESUME (optional)
    # ====================================================================
    start_step = 0
    start_epoch = 0
    if args.resume:
        start_step, start_epoch = load_checkpoint(
            pipeline, optimizer, scheduler, args.resume
        )

    # ====================================================================
    # 5. CSV LOG (truncate on resume to avoid duplicates)
    # ====================================================================
    if args.resume and os.path.exists(csv_path):
        # Keep only rows with step <= start_step
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
            writer.writerow(["step", "epoch", "avg_loss",
                             "learning_rate", "grad_norm", "samples_per_sec"])

    # ====================================================================
    # 6. TRAINING LOOP
    # ====================================================================
    print(f"\n{'='*70}")
    print("TRAINING")
    print("=" * 70)
    print(f"  lr={args.lr}  warmup={args.warmup_steps}")
    print(f"  max_grad_norm={args.max_grad_norm}  weight_decay={args.weight_decay}")
    print()

    pipeline.train()
    global_step = start_step
    micro_step = 0
    log_time = time.time()
    epoch_start_time = time.time()
    loss_window = deque(maxlen=100)  # windowed average (last 100 micro-steps)

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
            samples_so_far = epoch * total_samples + batch_idx
            if global_step > 0 and samples_so_far < start_step * args.grad_accum:
                continue

            # Move to device
            pixel_values   = batch["pixel_values"].to(device=dev, dtype=dtype)
            image_grid_thw = batch["image_grid_thw"].to(device=dev)
            depth_maps     = batch["depth_maps"].to(device=dev, dtype=dtype)
            input_ids      = batch["input_ids"].to(device=dev)
            labels         = batch["labels"].to(device=dev)

            # Forward (model already in bfloat16, no AMP needed)
            try:
                output = pipeline(
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    depth_maps=depth_maps,
                    input_ids=input_ids,
                    rle_list=batch["rle_list"][0],
                    mask_token_positions=batch["mask_positions"][0],
                    decoded_masks=batch["decoded_masks"][0],
                    use_gradient_checkpointing=not args.no_grad_ckpt,
                )
            except RuntimeError as e:
                if "out of memory" in str(e):
                    tqdm.write(f"  [!] OOM at batch {batch_idx}, skipping")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad(set_to_none=True)
                    continue
                raise

            logits = output["logits"]

            # Loss (autoregressive CE, scaled by grad_accum)
            loss = criterion(logits, labels) / args.grad_accum
            loss.backward()

            micro_step += 1
            epoch_samples += 1
            loss_val = loss.item() * args.grad_accum  # unscaled CE loss
            total_loss_sum += loss_val
            loss_window.append(loss_val)

            # Update tqdm on every micro-step (windowed average)
            window_avg = sum(loss_window) / len(loss_window)
            pbar.set_postfix({
                "step": global_step,
                "loss": f"{window_avg:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            })

            # Free memory
            del logits, output, loss, pixel_values, depth_maps

            # Optimizer step every grad_accum micro-steps
            if micro_step % args.grad_accum == 0:
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params, args.max_grad_norm
                ).item()

                # Optimizer step
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # LR warmup + cosine decay
                global_step += 1
                if global_step <= args.warmup_steps:
                    # Linear warmup
                    warmup_lr = args.lr * (global_step / args.warmup_steps)
                    for pg in optimizer.param_groups:
                        pg["lr"] = warmup_lr
                else:
                    scheduler.step()

                current_lr = optimizer.param_groups[0]["lr"]
                window_avg = sum(loss_window) / len(loss_window)

                # Logging
                if global_step % args.log_steps == 0:
                    elapsed = time.time() - log_time
                    samples_sec = (args.grad_accum * args.log_steps) / elapsed
                    current_epoch = (global_step * args.grad_accum) / total_samples

                    tqdm.write(
                        f"  step={global_step:>7d}  "
                        f"epoch={current_epoch:.2f}  "
                        f"loss={window_avg:.4f}  "
                        f"lr={current_lr:.2e}  "
                        f"grad_norm={grad_norm:.3f}  "
                        f"samples/s={samples_sec:.1f}"
                    )

                    # Write to CSV
                    with open(csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            global_step, f"{current_epoch:.4f}",
                            f"{window_avg:.6f}",
                            f"{current_lr:.8f}",
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

        # End of epoch
        epoch_elapsed = time.time() - epoch_start_time
        epoch_steps = epoch_samples // args.grad_accum
        avg_epoch_loss = total_loss_sum / max(epoch_samples, 1)
        print(f"\n  Epoch {epoch + 1} done: "
              f"avg_loss={avg_epoch_loss:.4f}  "
              f"steps={epoch_steps}  "
              f"time={epoch_elapsed/60:.1f}min")

        # Save end-of-epoch checkpoint
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch + 1}")
        save_checkpoint(
            pipeline, optimizer, scheduler,
            global_step, epoch + 1, avg_epoch_loss, ckpt_path,
        )
        print_vram_usage(f"epoch {epoch + 1}")

    # ====================================================================
    # 7. FINAL SUMMARY
    # ====================================================================
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Total steps:  {global_step}")
    print(f"  CSV log:      {os.path.abspath(csv_path)}")
    print(f"  Checkpoints:  {os.path.abspath(CHECKPOINT_DIR)}")
    print_vram_usage("final")


if __name__ == "__main__":
    main()
