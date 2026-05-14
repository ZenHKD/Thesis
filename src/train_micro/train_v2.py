"""
SpatialVLM Micro Training — Full Fine-tuning
==========================================================

All components trainable from epoch 1:
    - Vision Encoder (4 ViT blocks)       lr=5e-5
    - Token Embeddings + LM Head (tied)   lr=2e-5  (TRAINABLE, full vocab)
    - Text Decoder (24 layers, single pass)  lr=2e-5
    - RTI (Region Token Injector)         lr=5e-5
    - Number Head (xVal regression)       lr=5e-4

Loss:   L = CE + α·L_SmoothL1
        SmoothL1 on Number Head output for distance + count samples

Usage:
    python src/train_micro/train.py
    python src/train_micro/train.py --epochs 5 --batch-size 4
    python src/train_micro/train.py --split train_sample --epochs 2
    python src/train_micro/train.py --resume checkpoints/micro/step_20000
"""
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import os
import sys
import csv
import time
import math
import argparse
from collections import deque
import torch
from safetensors.torch import save_file, load_file
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.dataloader.dataloader import SpatialVLMDataset, get_dataloader
from model_micro.pipeline_v2 import SpatialVLM, print_vram_usage
from model_micro.loss import SpatialLoss
from src.train_micro.val import validate


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINT_BASE = os.path.join(PROJECT_DIR, "checkpoints", "micro_v2")


# =========================================================================
# Checkpoint Save / Load
# =========================================================================

def save_checkpoint(pipeline, optimizer, scheduler, step, epoch, loss, path):
    """Save full model checkpoint using safetensors."""
    os.makedirs(path, exist_ok=True)

    # Save model weights via safetensors
    model_state = {k: v for k, v in pipeline.state_dict().items()}
    
    # Safetensors vigorously rejects shared memory tensors (like tied embeddings/lm_head). 
    # We explicitly strip out duplicates based on their raw memory pointer.
    seen_ptrs = set()
    cleaned_state = {}
    for k, v in model_state.items():
        ptr = v.data_ptr()
        if ptr in seen_ptrs:
            continue
        seen_ptrs.add(ptr)
        cleaned_state[k] = v

    save_file(cleaned_state, os.path.join(path, "model.safetensors"))

    # Save training state via torch
    torch.save({
        "step": step,
        "epoch": epoch,
        "loss": loss,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }, os.path.join(path, "training_state.pt"))
    print(f"  [*] Checkpoint saved: {path}")


def load_checkpoint(pipeline, optimizer, scheduler, path):
    """Load checkpoint and restore model + optimizer + scheduler."""
    model_path = os.path.join(path, "model.safetensors")
    state_path = os.path.join(path, "training_state.pt")

    if os.path.exists(model_path) and os.path.exists(state_path):
        # Safetensors format
        model_state = load_file(model_path)
        pipeline.load_state_dict(model_state, strict=False)
        ckpt = torch.load(state_path, map_location="cpu", weights_only=True)
    else:
        # Fallback to old .pt format if resuming from earlier checkpoint
        ckpt_path = os.path.join(path, "checkpoint.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found in {path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        pipeline.load_state_dict(ckpt["model_state_dict"], strict=False)

    # Load optimizer + scheduler (skip if stage changed → different param groups)
    try:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        print(f"  [*] Optimizer + scheduler restored")
    except ValueError as e:
        if "different number of parameter groups" in str(e):
            print(f"  [!] Stage changed — optimizer/scheduler reset (fresh start for new stage)")
        else:
            raise

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
    parser.add_argument("--split",       default="train_balanced", choices=["train", "train_balanced", "train_sample"])
    parser.add_argument("--epochs",      type=int,   default=1)
    parser.add_argument("--stage",       type=int,   default=2, choices=[1, 2],
                        help="Training stage: 1=freeze decoder (train vision+RTI+Embed+LM_Head+CustomHeads), "
                             "2=full fine-tuning (all trainable)")
    parser.add_argument("--lr-vision",   type=float, default=5e-5)
    parser.add_argument("--lr-backbone", type=float, default=2e-5)
    parser.add_argument("--lr-rti",      type=float, default=1e-4)
    parser.add_argument("--lr-numhead",  type=float, default=1e-4)
    parser.add_argument("--lr-c",        type=float, default=1e-4,
                        help="Learning rate for Category Head")
    parser.add_argument("--weight-sl1",  type=float, default=1.0,
                        help="Weight for SmoothL1 loss")
    parser.add_argument("--weight-cat",  type=float, default=1.0,
                        help="Weight for Category Focal Loss")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Exponent for Focal Loss")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Label smoothing factor for CrossEntropy (0.0 for modern LLMs)")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size",  type=int,   default=1)
    parser.add_argument("--grad-accum",  type=int,   default=16)
    parser.add_argument("--max-grad-norm", type=float, default=2.0)
    parser.add_argument("--warmup-steps", type=int,  default=1000)
    parser.add_argument("--resolution",  default="320p",
                        choices=["1080p", "720p", "540p", "450p", "320p"])
    parser.add_argument("--grad-ckpt", action="store_true",
                        help="Enable gradient checkpointing (saves VRAM, slower)")
    # Validation
    parser.add_argument("--val-split",   default="val")
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument("--val-max-samples", type=int, default=None,
                        help="Limit val samples (None = full val set)")
    parser.add_argument("--val-steps", type=int, default=500,
                    help="Run validation every N steps (0 to disable)")
    # Logging & Checkpointing
    parser.add_argument("--log-steps",   type=int,   default=100)
    parser.add_argument("--save-steps",  type=int,   default=5000)
    parser.add_argument("--resume",      type=str,   default=None)
    parser.add_argument("--init-weights", type=str,  default=None,
                        help="Load model weights only (no optimizer/step). Use for stage transitions.")
    parser.add_argument("--num-workers", type=int,   default=4)
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile on the Qwen backbone")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    # Enable TF32 for Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    CHECKPOINT_DIR = os.path.join(CHECKPOINT_BASE, f"stage{args.stage}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    csv_path = os.path.join(CHECKPOINT_BASE, f"training_stage{args.stage}.csv")

    # ====================================================================
    # 1. LOAD MODEL
    # ====================================================================
    print("=" * 70)
    print("MICRO TRAINING: Full Fine-tuning")
    print("=" * 70)

    pipeline = SpatialVLM(
        dtype=dtype,
        device_map=args.device,
        attn_implementation=args.attn_impl,
    )
    print_vram_usage("after model load")

    if args.compile:
        print("  [*] Compiling Qwen backbone, custom heads, and RTI with torch.compile...")
        pipeline.qwen = torch.compile(pipeline.qwen)
        pipeline.cat_head = torch.compile(pipeline.cat_head)
        pipeline.num_head = torch.compile(pipeline.num_head)
        pipeline.region_token_extractor = torch.compile(pipeline.region_token_extractor)

    # Load weights from a previous stage (no optimizer/step restore)
    if args.init_weights:
        wt_path = os.path.join(args.init_weights, "model.safetensors")
        if os.path.exists(wt_path):
            from safetensors.torch import load_file as _lf
            pipeline.load_state_dict(_lf(wt_path), strict=False)
            print(f"  [*] Loaded weights from: {wt_path}")
        else:
            raise FileNotFoundError(f"No model.safetensors in {args.init_weights}")

    # ====================================================================
    # 2. CONFIGURE TRAINABLE PARAMETERS (5 groups)
    # ====================================================================
    print(f"\n{'='*70}")
    if args.stage == 1:
        print("PARAMETER GROUPS  [STAGE 1: Decoder FROZEN]")
    else:
        print("PARAMETER GROUPS  [STAGE 2: Full Fine-tuning]")
    print("=" * 70)

    # Stage 1: Freeze LLM decoder and final LayerNorm
    #          Vision encoder, RTI, Custom Heads, Embeddings, and LM Head are trainable
    # Stage 2: Everything is trainable
    if args.stage == 1:
        # Freeze decoder layers
        for param in pipeline.qwen.model.language_model.layers.parameters():
            param.requires_grad = False
        # Freeze final layernorm
        for param in pipeline.qwen.model.language_model.norm.parameters():
            param.requires_grad = False
        print("  [*] FROZEN: Decoder layers, LayerNorm")
        print("  [*] TRAINABLE: Vision Encoder, RTI, NumHead, CatHead, Embeddings, LM Head")

    # Build optimizer param groups (only include params with requires_grad)
    vision_params = [p for p in pipeline.qwen.model.visual.parameters() if p.requires_grad]
    embed_params = [p for p in pipeline.qwen.model.language_model.embed_tokens.parameters() if p.requires_grad]
    decoder_params = [p for p in pipeline.qwen.model.language_model.layers.parameters() if p.requires_grad] + \
                     [p for p in pipeline.qwen.model.language_model.norm.parameters() if p.requires_grad]
    rti_params = [p for p in pipeline.region_token_extractor.parameters() if p.requires_grad]
    numhead_params = [p for p in pipeline.num_head.parameters() if p.requires_grad]
    cathead_params = [p for p in pipeline.cat_head.parameters() if p.requires_grad]

    groups = [
        ("Vision Encoder",  vision_params,  args.lr_vision),
        ("Embeddings",      embed_params,   args.lr_backbone),  
        ("Decoder",         decoder_params, args.lr_backbone),
        ("RTI",               rti_params,   args.lr_rti),
        ("Number Head",     numhead_params, args.lr_numhead),
        ("Category Head",   cathead_params, args.lr_c),
    ]

    for name, params, lr in groups:
        n = sum(p.numel() for p in params)
        status = "FROZEN" if n == 0 else f"lr={lr}"
        print(f"  {name:20s}: {n:>12,} ({n/1e6:.2f}M)  {status}")

    total_trainable = sum(p.numel() for p in pipeline.parameters() if p.requires_grad)
    total_all = sum(p.numel() for p in pipeline.parameters())
    print(f"  {'─'*60}")
    print(f"  {'Total trainable':20s}: {total_trainable:>12,} ({total_trainable/1e6:.2f}M) / {total_all/1e6:.0f}M")
    n_layers = len(list(pipeline.qwen.model.language_model.layers))
    print(f"  Decoder: {n_layers} layers (single pass)")

    # ====================================================================
    # 3. LOAD DATA
    # ====================================================================
    print(f"\n{'='*70}")
    print("LOADING DATA")
    print("=" * 70)

    processor = pipeline.processor
    target_size = {"1080p": None, "720p": (1280, 720),
                   "540p": (960, 540), "450p": (800, 450),
                   "320p": (512, 320)}[args.resolution]

    dataset = SpatialVLMDataset(
        args.split, processor=processor, target_size=target_size
    )
    loader = get_dataloader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
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
    # Only include non-empty param groups in optimizer
    param_groups = []
    for name, params, lr in groups:
        if len(params) > 0:
            param_groups.append({"params": params, "lr": lr, "name": name})

    optimizer = AdamW(param_groups, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max(total_steps - args.warmup_steps, 1), eta_min=1e-6,
    )
    criterion = SpatialLoss(
        weight_sl1=args.weight_sl1,
        weight_cat=args.weight_cat,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing
    )
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
        "step", "epoch", "avg_loss", "val_loss", "val_ce", "val_sl1",
        "val_cat", "val_cat_acc", "val_num_acc",
        "lr_vision", "lr_decoder", "lr_rti", "lr_numhead", "lr_c",
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
    print(f"TRAINING  [Stage {args.stage}]")
    print("=" * 70)
    print(f"  stage={args.stage}  lr_vision={args.lr_vision}  lr_backbone={args.lr_backbone}  "
          f"lr_rti={args.lr_rti}  lr_numhead={args.lr_numhead}  lr_c={args.lr_c}")
    print(f"  warmup={args.warmup_steps}  max_grad_norm={args.max_grad_norm}  weight_sl1={args.weight_sl1}  weight_cat={args.weight_cat}")
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
            pixel_values   = batch["pixel_values"].to(device=dev, dtype=dtype, non_blocking=True)
            pixel_values_rgb = batch["pixel_values_rgb"].to(device=dev, dtype=dtype, non_blocking=True)
            image_grid_thw = batch["image_grid_thw"].to(device=dev, non_blocking=True)
            depth_maps     = batch["depth_maps"].to(device=dev, dtype=dtype, non_blocking=True)
            input_ids      = batch["input_ids"].to(device=dev, non_blocking=True)
            labels         = batch["labels"].to(device=dev, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device=dev, non_blocking=True)

            # Forward
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
            cat_logits = output.get("cat_logits", None)

            # Build is_distance mask for domain-aware MSE normalization
            is_distance = torch.tensor(
                [c == "distance" for c in batch["categories"]],
                dtype=torch.bool, device=dev,
            )

            loss = criterion(
                logits, labels,
                num_pred, batch["target_num"].to(dev),
                batch["is_numeric"].to(dev),
                num_is_distance=is_distance,
                cat_logits=cat_logits,
                cat_targets=batch.get("target_cat_index", torch.zeros(1)).to(dev),
                is_categorical=batch.get("is_categorical", torch.zeros(1, dtype=torch.bool)).to(dev),
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
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",  # first active lr
            })

            del logits, output, loss, pixel_values, pixel_values_rgb, depth_maps, num_pred, cat_logits

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

                    lrs = {pg["name"]: pg["lr"] for pg in optimizer.param_groups}
                    lr_v = lrs.get("Vision Encoder", 0.0)
                    lr_e = lrs.get("Embeddings", 0.0)
                    lr_d = lrs.get("Decoder", 0.0)
                    lr_rti = lrs.get("RTI", 0.0)
                    lr_n = lrs.get("Number Head", 0.0)
                    lr_c = lrs.get("Category Head", 0.0)

                    tqdm.write(
                        f"  step={global_step:>7d}  "
                        f"epoch={current_epoch:.2f}  "
                        f"loss={window_avg:.4f}  "
                        f"lr_e={lr_e:.2e}  "
                        f"lr_d={lr_d:.2e}  "
                        f"grad_norm={grad_norm:.3f}  "
                        f"samples/s={samples_sec:.1f}"
                    )

                    # Write to CSV (val_loss is empty at log steps, filled at epoch end)
                    with open(csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            global_step, f"{current_epoch:.4f}",
                            f"{window_avg:.6f}", "", "", "", "", "", "",  # val_loss..val_num_acc empty
                            f"{lr_v:.8f}", f"{lr_d:.8f}",
                            f"{lr_rti:.8f}", f"{lr_n:.8f}", f"{lr_c:.8f}",
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

                # Validation
                if args.val_steps > 0 and global_step % args.val_steps == 0 and global_step > 0:
                    print(f"\n  [{'='*70}]")
                    print(f"  Running validation at step {global_step}...")
                    torch.cuda.empty_cache()

                    val_results = validate(
                        pipeline, criterion, pipeline.processor,
                        resolution=args.resolution,
                        batch_size=args.val_batch_size,
                        num_workers=args.num_workers,
                        max_samples=args.val_max_samples,
                        split=args.val_split,
                    )
                    val_loss = val_results["val_loss"]
                    val_ce = val_results.get("val_ce", 0.0)
                    val_sl1 = val_results.get("val_sl1", 0.0)
                    pipeline.train()  # Switch back to train mode

                    current_epoch = (global_step * effective_batch) / total_samples
                    lrs = {pg["name"]: pg["lr"] for pg in optimizer.param_groups}
                    lr_v = lrs.get("Vision Encoder", 0.0)
                    lr_e = lrs.get("Embeddings", 0.0)
                    lr_d = lrs.get("Decoder", 0.0)
                    lr_rti = lrs.get("RTI", 0.0)
                    lr_n = lrs.get("Number Head", 0.0)
                    lr_c = lrs.get("Category Head", 0.0)

                    val_cat = val_results.get("val_cat", 0.0)
                    val_cat_acc = val_results.get("val_cat_acc", 0.0)
                    val_num_acc = val_results.get("val_num_acc", 0.0)

                    # Log to CSV with val_loss filled
                    with open(csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            global_step, f"{current_epoch:.4f}",
                            f"{window_avg:.6f}",
                            f"{val_loss:.6f}", f"{val_ce:.6f}", f"{val_sl1:.6f}",
                            f"{val_cat:.6f}", f"{val_cat_acc:.2f}", f"{val_num_acc:.2f}",
                            f"{lr_v:.8f}", f"{lr_d:.8f}",
                            f"{lr_rti:.8f}", f"{lr_n:.8f}", f"{lr_c:.8f}",
                            "", "",
                        ])
                    print(f"\nValidation complete: val_loss={val_loss:.4f} "
                          f"(CE={val_ce:.4f}, SL1={val_sl1:.4f}, Cat={val_cat:.4f}, "
                          f"CatAcc={val_cat_acc:.1f}%, NumAcc={val_num_acc:.1f}%)")

        # ==============================================================
        # END OF EPOCH 
        # ==============================================================
        epoch_elapsed = time.time() - epoch_start_time
        print(f"\nEpoch {epoch + 1} completed in {epoch_elapsed/60:.1f}min")

        # Save end-of-epoch checkpoint
        epoch_ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch + 1}")
        save_checkpoint(
            pipeline, optimizer, scheduler,
            global_step, epoch + 1, window_avg, epoch_ckpt_path,
        )

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
