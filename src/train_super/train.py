"""
SpatialVLM Super Training — 4-Head Architecture (DPT Pipeline)
==========================================================

2-Stage Training:
    Stage 1 (--stage 1): Freeze Qwen Vision Encoder
    Stage 2 (--stage 2): Unfreeze Qwen Vision Encoder

Both stages: Embeddings FROZEN (only <|mcq|>, <|lr|>, <|dist|>, <|count|> trainable)

Components:
    - Vision Encoder (12 ViT blocks, DPT multi-layer features)
    - Text Decoder (24 layers, single pass)
    - DPT-based RTI (Region Token Extractor)
    - SharedVisualFuser (Dual-Stream)
    - 4 Heads: MCQ, LeftRight, Distance, Count

Loss:   L = CE + w_dist·L_Dist + w_count·L_Count + w_mcq·L_MCQ + w_lr·L_LR

Usage:
    python src/train_super/train.py --stage 1
    python src/train_super/train.py --stage 2 --resume checkpoints/super/stage1/epoch_1
    python src/train_super/train.py --stage 1 --epochs 5 --batch-size 4
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

# ---- CUDA Performance Flags (free speed-ups) ----
torch.backends.cuda.matmul.allow_tf32 = True   # TF32 for matmul ops (Ampere+)
torch.backends.cudnn.allow_tf32 = True         # TF32 for cuDNN ops (Ampere+)
torch.backends.cudnn.benchmark = True          # Auto-tune convolution kernels (RTI U-Net)
torch.set_float32_matmul_precision("medium")   # Allow TF32 matmul in torch.compile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from super_model.dataloader import SpatialVLMDataset, get_dataloader
from super_model.pipeline import SpatialVLM, print_vram_usage
from super_model.loss import SpatialLoss
from src.train_super.val import validate


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINT_BASE = os.path.join(PROJECT_DIR, "checkpoints", "super")


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

    # Load optimizer + scheduler
    try:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        print(f"  [*] Optimizer + scheduler restored")
    except ValueError as e:
        print(f"  [!] Optimizer/scheduler mismatch — reset (fresh start)")
        import traceback; traceback.print_exc()

    print(f"  [*] Resumed from: {path} (step={ckpt['step']}, epoch={ckpt['epoch']})")
    return ckpt["step"], ckpt["epoch"]


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="SpatialVLM Super Training")
    # Model
    parser.add_argument("--device",      default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype",       default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--attn-impl",   default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])
    # Training
    parser.add_argument("--stage",       type=int, default=1, choices=[1, 2],
                        help="Training stage: 1=freeze vision, 2=unfreeze vision")
    parser.add_argument("--split",       default="train_balanced", choices=["train", "train_balanced", "train_sample"])
    parser.add_argument("--epochs",      type=int,   default=1)
    parser.add_argument("--lr-vision",   type=float, default=5e-5)
    parser.add_argument("--lr-backbone", type=float, default=5e-5)
    parser.add_argument("--lr-rti",      type=float, default=1e-4)
    parser.add_argument("--lr-dist",     type=float, default=1e-4)
    parser.add_argument("--lr-count",    type=float, default=1e-4)
    parser.add_argument("--weight-dist",  type=float, default=100.0)
    parser.add_argument("--weight-count", type=float, default=2.0)
    parser.add_argument("--weight-mcq",   type=float, default=2.0)
    parser.add_argument("--weight-lr",    type=float, default=2.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size",  type=int,   default=2)
    parser.add_argument("--grad-accum",  type=int,   default=16)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
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
                        help="Load model weights only (no optimizer/step restore).")
    parser.add_argument("--num-workers", type=int,   default=2)
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile on the Qwen backbone")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    CHECKPOINT_DIR = os.path.join(CHECKPOINT_BASE, f"stage{args.stage}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    csv_path = os.path.join(CHECKPOINT_BASE, "training.csv")

    # ====================================================================
    # 1. LOAD MODEL
    # ====================================================================
    print("=" * 70)
    print(f"SUPER TRAINING: Stage {args.stage} — {'Freeze' if args.stage == 1 else 'Unfreeze'} Vision")
    print("=" * 70)

    pipeline = SpatialVLM(
        dtype=dtype,
        device_map=args.device,
        attn_implementation=args.attn_impl,
    )
    print_vram_usage("after model load")

    if args.compile:
        print("  [*] Compiling Qwen + RTI + Heads with torch.compile...")
        pipeline.qwen = torch.compile(pipeline.qwen)
        pipeline.region_token_extractor = torch.compile(pipeline.region_token_extractor)
        pipeline.mcq_head = torch.compile(pipeline.mcq_head)
        pipeline.lr_head = torch.compile(pipeline.lr_head)
        pipeline.dist_head = torch.compile(pipeline.dist_head)
        pipeline.count_head = torch.compile(pipeline.count_head)

    # Load weights only (no optimizer/step restore)
    if args.init_weights:
        wt_path = os.path.join(args.init_weights, "model.safetensors")
        if os.path.exists(wt_path):
            from safetensors.torch import load_file as _lf
            pipeline.load_state_dict(_lf(wt_path), strict=False)
            print(f"  [*] Loaded weights from: {wt_path}")
        else:
            raise FileNotFoundError(f"No model.safetensors in {args.init_weights}")

    # ====================================================================
    # 2. CONFIGURE TRAINABLE PARAMETERS (8 groups)
    # ====================================================================
    print(f"\n{'='*70}")
    print(f"PARAMETER GROUPS  [Stage {args.stage}: {'Vision FROZEN' if args.stage == 1 else 'Vision TRAINABLE'}]")
    print("=" * 70)

    # --- Freeze text embeddings (only 4 special tokens trainable) ---
    embed_layer = pipeline.qwen.model.language_model.embed_tokens
    embed_layer.weight.requires_grad = False  # Freeze full embedding table

    # Create a trainable parameter for only the 4 special token rows
    special_ids = [pipeline.mcq_token_id, pipeline.lr_token_id,
                   pipeline.dist_token_id, pipeline.count_token_id]
    special_embed = embed_layer.weight.data[special_ids].clone().detach().requires_grad_(True)
    pipeline._special_embed = special_embed
    pipeline._special_ids = special_ids
    n_special = special_embed.numel()
    print(f"  [*] Embeddings FROZEN (full vocab)")
    print(f"  [*] Only 4 special tokens trainable: {special_ids} ({n_special:,} params)")
    print(f"  [*] Manual embed injection (no hook — compile-friendly)")

    # Stage-based Vision Encoder freezing
    if args.stage == 1:
        # Stage 1: Freeze vision encoder
        for p in pipeline.qwen.model.visual.parameters():
            p.requires_grad = False
        print(f"  [*] Vision Encoder FROZEN (Stage 1)")
    else:
        # Stage 2: Unfreeze vision encoder
        for p in pipeline.qwen.model.visual.parameters():
            p.requires_grad = True
        print(f"  [*] Vision Encoder TRAINABLE (Stage 2)")

    # Build optimizer param groups (only include params with requires_grad)
    vision_params = [p for p in pipeline.qwen.model.visual.parameters() if p.requires_grad]
    decoder_params = [p for p in pipeline.qwen.model.language_model.layers.parameters() if p.requires_grad] + \
                     [p for p in pipeline.qwen.model.language_model.norm.parameters() if p.requires_grad]
    rti_params = [p for p in pipeline.region_token_extractor.parameters() if p.requires_grad]
    dist_params = [p for p in pipeline.dist_head.parameters() if p.requires_grad]
    count_params = [p for p in pipeline.count_head.parameters() if p.requires_grad]

    groups = [
        ("Vision Encoder",  vision_params,        args.lr_vision),
        ("Special Embed",   [special_embed],       args.lr_backbone),
        ("Decoder",         decoder_params,        args.lr_backbone),
        ("RTI Modules",     rti_params,            args.lr_rti),
        ("Distance Head",   dist_params,           args.lr_rti),
        ("Count Head",      count_params,          args.lr_rti),
    ]

    for name, params, lr in groups:
        n = sum(p.numel() for p in params)
        status = "FROZEN" if n == 0 else f"lr={lr}"
        print(f"  {name:20s}: {n:>12,} ({n/1e6:.2f}M)  {status}")

    total_trainable = sum(p.numel() for p in pipeline.parameters() if p.requires_grad) + n_special
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

    optimizer = AdamW(param_groups, weight_decay=args.weight_decay, betas=(0.9, 0.999), fused=True)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max(total_steps - args.warmup_steps, 1), eta_min=1e-6,
    )
    criterion = SpatialLoss(
        weight_dist=args.weight_dist,
        weight_count=args.weight_count,
        weight_mcq=args.weight_mcq,
        weight_lr=args.weight_lr,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
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
        "step", "epoch", "avg_loss",
        "total_val_loss", "val_loss_ce",
        "val_loss_mcq", "val_acc_mcq",
        "val_loss_lr", "val_acc_lr",
        "val_loss_dist", "val_acc_dist",
        "val_loss_count", "val_acc_count",
        "lr_vision", "lr_embed", "lr_decoder", "lr_rti", "lr_fuser",
        "lr_dist", "lr_count",
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
    print(f"  lr_vision={args.lr_vision}  lr_backbone={args.lr_backbone}  lr_rti={args.lr_rti}")
    print(f"  lr_dist={args.lr_dist}  lr_count={args.lr_count}  (MCQ/LR use SharedVisualFuser @ lr_rti)")
    print(f"  warmup={args.warmup_steps}  max_grad_norm={args.max_grad_norm}")
    print(f"  w_dist={args.weight_dist}  w_count={args.weight_count}  w_mcq={args.weight_mcq}  w_lr={args.weight_lr}")
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
            image_grid_thw = batch["image_grid_thw"].to(device=dev, non_blocking=True)
            depth_maps     = batch["depth_maps"].to(device=dev, dtype=dtype, non_blocking=True)
            input_ids      = batch["input_ids"].to(device=dev, non_blocking=True)
            labels         = batch["labels"].to(device=dev, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device=dev, non_blocking=True)

            # Inject special token embeddings before forward (no hook → compile-friendly)
            with torch.no_grad():
                embed_layer.weight.data[special_ids] = pipeline._special_embed.data

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
                    mcq_token_positions=batch.get("mcq_token_positions"),
                    lr_token_positions=batch.get("lr_token_positions"),
                    dist_token_positions=batch.get("dist_token_positions"),
                    count_token_positions=batch.get("count_token_positions"),
                    attention_mask=attention_mask,
                    use_gradient_checkpointing=args.grad_ckpt,
                    vision_requires_grad=(args.stage == 2),
                )
            except RuntimeError as e:
                if "out of memory" in str(e):
                    tqdm.write(f"  [!] OOM at batch {batch_idx}, skipping")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad(set_to_none=True)
                    continue
                raise

            logits     = output["logits"]
            dist_pred  = output["dist_pred"]
            count_pred = output["count_pred"]
            mcq_logits = output.get("mcq_logits", None)
            lr_logits  = output.get("lr_logits", None)

            categories = batch["categories"]
            B = len(categories)
            is_distance = torch.tensor([c == "distance" for c in categories], dtype=torch.bool, device=dev)
            is_count    = torch.tensor([c == "count" for c in categories], dtype=torch.bool, device=dev)
            is_mcq      = torch.tensor([c == "mcq" for c in categories], dtype=torch.bool, device=dev)
            is_lr       = torch.tensor([c == "left_right" for c in categories], dtype=torch.bool, device=dev)

            loss = criterion(
                logits, labels,
                dist_pred=dist_pred,
                dist_gt=batch["target_num"].to(dev),
                is_distance=is_distance,
                count_pred=count_pred,
                count_gt=batch["target_num"].to(dev),
                is_count=is_count,
                mcq_logits=mcq_logits,
                mcq_targets=batch.get("target_cat_index", torch.zeros(B, dtype=torch.long)).to(dev),
                is_mcq=is_mcq,
                lr_logits=lr_logits,
                lr_targets=batch.get("target_cat_index", torch.zeros(B, dtype=torch.long)).to(dev),
                is_lr=is_lr,
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
            })

            del logits, output, loss, pixel_values, depth_maps
            del dist_pred, count_pred, mcq_logits, lr_logits

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
                    lr_v    = lrs.get("Vision Encoder", 0.0)
                    lr_e    = lrs.get("Special Embed", 0.0)
                    lr_d    = lrs.get("Decoder", 0.0)
                    lr_rti  = lrs.get("RTI", 0.0)
                    lr_fus  = lrs.get("Visual Fuser", 0.0)
                    lr_dist = lrs.get("Distance Head", 0.0)
                    lr_cnt  = lrs.get("Count Head", 0.0)

                    tqdm.write(
                        f"  step={global_step:>7d}  "
                        f"epoch={current_epoch:.2f}  "
                        f"loss={window_avg:.4f}  "
                        f"grad_norm={grad_norm:.3f}  "
                        f"samples/s={samples_sec:.1f}"
                    )

                    # CSV: val columns empty at log steps
                    with open(csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            global_step, f"{current_epoch:.4f}",
                            f"{window_avg:.6f}",
                            "", "",               # total_val_loss, val_loss_ce
                            "", "", "", "",       # mcq loss/acc, lr loss/acc
                            "", "", "", "",       # dist loss/acc, count loss/acc
                            f"{lr_v:.8f}", f"{lr_e:.8f}", f"{lr_d:.8f}", f"{lr_rti:.8f}", f"{lr_fus:.8f}",
                            f"{lr_dist:.8f}", f"{lr_cnt:.8f}",
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
                    pipeline.train()

                    vl      = val_results["val_loss"]
                    vce     = val_results.get("val_ce", 0.0)
                    vdist   = val_results.get("val_dist", 0.0)
                    vcnt    = val_results.get("val_count", 0.0)
                    vmcq    = val_results.get("val_mcq", 0.0)
                    vmcq_a  = val_results.get("val_mcq_acc", 0.0)
                    vlr     = val_results.get("val_lr", 0.0)
                    vlr_a   = val_results.get("val_lr_acc", 0.0)
                    vdist_a = val_results.get("val_dist_acc", 0.0)
                    vcnt_a  = val_results.get("val_count_acc", 0.0)

                    current_epoch = (global_step * effective_batch) / total_samples
                    lrs = {pg["name"]: pg["lr"] for pg in optimizer.param_groups}
                    lr_v    = lrs.get("Vision Encoder", 0.0)
                    lr_e    = lrs.get("Special Embed", 0.0)
                    lr_d    = lrs.get("Decoder", 0.0)
                    lr_rti  = lrs.get("RTI", 0.0)
                    lr_fus  = lrs.get("Visual Fuser", 0.0)
                    lr_dist = lrs.get("Distance Head", 0.0)
                    lr_cnt  = lrs.get("Count Head", 0.0)

                    with open(csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            global_step, f"{current_epoch:.4f}",
                            f"{window_avg:.6f}",
                            f"{vl:.6f}", f"{vce:.6f}",
                            f"{vmcq:.6f}", f"{vmcq_a:.2f}",
                            f"{vlr:.6f}", f"{vlr_a:.2f}",
                            f"{vdist:.6f}", f"{vdist_a:.2f}",
                            f"{vcnt:.6f}", f"{vcnt_a:.2f}",
                            f"{lr_v:.8f}", f"{lr_e:.8f}", f"{lr_d:.8f}", f"{lr_rti:.8f}", f"{lr_fus:.8f}",
                            f"{lr_dist:.8f}", f"{lr_cnt:.8f}",
                            "", "",
                        ])
                    print(f"\nValidation Accuracy: "
                          f"MCQ={vmcq_a:.1f}% | LR={vlr_a:.1f}% | "
                          f"Dist={vdist_a:.1f}% | Count={vcnt_a:.1f}%")

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
