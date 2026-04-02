"""
SpatialVLM Phase 2 Training -- Full Fine-tuning with LoRA
===========================================================

Phase 2: LoRA on Vision Encoder (rank=32) + Backbone (rank=64),
         GSA + RTI continue training at reduced LR.
         Text embeddings (embed_tokens / lm_head) remain frozen.

Loss:    L_lm (autoregressive CrossEntropy on structured target)
Goal:    Fine-tune entire pipeline toward spatial reasoning.

Prerequisites:
    Phase 1 checkpoint with trained GSA + RTI weights.

Usage:
    python src/train_phase2/train.py --phase1-ckpt checkpoints/phase1/step_25000
    python src/train_phase2/train.py --phase1-ckpt checkpoints/phase1/step_10000 --epochs 5
    python src/train_phase2/train.py --phase1-ckpt checkpoints/phase1/step_25000 --resume checkpoints/phase2/step_5000
    python src/train_phase2/train.py --phase1-ckpt checkpoints/phase1/step_25000 --split train_sample --epochs 1

LoRA Targets (self-implemented):
    Vision Encoder (12 ViT blocks, rank=32):
        attn.qkv, attn.proj, mlp.linear_fc1, mlp.linear_fc2

    Backbone DeltaNet (layers 0,1,2, 4,5,6, ... rank=64):
        linear_attn.in_proj_qkv, in_proj_z, out_proj
        mlp.gate_proj, up_proj, down_proj

    Backbone GatedAttn (layers 3, 7, 11, 15, 19, 23, rank=64):
        self_attn.q_proj, k_proj, v_proj, o_proj
        mlp.gate_proj, up_proj, down_proj
"""

import os
import warnings
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", FutureWarning)

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

from src.dataloader.dataloader import SpatialVLMDataset, get_dataloader
from model.pipeline import SpatialVLM, print_vram_usage
from model.loss import SpatialVLMLoss
from model.lora import (
    apply_lora, get_lora_state_dict, load_lora_state_dict,
    count_lora_params, print_lora_summary, LoRALinear,
)


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints", "phase2")

# LoRA target suffixes -- vision encoder vs backbone
VISION_TARGETS = [
    "attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2",
]
BACKBONE_TARGETS = [
    # DeltaNet
    "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
    # GatedAttn
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    # MLP (all 24 layers)
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


# =========================================================================
# Phase 1 Checkpoint Loading
# =========================================================================

def load_phase1_checkpoint(pipeline, ckpt_dir):
    """Load GSA + RTI weights from a Phase 1 checkpoint."""
    ckpt_path = os.path.join(ckpt_dir, "checkpoint.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Phase 1 checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    pipeline.gsa.load_state_dict(ckpt["gsa_state_dict"])
    pipeline.region_token_extractor.load_state_dict(ckpt["rti_state_dict"])

    step = ckpt.get("step", "?")
    epoch = ckpt.get("epoch", "?")
    loss = ckpt.get("loss", "?")
    print(f"  [*] Loaded Phase 1 GSA+RTI from: {ckpt_dir}")
    print(f"      Phase 1 step={step}, epoch={epoch}, loss={loss}")


# =========================================================================
# Apply LoRA to Qwen (Vision + Backbone)
# =========================================================================

def apply_lora_to_qwen(pipeline, vision_rank=32, backbone_rank=64,
                       alpha_ratio=2.0, dropout=0.05):
    """Apply self-implemented LoRA to Vision Encoder and Backbone.

    Vision modules get rank=32, backbone modules get rank=64.
    Merger and embeddings are excluded.
    """
    qwen = pipeline.qwen

    # Vision Encoder LoRA (rank=32)
    n_vision, vision_modules = apply_lora(
        qwen.model.visual,
        target_suffixes=VISION_TARGETS,
        rank=vision_rank,
        alpha=float(vision_rank * alpha_ratio),
        dropout=dropout,
    )

    # Backbone LoRA (rank=64)
    n_backbone, backbone_modules = apply_lora(
        qwen.model.language_model,
        target_suffixes=BACKBONE_TARGETS,
        rank=backbone_rank,
        alpha=float(backbone_rank * alpha_ratio),
        dropout=dropout,
    )

    print(f"  Vision LoRA:   {len(vision_modules)} modules, {n_vision:,} params (rank={vision_rank})")
    print(f"  Backbone LoRA: {len(backbone_modules)} modules, {n_backbone:,} params (rank={backbone_rank})")
    print(f"  Total LoRA:    {n_vision + n_backbone:,} params")

    return n_vision, n_backbone


# =========================================================================
# Checkpoint Save / Load for Phase 2
# =========================================================================

def save_checkpoint(pipeline, optimizer, scheduler, step, epoch, loss, path,
                    phase1_ckpt_path=""):
    """Save Phase 2 checkpoint: LoRA adapters + GSA + RTI + optimizer state."""
    os.makedirs(path, exist_ok=True)

    # Save everything in one file (LoRA weights are tiny)
    vision_lora_sd = get_lora_state_dict(pipeline.qwen.model.visual)
    backbone_lora_sd = get_lora_state_dict(pipeline.qwen.model.language_model)

    torch.save({
        "step": step,
        "epoch": epoch,
        "loss": loss,
        "phase1_ckpt_path": phase1_ckpt_path,
        "gsa_state_dict": pipeline.gsa.state_dict(),
        "rti_state_dict": pipeline.region_token_extractor.state_dict(),
        "vision_lora_state_dict": vision_lora_sd,
        "backbone_lora_state_dict": backbone_lora_sd,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }, os.path.join(path, "checkpoint.pt"))
    print(f"  [*] Checkpoint saved: {path}")


def load_checkpoint(pipeline, optimizer, scheduler, path):
    """Load Phase 2 checkpoint: LoRA + GSA + RTI + optimizer state."""
    ckpt_path = os.path.join(path, "checkpoint.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Phase 2 checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # Load LoRA weights
    load_lora_state_dict(pipeline.qwen.model.visual,
                         ckpt["vision_lora_state_dict"])
    load_lora_state_dict(pipeline.qwen.model.language_model,
                         ckpt["backbone_lora_state_dict"])

    # Load GSA + RTI
    pipeline.gsa.load_state_dict(ckpt["gsa_state_dict"])
    pipeline.region_token_extractor.load_state_dict(ckpt["rti_state_dict"])

    # Load optimizer + scheduler
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    p1_path = ckpt.get("phase1_ckpt_path", "?")
    print(f"  [*] Resumed Phase 2 from: {path} (step={ckpt['step']}, epoch={ckpt['epoch']})")
    print(f"      Original Phase 1 base: {p1_path}")
    return ckpt["step"], ckpt["epoch"]


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="SpatialVLM Phase 2 Training")
    # Model
    parser.add_argument("--device",     default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype",      default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--attn-impl",  default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])
    # Phase 1 checkpoint (required)
    parser.add_argument("--phase1-ckpt", type=str, required=True,
                        help="Path to Phase 1 checkpoint directory")
    # LoRA
    parser.add_argument("--vision-lora-rank", type=int, default=32)
    parser.add_argument("--backbone-lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha-ratio", type=float, default=2.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    # Training
    parser.add_argument("--split",      default="train", choices=["train", "train_sample"])
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--lr-vision",  type=float, default=5e-5)
    parser.add_argument("--lr-backbone", type=float, default=2e-5)
    parser.add_argument("--lr-custom",  type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int,   default=1)
    parser.add_argument("--grad-accum", type=int,   default=8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--resolution",  default="1080p",
                        choices=["1080p", "720p", "540p", "450p"])
    parser.add_argument("--no-grad-ckpt", action="store_true")
    # Logging & Checkpointing
    parser.add_argument("--log-steps",  type=int,   default=100)
    parser.add_argument("--resume",     type=str,   default=None)
    parser.add_argument("--save-steps", type=int,   default=5000)
    parser.add_argument("--num-workers", type=int,  default=4)
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
    print("PHASE 2 TRAINING: Full Fine-tuning with LoRA")
    print("=" * 70)

    pipeline = SpatialVLM(
        dtype=dtype,
        device_map=args.device,
        attn_implementation=args.attn_impl,
    )
    print_vram_usage("after model load")

    # ====================================================================
    # 2. LOAD PHASE 1 WEIGHTS (GSA + RTI)
    # ====================================================================
    print(f"\n{'='*70}")
    print("LOADING PHASE 1 CHECKPOINT")
    print("=" * 70)

    load_phase1_checkpoint(pipeline, args.phase1_ckpt)

    # ====================================================================
    # 3. APPLY LoRA
    # ====================================================================
    print(f"\n{'='*70}")
    print("APPLYING LoRA ADAPTERS")
    print("=" * 70)

    n_vision, n_backbone = apply_lora_to_qwen(
        pipeline,
        vision_rank=args.vision_lora_rank,
        backbone_rank=args.backbone_lora_rank,
        alpha_ratio=args.lora_alpha_ratio,
        dropout=args.lora_dropout,
    )
    print_vram_usage("after LoRA")

    # ====================================================================
    # 4. CONFIGURE TRAINABLE PARAMETERS (3 groups)
    # ====================================================================
    print(f"\n{'='*70}")
    print("PARAMETER GROUPS")
    print("=" * 70)

    # Freeze everything in Qwen first
    for param in pipeline.qwen.parameters():
        param.requires_grad = False

    # Unfreeze LoRA adapters only
    for module in pipeline.qwen.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.weight.requires_grad_(True)
            module.lora_B.weight.requires_grad_(True)

    # GSA + RTI fully trainable
    for param in pipeline.gsa.parameters():
        param.requires_grad = True
    for param in pipeline.region_token_extractor.parameters():
        param.requires_grad = True

    # Build 3 optimizer param groups
    vision_lora_params = []
    backbone_lora_params = []

    for name, module in pipeline.qwen.model.visual.named_modules():
        if isinstance(module, LoRALinear):
            vision_lora_params.extend([module.lora_A.weight, module.lora_B.weight])

    for name, module in pipeline.qwen.model.language_model.named_modules():
        if isinstance(module, LoRALinear):
            backbone_lora_params.extend([module.lora_A.weight, module.lora_B.weight])

    custom_params = list(pipeline.gsa.parameters()) + \
                    list(pipeline.region_token_extractor.parameters())

    n_vision_p  = sum(p.numel() for p in vision_lora_params)
    n_backbone_p = sum(p.numel() for p in backbone_lora_params)
    n_custom  = sum(p.numel() for p in custom_params)
    n_total   = n_vision_p + n_backbone_p + n_custom
    n_frozen  = sum(p.numel() for p in pipeline.parameters() if not p.requires_grad)

    print(f"  Group 1 (Vision LoRA r={args.vision_lora_rank}):  {n_vision_p:>12,} ({n_vision_p/1e6:.2f}M)  lr={args.lr_vision}")
    print(f"  Group 2 (Backbone LoRA r={args.backbone_lora_rank}): {n_backbone_p:>12,} ({n_backbone_p/1e6:.2f}M)  lr={args.lr_backbone}")
    print(f"  Group 3 (GSA + RTI):         {n_custom:>12,} ({n_custom/1e6:.2f}M)  lr={args.lr_custom}")
    print(f"  {'─'*60}")
    print(f"  Total trainable:             {n_total:>12,} ({n_total/1e6:.2f}M)")
    print(f"  Frozen:                      {n_frozen:>12,} ({n_frozen/1e6:.2f}M)")

    # ====================================================================
    # 5. LOAD DATA
    # ====================================================================
    print(f"\n{'='*70}")
    print("LOADING DATA")
    print("=" * 70)

    processor = pipeline.processor
    target_size = {"1080p": None, "720p": (1280, 720), "540p": (960, 540), "450p": (800, 450)}[args.resolution]
    dataset = SpatialVLMDataset(args.split, processor=processor, target_size=target_size)
    loader = get_dataloader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False,
    )

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
    # 6. OPTIMIZER + SCHEDULER
    # ====================================================================
    param_groups = [
        {"params": vision_lora_params,   "lr": args.lr_vision,   "name": "vision_lora"},
        {"params": backbone_lora_params, "lr": args.lr_backbone, "name": "backbone_lora"},
        {"params": custom_params,        "lr": args.lr_custom,   "name": "gsa_rti"},
    ]

    optimizer = AdamW(param_groups, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - args.warmup_steps, eta_min=1e-6)
    criterion = SpatialVLMLoss()
    dev = pipeline.device

    # ====================================================================
    # 7. RESUME PHASE 2 (optional)
    # ====================================================================
    start_step = 0
    start_epoch = 0
    if args.resume:
        start_step, start_epoch = load_checkpoint(
            pipeline, optimizer, scheduler, args.resume
        )

    # ====================================================================
    # 8. CSV LOG
    # ====================================================================
    if args.resume and os.path.exists(csv_path):
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
                             "lr_vision", "lr_backbone", "lr_custom",
                             "grad_norm", "samples_per_sec"])

    # ====================================================================
    # 9. TRAINING LOOP
    # ====================================================================
    print(f"\n{'='*70}")
    print("TRAINING")
    print("=" * 70)
    print(f"  Phase 1 base:     {args.phase1_ckpt}")
    print(f"  lr_vision={args.lr_vision}  lr_backbone={args.lr_backbone}  lr_custom={args.lr_custom}")
    print(f"  warmup={args.warmup_steps}  max_grad_norm={args.max_grad_norm}")
    print(f"  Vision LoRA rank={args.vision_lora_rank}  Backbone LoRA rank={args.backbone_lora_rank}")
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
            samples_so_far = epoch * total_samples + batch_idx
            if global_step > 0 and samples_so_far < start_step * args.grad_accum:
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
                    rle_list=batch["rle_list"][0],
                    mask_token_positions=batch["mask_positions"][0],
                    decoded_masks=batch["decoded_masks"][0],
                    use_gradient_checkpointing=not args.no_grad_ckpt,
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
            loss = criterion(logits, labels) / args.grad_accum
            loss.backward()

            micro_step += 1
            epoch_samples += 1
            loss_val = loss.item() * args.grad_accum
            total_loss_sum += loss_val
            loss_window.append(loss_val)

            window_avg = sum(loss_window) / len(loss_window)
            pbar.set_postfix({
                "step": global_step,
                "loss": f"{window_avg:.4f}",
                "lr_bb": f"{optimizer.param_groups[1]['lr']:.2e}",
            })

            del logits, output, loss, pixel_values, depth_maps

            # Optimizer step every grad_accum micro-steps
            if micro_step % args.grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    all_trainable, args.max_grad_norm
                ).item()

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
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
                    samples_sec = (args.grad_accum * args.log_steps) / elapsed
                    current_epoch = (global_step * args.grad_accum) / total_samples

                    lr_v = optimizer.param_groups[0]["lr"]
                    lr_b = optimizer.param_groups[1]["lr"]
                    lr_c = optimizer.param_groups[2]["lr"]

                    tqdm.write(
                        f"  step={global_step:>7d}  "
                        f"epoch={current_epoch:.2f}  "
                        f"loss={window_avg:.4f}  "
                        f"lr_v={lr_v:.2e}  "
                        f"lr_b={lr_b:.2e}  "
                        f"lr_c={lr_c:.2e}  "
                        f"grad_norm={grad_norm:.3f}  "
                        f"samples/s={samples_sec:.1f}"
                    )

                    with open(csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            global_step, f"{current_epoch:.4f}",
                            f"{window_avg:.6f}",
                            f"{lr_v:.8f}", f"{lr_b:.8f}", f"{lr_c:.8f}",
                            f"{grad_norm:.6f}", f"{samples_sec:.2f}",
                        ])

                    log_time = time.time()

                # Mid-epoch checkpoint
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    ckpt_path = os.path.join(CHECKPOINT_DIR, f"step_{global_step}")
                    save_checkpoint(
                        pipeline, optimizer, scheduler,
                        global_step, epoch, window_avg, ckpt_path,
                        phase1_ckpt_path=args.phase1_ckpt,
                    )

        # End of epoch
        epoch_elapsed = time.time() - epoch_start_time
        epoch_steps = epoch_samples // args.grad_accum
        avg_epoch_loss = total_loss_sum / max(epoch_samples, 1)
        print(f"\n  Epoch {epoch + 1} done: "
              f"avg_loss={avg_epoch_loss:.4f}  "
              f"steps={epoch_steps}  "
              f"time={epoch_elapsed/60:.1f}min")

        ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch + 1}")
        save_checkpoint(
            pipeline, optimizer, scheduler,
            global_step, epoch + 1, avg_epoch_loss, ckpt_path,
            phase1_ckpt_path=args.phase1_ckpt,
        )
        print_vram_usage(f"epoch {epoch + 1}")

    # ====================================================================
    # 10. FINAL SUMMARY
    # ====================================================================
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Phase 1 base:     {args.phase1_ckpt}")
    print(f"  Total steps:      {global_step}")
    print(f"  Vision LoRA:      rank={args.vision_lora_rank}")
    print(f"  Backbone LoRA:    rank={args.backbone_lora_rank}")
    print(f"  CSV log:          {os.path.abspath(csv_path)}")
    print(f"  Checkpoints:      {os.path.abspath(CHECKPOINT_DIR)}")
    print_vram_usage("final")


if __name__ == "__main__":
    main()
