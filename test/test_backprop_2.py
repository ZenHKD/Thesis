"""
Test SpatialVLM Phase 2 backpropagation.

Loads Phase 1 checkpoint, applies self-implemented LoRA to
Vision Encoder (rank=32) + Backbone (rank=64), runs 1 forward + backward,
and verifies gradient flow.

Usage:
    python test/test_backprop_2.py --phase1-ckpt checkpoints/phase1/step_30000
    python test/test_backprop_2.py --phase1-ckpt checkpoints/phase1/step_30000 --resolution 450p
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataloader.dataloader import SpatialVLMDataset, get_dataloader
from model.pipeline import SpatialVLM, print_vram_usage
from model.loss import SpatialVLMLoss
from model.lora import apply_lora, LoRALinear


# LoRA targets (same as train_phase2/train.py)
VISION_TARGETS = [
    "attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2",
]
BACKBONE_TARGETS = [
    "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


def check_gradients(module, module_name, filter_fn=None):
    """Check gradients. Returns (has_grad, has_issues)."""
    has_grad = False
    has_issues = False
    count_ok = count_zero = count_none = count_bad = 0

    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if filter_fn and not filter_fn(name, param):
            continue

        if param.grad is None:
            print(f"    {name:55s}: grad=None  [FAIL]")
            count_none += 1
            has_issues = True
        else:
            grad_norm = param.grad.norm().item()
            has_nan = torch.isnan(param.grad).any().item()
            has_inf = torch.isinf(param.grad).any().item()

            if has_nan:
                status = "[NaN]"; count_bad += 1; has_issues = True
            elif has_inf:
                status = "[Inf]"; count_bad += 1; has_issues = True
            elif grad_norm == 0.0:
                status = "[ZERO]"; count_zero += 1
            else:
                status = "[OK]"; count_ok += 1; has_grad = True

            print(f"    {name:55s}: grad_norm={grad_norm:.6f}  {status}")

    total = count_ok + count_zero + count_none + count_bad
    print(f"    -- {module_name}: {count_ok}/{total} params have non-zero grad"
          f" ({count_zero} zero, {count_none} None, {count_bad} NaN/Inf)")
    return has_grad, has_issues


def main():
    parser = argparse.ArgumentParser(description="Test Phase 2 backpropagation")
    parser.add_argument("--device",      default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype",       default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--attn-impl",   default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--phase1-ckpt", type=str, required=True)
    parser.add_argument("--vision-rank", type=int, default=32)
    parser.add_argument("--backbone-rank", type=int, default=64)
    parser.add_argument("--resolution",  default="450p",
                        choices=["1080p", "720p", "540p", "450p"])
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    target_size = {"1080p": None, "720p": (1280, 720), "540p": (960, 540), "450p": (800, 450)}[args.resolution]

    # ====================================================================
    # 1. LOAD MODEL
    # ====================================================================
    print("=" * 70)
    print("TEST: Phase 2 Backpropagation (self-implemented LoRA)")
    print("=" * 70)

    pipeline = SpatialVLM(dtype=dtype, device_map=args.device, attn_implementation=args.attn_impl)
    print_vram_usage("after model load")

    # ====================================================================
    # 2. LOAD PHASE 1 CHECKPOINT
    # ====================================================================
    print(f"\n{'='*70}")
    print("LOADING PHASE 1 CHECKPOINT")
    print("=" * 70)

    ckpt_path = os.path.join(args.phase1_ckpt, "checkpoint.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    pipeline.gsa.load_state_dict(ckpt["gsa_state_dict"])
    pipeline.region_token_extractor.load_state_dict(ckpt["rti_state_dict"])
    print(f"  Loaded from: {args.phase1_ckpt}")
    print(f"  Phase 1 step={ckpt.get('step', '?')}, loss={ckpt.get('loss', '?')}")

    # ====================================================================
    # 3. APPLY LoRA (self-implemented)
    # ====================================================================
    print(f"\n{'='*70}")
    print("APPLYING LoRA ADAPTERS")
    print("=" * 70)

    n_vis, vis_modules = apply_lora(
        pipeline.qwen.model.visual, VISION_TARGETS,
        rank=args.vision_rank, alpha=float(args.vision_rank * 2), dropout=0.05,
    )
    n_bb, bb_modules = apply_lora(
        pipeline.qwen.model.language_model, BACKBONE_TARGETS,
        rank=args.backbone_rank, alpha=float(args.backbone_rank * 2), dropout=0.05,
    )
    print(f"  Vision:   {len(vis_modules)} modules, {n_vis:,} params (rank={args.vision_rank})")
    print(f"  Backbone: {len(bb_modules)} modules, {n_bb:,} params (rank={args.backbone_rank})")
    print_vram_usage("after LoRA")

    # ====================================================================
    # 4. CONFIGURE TRAINABLE PARAMETERS
    # ====================================================================
    print(f"\n{'='*70}")
    print("PARAMETER SETUP")
    print("=" * 70)

    # Freeze everything, then unfreeze LoRA + custom
    for param in pipeline.qwen.parameters():
        param.requires_grad = False
    for module in pipeline.qwen.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.weight.requires_grad_(True)
            module.lora_B.weight.requires_grad_(True)
    for param in pipeline.gsa.parameters():
        param.requires_grad = True
    for param in pipeline.region_token_extractor.parameters():
        param.requires_grad = True

    # Count
    n_custom = sum(p.numel() for p in pipeline.gsa.parameters()) + \
               sum(p.numel() for p in pipeline.region_token_extractor.parameters())
    n_total = n_vis + n_bb + n_custom
    n_frozen = sum(p.numel() for p in pipeline.parameters() if not p.requires_grad)

    print(f"  Vision LoRA:     {n_vis:>12,} ({n_vis/1e6:.2f}M)")
    print(f"  Backbone LoRA:   {n_bb:>12,} ({n_bb/1e6:.2f}M)")
    print(f"  GSA + RTI:       {n_custom:>12,} ({n_custom/1e6:.2f}M)")
    print(f"  Total trainable: {n_total:>12,} ({n_total/1e6:.2f}M)")
    print(f"  Frozen:          {n_frozen:>12,} ({n_frozen/1e6:.2f}M)")

    # ====================================================================
    # 5. VERIFY MERGER IS EXCLUDED
    # ====================================================================
    print(f"\n{'='*70}")
    print("MERGER EXCLUSION CHECK")
    print("=" * 70)

    merger_has_lora = False
    for name, module in pipeline.qwen.model.visual.merger.named_modules():
        if isinstance(module, LoRALinear):
            merger_has_lora = True
            print(f"  [!] LoRA on merger: {name}")

    print("  [OK] No LoRA on merger" if not merger_has_lora else "  [FAIL] Merger has LoRA")

    # ====================================================================
    # 6. LOAD DATA
    # ====================================================================
    print(f"\n{'='*70}")
    print("LOADING DATA")
    print("=" * 70)

    dataset = SpatialVLMDataset("train_sample", processor=pipeline.processor,
                                max_samples=1, target_size=target_size)
    loader = get_dataloader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    batch = next(iter(loader))

    print(f"  Resolution: {args.resolution}")
    print(f"  Category:   {batch['categories'][0]}")
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key:18s}: {list(val.shape)}  dtype={val.dtype}")

    # ====================================================================
    # 7. FORWARD + BACKWARD
    # ====================================================================
    print(f"\n{'='*70}")
    print("FORWARD + BACKWARD")
    print("=" * 70)

    dev = pipeline.device
    pixel_values   = batch["pixel_values"].to(device=dev, dtype=dtype)
    image_grid_thw = batch["image_grid_thw"].to(device=dev)
    depth_maps     = batch["depth_maps"].to(device=dev, dtype=dtype)
    input_ids      = batch["input_ids"].to(device=dev)
    labels         = batch["labels"].to(device=dev)

    pipeline.train()
    output = pipeline(
        pixel_values=pixel_values, image_grid_thw=image_grid_thw,
        depth_maps=depth_maps, input_ids=input_ids,
        rle_list=batch["rle_list"][0],
        mask_token_positions=batch["mask_positions"][0],
        decoded_masks=batch["decoded_masks"][0],
        use_gradient_checkpointing=True,
        vision_requires_grad=True,
    )

    criterion = SpatialVLMLoss()
    loss = criterion(output["logits"], labels)
    print(f"  Loss = {loss.item():.4f}")
    loss.backward()
    print_vram_usage("after backward")

    # ====================================================================
    # 8. GRADIENT CHECKS
    # ====================================================================
    print(f"\n{'='*70}")
    print("GRADIENT CHECK -- Vision LoRA")
    print("=" * 70)

    vision_ok, vision_issues = check_gradients(
        pipeline.qwen.model.visual, "Vision LoRA",
        filter_fn=lambda n, p: "lora_" in n,
    )

    print(f"\n{'='*70}")
    print("GRADIENT CHECK -- Backbone LoRA")
    print("=" * 70)

    backbone_ok, backbone_issues = check_gradients(
        pipeline.qwen.model.language_model, "Backbone LoRA",
        filter_fn=lambda n, p: "lora_" in n,
    )

    print(f"\n{'='*70}")
    print("GRADIENT CHECK -- GSA")
    print("=" * 70)
    gsa_ok, gsa_issues = check_gradients(pipeline.gsa, "GSA")

    print(f"\n{'='*70}")
    print("GRADIENT CHECK -- RTI")
    print("=" * 70)
    rti_ok, rti_issues = check_gradients(pipeline.region_token_extractor, "RTI")

    # ====================================================================
    # 9. FROZEN CHECKS
    # ====================================================================
    print(f"\n{'='*70}")
    print("FROZEN CHECK -- Base weights")
    print("=" * 70)

    spot_checks = [
        ("embed_tokens.weight", pipeline.qwen.model.language_model.embed_tokens.weight),
        ("visual.merger.linear_fc1.weight", pipeline.qwen.model.visual.merger.linear_fc1.weight),
        ("visual.merger.linear_fc2.weight", pipeline.qwen.model.visual.merger.linear_fc2.weight),
        ("layers.0.mlp.gate_proj.base.weight", pipeline.qwen.model.language_model.layers[0].mlp.gate_proj.base.weight),
    ]

    frozen_leak = False
    for name, param in spot_checks:
        if param.grad is not None and param.grad.norm().item() > 0:
            print(f"    {name:55s}: grad_norm={param.grad.norm().item():.6f}  [LEAK]")
            frozen_leak = True
        else:
            print(f"    {name:55s}: grad=None  [OK - frozen]")

    # ====================================================================
    # 10. SUMMARY
    # ====================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)

    all_ok = True
    checks = [
        (vision_ok, "Vision LoRA has non-zero gradients"),
        (backbone_ok, "Backbone LoRA has non-zero gradients"),
        (gsa_ok, "GSA has non-zero gradients"),
        (rti_ok, "RTI has non-zero gradients"),
        (not frozen_leak, "Base weights are frozen (no gradient leaks)"),
        (not merger_has_lora, "Merger excluded from LoRA"),
        (not (vision_issues or backbone_issues or gsa_issues or rti_issues), "No NaN/Inf in any gradients"),
        (torch.isfinite(loss).item(), f"Loss is finite ({loss.item():.4f})"),
    ]

    for ok, msg in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {msg}")
        if not ok:
            all_ok = False

    print(f"\n{'='*70}")
    print(f"  Phase 2 Backprop Test [{'OK' if all_ok else 'FAIL'}]")
    print(f"{'='*70}")
    print_vram_usage("final")


if __name__ == "__main__":
    main()
