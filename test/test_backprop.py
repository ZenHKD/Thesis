"""
Test SpatialVLM single backpropagation step.

Loads 1 batch from the dataloader, runs forward through the pipeline,
computes loss (L_lm), calls backward, and verifies gradients
flow to GSA and RTI (the trainable custom modules).

Usage:
    python test/test_backprop.py
    python test/test_backprop.py --resolution 450p
    python test/test_backprop.py --attn-impl sdpa
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoProcessor
from src.dataloader.dataloader import SpatialVLMDataset, get_dataloader
from model.pipeline import SpatialVLM, print_vram_usage
from model.loss import SpatialVLMLoss


MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "model", "qwen3.5-0.8b")


def main():
    parser = argparse.ArgumentParser(description="Test SpatialVLM backpropagation")
    parser.add_argument("--device",     default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype",      default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--attn-impl",  default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--resolution", default="450p",
                        choices=["1080p", "720p", "540p", "450p"],
                        help="Image resolution (default: 450p to fit in 12GB VRAM)")

    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    target_size = {
        "1080p": None,
        "720p":  (1280, 720),
        "540p":  (960, 540),
        "450p":  (800, 450),
    }[args.resolution]

    # ====================================================================
    # 1. LOAD MODEL
    # ====================================================================
    print("=" * 70)
    print("LOADING MODEL")
    print("=" * 70)

    pipeline = SpatialVLM(
        dtype=dtype,
        device_map=args.device,
        attn_implementation=args.attn_impl,
    )
    print_vram_usage("after model load")

    # Freeze Qwen (Phase 1 setup: only GSA + RTI are trainable)
    for param in pipeline.qwen.parameters():
        param.requires_grad = False
    for param in pipeline.gsa.parameters():
        param.requires_grad = True
    for param in pipeline.region_token_extractor.parameters():
        param.requires_grad = True

    n_trainable = sum(p.numel() for p in pipeline.parameters() if p.requires_grad)
    n_frozen    = sum(p.numel() for p in pipeline.parameters() if not p.requires_grad)
    print(f"\n  Trainable: {n_trainable:,} ({n_trainable/1e6:.2f}M)")
    print(f"  Frozen:    {n_frozen:,} ({n_frozen/1e6:.2f}M)")

    # ====================================================================
    # 2. LOAD DATA (1 batch)
    # ====================================================================
    print(f"\n{'='*70}")
    print("LOADING DATA")
    print("=" * 70)

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    dataset = SpatialVLMDataset("train_sample", processor=processor,
                                max_samples=args.batch_size, target_size=target_size)
    loader = get_dataloader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=False)
    batch = next(iter(loader))

    print(f"  Resolution: {args.resolution}")
    print(f"  Batch size: {args.batch_size}")
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key:18s}: {list(val.shape)}  dtype={val.dtype}")
        elif isinstance(val, list):
            print(f"  {key:18s}: list[{len(val)}]")

    # ====================================================================
    # 3. FORWARD PASS
    # ====================================================================
    print(f"\n{'='*70}")
    print("FORWARD PASS")
    print("=" * 70)

    dev = pipeline.device
    pixel_values   = batch["pixel_values"].to(device=dev, dtype=dtype)
    image_grid_thw = batch["image_grid_thw"].to(device=dev)
    depth_maps     = batch["depth_maps"].to(device=dev, dtype=dtype)
    input_ids      = batch["input_ids"].to(device=dev)
    labels         = batch["labels"].to(device=dev)

    # Forward through pipeline (training mode, gradient checkpointing for VRAM)
    pipeline.train()
    output = pipeline(
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        depth_maps=depth_maps,
        input_ids=input_ids,
        rle_list=batch["rle_list"][0],
        mask_token_positions=batch["mask_positions"][0],
        decoded_masks=batch["decoded_masks"][0],
        use_gradient_checkpointing=True,
    )
    logits = output["logits"]
    print(f"  Logits shape: {list(logits.shape)}")
    print(f"  Labels shape: {list(labels.shape)}")
    print_vram_usage("after forward")

    # ====================================================================
    # 4. COMPUTE LOSS
    # ====================================================================
    print(f"\n{'='*70}")
    print("LOSS COMPUTATION")
    print("=" * 70)

    criterion = SpatialVLMLoss()

    loss = criterion(logits, labels)

    print(f"  Loss = {loss.item():.4f}")

    # ====================================================================
    # 5. BACKWARD PASS
    # ====================================================================
    print(f"\n{'='*70}")
    print("BACKWARD PASS")
    print("=" * 70)

    loss.backward()
    print_vram_usage("after backward")

    # ====================================================================
    # 6. GRADIENT CHECK
    # ====================================================================
    print(f"\n{'='*70}")
    print("GRADIENT CHECK")
    print("=" * 70)

    # Check GSA gradients
    print("\n  GSA gradients:")
    gsa_has_grad = False
    for name, param in pipeline.gsa.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                has_nan = torch.isnan(param.grad).any().item()
                has_inf = torch.isinf(param.grad).any().item()
                status = "[OK]"
                if has_nan:
                    status = "[NaN]"
                elif has_inf:
                    status = "[Inf]"
                elif grad_norm == 0.0:
                    status = "[ZERO]"
                else:
                    gsa_has_grad = True
                print(f"    {name:45s}: grad_norm={grad_norm:.6f}  {status}")
            else:
                print(f"    {name:45s}: grad=None  [FAIL]")

    # Check RTI gradients
    print("\n  RTI gradients:")
    rti_has_grad = False
    for name, param in pipeline.region_token_extractor.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                has_nan = torch.isnan(param.grad).any().item()
                has_inf = torch.isinf(param.grad).any().item()
                status = "[OK]"
                if has_nan:
                    status = "[NaN]"
                elif has_inf:
                    status = "[Inf]"
                elif grad_norm == 0.0:
                    status = "[ZERO]"
                else:
                    rti_has_grad = True
                print(f"    {name:45s}: grad_norm={grad_norm:.6f}  {status}")
            else:
                print(f"    {name:45s}: grad=None  [FAIL]")

    # Check Qwen is frozen (no grads)
    print("\n  Qwen (frozen) -- spot check:")
    qwen_leak = False
    qwen = pipeline.qwen
    spot_checks = [
        ("visual.blocks.0.attn.qkv.weight", qwen.model.visual.blocks[0].attn.qkv.weight),
        ("language_model.layers.0.mlp.gate_proj.weight",
         qwen.model.language_model.layers[0].mlp.gate_proj.weight),
    ]
    for name, param in spot_checks:
        if param.grad is not None:
            print(f"    {name:45s}: grad exists  [LEAK]")
            qwen_leak = True
        else:
            print(f"    {name:45s}: grad=None  [OK - frozen]")

    # ====================================================================
    # 7. SUMMARY
    # ====================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)

    all_ok = True

    if gsa_has_grad:
        print("  [OK] GSA has non-zero gradients")
    else:
        print("  [FAIL] GSA has no gradients")
        all_ok = False

    if rti_has_grad:
        print("  [OK] RTI has non-zero gradients")
    else:
        print("  [FAIL] RTI has no gradients")
        all_ok = False

    if not qwen_leak:
        print("  [OK] Qwen backbone is frozen (no gradient leaks)")
    else:
        print("  [FAIL] Qwen backbone has gradient leaks")
        all_ok = False

    loss_finite = torch.isfinite(loss).item()
    if loss_finite:
        print("  [OK] Loss is finite")
    else:
        print("  [FAIL] Loss is NaN or Inf")
        all_ok = False

    print(f"\n{'='*70}")
    if all_ok:
        print("  Backprop test [OK]")
    else:
        print("  Backprop test [FAIL]")
    print(f"{'='*70}")
    print_vram_usage("final")


if __name__ == "__main__":
    main()
