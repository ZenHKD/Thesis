"""
Test SpatialVLM Micro full pipeline backpropagation.

Full fine-tuning from scratch.
Loads pruned Micro checkpoint, runs 1 forward + backward on real data,
verifies gradient flow to ALL components.

Usage:
    python test_micro/test_backprop.py
    python test_micro/test_backprop.py --resolution 450p
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataloader.dataloader_new import SpatialVLMDataset, get_dataloader
from model_micro.pipeline import SpatialVLM, print_vram_usage
from model_micro.loss import SpatialLoss


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
    parser = argparse.ArgumentParser(description="Test Micro backpropagation (full fine-tuning)")
    parser.add_argument("--device",    default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype",     default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--attn-impl", default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--resolution", default="450p",
                        choices=["1080p", "720p", "540p", "450p"])
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size for testing (default: 2)")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    target_size = {"1080p": None, "720p": (1280, 720),
                   "540p": (960, 540), "450p": (800, 450)}[args.resolution]

    # ====================================================================
    # 1. LOAD MODEL
    # ====================================================================
    print("=" * 70)
    print("TEST: Micro Full Fine-tuning Backpropagation")
    print("=" * 70)

    pipeline = SpatialVLM(dtype=dtype, device_map=args.device,
                          attn_implementation=args.attn_impl)
    print_vram_usage("after model load")

    # ====================================================================
    # 2. CONFIGURE — ALL TRAINABLE (full fine-tuning)
    # ====================================================================
    print(f"\n{'='*70}")
    print("PARAMETER SETUP (full fine-tuning — everything trainable)")
    print("=" * 70)

    for param in pipeline.parameters():
        param.requires_grad = True

    # Count per component
    components = {
        "Vision Encoder":  pipeline.qwen.model.visual,
        "Embeddings":      pipeline.qwen.model.language_model.embed_tokens,
        "Decoder (8 layers)": pipeline.qwen.model.language_model.layers,
        "LM Head (tied)":  pipeline.qwen.lm_head,
        "GSA":             pipeline.gsa,
        "RTI":             pipeline.region_token_extractor,
        "Number Head":     pipeline.num_head,
    }

    total_trainable = 0
    for name, module in components.items():
        n = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total_trainable += n
        print(f"  {name:25s}: {n:>12,} ({n/1e6:.2f}M)")
    print(f"  {'Total trainable':25s}: {total_trainable:>12,} ({total_trainable/1e6:.2f}M)")

    # ====================================================================
    # 3. LOAD DATA
    # ====================================================================
    print(f"\n{'='*70}")
    print("LOADING DATA")
    print("=" * 70)

    dataset = SpatialVLMDataset("train_sample", processor=pipeline.processor,
                                max_samples=args.batch_size * 2,
                                target_size=target_size)
    loader = get_dataloader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=False)
    batch = next(iter(loader))

    print(f"  Resolution:   {args.resolution}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Categories:   {batch['categories']}")
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key:22s}: {list(val.shape)}  dtype={val.dtype}")

    # ====================================================================
    # 4. FORWARD + BACKWARD
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
        rle_list=batch["rle_list"],
        mask_token_positions=batch["mask_positions"],
        decoded_masks=batch["decoded_masks"],
        num_token_positions=batch.get("num_token_positions"),
        use_gradient_checkpointing=True,
        vision_requires_grad=True,
    )

    logits = output["logits"]
    num_pred = output["num_pred"]
    print(f"  logits shape: {list(logits.shape)}")
    print(f"  num_pred:     {num_pred.tolist()}")

    # Loss with label remapping
    criterion = SpatialLoss(alpha=1.0, remap_fn=pipeline.remap_to_new)
    loss = criterion(
        logits, labels,
        num_pred, batch["target_num"].to(dev),
        batch["is_numeric"].to(dev),
    )
    print(f"  Loss = {loss.item():.4f}")
    loss.backward()
    print_vram_usage("after backward")

    # ====================================================================
    # 5. GRADIENT CHECKS
    # ====================================================================
    print(f"\n{'='*70}")
    print("GRADIENT CHECK — Vision Encoder")
    print("=" * 70)
    vision_ok, vision_issues = check_gradients(
        pipeline.qwen.model.visual, "Vision Encoder")

    print(f"\n{'='*70}")
    print("GRADIENT CHECK — Decoder (8 layers)")
    print("=" * 70)
    decoder_ok, decoder_issues = check_gradients(
        pipeline.qwen.model.language_model.layers, "Decoder")

    print(f"\n{'='*70}")
    print("GRADIENT CHECK — Embeddings")
    print("=" * 70)
    embed_ok, embed_issues = check_gradients(
        pipeline.qwen.model.language_model.embed_tokens, "Embeddings")

    print(f"\n{'='*70}")
    print("GRADIENT CHECK — GSA")
    print("=" * 70)
    gsa_ok, gsa_issues = check_gradients(pipeline.gsa, "GSA")

    print(f"\n{'='*70}")
    print("GRADIENT CHECK — RTI")
    print("=" * 70)
    rti_ok, rti_issues = check_gradients(pipeline.region_token_extractor, "RTI")

    print(f"\n{'='*70}")
    print("GRADIENT CHECK — Number Head")
    print("=" * 70)
    numhead_ok, numhead_issues = check_gradients(pipeline.num_head, "Number Head")

    # ====================================================================
    # 6. SUMMARY
    # ====================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)

    all_ok = True
    has_numeric = batch["is_numeric"].any().item()
    checks = [
        (vision_ok, "Vision Encoder has non-zero gradients"),
        (decoder_ok, "Decoder layers have non-zero gradients"),
        (embed_ok, "Embeddings have non-zero gradients"),
        (gsa_ok, "GSA has non-zero gradients"),
        (rti_ok, "RTI has non-zero gradients"),
        (numhead_ok or not has_numeric,
         f"Number Head has gradients (has_numeric={has_numeric})"),
        (not (vision_issues or decoder_issues or embed_issues
              or gsa_issues or rti_issues or (numhead_issues and has_numeric)),
         "No NaN/Inf in any gradients"),
        (torch.isfinite(loss).item(), f"Loss is finite ({loss.item():.4f})"),
    ]

    for ok, msg in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {msg}")
        if not ok:
            all_ok = False

    print(f"\n{'='*70}")
    print(f"  Micro Backprop Test [{'OK' if all_ok else 'FAIL'}]")
    print(f"{'='*70}")
    print_vram_usage("final")


if __name__ == "__main__":
    main()
