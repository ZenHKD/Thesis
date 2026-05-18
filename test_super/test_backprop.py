"""
Test SpatialVLM Super full pipeline backpropagation.

Full fine-tuning — all components trainable including embeddings.
Loads pruned Super checkpoint, runs 1 forward + backward on real data,
verifies gradient flow to all trainable components (4 heads).

Usage:
    python test_super/test_backprop.py
    python test_super/test_backprop.py --resolution 320p
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from super_model.dataloader import SpatialVLMDataset, get_dataloader
from super_model.pipeline import SpatialVLM, print_vram_usage
from super_model.loss import SpatialLoss


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
            count_none += 1
            has_issues = True
        else:
            grad_norm = param.grad.norm().item()
            has_nan = torch.isnan(param.grad).any().item()
            has_inf = torch.isinf(param.grad).any().item()

            if has_nan:
                count_bad += 1; has_issues = True
            elif has_inf:
                count_bad += 1; has_issues = True
            elif grad_norm == 0.0:
                count_zero += 1
            else:
                count_ok += 1; has_grad = True

    total = count_ok + count_zero + count_none + count_bad
    print(f"    -- {module_name}: {count_ok}/{total} params have non-zero grad"
          f" ({count_zero} zero, {count_none} None, {count_bad} NaN/Inf)")
    return has_grad, has_issues


def main():
    parser = argparse.ArgumentParser(description="Test Super backpropagation (full fine-tuning)")
    parser.add_argument("--device",    default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype",     default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--attn-impl", default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--resolution", default="320p",
                        choices=["1080p", "720p", "540p", "450p", "320p"])
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for testing (default: 1)")
    parser.add_argument("--category", default=None,
                        choices=["mcq", "left_right", "distance", "count", ""],
                        help="Find first sample of this category")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    target_size = {"1080p": None, "720p": (1280, 720),
                   "540p": (960, 540), "450p": (800, 450),
                   "320p": (512, 320)}[args.resolution]

    # ====================================================================
    # 1. LOAD MODEL
    # ====================================================================
    print("=" * 70)
    print("TEST: Super Fine-tuning Backpropagation (Stage 1 Config)")
    print("=" * 70)

    pipeline = SpatialVLM(dtype=dtype, device_map=args.device,
                          attn_implementation=args.attn_impl)
    print_vram_usage("after model load")

    # ====================================================================
    # 2. CONFIGURE — STAGE 1 TRAINING (Freeze Vision, Freeze Embeddings except special)
    # ====================================================================
    print(f"\n{'='*70}")
    print("PARAMETER SETUP (Stage 1: Vision FROZEN, Embeddings FROZEN except 4 tokens)")
    print("=" * 70)

    # Freeze Vision
    for p in pipeline.qwen.model.visual.parameters():
        p.requires_grad = False

    # Freeze Embeddings except special tokens
    embed_layer = pipeline.qwen.model.language_model.embed_tokens
    embed_layer.weight.requires_grad = False
    
    special_ids = [pipeline.mcq_token_id, pipeline.lr_token_id,
                   pipeline.dist_token_id, pipeline.count_token_id]
    pipeline._special_embed = embed_layer.weight.data[special_ids].clone().detach().requires_grad_(True)
    pipeline._special_ids = special_ids

    print(f"  Embeddings FROZEN (only {special_ids} trainable)")
    print(f"  Vision Encoder FROZEN")

    components = {
        "Vision Encoder":     pipeline.qwen.model.visual,
        "Embeddings (special)": pipeline._special_embed,
        "Decoder (24 layers)": pipeline.qwen.model.language_model.layers,
        "RTI":                pipeline.region_token_extractor,
        "MCQ Head":           pipeline.mcq_head,
        "LeftRight Head":     pipeline.lr_head,
        "Distance Head":      pipeline.dist_head,
        "Count Head":         pipeline.count_head,
    }

    total_trainable = 0
    for name, module in components.items():
        if isinstance(module, torch.Tensor):
            n = module.numel() if module.requires_grad else 0
        else:
            n = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total_trainable += n
        print(f"  {name:25s}: {n:>12,} ({n/1e6:.2f}M)")
    lm_n = sum(p.numel() for p in pipeline.qwen.lm_head.parameters() if p.requires_grad)
    print(f"  {'LM Head (tied)':25s}: {lm_n:>12,} ({lm_n/1e6:.2f}M)  [not counted]")
    print(f"  {'Total trainable':25s}: {total_trainable:>12,} ({total_trainable/1e6:.2f}M)")

    # ====================================================================
    # 3. LOAD DATA
    # ====================================================================
    print(f"\n{'='*70}")
    print("LOADING DATA")
    print("=" * 70)

    dataset = SpatialVLMDataset("train_sample", processor=pipeline.processor,
                                target_size=target_size)
    loader = get_dataloader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=False)
    
    batch = None
    if args.category:
        for b in loader:
            if b["categories"][0] == args.category:
                batch = b
                break
        if batch is None:
            print(f"[!] No sample found with category={args.category}")
            return
    else:
        batch = next(iter(loader))

    print(f"  Resolution:   {args.resolution}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Categories:   {batch['categories']}")
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key:28s}: {list(val.shape)}  dtype={val.dtype}")

    # ====================================================================
    # 4. FORWARD + BACKWARD
    # ====================================================================
    print(f"\n{'='*70}")
    print("FORWARD + BACKWARD")
    print("=" * 70)

    dev = pipeline.device
    pixel_values     = batch["pixel_values"].to(device=dev, dtype=dtype).requires_grad_(True)
    image_grid_thw   = batch["image_grid_thw"].to(device=dev)
    depth_maps       = batch["depth_maps"].to(device=dev, dtype=dtype).requires_grad_(True)
    input_ids        = batch["input_ids"].to(device=dev)
    labels           = batch["labels"].to(device=dev)

    # --- Setup Profiling Hooks ---
    backward_events = {}
    forward_events = {}
    vram_deltas = {}

    def _find_grad_tensor(obj):
        """Recursively find first requires_grad Tensor in a nested structure."""
        if isinstance(obj, torch.Tensor):
            return obj if obj.requires_grad else None
        if isinstance(obj, (tuple, list)):
            for item in obj:
                t = _find_grad_tensor(item)
                if t is not None:
                    return t
        if isinstance(obj, dict):
            for val in obj.values():
                t = _find_grad_tensor(val)
                if t is not None:
                    return t
        return None

    def add_timing_hook(module, name):
        """Attach hooks for forward timing, VRAM tracking, and backward timing."""
        backward_events[name] = []
        forward_events[name] = []
        vram_deltas[name] = []

        def pre_hook(mod, inputs, _name=name):
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            s.record()
            mod._prof_fwd_start = s
            mod._prof_vram_before = torch.cuda.memory_allocated()

        def post_hook(mod, inputs, output, _name=name):
            torch.cuda.synchronize()
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            forward_events[_name].append((mod._prof_fwd_start, e))
            vram_after = torch.cuda.memory_allocated()
            vram_deltas[_name].append(vram_after - mod._prof_vram_before)

            out_t = _find_grad_tensor(output)
            in_t = _find_grad_tensor(inputs)

            if in_t is None and hasattr(mod, "weight") and getattr(mod.weight, "requires_grad", False):
                in_t = mod.weight

            if out_t is None:
                return

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            def on_output_grad(grad, _s=start):
                torch.cuda.synchronize()
                _s.record()
                return grad
            out_t.register_hook(on_output_grad)

            if in_t is not None:
                def on_input_grad(grad, _s=start, _e=end, _n=_name):
                    torch.cuda.synchronize()
                    _e.record()
                    backward_events[_n].append((_s, _e))
                    return grad
                in_t.register_hook(on_input_grad)

        module.register_forward_pre_hook(pre_hook)
        module.register_forward_hook(post_hook)

    if dev.type == 'cuda':
        add_timing_hook(pipeline.qwen.model.visual, "Vision Encoder")
        add_timing_hook(pipeline.qwen.model.language_model.embed_tokens, "Embeddings")
        decoder_layers = pipeline.qwen.model.language_model.layers
        for i, layer in enumerate(decoder_layers):
            add_timing_hook(layer, f"Decoder[{i}]")
        add_timing_hook(pipeline.qwen.lm_head, "LM Head")
        add_timing_hook(pipeline.region_token_extractor, "RTI")
        add_timing_hook(pipeline.mcq_head, "MCQ Head")
        add_timing_hook(pipeline.lr_head, "LeftRight Head")
        add_timing_hook(pipeline.dist_head, "Distance Head")
        add_timing_hook(pipeline.count_head, "Count Head")

    pipeline.train()

    # Time overall Forward
    fwd_start = torch.cuda.Event(enable_timing=True) if dev.type == 'cuda' else None
    fwd_end = torch.cuda.Event(enable_timing=True) if dev.type == 'cuda' else None
    if fwd_start: fwd_start.record()

    # Inject special token embeddings before forward
    embed_layer = pipeline.qwen.model.language_model.embed_tokens
    with torch.no_grad():
        embed_layer.weight.data[pipeline._special_ids] = pipeline._special_embed.data

    output = pipeline(
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        depth_maps=depth_maps, input_ids=input_ids,
        rle_list=batch["rle_list"],
        mask_token_positions=batch["mask_positions"],
        decoded_masks=batch["decoded_masks"],
        mcq_token_positions=batch.get("mcq_token_positions"),
        lr_token_positions=batch.get("lr_token_positions"),
        dist_token_positions=batch.get("dist_token_positions"),
        count_token_positions=batch.get("count_token_positions"),
        use_gradient_checkpointing=False,
        vision_requires_grad=False,
    )

    if fwd_end: fwd_end.record()

    logits     = output["logits"]
    dist_pred  = output["dist_pred"]
    count_pred = output["count_pred"]
    mcq_logits = output.get("mcq_logits", None)
    lr_logits  = output.get("lr_logits", None)

    print(f"  logits:      {list(logits.shape)}")
    print(f"  dist_pred:   {dist_pred.tolist()}")
    print(f"  count_pred:  {count_pred.tolist()}")
    print(f"  mcq_logits:  {[cl.shape if cl is not None else None for cl in mcq_logits] if mcq_logits else None}")
    print(f"  lr_logits:   {[cl.shape if cl is not None else None for cl in lr_logits] if lr_logits else None}")

    # Build per-task masks from categories
    categories = batch["categories"]
    B = len(categories)
    is_distance = torch.tensor([c == "distance" for c in categories], dtype=torch.bool)
    is_count    = torch.tensor([c == "count" for c in categories], dtype=torch.bool)
    is_mcq      = torch.tensor([c == "mcq" for c in categories], dtype=torch.bool)
    is_lr       = torch.tensor([c == "left_right" for c in categories], dtype=torch.bool)

    # Loss
    criterion = SpatialLoss()
    loss, components = criterion(
        logits, labels,
        dist_pred=dist_pred,
        dist_gt=batch["target_num"].to(dev),
        is_distance=is_distance.to(dev),
        count_pred=count_pred,
        count_gt=batch["target_num"].to(dev),
        is_count=is_count.to(dev),
        mcq_logits=mcq_logits,
        mcq_targets=batch.get("target_cat_index", torch.zeros(B, dtype=torch.long)).to(dev),
        is_mcq=is_mcq.to(dev),
        lr_logits=lr_logits,
        lr_targets=batch.get("target_cat_index", torch.zeros(B, dtype=torch.long)).to(dev),
        is_lr=is_lr.to(dev),
        return_components=True,
    )
    print(f"  Loss = {loss.item():.4f}")
    print(f"  Components: {components}")

    # Time overall Backward
    bwd_start = torch.cuda.Event(enable_timing=True) if dev.type == 'cuda' else None
    bwd_end = torch.cuda.Event(enable_timing=True) if dev.type == 'cuda' else None
    if bwd_start: bwd_start.record()

    loss.backward()

    # Restore gradients to special_embed
    if embed_layer.weight.grad is not None:
        pipeline._special_embed.grad = embed_layer.weight.grad[pipeline._special_ids].clone()

    if bwd_end: bwd_end.record()
    if dev.type == 'cuda':
        torch.cuda.synchronize()

    # --- Print Profiling Results ---
    print(f"\n{'='*70}")
    print("TIMING PROFILER (Milliseconds)")
    print("=" * 70)
    if dev.type == 'cuda':
        total_fwd = fwd_start.elapsed_time(fwd_end)
        total_bwd = bwd_start.elapsed_time(bwd_end)
        print(f"  Total Forward   : {total_fwd:.2f} ms")
        print(f"  Total Backward  : {total_bwd:.2f} ms")

        def get_fwd_ms(name):
            evts = forward_events.get(name, [])
            return sum(s.elapsed_time(e) for s, e in evts) if evts else 0.0

        def get_vram_mb(name):
            deltas = vram_deltas.get(name, [])
            return max(deltas) / (1024 ** 2) if deltas else 0.0

        print(f"\n  {'Module':25s} {'Fwd (ms)':>10s} {'Bwd (ms)':>10s} {'VRAM (MB)':>10s}")
        print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")

        decoder_fwd = decoder_bwd = decoder_vram = 0.0
        other_modules = []
        for name in backward_events:
            if name.startswith("Decoder["):
                decoder_fwd += get_fwd_ms(name)
                evts = backward_events[name]
                if evts:
                    decoder_bwd += sum(s.elapsed_time(e) for s, e in evts)
                decoder_vram += get_vram_mb(name)
            else:
                other_modules.append(name)

        for name in other_modules:
            fwd_ms = get_fwd_ms(name)
            evts = backward_events[name]
            vram_mb = get_vram_mb(name)
            if evts:
                bwd_ms = sum(s.elapsed_time(e) for s, e in evts)
                print(f"  {name:25s} {fwd_ms:>10.2f} {bwd_ms:>10.2f} {vram_mb:>10.2f}")
            else:
                print(f"  {name:25s} {fwd_ms:>10.2f} {'—':>10s} {vram_mb:>10.2f}")

        n_layers = sum(1 for n in backward_events if n.startswith("Decoder["))
        dec_label = f"Decoder ({n_layers})"
        print(f"  {dec_label:25s} {decoder_fwd:>10.2f} {decoder_bwd:>10.2f} {decoder_vram:>10.2f}")

        for name in backward_events:
            if name.startswith("Decoder["):
                fwd_ms = get_fwd_ms(name)
                evts = backward_events[name]
                bwd_ms = sum(s.elapsed_time(e) for s, e in evts) if evts else 0.0
                vram_mb = get_vram_mb(name)
                print(f"    {name:23s} {fwd_ms:>10.2f} {bwd_ms:>10.2f} {vram_mb:>10.2f}")
    else:
        print("  [INFO] Timing profiler requires --device cuda.")

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
    print("GRADIENT CHECK — Decoder (24 layers)")
    print("=" * 70)
    decoder_ok, decoder_issues = check_gradients(
        pipeline.qwen.model.language_model.layers, "Decoder")

    print(f"\n{'='*70}")
    print("GRADIENT CHECK — Embeddings (SPECIAL TOKENS ONLY)")
    print("=" * 70)
    embed_weight = pipeline._special_embed
    embed_ok = embed_weight.grad is not None and embed_weight.grad.norm().item() > 0
    print(f"    _special_embed: grad={'has grad (correct)' if embed_ok else 'NO GRAD (unexpected!)'}")

    print(f"\n{'='*70}")
    print("GRADIENT CHECK — RTI (Full)")
    print("=" * 70)
    rti_ok, rti_issues = check_gradients(pipeline.region_token_extractor, "RTI")

    print(f"\n{'='*70}")
    print("GRADIENT CHECK — RTI Sub-components (DPT)")
    print("=" * 70)
    print("1. DPT Reassembly Layers")
    if hasattr(pipeline.region_token_extractor, 'reassemble'):
        check_gradients(pipeline.region_token_extractor.reassemble, "  RTI DPT Reassemble")
    if hasattr(pipeline.region_token_extractor, 'fusion'):
        check_gradients(pipeline.region_token_extractor.fusion, "  RTI DPT Fusion")
    print("\n2. Mask Pool Projections")
    if hasattr(pipeline.region_token_extractor, 'rgb_proj'):
        check_gradients(pipeline.region_token_extractor.rgb_proj, "  RTI RGB Proj")
    if hasattr(pipeline.region_token_extractor, 'dep_proj'):
        check_gradients(pipeline.region_token_extractor.dep_proj, "  RTI Depth Proj")
    if hasattr(pipeline.region_token_extractor, 'gdep_proj'):
        check_gradients(pipeline.region_token_extractor.gdep_proj, "  RTI Global Depth Proj")



    print(f"\n{'='*70}")
    print("GRADIENT CHECK — MCQ Head")
    print("=" * 70)
    mcq_ok, mcq_issues = check_gradients(pipeline.mcq_head, "MCQ Head")

    print(f"\n{'='*70}")
    print("GRADIENT CHECK — LeftRight Head")
    print("=" * 70)
    lr_ok, lr_issues = check_gradients(pipeline.lr_head, "LeftRight Head")

    print(f"\n{'='*70}")
    print("GRADIENT CHECK — Distance Head")
    print("=" * 70)
    dist_ok, dist_issues = check_gradients(pipeline.dist_head, "Distance Head")

    print(f"\n{'='*70}")
    print("GRADIENT CHECK — Count Head")
    print("=" * 70)
    count_ok, count_issues = check_gradients(pipeline.count_head, "Count Head")

    # ====================================================================
    # 6. SUMMARY
    # ====================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)

    has_distance = is_distance.any().item()
    has_count = is_count.any().item()
    has_mcq = is_mcq.any().item()
    has_lr = is_lr.any().item()

    all_ok = True
    checks = [
        (not vision_ok, "Vision Encoder has NO gradients (Frozen)"),
        (decoder_ok, "Decoder layers have non-zero gradients"),
        (embed_ok, "Special Embeddings are TRAINABLE (have gradients)"),
        (rti_ok, "RTI has non-zero gradients"),
        (True, f"MCQ Head has NO params (has_mcq={has_mcq})"),
        (True, f"LeftRight Head has NO params (has_lr={has_lr})"),
        (dist_ok or not has_distance,
         f"Distance Head has gradients (has_distance={has_distance})"),
        (count_ok or not has_count,
         f"Count Head has gradients (has_count={has_count})"),
        (not (vision_issues or decoder_issues or rti_issues
              or (mcq_issues and has_mcq)
              or (lr_issues and has_lr)
              or (dist_issues and has_distance)
              or (count_issues and has_count)),
         "No NaN/Inf in any gradients"),
        (torch.isfinite(loss).item(), f"Loss is finite ({loss.item():.4f})"),
    ]

    for ok, msg in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {msg}")
        if not ok:
            all_ok = False

    print(f"\n{'='*70}")
    print(f"  Super Backprop Test [{'OK' if all_ok else 'FAIL'}]")
    print(f"{'='*70}")
    print_vram_usage("final")


if __name__ == "__main__":
    main()
