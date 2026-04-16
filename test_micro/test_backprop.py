"""
Test SpatialVLM Micro full pipeline backpropagation.

Full fine-tuning — all components trainable including embeddings.
Loads pruned Micro checkpoint, runs 1 forward + backward on real data,
verifies gradient flow to all trainable components.

Usage:
    python test_micro/test_backprop.py
    python test_micro/test_backprop.py --resolution 320p
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataloader.dataloader import SpatialVLMDataset, get_dataloader
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
    parser.add_argument("--resolution", default="320p",
                        choices=["1080p", "720p", "540p", "450p", "320p"])
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for testing (default: 4)")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    target_size = {"1080p": None, "720p": (1280, 720),
                   "540p": (960, 540), "450p": (800, 450),
                   "320p": (512, 320)}[args.resolution]

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
    print("PARAMETER SETUP (all trainable, full fine-tuning)")
    print("=" * 70)

    # Embeddings are TRAINABLE
    embed_trainable = pipeline.qwen.model.language_model.embed_tokens.weight.requires_grad
    print(f"  Embeddings trainable: {embed_trainable}")

    # Count per component
    components = {
        "Vision Encoder":  pipeline.qwen.model.visual,
        "Embeddings":      pipeline.qwen.model.language_model.embed_tokens,
        "Decoder (24 layers)": pipeline.qwen.model.language_model.layers,
        "RTI":             pipeline.region_token_extractor,
        "Number Head":     pipeline.num_head,
    }

    total_trainable = 0
    for name, module in components.items():
        n = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total_trainable += n
        print(f"  {name:25s}: {n:>12,} ({n/1e6:.2f}M)")
    # LM Head (tied to Embeddings — same weight, not counted in total)
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
    pixel_values   = batch["pixel_values"].to(device=dev, dtype=dtype).requires_grad_(True)
    pixel_values_rgb = batch["pixel_values_rgb"].to(device=dev, dtype=dtype).requires_grad_(True)
    image_grid_thw = batch["image_grid_thw"].to(device=dev)
    depth_maps     = batch["depth_maps"].to(device=dev, dtype=dtype).requires_grad_(True)
    input_ids      = batch["input_ids"].to(device=dev)
    labels         = batch["labels"].to(device=dev)

    # --- Setup Profiling Hooks (Warning-Free) ---
    # Uses register_forward_hook + tensor.register_hook() instead of
    # register_full_backward_hook, which warns on non-Tensor module outputs
    # (e.g., BaseModelOutputWithPooling, list, ModelOutput).
    backward_events = {}
    forward_events = {}    # {name: [(start_event, end_event), ...]}
    vram_deltas = {}       # {name: [delta_bytes, ...]}

    def _find_grad_tensor(obj):
        """Recursively find first requires_grad Tensor in a nested structure."""
        if isinstance(obj, torch.Tensor):
            return obj if obj.requires_grad else None
        if isinstance(obj, (tuple, list)):
            for item in obj:
                t = _find_grad_tensor(item)
                if t is not None:
                    return t
        # Handle transformers ModelOutput (OrderedDict subclass) and dicts
        # Do NOT use hasattr(obj, 'values') — Tensors have .values() (sparse API)
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

        # --- Forward: pre-hook records start event + VRAM before ---
        def pre_hook(mod, inputs, _name=name):
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            s.record()
            # Stash start event + VRAM snapshot on the module temporarily
            mod._prof_fwd_start = s
            mod._prof_vram_before = torch.cuda.memory_allocated()

        # --- Forward: post-hook records end event + VRAM after ---
        def post_hook(mod, inputs, output, _name=name):
            torch.cuda.synchronize()
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            forward_events[_name].append((mod._prof_fwd_start, e))
            vram_after = torch.cuda.memory_allocated()
            vram_deltas[_name].append(vram_after - mod._prof_vram_before)

            # --- Backward timing (tensor-level hooks) ---
            out_t = _find_grad_tensor(output)
            in_t = _find_grad_tensor(inputs)

            # Fallback for modules like Embeddings whose actual inputs (e.g., token IDs)
            # do not require gradients: use their trainable weight to record backward end time.
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
        add_timing_hook(pipeline.num_head, "Number Head")

    pipeline.train()

    # Time overall Forward
    fwd_start = torch.cuda.Event(enable_timing=True) if dev.type == 'cuda' else None
    fwd_end = torch.cuda.Event(enable_timing=True) if dev.type == 'cuda' else None
    if fwd_start: fwd_start.record()

    output = pipeline(
        pixel_values=pixel_values, pixel_values_rgb=pixel_values_rgb, image_grid_thw=image_grid_thw,
        depth_maps=depth_maps, input_ids=input_ids,
        rle_list=batch["rle_list"],
        mask_token_positions=batch["mask_positions"],
        decoded_masks=batch["decoded_masks"],
        num_token_positions=batch.get("num_token_positions"),
        use_gradient_checkpointing=False,
        vision_requires_grad=True,
    )

    if fwd_end: fwd_end.record()

    logits = output["logits"]
    num_pred = output["num_pred"]
    print(f"  logits:          {list(logits.shape)}")
    print(f"  num_pred:        {num_pred.tolist()}")

    # Loss (Uniform CE + SmoothL1)
    criterion = SpatialLoss(alpha=0.1)
    loss = criterion(
        logits, labels,
        num_pred, batch["target_num"].to(dev),
        batch["is_numeric"].to(dev),
    )
    print(f"  Loss = {loss.item():.4f}")
    
    # Time overall Backward
    bwd_start = torch.cuda.Event(enable_timing=True) if dev.type == 'cuda' else None
    bwd_end = torch.cuda.Event(enable_timing=True) if dev.type == 'cuda' else None
    if bwd_start: bwd_start.record()

    loss.backward()

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

        # Helper: get forward ms for a module
        def get_fwd_ms(name):
            evts = forward_events.get(name, [])
            return sum(s.elapsed_time(e) for s, e in evts) if evts else 0.0

        # Helper: get VRAM delta for a module (max across calls, in MB)
        def get_vram_mb(name):
            deltas = vram_deltas.get(name, [])
            return max(deltas) / (1024 ** 2) if deltas else 0.0

        print(f"\n  {'Module':25s} {'Fwd (ms)':>10s} {'Bwd (ms)':>10s} {'VRAM (MB)':>10s}")
        print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")

        # Aggregate decoder layers
        decoder_fwd = 0.0
        decoder_bwd = 0.0
        decoder_vram = 0.0
        decoder_calls = 0
        other_modules = []
        for name in backward_events:
            if name.startswith("Decoder["):
                decoder_fwd += get_fwd_ms(name)
                evts = backward_events[name]
                if evts:
                    decoder_bwd += sum(s.elapsed_time(e) for s, e in evts)
                    decoder_calls += len(evts)
                decoder_vram += get_vram_mb(name)
            else:
                other_modules.append(name)

        # Print non-decoder modules
        for name in other_modules:
            fwd_ms = get_fwd_ms(name)
            evts = backward_events[name]
            vram_mb = get_vram_mb(name)
            if evts:
                bwd_ms = sum(s.elapsed_time(e) for s, e in evts)
                pct = bwd_ms / total_bwd * 100
                print(f"  {name:25s} {fwd_ms:>10.2f} {bwd_ms:>10.2f} {vram_mb:>10.2f}")
            else:
                print(f"  {name:25s} {fwd_ms:>10.2f} {'—':>10s} {vram_mb:>10.2f}")

        # Print aggregated decoder
        n_layers = sum(1 for n in backward_events if n.startswith("Decoder["))
        n_loops = decoder_calls // max(n_layers, 1)
        dec_label = f"Decoder ({n_layers}×{n_loops})"
        print(f"  {dec_label:25s} {decoder_fwd:>10.2f} {decoder_bwd:>10.2f} {decoder_vram:>10.2f}")

        # Per-layer detail
        for name in backward_events:
            if name.startswith("Decoder["):
                fwd_ms = get_fwd_ms(name)
                evts = backward_events[name]
                bwd_ms = sum(s.elapsed_time(e) for s, e in evts) if evts else 0.0
                vram_mb = get_vram_mb(name)
                calls = len(evts) if evts else 0
                print(f"    {name:23s} {fwd_ms:>10.2f} {bwd_ms:>10.2f} {vram_mb:>10.2f}  ({calls} calls)")
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
    print("GRADIENT CHECK — Embeddings (TRAINABLE — should have grad)")
    print("=" * 70)
    embed_weight = pipeline.qwen.model.language_model.embed_tokens.weight
    embed_ok = embed_weight.grad is not None and embed_weight.grad.norm().item() > 0
    print(f"    embed_tokens.weight: grad={'has grad (TRAINABLE, correct)' if embed_ok else 'NO GRAD (unexpected!)'}")

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
        (embed_ok, "Embeddings are TRAINABLE (have gradients)"),
        (rti_ok, "RTI has non-zero gradients"),
        (numhead_ok or not has_numeric,
         f"Number Head has gradients (has_numeric={has_numeric})"),
        (not (vision_issues or decoder_issues
              or rti_issues
              or (numhead_issues and has_numeric)),
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
