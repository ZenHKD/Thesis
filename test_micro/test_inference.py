"""
Test SpatialVLM Micro inference.

Usage:
    # No checkpoint (untrained pruned model):
    python test_micro/test_inference.py

    # With checkpoint:
    python test_micro/test_inference.py --step 20000

    # More options:
    python test_micro/test_inference.py --step 20000 --num-samples 10 --split val
    python test_micro/test_inference.py --step 20000 --sample-idx 0 5 10 15 20
"""

import sys
import os
import argparse
import torch
import re as _re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_micro.pipeline import SpatialVLM, print_vram_usage, find_mask_positions
from src.dataloader.dataloader import SpatialVLMDataset

# Paths
ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(ROOT, "checkpoints", "micro")


# =========================================================================
# Checkpoint Loading
# =========================================================================

def _fix_checkpoint_state_dict(state_dict: dict) -> dict:
    """Fix checkpoint state dict for loading into an uncompiled model.

    Handles two issues:
    1. torch.compile() wraps parameter names with `_orig_mod.` prefix — strip it.
    2. save_checkpoint deduplicates tied tensors (embed_tokens ↔ lm_head share
       the same memory pointer), so lm_head.weight is missing — restore it.
    """
    cleaned = {}
    n_stripped = 0
    for k, v in state_dict.items():
        new_k = k.replace("._orig_mod.", ".").replace("_orig_mod.", "")
        if new_k != k:
            n_stripped += 1
        cleaned[new_k] = v
    if n_stripped > 0:
        print(f"  [*] Stripped _orig_mod. prefix from {n_stripped} keys (torch.compile checkpoint)")

    # Restore tied lm_head.weight from embed_tokens.weight if missing
    embed_key = "qwen.model.language_model.embed_tokens.weight"
    lm_head_key = "qwen.lm_head.weight"
    if embed_key in cleaned and lm_head_key not in cleaned:
        cleaned[lm_head_key] = cleaned[embed_key]
        print(f"  [*] Restored tied weight: {lm_head_key} <- {embed_key}")

    return cleaned


def load_checkpoint_weights(pipeline, path: str):
    """Load only model weights from a training checkpoint (no optimizer).

    Handles torch.compile() checkpoints by stripping `_orig_mod.` prefix.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint directory not found: {path}")

    print(f"  Loading checkpoint from: {path}")
    
    model_path = os.path.join(path, "model.safetensors")
    if os.path.exists(model_path):
        from safetensors.torch import load_file
        model_state = load_file(model_path)
        model_state = _fix_checkpoint_state_dict(model_state)
        missing, unexpected = pipeline.load_state_dict(model_state, strict=False)
        info = ("?", "?", "?")
        
        # Try to read info from training_state.pt
        state_path = os.path.join(path, "training_state.pt")
        if os.path.exists(state_path):
            ckpt = torch.load(state_path, map_location="cpu", weights_only=True)
            info = ckpt.get("step", "?"), ckpt.get("epoch", "?"), ckpt.get("loss", "?")
    else:
        # Fallback to .pt
        ckpt_path = os.path.join(path, "checkpoint.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found in {path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model_state = ckpt["model_state_dict"]
        model_state = _fix_checkpoint_state_dict(model_state)
        missing, unexpected = pipeline.load_state_dict(model_state, strict=False)
        info = ckpt.get("step", "?"), ckpt.get("epoch", "?"), ckpt.get("loss", "?")

    if missing:
        print(f"  Warning: Missing keys: {len(missing)}")
        for k in missing[:5]:
            print(f"    - {k}")
    if unexpected:
        print(f"  Warning: Unexpected keys: {len(unexpected)}")
        for k in unexpected[:5]:
            print(f"    - {k}")

    # Format epoch/loss nicely if they are numbers
    epoch_str = f"{info[1]:.4f}" if isinstance(info[1], (float, int)) else str(info[1])
    loss_str = f"{info[2]:.6f}" if isinstance(info[2], (float, int)) else str(info[2])
    print(f"  Loaded: step={info[0]}, epoch={epoch_str}, loss={loss_str}")
    return info


# =========================================================================
# Inference Runner
# =========================================================================

def run_inference(pipeline, sample: dict, do_sample: bool = False, top_p: float = 0.9, top_k: int = 50, max_new_tokens: int = 150, temperature: float = 1.0, repetition_penalty: float = 1.0) -> dict:
    """Run inference on a single dataloader sample.

    Uses pipeline.generate() which always runs T_max loops.

    For distance/count: also runs a forward pass to get num_pred from
    the Number Head.
    """
    dev   = pipeline.device
    dtype = next(pipeline.qwen.parameters()).dtype

    # pixel_values: [num_patches, C] — do NOT unsqueeze (Qwen visual expects 2D)
    # image_grid_thw: [1, 3] — already has image dim
    # depth_map: [H, W] -> [1, H, W] — needs batch dim
    pixel_values   = sample["pixel_values"].to(device=dev, dtype=dtype)
    pixel_values_rgb = sample["pixel_values_rgb"].unsqueeze(0).to(device=dev, dtype=dtype)
    image_grid_thw = sample["image_grid_thw"].to(device=dev)
    depth_map      = sample["depth_map"].unsqueeze(0).to(device=dev, dtype=dtype)

    # Re-tokenize without the answer (prompt only, direct tokenization)
    question = sample["_question"]

    import re
    question = re.sub(r'(<mask.*?>)', r'<|object_ref_start|>\1<|object_ref_end|>', question)

    sys_str = (
        "<|im_start|>system\n"
        "You are an expert AI assistant for warehouse spatial reasoning. "
        "Analyze the image and the specific object regions carefully. "
        "First, output your step-by-step reasoning inside <think></think> tags. "
        "Finally, output your exact answer strictly using the format: 'category | value'.<|im_end|>\n"
    )
    
    h_p, w_p = image_grid_thw[0, 1].item(), image_grid_thw[0, 2].item()
    h_vis, w_vis = h_p // 2, w_p // 2
    num_visual_tokens = int(h_vis * w_vis)
    
    vision_str = "Picture 1: <|vision_start|>" + "<|image_pad|>" * num_visual_tokens + "<|vision_end|>\n"
    user_str = f"<|im_start|>user\n{vision_str}{question}<|im_end|>\n"
    eval_prompt = f"<|im_start|>assistant\n"
    
    full_prompt = sys_str + user_str + eval_prompt

    input_ids = pipeline.processor.tokenizer(
        full_prompt, return_tensors="pt", padding=False
    ).input_ids.to(device=dev)

    # Find <mask> positions in the inference prompt
    mask_positions = find_mask_positions(input_ids, pipeline.processor.tokenizer)

    rle_list = sample["rle_list"]
    decoded_masks = sample["decoded_masks"]

    n = min(len(mask_positions), len(rle_list))
    mask_positions = mask_positions[:n]
    rle_list = rle_list[:n]
    decoded_masks = decoded_masks[:n]

    # Step 1: Generate text tokens (always T_max loops)
    output_ids = pipeline.generate(
        pixel_values, pixel_values_rgb, image_grid_thw, depth_map, input_ids,
        rle_list=[rle_list],
        mask_token_positions=[mask_positions],
        decoded_masks=[decoded_masks],
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )
    raw_full = pipeline.processor.tokenizer.decode(
        output_ids[0], skip_special_tokens=False
    ).replace("<|endoftext|>", "").replace("<|im_end|>", "").strip()
    
    think_match = _re.search(r'<think>.*?</think>', raw_full, flags=_re.DOTALL)
    think_str = think_match.group(0).strip() if think_match else ""
    
    raw_output = _re.sub(r'<think>.*?</think>\s*', '', raw_full, flags=_re.DOTALL).strip()
    parsed = pipeline.parse_output(raw_output)

    # Step 2: For numeric categories, get num_pred via forward pass
    num_pred_val = None

    _clean_pattern = _re.compile(
        r'^<think>.+?</think>\s*(distance|count)\s*\|\s*<\|num\|>$', _re.DOTALL
    )

    # Check raw output before stripping think tags
    raw_full = pipeline.processor.tokenizer.decode(
        output_ids[0], skip_special_tokens=False
    ).replace("<|endoftext|>", "").replace("<|im_end|>", "").strip()
    is_clean = bool(_clean_pattern.match(raw_full))

    if is_clean and parsed["category"] in ("distance", "count"):
        full_input_ids = sample["input_ids"].unsqueeze(0).to(device=dev)
        num_token_pos = sample.get("num_token_pos", -1)

        if num_token_pos >= 0:
            output = pipeline(
                pixel_values=pixel_values,
                pixel_values_rgb=pixel_values_rgb,
                image_grid_thw=image_grid_thw,
                depth_maps=depth_map,
                input_ids=full_input_ids,
                rle_list=[rle_list],
                mask_token_positions=[mask_positions],
                decoded_masks=[decoded_masks],
                num_token_positions=[num_token_pos],
            )
            num_pred_val = output["num_pred"][0].item()

    return {
        "category":   parsed["category"],
        "answer":     parsed["answer"],
        "num_pred":   num_pred_val,
        "raw":        raw_output,
        "think":      think_str,
    }


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Test SpatialVLM Micro inference")
    parser.add_argument("--checkpoint",     type=str, default=None,
                        help="Path to full checkpoint dir (e.g. checkpoints/micro/epoch_1)")
    parser.add_argument("--step",           type=int, default=None,
                        help="Legacy: Checkpoint step to load (e.g. 20000). None = untrained.")
    parser.add_argument("--split",          default="train_sample",
                        choices=["train", "val", "test", "train_sample"])
    parser.add_argument("--device",         default="cuda", choices=["cuda"],
                        help="Device to run on (only cuda is supported)")
    parser.add_argument("--dtype",          default="bfloat16",
                        choices=["bfloat16", "float32"])
    parser.add_argument("--attn-impl",      default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--resolution",     default="320p",
                        choices=["1080p", "720p", "540p", "450p", "320p"])
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--do-sample",      action="store_true", help="Use sampling (debug only) instead of greedy")
    parser.add_argument("--temperature",    type=float, default=1.0, help="Temperature (only used with --do-sample)")
    parser.add_argument("--top-p",          type=float, default=0.9, help="Top-p sampling (only used with --do-sample)")
    parser.add_argument("--top-k",          type=int, default=50, help="Top-k sampling (only used with --do-sample)")
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="Repetition penalty (1.0 = disabled, recommended for strict eval)")
    parser.add_argument("--num-samples",    type=int, default=5,
                        help="Number of samples to test (from start of split)")
    parser.add_argument("--sample-idx",     type=int, nargs="+", default=None,
                        help="Specific sample indices (overrides --num-samples)")
    parser.add_argument("--category",       default=None,
                        choices=["mcq", "left_right", "distance", "count"],
                        help="Filter by category (test all if None)")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    target_size = {"1080p": None, "720p": (1280, 720),
                   "540p": (960, 540), "450p": (800, 450),
                   "320p": (512, 320)}[args.resolution]

    # ------------------------------------------------------------------ #
    # Load model
    # ------------------------------------------------------------------ #
    print(f"{'='*70}")
    print("LOADING MODEL")
    print(f"{'='*70}")

    pipeline = SpatialVLM(
        dtype=dtype,
        device_map=args.device,
        attn_implementation=args.attn_impl,
    )
    print_vram_usage("after model load")

    ckpt_label = "untrained"
    if args.checkpoint is not None:
        print(f"\n  Loading checkpoint: {args.checkpoint}...")
        load_checkpoint_weights(pipeline, args.checkpoint)
        ckpt_label = os.path.basename(args.checkpoint.rstrip("/"))
    elif args.step is not None:
        ckpt_path = os.path.join(CKPT_DIR, f"step_{args.step}")
        print(f"\n  Loading checkpoint step={args.step}...")
        load_checkpoint_weights(pipeline, ckpt_path)
        ckpt_label = f"step_{args.step}"
    else:
        print("  Using untrained pruned model (no checkpoint)")
    pipeline.eval()

    print(f"\n  Decoder Layers: 24 (single pass)")

    # ------------------------------------------------------------------ #
    # Load dataset
    # ------------------------------------------------------------------ #
    print(f"\n{'='*70}")
    print(f"LOADING DATASET  (split={args.split}, resolution={args.resolution})")
    print(f"{'='*70}")

    dataset = SpatialVLMDataset(
        args.split,
        processor=pipeline.processor,
        target_size=target_size,
    )
    print(f"  Total entries: {len(dataset)}")

    # Filter by category if requested
    if args.category:
        indices = [i for i in range(len(dataset))
                   if dataset.data[i].get("category") == args.category]
        print(f"  Filtered to {len(indices)} samples with category={args.category}")
    else:
        indices = list(range(len(dataset)))

    # Select sample indices
    if args.sample_idx is not None:
        selected = [i for i in args.sample_idx if i < len(indices)]
    else:
        selected = indices[:args.num_samples]

    N = len(selected)
    print(f"  Selected {N} samples: {selected[:10]}{'...' if N > 10 else ''}")

    # Load samples
    samples = []
    for idx in selected:
        try:
            s = dataset[idx]
            entry = dataset.data[idx]
            from PIL import Image
            img = Image.open(
                os.path.join(dataset.image_dir, entry["image"])
            ).convert("RGB")
            if target_size:
                img = img.resize(target_size, Image.LANCZOS)
            question_raw = entry["conversations"][0]["value"]
            question = question_raw.replace("<image>\n", "").replace("<image>", "").strip()
            s["_image"] = img
            s["_question"] = question
            s["idx"] = idx
            samples.append(s)
        except Exception as e:
            print(f"  Warning: Failed to load sample {idx}: {e}")

    for s in samples:
        print(f"  [{s['idx']:>5}] cat={s['category']:10s}  "
              f"answer={s['answer']:>12s}  "
              f"masks={len(s['rle_list'])}  image={s['image_name']}")

    # ------------------------------------------------------------------ #
    # Run inference
    # ------------------------------------------------------------------ #
    print(f"\n{'='*70}")
    print(f"INFERENCE  ({N} samples, {ckpt_label})")
    print(f"{'='*70}")

    results_by_category = {}
    correct = 0

    for i, s in enumerate(samples):
        cat = s["category"]
        gt = s["answer"]
        if gt.startswith("<|num|>="):
            gt_clean = gt[8:]
        else:
            gt_clean = gt.strip('"')

        print(f"\n{'--'*35}")
        print(f"  Sample [{s['idx']}]: {s['image_name']}  |  "
              f"{cat}  |  GT={gt}")
        print(f"  Q: {s['_question'][:120]}{'...' if len(s['_question']) > 120 else ''}")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        try:
            with torch.no_grad():
                result = run_inference(
                    pipeline, s,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_new_tokens=args.max_new_tokens,
                    repetition_penalty=args.repetition_penalty,
                )
        except Exception as e:
            print(f"  Warning: Inference failed: {e}")
            import traceback
            traceback.print_exc()
            result = {"category": "error", "answer": None, "raw": str(e)}

        # Compare answers
        pred_answer = str(result.get("answer", "")).strip('"')
        num_pred = result.get("num_pred")

        if cat in ("distance", "count"):
            if num_pred is not None:
                try:
                    gt_num = float(gt_clean)
                    if cat == "count":
                        num_pred = round(num_pred)
                        rel_err = abs(num_pred - gt_num) / (abs(gt_num) + 1e-6)
                        match = (num_pred == round(gt_num))
                        match_detail = f"num_pred={num_pred} gt={round(gt_num)} (Exact Match)"
                    else:
                        rel_err = abs(num_pred - gt_num) / (abs(gt_num) + 1e-6)
                        match = rel_err < 0.10
                        match_detail = f"num_pred={num_pred:.2f} gt={gt_num:.2f} rel_err={rel_err*100:.1f}%"
                except (ValueError, TypeError):
                    match = False
                    match_detail = f"num_pred={num_pred:.2f} (GT not numeric)"
            else:
                match = False
                match_detail = f"text_answer={pred_answer!r} (no num_pred)"
        elif cat == "mcq":
            try:
                pred_int = round(float(pred_answer))
                gt_int = round(float(gt_clean))
                match = pred_int == gt_int
                match_detail = f"pred={pred_int} gt={gt_int}"
            except (ValueError, TypeError):
                match = pred_answer == gt_clean
                match_detail = f"pred={pred_answer!r} gt={gt_clean!r}"
        else:
            match = pred_answer == gt_clean
            match_detail = ""

        match_flag = "[OK]" if match else "[FAIL]"
        if match:
            correct += 1

        if cat not in results_by_category:
            results_by_category[cat] = {"correct": 0, "total": 0}
        results_by_category[cat]["total"] += 1
        if match:
            results_by_category[cat]["correct"] += 1

        print(f"\n  Reasoning:    {result.get('think', '')}")
        print(f"  Raw output:   {result['raw'][:150]}")
        print(f"  Parsed:       category={result['category']!r}  answer={result['answer']!r}")
        if num_pred is not None:
            print(f"  Number Head:  num_pred={num_pred:.4f}")
        print(f"  Ground truth: category={cat!r}  answer={gt!r}")
        print(f"  Match: {match_flag}  {match_detail}")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY  ({ckpt_label})")
    print(f"{'='*70}")
    print(f"\n  Overall: {correct}/{N} correct ({correct/max(N,1)*100:.1f}%)")

    if results_by_category:
        print(f"\n  {'Category':<12}  {'Correct':>8}  {'Total':>6}  {'Accuracy':>8}")
        print(f"  {'--'*25}")
        for cat in sorted(results_by_category.keys()):
            r = results_by_category[cat]
            acc = r["correct"] / max(r["total"], 1) * 100
            print(f"  {cat:<12}  {r['correct']:>8}  {r['total']:>6}  {acc:>7.1f}%")

    print_vram_usage("final")


if __name__ == "__main__":
    main()
