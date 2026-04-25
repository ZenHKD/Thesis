"""
Evaluate SpatialVLM Micro on a full dataset split.

Usage:
    python src/train_micro/evaluation.py --checkpoint checkpoints/micro/epoch_2 --split val
"""

import sys
import os
import argparse
import torch
import re as _re
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model_micro.pipeline import SpatialVLM, print_vram_usage, find_mask_positions, NUM_TOKEN_ID, CAT_TOKEN_ID
from src.dataloader.dataloader import SpatialVLMDataset

# Paths
ROOT     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CKPT_DIR = os.path.join(ROOT, "checkpoints", "micro")

# =========================================================================
# Checkpoint Loading
# =========================================================================

def _fix_checkpoint_state_dict(state_dict: dict) -> dict:
    cleaned = {}
    n_stripped = 0
    for k, v in state_dict.items():
        new_k = k.replace("._orig_mod.", ".").replace("_orig_mod.", "")
        if new_k != k:
            n_stripped += 1
        cleaned[new_k] = v
    if n_stripped > 0:
        print(f"  [*] Stripped _orig_mod. prefix from {n_stripped} keys")

    embed_key = "qwen.model.language_model.embed_tokens.weight"
    lm_head_key = "qwen.lm_head.weight"
    if embed_key in cleaned and lm_head_key not in cleaned:
        cleaned[lm_head_key] = cleaned[embed_key]
        print(f"  [*] Restored tied weight: {lm_head_key} <- {embed_key}")

    return cleaned


def load_checkpoint_weights(pipeline, path: str):
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
        
        state_path = os.path.join(path, "training_state.pt")
        if os.path.exists(state_path):
            ckpt = torch.load(state_path, map_location="cpu", weights_only=True)
            info = ckpt.get("step", "?"), ckpt.get("epoch", "?"), ckpt.get("loss", "?")
    else:
        ckpt_path = os.path.join(path, "checkpoint.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found in {path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model_state = ckpt["model_state_dict"]
        model_state = _fix_checkpoint_state_dict(model_state)
        missing, unexpected = pipeline.load_state_dict(model_state, strict=False)
        info = ckpt.get("step", "?"), ckpt.get("epoch", "?"), ckpt.get("loss", "?")

    epoch_str = f"{info[1]:.4f}" if isinstance(info[1], (float, int)) else str(info[1])
    loss_str = f"{info[2]:.6f}" if isinstance(info[2], (float, int)) else str(info[2])
    print(f"  Loaded: step={info[0]}, epoch={epoch_str}, loss={loss_str}")
    return info


# =========================================================================
# Inference Runner
# =========================================================================

def run_inference(pipeline, sample: dict, do_sample: bool = False, top_p: float = 0.9, top_k: int = 50, max_new_tokens: int = 150, temperature: float = 1.0, repetition_penalty: float = 1.0) -> dict:
    dev   = pipeline.device
    dtype = next(pipeline.qwen.parameters()).dtype

    pixel_values   = sample["pixel_values"].to(device=dev, dtype=dtype)
    pixel_values_rgb = sample["pixel_values_rgb"].unsqueeze(0).to(device=dev, dtype=dtype)
    image_grid_thw = sample["image_grid_thw"].to(device=dev)
    depth_map      = sample["depth_map"].unsqueeze(0).to(device=dev, dtype=dtype)

    question = sample["_question"]

    import re
    mask_idx = [0]
    def replace_mask(m):
        i = mask_idx[0]
        mask_idx[0] += 1
        return f"[Region {i}]: <|object_ref_start|>{m.group(1)}<|object_ref_end|>"
    question = re.sub(r'(<mask.*?>)', replace_mask, question)

    sys_str = (
        "<|im_start|>system\n"
        "You are an expert AI assistant for warehouse spatial reasoning. "
        "Analyze the image and the specific object regions carefully. "
        "First, output your step-by-step reasoning inside <think></think> tags. "
        "Then, output your answer using EXACTLY one of these formats:\n"
        "  mcq | <|cat|>\n"
        "  left_right | <|cat|>\n"
        "  distance | <|num|>\n"
        "  count | <|num|><|im_end|>\n"
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

    mask_positions = find_mask_positions(input_ids, pipeline.processor.tokenizer)

    rle_list = sample["rle_list"]
    decoded_masks = sample["decoded_masks"]

    n = min(len(mask_positions), len(rle_list))
    mask_positions = mask_positions[:n]
    rle_list = rle_list[:n]
    decoded_masks = decoded_masks[:n]

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

    num_pred_val = None
    cat_pred_idx = None

    _clean_pattern = _re.compile(
        r'^<think>.+?</think>\s*(distance|count)\s*\|\s*<\|num\|>$', _re.DOTALL
    )
    _cat_pattern = _re.compile(
        r'^<think>.+?</think>\s*(mcq|left_right)\s*\|\s*<\|cat\|>$', _re.DOTALL
    )

    is_clean = bool(_clean_pattern.match(raw_full))
    is_cat_clean = bool(_cat_pattern.match(raw_full))

    if is_clean and parsed["category"] in ("distance", "count"):
        full_generated_ids = torch.cat([input_ids, output_ids], dim=1)
        gen_ids_list = full_generated_ids[0].tolist()
        num_token_pos = -1
        
        for idx_pos in range(len(gen_ids_list) - 1, -1, -1):
            if gen_ids_list[idx_pos] == NUM_TOKEN_ID:
                num_token_pos = idx_pos
                break

        if num_token_pos >= 0:
            output = pipeline(
                pixel_values=pixel_values,
                pixel_values_rgb=pixel_values_rgb,
                image_grid_thw=image_grid_thw,
                depth_maps=depth_map,
                input_ids=full_generated_ids,
                rle_list=[rle_list],
                mask_token_positions=[mask_positions],
                decoded_masks=[decoded_masks],
                num_token_positions=[num_token_pos],
            )
            num_pred_val = output["num_pred"][0].item()

    elif is_cat_clean and parsed["category"] in ("mcq", "left_right"):
        full_generated_ids = torch.cat([input_ids, output_ids], dim=1)
        gen_ids_list = full_generated_ids[0].tolist()
        cat_token_pos = -1
        
        for idx_pos in range(len(gen_ids_list) - 1, -1, -1):
            if gen_ids_list[idx_pos] == CAT_TOKEN_ID:
                cat_token_pos = idx_pos
                break

        if cat_token_pos >= 0:
            output = pipeline(
                pixel_values=pixel_values,
                pixel_values_rgb=pixel_values_rgb,
                image_grid_thw=image_grid_thw,
                depth_maps=depth_map,
                input_ids=full_generated_ids,
                rle_list=[rle_list],
                mask_token_positions=[mask_positions],
                decoded_masks=[decoded_masks],
                cat_token_positions=[cat_token_pos],
            )
            cat_logits_out = output.get("cat_logits")
            if cat_logits_out and cat_logits_out[0] is not None:
                cat_pred_idx = cat_logits_out[0].argmax().item()

    return {
        "category":   parsed["category"],
        "answer":     parsed["answer"],
        "num_pred":   num_pred_val,
        "cat_pred":   cat_pred_idx,
        "raw":        raw_output,
        "think":      think_str,
    }


# =========================================================================
# Main Evaluation Loop
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate SpatialVLM Micro")
    parser.add_argument("--checkpoint",     type=str, default=None, required=True,
                        help="Path to full checkpoint dir (e.g. checkpoints/micro/epoch_1)")
    parser.add_argument("--split",          default="val",
                        choices=["train", "val", "test", "train_sample"])
    parser.add_argument("--device",         default="cuda", choices=["cuda"])
    parser.add_argument("--attn-impl",      default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--resolution",     type=int, default=320)
    args = parser.parse_args()

    print("=" * 70)
    print("EVALUATION: SpatialVLM Micro")
    print("=" * 70)

    target_size = (args.resolution, args.resolution) if args.resolution else None

    # Load Model
    pipeline = SpatialVLM(
        dtype=torch.bfloat16,
        device_map=args.device,
        attn_implementation=args.attn_impl,
    )
    
    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint {ckpt_path} not found.")
        sys.exit(1)

    print(f"\n  Loading checkpoint...")
    load_checkpoint_weights(pipeline, ckpt_path)
    pipeline.eval()

    # Load dataset
    print(f"\n{'='*70}")
    print(f"LOADING DATASET  (split={args.split}, resolution={args.resolution})")
    print(f"{'='*70}")

    dataset = SpatialVLMDataset(
        args.split,
        processor=pipeline.processor,
        target_size=target_size,
    )
    
    N = len(dataset)
    print(f"  Total validation entries: {N}")

    samples = []
    for idx in tqdm(range(N), desc="Loading items into memory"):
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

    print(f"\n{'='*70}")
    print(f"EVALUATING SAMPLES")
    print(f"{'='*70}")

    results_by_category = {}
    correct = 0
    
    categories = ["count", "distance", "mcq", "left_right"]
    confusion_matrix = {t: {p: 0 for p in categories + ["unknown"]} for t in categories}

    for idx, s in enumerate(tqdm(samples, desc="Inference Progress")):
        cat = s["category"]
        gt = s["answer"]
        if gt.startswith("<|num|>="):
            gt_clean = gt[8:]
        elif gt.startswith("<|cat|>="):
            gt_clean = gt[8:]
        else:
            gt_clean = gt.strip('"')

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        try:
            with torch.no_grad():
                result = run_inference(pipeline, s)
        except Exception as e:
            result = {"category": "error", "answer": None, "raw": str(e), "num_pred": None}

        # Confusion Matrix Tracking
        pred_cat = result.get("category", "unknown")
        if pred_cat not in categories:
            pred_cat = "unknown"
        confusion_matrix[cat][pred_cat] += 1

        # Accuracy Tracking (Requires matching categorical prediction)
        match = False
        if pred_cat == cat:
            pred_answer = str(result.get("answer", "")).strip('"')
            num_pred = result.get("num_pred")
            cat_pred = result.get("cat_pred")

            if cat in ("distance", "count"):
                if num_pred is not None:
                    try:
                        gt_num = float(gt_clean)
                        if cat == "count":
                            num_pred = round(num_pred)
                            match = (num_pred == round(gt_num))
                        else:
                            rel_err = abs(num_pred - gt_num) / (abs(gt_num) + 1e-6)
                            match = rel_err < 0.10
                    except (ValueError, TypeError):
                        match = False
            elif cat == "mcq":
                if cat_pred is not None:
                    try:
                        gt_int = round(float(gt_clean))
                        match = cat_pred == gt_int
                    except (ValueError, TypeError):
                        match = False
                else:
                    try:
                        pred_int = round(float(pred_answer))
                        gt_int = round(float(gt_clean))
                        match = pred_int == gt_int
                    except (ValueError, TypeError):
                        match = pred_answer == gt_clean
            elif cat == "left_right":
                if cat_pred is not None:
                    pred_lr = "left" if cat_pred == 0 else "right"
                    match = pred_lr == gt_clean
                else:
                    match = pred_answer == gt_clean
            else:
                match = pred_answer == gt_clean

        if match:
            correct += 1

        if cat not in results_by_category:
            results_by_category[cat] = {"correct": 0, "total": 0}
        results_by_category[cat]["total"] += 1
        if match:
            results_by_category[cat]["correct"] += 1

    # SUMMARY
    print(f"\n{'='*70}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Overall Category+Answer Correct: {correct}/{N} ({correct/max(N,1)*100:.1f}%)")

    if results_by_category:
        print(f"\n  {'Category':<12}  {'Correct':>8}  {'Total':>6}  {'Accuracy':>8}")
        print(f"  {'-'*40}")
        for cat in sorted(results_by_category.keys()):
            r = results_by_category[cat]
            acc = r["correct"] / max(r["total"], 1) * 100
            print(f"  {cat:<12}  {r['correct']:>8}  {r['total']:>6}  {acc:>7.1f}%")

    print(f"\n{'='*70}")
    print(f"CATEGORY CONFUSION MATRIX (Row=GT, Col=Pred)")
    print(f"{'='*70}")
    
    header = f"  {'GT / Pred':<12} | " + " | ".join(f"{c:>10}" for c in categories + ["unknown"])
    print(header)
    print("  " + "-" * len(header))
    
    for gt_cat in categories:
        row_counts = [confusion_matrix[gt_cat][p] for p in categories + ["unknown"]]
        row_str = f"  {gt_cat:<12} | " + " | ".join(f"{count:>10}" for count in row_counts)
        print(row_str)

    print("\n")
    print_vram_usage("final")

if __name__ == "__main__":
    main()
