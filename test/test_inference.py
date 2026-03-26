"""
Test SpatialVLM pipeline inference with real samples from train_sample.json.

Usage:
    python test/test_inference.py
    python test/test_inference.py --num-samples 10
    python test/test_inference.py --attn-impl sdpa    # SDPA attention
"""

import sys
import os
import json
import re
import argparse
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.pipeline import SpatialVLM, SYSTEM_PROMPT, print_vram_usage

# Paths
DATA_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "nvidia_warehouse_dataset", "train_sample")
JSON_PATH  = os.path.join(DATA_DIR, "train_sample.json")
IMAGE_DIR  = os.path.join(DATA_DIR, "images")
DEPTH_DIR  = os.path.join(DATA_DIR, "depths")



def load_sample(entry: dict) -> dict:
    """Load a single dataset entry -> dict with all tensors/data ready."""
    image_name = entry["image"]
    image_path = os.path.join(IMAGE_DIR, image_name)
    depth_path = os.path.join(DEPTH_DIR, image_name.replace(".png", "_depth.png"))

    # Load RGB image
    image = Image.open(image_path).convert("RGB")

    # Load depth map (8-bit grayscale PNG, mode=L, range 0-255)
    depth_np = np.array(Image.open(depth_path), dtype=np.float32)
    depth_map = torch.from_numpy(depth_np)  # [H, W]

    # Extract question text (remove leading <image>\n if present)
    question_raw = entry["conversations"][0]["value"]
    question = question_raw.replace("<image>\n", "").replace("<image>", "").strip()

    # Find <mask> positions in the tokenized question (will be found after tokenization)
    rle_list = entry["rle"]

    # Format normalized_answer per spec:
    #   mcq -> "5" (quoted int), left_right -> "left" (quoted),
    #   distance -> 5.2 (float), count -> 3 (int)
    category = entry["category"]
    raw_answer = str(entry["normalized_answer"])
    if category in ("mcq", "left_right"):
        normalized_answer = f'"{ raw_answer}"'
    else:
        normalized_answer = raw_answer

    return {
        "image":             image,
        "question":          question,
        "depth_map":         depth_map,
        "rle_list":          rle_list,
        "category":          category,
        "normalized_answer": normalized_answer,
        "freeform_answer":   entry["freeform_answer"],
        "image_name":        image_name,
    }


def find_mask_positions(input_ids: torch.Tensor, tokenizer) -> list:
    """Find token positions of <mask> in input_ids.

    Qwen BPE tokenizes <mask> as 3 subtokens, but the first token varies:
      - isolated: ['<', 'mask', '>']  = [27, 10931, 29]
      - in context: [' <', 'mask', '>'] = [361, 10931, 29]
    We match by decoded text to handle both cases robustly.
    """
    input_list = input_ids[0].tolist()
    decoded = [tokenizer.decode([t]) for t in input_list]

    positions = []
    i = 0
    while i < len(decoded) - 2:
        # Match: token ending with '<' + 'mask' + token starting with '>'
        # The '>' may be merged with following punctuation (e.g. '>,', '> ')
        if (decoded[i].rstrip().endswith('<')
                and decoded[i+1] == 'mask'
                and decoded[i+2].lstrip().startswith('>')):
            positions.append(i)
            i += 3  # skip past this <mask>
        else:
            i += 1
    return positions


def main():
    parser = argparse.ArgumentParser(description="Test SpatialVLM with 5 real samples")
    parser.add_argument("--device",    default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype",     default="bfloat16", choices=["bfloat16", "float32"])

    parser.add_argument("--attn-impl", default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of samples to test (default: 5)")
    args = parser.parse_args()

    N_SAMPLES = args.num_samples

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    # Load dataset
    print(f"{'='*70}")
    print("LOADING DATASET")
    print(f"{'='*70}")
    with open(JSON_PATH, "r") as f:
        dataset = json.load(f)
    print(f"  Total samples in JSON: {len(dataset)}")
    print(f"  Using first {N_SAMPLES} samples")

    samples = [load_sample(dataset[i]) for i in range(N_SAMPLES)]
    print(f"  Loaded {len(samples)} samples [OK]")
    for i, s in enumerate(samples):
        print(f"  [{i}] cat={s['category']:10s}  answer={s['normalized_answer']:6s}  "
              f"masks={len(s['rle_list'])}  image={s['image_name']}")

    # Load model
    print(f"\n{'='*70}")
    print("LOADING MODEL")
    print(f"{'='*70}")
    pipeline = SpatialVLM(
        dtype=dtype,
        device_map=args.device,
        attn_implementation=args.attn_impl,
    )
    print_vram_usage("after model load")

    # Run inference
    print(f"\n{'='*70}")
    print(f"INFERENCE ({N_SAMPLES} samples)")
    print(f"{'='*70}")

    dev = pipeline.device
    tokenizer = pipeline.processor.tokenizer
    correct = 0

    for i, s in enumerate(samples):
        print(f"\n{'-'*70}")
        print(f"  Sample [{i}]: {s['image_name']}  |  {s['category']}  |  GT={s['normalized_answer']}")
        print(f"  Q: {s['question'][:100]}...")

        # Prepare inputs via processor
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": s["image"]},
                {"type": "text",  "text": s["question"]},
            ]},
        ]
        text = pipeline.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = pipeline.processor(
            text=[text], images=[s["image"]], return_tensors="pt", padding=True
        )

        pixel_values   = inputs["pixel_values"].to(device=dev, dtype=dtype)
        image_grid_thw = inputs["image_grid_thw"].to(device=dev)
        input_ids      = inputs["input_ids"].to(device=dev)
        depth_batch    = s["depth_map"].unsqueeze(0).to(device=dev, dtype=dtype)

        # Find <mask> positions
        mask_positions = find_mask_positions(input_ids, tokenizer)
        rle_list = s["rle_list"]

        print(f"  pixel_values:   {list(pixel_values.shape)}")
        print(f"  image_grid_thw: {image_grid_thw.tolist()}")
        print(f"  depth_map:      {list(depth_batch.shape)}")
        print(f"  input_ids:      {list(input_ids.shape)}")
        print(f"  mask_positions: {mask_positions}  ({len(mask_positions)} found, {len(rle_list)} RLEs)")

        # Show question after <mask> -> <mask_rgb> <mask_depth> replacement
        q_injected = re.sub(r'<mask>', '<mask_rgb> <mask_depth>', s['question'])
        print(f"  Q (injected): {q_injected[:120]}...")

        # Match mask count
        if len(mask_positions) != len(rle_list):
            print(f"  [!] Mask count mismatch: {len(mask_positions)} positions vs {len(rle_list)} RLEs")
            # Use minimum
            n = min(len(mask_positions), len(rle_list))
            mask_positions = mask_positions[:n]
            rle_list = rle_list[:n]

        # Run inference
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            output_ids = pipeline.generate(
                pixel_values, image_grid_thw, depth_batch, input_ids,
                rle_list=rle_list if len(rle_list) > 0 else None,
                mask_token_positions=mask_positions if len(mask_positions) > 0 else None,
                max_new_tokens=args.max_new_tokens,
            )

        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        raw_output = decoded[0]

        parsed = SpatialVLM.parse_output(raw_output)

        type_flag = "[OK]" if parsed["type_ok"] else "[FAIL]"
        match = str(parsed.get("answer", "")).strip('"') == str(s["normalized_answer"]).strip('"')
        match_flag = "[OK]" if match else "[FAIL]"
        if match:
            correct += 1

        print(f"\n  Raw output:   {raw_output[:150]}")
        print(f"  Parsed:       category={parsed['category']!r}  answer={parsed['answer']!r}  type_ok={type_flag}")
        print(f"  Ground truth: category={s['category']!r}  answer={s['normalized_answer']!r}")
        print(f"  Match: {match_flag}")
        print_vram_usage(f"sample {i}")

    # Summary
    print(f"\n{'='*70}")
    print(f"RESULTS: {correct}/{N_SAMPLES} correct ({correct/N_SAMPLES*100:.1f}%)")
    print(f"{'='*70}")
    print_vram_usage("final")


if __name__ == "__main__":
    main()
