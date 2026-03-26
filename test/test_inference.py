"""
Test SpatialVLM pipeline inference with real samples from train_sample.json.

Usage:
    python test/test_inference.py
    python test/test_inference.py --num-samples 10
    python test/test_inference.py --attn-impl sdpa
"""

import sys
import os
import json
import argparse
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.pipeline import SpatialVLM, print_vram_usage

# Paths
DATA_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "nvidia_warehouse_dataset", "train_sample")
JSON_PATH = os.path.join(DATA_DIR, "train_sample.json")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
DEPTH_DIR = os.path.join(DATA_DIR, "depths")


def load_sample(entry: dict) -> dict:
    """Load a single dataset entry -> dict with all tensors/data ready."""
    image_name = entry["image"]
    image_path = os.path.join(IMAGE_DIR, image_name)
    depth_path = os.path.join(DEPTH_DIR, image_name.replace(".png", "_depth.png"))

    image = Image.open(image_path).convert("RGB")

    depth_np = np.array(Image.open(depth_path), dtype=np.float32)
    depth_map = torch.from_numpy(depth_np)  # [H, W]

    question_raw = entry["conversations"][0]["value"]
    question = question_raw.replace("<image>\n", "").replace("<image>", "").strip()

    category = entry["category"]
    raw_answer = str(entry["normalized_answer"])
    if category in ("mcq", "left_right"):
        normalized_answer = f'"{raw_answer}"'
    else:
        normalized_answer = raw_answer

    return {
        "image":             image,
        "question":          question,
        "depth_map":         depth_map,
        "rle_list":          entry["rle"],
        "category":          category,
        "normalized_answer": normalized_answer,
        "image_name":        image_name,
    }


def main():
    parser = argparse.ArgumentParser(description="Test SpatialVLM inference")
    parser.add_argument("--device",         default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype",          default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--attn-impl",      default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--num-samples",    type=int, default=5)
    args = parser.parse_args()

    N = args.num_samples
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    # Load dataset
    print(f"{'='*70}")
    print("LOADING DATASET")
    print(f"{'='*70}")
    with open(JSON_PATH, "r") as f:
        dataset = json.load(f)
    samples = [load_sample(dataset[i]) for i in range(N)]
    print(f"  Loaded {N}/{len(dataset)} samples")
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
    print(f"INFERENCE ({N} samples)")
    print(f"{'='*70}")

    correct = 0
    for i, s in enumerate(samples):
        print(f"\n{'-'*70}")
        print(f"  Sample [{i}]: {s['image_name']}  |  {s['category']}  |  GT={s['normalized_answer']}")
        print(f"  Q: {s['question'][:100]}...")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        result = pipeline.predict(
            image=s["image"],
            question=s["question"],
            depth_map=s["depth_map"],
            rle_list=s["rle_list"],
            max_new_tokens=args.max_new_tokens,
        )

        type_flag = "[OK]" if result["type_ok"] else "[FAIL]"
        match = str(result.get("answer", "")).strip('"') == str(s["normalized_answer"]).strip('"')
        match_flag = "[OK]" if match else "[FAIL]"
        if match:
            correct += 1

        print(f"\n  Raw output:   {result['raw'][:150]}")
        print(f"  Parsed:       category={result['category']!r}  answer={result['answer']!r}  type_ok={type_flag}")
        print(f"  Ground truth: category={s['category']!r}  answer={s['normalized_answer']!r}")
        print(f"  Match: {match_flag}")
        print_vram_usage(f"sample {i}")

    # Summary
    print(f"\n{'='*70}")
    print(f"RESULTS: {correct}/{N} correct ({correct/N*100:.1f}%)")
    print(f"{'='*70}")
    print_vram_usage("final")


if __name__ == "__main__":
    main()
