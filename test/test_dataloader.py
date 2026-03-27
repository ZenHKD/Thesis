"""
Test SpatialVLM DataLoader.

Usage:
    python test/test_dataloader.py
    python test/test_dataloader.py --num-samples 10
    python test/test_dataloader.py --batch-size 4
"""

import sys
import os
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoProcessor
from src.dataloader.dataloader import SpatialVLMDataset, get_dataloader

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "model", "qwen3.5-0.8b")


def main():
    parser = argparse.ArgumentParser(description="Test SpatialVLM DataLoader")
    parser.add_argument("--split", default="train_sample",
                        choices=["train", "val", "test", "train_sample"])
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    print("=" * 70)
    print("DATALOADER TEST")
    print("=" * 70)

    # Load processor
    print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer = processor.tokenizer

    # Create dataset
    print(f"\nCreating dataset ({args.split}, max {args.num_samples})...")
    dataset = SpatialVLMDataset(args.split, processor=processor, max_samples=args.num_samples)
    print(f"  Dataset size: {len(dataset)}")

    # Test one sample per category
    TARGET_CATS = ["mcq", "distance", "count", "left_right"]
    found = {}
    for idx in range(len(dataset)):
        cat = dataset.data[idx]["category"]
        if cat in TARGET_CATS and cat not in found:
            found[cat] = idx
        if len(found) == len(TARGET_CATS):
            break

    for cat in TARGET_CATS:
        if cat not in found:
            print(f"\n  [!] Category '{cat}' not found in dataset, skipping")
            continue

        idx = found[cat]
        print(f"\n{'-'*70}")
        print(f"SAMPLE TEST: category={cat}  (index={idx})")
        print(f"{'-'*70}")

        sample = dataset[idx]
        for key, val in sample.items():
            if isinstance(val, torch.Tensor):
                print(f"  {key:18s}: {list(val.shape)}  dtype={val.dtype}")
            elif isinstance(val, list):
                print(f"  {key:18s}: list[{len(val)}]")
            else:
                print(f"  {key:18s}: {val!r}")

        # Check labels
        n_masked = (sample["labels"] == -100).sum().item()
        n_active = (sample["labels"] != -100).sum().item()
        total = sample["labels"].shape[0]
        print(f"\n  Labels: {n_masked} masked (prompt) + {n_active} active (answer) = {total} total")

        # Decode the active (answer) portion
        active_ids = sample["labels"][sample["labels"] != -100]
        decoded_answer = tokenizer.decode(active_ids, skip_special_tokens=True)
        print(f"  Decoded answer: {decoded_answer[:120]}")

    # Test DataLoader
    print(f"\n{'-'*70}")
    print(f"DATALOADER TEST (batch_size={args.batch_size})")
    print(f"{'-'*70}")

    loader = get_dataloader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    for i, batch in enumerate(loader):
        print(f"\n  Batch {i}:")
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                print(f"    {key:18s}: {list(val.shape)}  dtype={val.dtype}")
            elif isinstance(val, list):
                print(f"    {key:18s}: list[{len(val)}]")

    print(f"\n{'='*70}")
    print("  Dataloader [OK]")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
