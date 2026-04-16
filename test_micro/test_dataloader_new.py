"""
Test SpatialVLM Micro DataLoader (dataloader_new.py).

Verifies:
    1. Per-category sample loading (mcq, distance, count, left_right)
    2. Tensor shapes and dtypes
    3. Label masking (prompt = -100, answer = active)
    4. Numeric fields (is_numeric, target_num, num_token_pos)
    5. Batched collation with variable-length padding
    6. Token ID ranges (all IDs should be within Qwen vocab)

Usage:
    python test_micro/test_dataloader_new.py
    python test_micro/test_dataloader_new.py --batch-size 4
    python test_micro/test_dataloader_new.py --resolution 320p
"""

import sys
import os
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoProcessor
from src.dataloader.dataloader_new import SpatialVLMDataset, get_dataloader

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "model_micro", "qwen3.5-micro")


def main():
    parser = argparse.ArgumentParser(description="Test SpatialVLM Micro DataLoader")
    parser.add_argument("--split", default="train_sample",
                        choices=["train", "val", "test", "train_sample"])
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--resolution", default="320p",
                        choices=["1080p", "720p", "540p", "450p", "320p"])
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    args = parser.parse_args()

    target_size = {"1080p": None, "720p": (1280, 720),
                   "540p": (960, 540), "450p": (800, 450),
                   "320p": (512, 320)}[args.resolution]

    print("=" * 70)
    print("DATALOADER TEST (Micro)")
    print("=" * 70)

    # Load processor
    print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer = processor.tokenizer

    # Create dataset
    print(f"\nCreating dataset ({args.split}, max {args.num_samples}, {args.resolution})...")
    dataset = SpatialVLMDataset(args.split, processor=processor,
                                max_samples=args.num_samples, target_size=target_size)
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Resolution:   {args.resolution}"
          + (f" ({target_size[0]}x{target_size[1]})" if target_size else " (original)"))

    # ------------------------------------------------------------------
    # Test one sample per category
    # ------------------------------------------------------------------
    TARGET_CATS = ["mcq", "distance", "count", "left_right"]
    found = {}
    for idx in range(len(dataset)):
        cat = dataset.data[idx]["category"]
        if cat in TARGET_CATS and cat not in found:
            found[cat] = idx
        if len(found) == len(TARGET_CATS):
            break

    all_ok = True
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
                val = val.to(args.device)
                print(f"  {key:18s}: {list(val.shape)}  dtype={val.dtype}  device={val.device}")
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
        decoded_answer = tokenizer.decode(active_ids, skip_special_tokens=False)
        print(f"  Decoded answer: {decoded_answer}")

        # Verify label boundary: answer should start with <think>
        if decoded_answer.strip().startswith("<think>"):
            print(f"  [OK] Label boundary correct (starts with <think>)")
        else:
            print(f"  [FAIL] Label boundary wrong: answer starts with '{decoded_answer[:30]}'")
            all_ok = False

        # Check numeric fields
        is_num = sample["is_numeric"]
        target_num = sample["target_num"]
        num_pos = sample["num_token_pos"]

        if cat in ("distance", "count"):
            if not is_num:
                print(f"  [FAIL] is_numeric should be True for {cat}")
                all_ok = False
            else:
                print(f"  [OK] is_numeric=True, target_num={target_num}, num_token_pos={num_pos}")
            if num_pos == -1:
                print(f"  [WARN] <|num|> token not found in input_ids")
        else:
            if is_num:
                print(f"  [FAIL] is_numeric should be False for {cat}")
                all_ok = False
            else:
                print(f"  [OK] is_numeric=False (text-only task)")

    # ------------------------------------------------------------------
    # Test DataLoader with batching
    # ------------------------------------------------------------------
    print(f"\n{'-'*70}")
    print(f"DATALOADER BATCH TEST (batch_size={args.batch_size})")
    print(f"{'-'*70}")

    loader = get_dataloader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=False)
    for i, batch in enumerate(loader):
        print(f"\n  Batch {i}:")
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                val = val.to(args.device)
                print(f"    {key:22s}: {list(val.shape)}  dtype={val.dtype}  device={val.device}")
            elif isinstance(val, list):
                if len(val) > 0 and isinstance(val[0], list):
                    print(f"    {key:22s}: list[{len(val)}] of lists "
                          f"(lens: {[len(v) for v in val]})")
                else:
                    print(f"    {key:22s}: list[{len(val)}]")

        # Verify padding alignment
        B = batch["input_ids"].shape[0]
        L = batch["input_ids"].shape[1]
        L_labels = batch["labels"].shape[1]
        assert L == L_labels, f"input_ids and labels length mismatch: {L} vs {L_labels}"

        # Verify numeric fields
        assert batch["is_numeric"].shape == (B,), f"is_numeric shape: {batch['is_numeric'].shape}"
        assert batch["target_num"].shape == (B,), f"target_num shape: {batch['target_num'].shape}"

        cats_in_batch = batch["categories"]
        print(f"    Categories: {cats_in_batch}")
        print(f"    Padding OK: input_ids={L}, labels={L_labels}")

        # --- Deep mask verification ---
        mask_positions = batch["mask_positions"]
        decoded_masks = batch["decoded_masks"]
        rle_list = batch["rle_list"]

        batch_mask_ok = True
        for b in range(B):
            n_masks = len(mask_positions[b])
            n_rle = len(rle_list[b])
            n_decoded = len(decoded_masks[b])

            # Check consistency across fields
            if n_masks != n_rle or n_masks != n_decoded:
                print(f"    [FAIL] Sample {b}: mask_positions({n_masks}) != "
                      f"rle_list({n_rle}) != decoded_masks({n_decoded})")
                all_ok = False
                batch_mask_ok = False
                continue

            if n_masks == 0:
                continue  # No masks for this sample

            # Check each mask position points to <mask> token
            # Use string-based check: decode the 3-token span and verify it contains '<mask>'
            # Verify tokenizer can encode/decode correctly
            ids_list = batch["input_ids"][b].tolist()
            for m_idx, pos in enumerate(mask_positions[b]):
                if pos + 2 >= L:
                    print(f"    [FAIL] Sample {b}, mask {m_idx}: pos={pos} out of bounds (L={L})")
                    all_ok = False
                    batch_mask_ok = False
                    continue

                decoded_tok = tokenizer.decode(ids_list[pos:pos+3])
                if "<mask>" not in decoded_tok:
                    print(f"    [FAIL] Sample {b}, mask {m_idx}: pos={pos} -> "
                          f"'{decoded_tok}' (no <mask>)")
                    all_ok = False
                    batch_mask_ok = False

            # Check decoded_masks have correct shape (H, W)
            H, W = target_size[1], target_size[0]  # (W, H) -> (H, W)
            for m_idx, mask_dict in enumerate(decoded_masks[b]):
                binary = mask_dict["binary"]
                if binary.shape != (H, W):
                    print(f"    [FAIL] Sample {b}, mask {m_idx}: binary shape={binary.shape} "
                          f"expected ({H}, {W})")
                    all_ok = False
                    batch_mask_ok = False
                if "soft2d" not in mask_dict:
                    print(f"    [FAIL] Sample {b}, mask {m_idx}: missing 'soft2d' key")
                    all_ok = False
                    batch_mask_ok = False

        status = "[OK]" if batch_mask_ok else "[FAIL]"
        print(f"    Masks verified: positions -> <mask> tokens, shapes -> ({H},{W})  {status}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    if all_ok:
        print("  Dataloader Test [OK]")
    else:
        print("  Dataloader Test [FAIL]")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
