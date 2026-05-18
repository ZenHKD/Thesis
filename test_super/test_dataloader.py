"""
Test SpatialVLM Super DataLoader (super_model/dataloader.py).

Verifies:
    1. Per-category sample loading (mcq, distance, count, left_right)
    2. Tensor shapes and dtypes
    3. Label masking (prompt = -100, answer = active)
    4. 4 dedicated token position fields (mcq/lr/dist/count)
    5. Batched collation with variable-length padding
    6. Token ID ranges (all IDs within Qwen vocab)

Usage:
    python test_super/test_dataloader.py
    python test_super/test_dataloader.py --batch-size 4
    python test_super/test_dataloader.py --resolution 320p
"""

import sys
import os
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoProcessor
from super_model.dataloader import SpatialVLMDataset, get_dataloader

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "super_model", "qwen3.5-super")


def main():
    parser = argparse.ArgumentParser(description="Test SpatialVLM Super DataLoader")
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
    print("DATALOADER TEST (Super)")
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
                print(f"  {key:22s}: {list(val.shape)}  dtype={val.dtype}  device={val.device}")
            elif isinstance(val, list):
                print(f"  {key:22s}: list[{len(val)}]")
            else:
                print(f"  {key:22s}: {val!r}")

        # Check labels
        n_masked = (sample["labels"] == -100).sum().item()
        n_active = (sample["labels"] != -100).sum().item()
        total = sample["labels"].shape[0]
        print(f"\n  Labels: {n_masked} masked (prompt) + {n_active} active (answer) = {total} total")

        # Decode the active (answer) portion
        active_ids = sample["labels"][sample["labels"] != -100]
        decoded_answer = tokenizer.decode(active_ids, skip_special_tokens=False)
        print(f"  Decoded answer: {decoded_answer}")

        # Verify label boundary: answer should start with the corresponding special token
        expected_token_id = {
            "mcq": dataset.mcq_token_id,
            "left_right": dataset.lr_token_id,
            "distance": dataset.dist_token_id,
            "count": dataset.count_token_id,
        }[cat]

        if len(active_ids) > 0 and active_ids[0].item() == expected_token_id:
            print(f"  [OK] Label boundary correct (starts with token ID {expected_token_id})")
        else:
            actual_id = active_ids[0].item() if len(active_ids) > 0 else -1
            print(f"  [FAIL] Label boundary wrong: answer starts with token ID {actual_id}, expected {expected_token_id}")
            all_ok = False

        # Check 4 dedicated token positions
        mcq_pos   = sample["mcq_token_pos"]
        lr_pos    = sample["lr_token_pos"]
        dist_pos  = sample["dist_token_pos"]
        count_pos = sample["count_token_pos"]

        if cat == "mcq":
            if mcq_pos == -1:
                print(f"  [FAIL] <|mcq|> token not found for mcq task")
                all_ok = False
            else:
                print(f"  [OK] mcq_token_pos={mcq_pos}")
            if lr_pos != -1 or dist_pos != -1 or count_pos != -1:
                print(f"  [FAIL] Other token positions should be -1 for mcq task")
                all_ok = False
        elif cat == "left_right":
            if lr_pos == -1:
                print(f"  [FAIL] <|lr|> token not found for left_right task")
                all_ok = False
            else:
                print(f"  [OK] lr_token_pos={lr_pos}")
            if mcq_pos != -1 or dist_pos != -1 or count_pos != -1:
                print(f"  [FAIL] Other token positions should be -1 for left_right task")
                all_ok = False
        elif cat == "distance":
            if dist_pos == -1:
                print(f"  [FAIL] <|dist|> token not found for distance task")
                all_ok = False
            else:
                print(f"  [OK] dist_token_pos={dist_pos}")
            if mcq_pos != -1 or lr_pos != -1 or count_pos != -1:
                print(f"  [FAIL] Other token positions should be -1 for distance task")
                all_ok = False
        elif cat == "count":
            if count_pos == -1:
                print(f"  [FAIL] <|count|> token not found for count task")
                all_ok = False
            else:
                print(f"  [OK] count_token_pos={count_pos}")
            if mcq_pos != -1 or lr_pos != -1 or dist_pos != -1:
                print(f"  [FAIL] Other token positions should be -1 for count task")
                all_ok = False

        # Check numeric/categorical flags
        is_num = sample["is_numeric"]
        is_cat_flag = sample["is_categorical"]

        if cat in ("distance", "count"):
            if not is_num:
                print(f"  [FAIL] is_numeric should be True for {cat}")
                all_ok = False
            else:
                print(f"  [OK] is_numeric=True, target_num={sample['target_num']}")
        else:
            if is_num:
                print(f"  [FAIL] is_numeric should be False for {cat}")
                all_ok = False

        if cat in ("mcq", "left_right"):
            if not is_cat_flag:
                print(f"  [FAIL] is_categorical should be True for {cat}")
                all_ok = False
            else:
                print(f"  [OK] is_categorical=True, target_cat_index={sample['target_cat_index']}")
        else:
            if is_cat_flag:
                print(f"  [FAIL] is_categorical should be False for {cat}")
                all_ok = False

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
                print(f"    {key:28s}: {list(val.shape)}  dtype={val.dtype}  device={val.device}")
            elif isinstance(val, list):
                if len(val) > 0 and isinstance(val[0], list):
                    print(f"    {key:28s}: list[{len(val)}] of lists "
                          f"(lens: {[len(v) for v in val]})")
                else:
                    print(f"    {key:28s}: list[{len(val)}]")

        # Verify padding alignment
        B = batch["input_ids"].shape[0]
        L = batch["input_ids"].shape[1]
        L_labels = batch["labels"].shape[1]
        assert L == L_labels, f"input_ids and labels length mismatch: {L} vs {L_labels}"

        # Verify numeric/categorical fields
        assert batch["is_numeric"].shape == (B,)
        assert batch["is_categorical"].shape == (B,)
        assert batch["target_num"].shape == (B,)
        assert batch["target_cat_index"].shape == (B,)

        # Verify 4 token position lists
        assert len(batch["mcq_token_positions"]) == B
        assert len(batch["lr_token_positions"]) == B
        assert len(batch["dist_token_positions"]) == B
        assert len(batch["count_token_positions"]) == B

        cats_in_batch = batch["categories"]
        print(f"    Categories:           {cats_in_batch}")
        print(f"    mcq_token_positions:  {batch['mcq_token_positions']}")
        print(f"    lr_token_positions:   {batch['lr_token_positions']}")
        print(f"    dist_token_positions: {batch['dist_token_positions']}")
        print(f"    count_token_positions:{batch['count_token_positions']}")
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

            if n_masks != n_rle or n_masks != n_decoded:
                print(f"    [FAIL] Sample {b}: mask_positions({n_masks}) != "
                      f"rle_list({n_rle}) != decoded_masks({n_decoded})")
                all_ok = False
                batch_mask_ok = False
                continue

            if n_masks == 0:
                continue

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

            if target_size:
                H, W = target_size[1], target_size[0]
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
        if target_size:
            print(f"    Masks verified: positions -> <mask> tokens, shapes -> ({H},{W})  {status}")
        else:
            print(f"    Masks verified: positions -> <mask> tokens  {status}")

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
