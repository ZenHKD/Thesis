"""
prune.py — Create Qwen 3.5 Micro (211M) from the original 0.8B checkpoint.

Pruning strategy:
    1. Vision: keep ViT blocks [8,9,10,11] → renumber [0,1,2,3]
    2. Decoder: keep layers [0,1,2,3, 20,21,22,23] → renumber [0..7]
    3. Vocabulary: slice embeddings to 318 kept tokens + 1 [NUM] = 319
    4. Config: update num_hidden_layers, layer_types, depth, vocab_size

Input:
    - model/qwen3.5-0.8b/                     (original HF checkpoint)
    - model_micro/micro_token_mapping.json     (from train_tokenizer.py)

Output:
    - model_micro/qwen3.5-micro/               (pruned HF-compatible checkpoint)

Usage:
    python model_micro/prune.py
"""

import os
import sys
import json
import copy
import torch
import shutil

from safetensors.torch import load_file, save_file

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

SRC_DIR      = os.path.join(PROJECT_ROOT, "model", "qwen3.5-0.8b")
DST_DIR      = os.path.join(SCRIPT_DIR, "qwen3.5-micro")
MAPPING_PATH = os.path.join(SCRIPT_DIR, "micro_token_mapping.json")

# ---------------------------------------------------------------------------
# Pruning config
# ---------------------------------------------------------------------------
KEEP_VISION_BLOCKS = [8, 9, 10, 11]           # 4 blocks (late ViT layers)
KEEP_DECODER_LAYERS = [0, 1, 2, 3, 20, 21, 22, 23]  # 8 layers (groups 0+5)

MICRO_LAYER_TYPES = [
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
]


def prune():
    print("=" * 60)
    print("  Qwen 3.5 0.8B → Micro (211M) Pruning")
    print("=" * 60)

    # --- Load token mapping ---
    with open(MAPPING_PATH, "r") as f:
        mapping = json.load(f)
    kept_old_ids = mapping["kept_old_ids"]          # 318 original Qwen token IDs
    total_vocab  = mapping["total_vocab"]            # 319 (318 + [NUM])
    print(f"\nToken mapping: {len(kept_old_ids)} kept tokens + [NUM] = {total_vocab} total")

    # --- Load original weights (safetensors) ---
    st_file = os.path.join(SRC_DIR, "model.safetensors-00001-of-00001.safetensors")
    print(f"\nLoading weights from: {st_file}")
    state_dict = load_file(st_file)
    print(f"  Original keys: {len(state_dict)}")

    new_state = {}
    kept_keys = set()
    skipped_keys = set()

    # ------------------------------------------------------------------
    # 1. VISION: keep blocks [8,9,10,11] → renumber [0,1,2,3]
    # ------------------------------------------------------------------
    print(f"\n--- Vision: keeping blocks {KEEP_VISION_BLOCKS} ---")
    vision_kept = 0
    vision_skipped = 0

    for key, val in state_dict.items():
        if not key.startswith("model.visual."):
            continue

        if "blocks." in key:
            # Extract block ID: model.visual.blocks.X.xxx
            parts = key.split("blocks.")
            block_id = int(parts[1].split(".")[0])

            if block_id in KEEP_VISION_BLOCKS:
                new_id = KEEP_VISION_BLOCKS.index(block_id)
                new_key = key.replace(f"blocks.{block_id}.", f"blocks.{new_id}.")
                new_state[new_key] = val
                kept_keys.add(key)
                vision_kept += 1
            else:
                skipped_keys.add(key)
                vision_skipped += 1
        else:
            # patch_embed, pos_embed, merger — keep as-is
            new_state[key] = val
            kept_keys.add(key)
            vision_kept += 1

    print(f"  Kept: {vision_kept} keys, Skipped: {vision_skipped} keys")

    # ------------------------------------------------------------------
    # 2. DECODER: keep layers [0,1,2,3, 20,21,22,23] → renumber [0..7]
    # ------------------------------------------------------------------
    print(f"\n--- Decoder: keeping layers {KEEP_DECODER_LAYERS} ---")
    decoder_kept = 0
    decoder_skipped = 0

    for key, val in state_dict.items():
        if not key.startswith("model.language_model."):
            continue

        if "layers." in key:
            # Extract layer ID: model.language_model.layers.X.xxx
            parts = key.split("layers.")
            layer_id = int(parts[1].split(".")[0])

            if layer_id in KEEP_DECODER_LAYERS:
                new_id = KEEP_DECODER_LAYERS.index(layer_id)
                new_key = key.replace(f"layers.{layer_id}.", f"layers.{new_id}.")
                new_state[new_key] = val
                kept_keys.add(key)
                decoder_kept += 1
            else:
                skipped_keys.add(key)
                decoder_skipped += 1

        elif "embed_tokens" in key:
            # --- Prune embedding to kept tokens ---
            old_embed = val  # [248320, 1024]
            print(f"\n--- Vocabulary: {old_embed.shape[0]} → {total_vocab} ---")

            # Slice kept rows
            kept_ids_tensor = torch.tensor(kept_old_ids, dtype=torch.long)
            pruned_embed = old_embed[kept_ids_tensor]           # [318, 1024]

            # Append [NUM] token with random init (scaled like Qwen embeddings)
            num_embed = torch.randn(1, old_embed.shape[1]) * 0.02  # [1, 1024]
            final_embed = torch.cat([pruned_embed, num_embed], dim=0)  # [319, 1024]
            print(f"  Embedding: [{old_embed.shape[0]}, {old_embed.shape[1]}] → [{final_embed.shape[0]}, {final_embed.shape[1]}]")
            print(f"  [NUM] token at new_id={len(kept_old_ids)} (random init)")

            new_state[key] = final_embed
            kept_keys.add(key)
            decoder_kept += 1
        else:
            # norm, rotary_emb, etc
            new_state[key] = val
            kept_keys.add(key)
            decoder_kept += 1

    print(f"  Kept: {decoder_kept} keys, Skipped: {decoder_skipped} keys")

    # ------------------------------------------------------------------
    # 3. LM HEAD (tied with embed_tokens, but may have separate key)
    # ------------------------------------------------------------------
    for key, val in state_dict.items():
        if key.startswith("lm_head."):
            # For tied embeddings, lm_head should use the same pruned embedding
            if val.shape[0] == state_dict["model.language_model.embed_tokens.weight"].shape[0]:
                # It's tied — prune the same way
                kept_ids_tensor = torch.tensor(kept_old_ids, dtype=torch.long)
                pruned_lm = val[kept_ids_tensor]
                num_lm = torch.randn(1, val.shape[1]) * 0.02
                new_state[key] = torch.cat([pruned_lm, num_lm], dim=0)
                print(f"\n  lm_head: pruned [{val.shape[0]}] → [{new_state[key].shape[0]}]")
            else:
                new_state[key] = val
            kept_keys.add(key)

    # ------------------------------------------------------------------
    # 4. Catch any remaining top-level keys (skip MTP)
    # ------------------------------------------------------------------
    mtp_skipped = 0
    for key, val in state_dict.items():
        if key not in kept_keys and key not in skipped_keys:
            # Skip MTP (Multi-Token Prediction) — speculative decoding weights
            # Not needed for training or standard inference
            if key.startswith("mtp."):
                mtp_skipped += 1
                skipped_keys.add(key)
                continue
            # Keep any other unmatched key
            new_state[key] = val
            print(f"  [info] Keeping unmatched key: {key} {val.shape}")
    if mtp_skipped > 0:
        print(f"\n  Skipped {mtp_skipped} MTP (speculative decoding) keys — not needed for training")

    # ------------------------------------------------------------------
    # 5. CONFIG
    # ------------------------------------------------------------------
    print(f"\n--- Config ---")
    with open(os.path.join(SRC_DIR, "config.json"), "r") as f:
        config = json.load(f)

    new_config = copy.deepcopy(config)

    # Text config
    new_config["text_config"]["num_hidden_layers"] = len(KEEP_DECODER_LAYERS)
    new_config["text_config"]["layer_types"] = MICRO_LAYER_TYPES
    new_config["text_config"]["vocab_size"] = total_vocab
    new_config["text_config"]["max_position_embeddings"] = 2048

    # Vision config
    new_config["vision_config"]["depth"] = len(KEEP_VISION_BLOCKS)

    print(f"  num_hidden_layers: {config['text_config']['num_hidden_layers']} → {new_config['text_config']['num_hidden_layers']}")
    print(f"  layer_types: {len(config['text_config']['layer_types'])} → {len(new_config['text_config']['layer_types'])}")
    print(f"  vocab_size: {config['text_config']['vocab_size']} → {new_config['text_config']['vocab_size']}")
    print(f"  vision depth: {config['vision_config']['depth']} → {new_config['vision_config']['depth']}")
    print(f"  max_position_embeddings: {config['text_config']['max_position_embeddings']} → {new_config['text_config']['max_position_embeddings']}")

    # ------------------------------------------------------------------
    # 6. SAVE
    # ------------------------------------------------------------------
    os.makedirs(DST_DIR, exist_ok=True)

    # Save pruned weights as safetensors
    out_path = os.path.join(DST_DIR, "model.safetensors")
    print(f"\nSaving pruned weights to: {out_path}")
    save_file(new_state, out_path)

    # Save config
    config_path = os.path.join(DST_DIR, "config.json")
    with open(config_path, "w") as f:
        json.dump(new_config, f, indent=2)
    print(f"Saved config to: {config_path}")

    # Copy tokenizer files (needed by AutoProcessor)
    tokenizer_files = [
        "tokenizer.json", "tokenizer_config.json",
        "merges.txt", "vocab.json",
        "chat_template.jinja",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    ]
    for fname in tokenizer_files:
        src = os.path.join(SRC_DIR, fname)
        dst = os.path.join(DST_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Copied: {fname}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  Pruning Complete!")
    print("=" * 60)

    total_params = sum(v.numel() for v in new_state.values())
    orig_params  = sum(v.numel() for v in state_dict.values())
    print(f"\n  Original: {orig_params:>12,} params ({orig_params * 2 / 1024**3:.2f} GB in FP16)")
    print(f"  Micro:    {total_params:>12,} params ({total_params * 2 / 1024**3:.2f} GB in FP16)")
    print(f"  Pruned:   {orig_params - total_params:>12,} params ({(orig_params - total_params) / orig_params * 100:.1f}%)")
    print(f"\n  Output directory: {DST_DIR}")
    print(f"  Files: {os.listdir(DST_DIR)}")


if __name__ == "__main__":
    prune()
