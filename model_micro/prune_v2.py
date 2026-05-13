"""
SpatialVLM Micro Token Injection Pipeline (No Architectural Pruning)
- Vision: Keep ALL original blocks
- Decoder: Keep ALL original layers
- Vocab: FULL original vocabulary + <|num|> + <|cat|> tokens appended
- Add <|num|> token (ID = 248077) for NumberHead
- Add <|cat|> token (ID = 248078) for CategoryHead
"""

import json
import shutil
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, AutoModelForImageTextToText

# ============== CONFIG ==============
ORIGINAL_MODEL_PATH = Path(__file__).parent.parent / "model" / "qwen3.5-0.8b"
OUTPUT_PATH = Path(__file__).parent / "qwen3.5-micro-fullvit"

NUM_TOKEN = "<|num|>"
CAT_TOKEN = "<|cat|>"

# Copy these auxiliary files from original model
COPY_FILES = ["preprocessor_config.json", "video_preprocessor_config.json", "chat_template.jinja"]


# ============== TOKENIZER (no pruning) ==============

def setup_tokenizer_with_special_tokens(tokenizer):
    """Copy original tokenizer files and append <|num|> + <|cat|> tokens.
    
    No vocabulary pruning — keeps all 248,320 original tokens.
    Adds <|num|> (ID=248077) and <|cat|> (ID=248078) at the end.
    
    Returns new_vocab_size, num_token_id, cat_token_id.
    """
    # Copy ALL tokenizer files from original (fully intact)
    tok_files = ["vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json"]
    for fname in tok_files:
        src = ORIGINAL_MODEL_PATH / fname
        if src.exists():
            shutil.copy2(src, OUTPUT_PATH / fname)

    # len(tokenizer) is 248077 (base vocab + special tokens)
    # The original embedding matrix is size 248320 (padded for tensor cores)
    num_token_id = 248077  # <|num|> for NumberHead (distance + count)
    cat_token_id = 248078  # <|cat|> for CategoryHead (mcq + left_right)
    new_vocab_size = 248320  # the physical size of the embeddings

    # Define special tokens to add
    special_tokens = [
        (num_token_id, NUM_TOKEN),
        (cat_token_id, CAT_TOKEN),
    ]

    # Patch tokenizer_config.json: add special tokens
    tc_path = OUTPUT_PATH / "tokenizer_config.json"
    if tc_path.exists():
        with open(tc_path, "r", encoding="utf-8") as f:
            tc = json.load(f)
        
        if "added_tokens_decoder" not in tc:
            tc["added_tokens_decoder"] = {}
        if "added_tokens_encoder" not in tc:
            tc["added_tokens_encoder"] = {}
        
        for token_id, token_str in special_tokens:
            tc["added_tokens_decoder"][str(token_id)] = {
                "content": token_str,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "single_word": False,
                "special": True
            }
            tc["added_tokens_encoder"][token_str] = token_id
    
        with open(tc_path, "w", encoding="utf-8") as f:
            json.dump(tc, f, indent=2, ensure_ascii=False)
        print(f"  Added <|num|> and <|cat|> to tokenizer_config.json")

    # Patch tokenizer.json: sync added_tokens from tokenizer_config.json perfectly
    tj_path = OUTPUT_PATH / "tokenizer.json"
    if tj_path.exists() and tc_path.exists():
        with open(tj_path, "r", encoding="utf-8") as f:
            tj = json.load(f)
        
        tj["added_tokens"] = []
        for tid_str, meta in tc["added_tokens_decoder"].items():
            tj["added_tokens"].append({
                "id": int(tid_str),
                "content": meta["content"],
                "single_word": meta.get("single_word", False),
                "lstrip": meta.get("lstrip", False),
                "rstrip": meta.get("rstrip", False),
                "normalized": meta.get("normalized", False),
                "special": meta.get("special", True)
            })
        
        # Sort by ID to ensure contiguous block (prevents HF renumbering)
        tj["added_tokens"].sort(key=lambda x: x["id"])
        
        with open(tj_path, "w", encoding="utf-8") as f:
            json.dump(tj, f, ensure_ascii=False)


    print(f"  Physical Embeddings Size: {new_vocab_size}")
    print(f"  <|num|> token ID: {num_token_id}")
    print(f"  <|cat|> token ID: {cat_token_id}")
    print(f"  Final vocab size: {new_vocab_size}")

    return new_vocab_size, num_token_id, cat_token_id


# ============== MAIN ==============

def main():
    print("=" * 60)
    print("SpatialVLM Micro Token Injection Pipeline (No Pruning)")
    print("=" * 60)
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # 1. Load original model
    print("\n[1/5] Loading original model...")
    tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL_PATH)
    config = AutoConfig.from_pretrained(ORIGINAL_MODEL_PATH)
    model = AutoModelForImageTextToText.from_pretrained(
        ORIGINAL_MODEL_PATH, dtype=torch.bfloat16, device_map="cpu"
    )
    print(f"  Loaded: {ORIGINAL_MODEL_PATH}")

    # 2. Setup tokenizer with <|num|> + <|cat|> tokens (no vocab pruning)
    print("\n[2/5] Setting up tokenizer (full vocab + <|num|> + <|cat|>)...")
    new_vocab_size, num_token_id, cat_token_id = setup_tokenizer_with_special_tokens(tokenizer)

    print(f"\n[3/5] Initializing special token embeddings...")
    # No need to resize, 248320 is already large enough to hold 248077-248078
    with torch.no_grad():
        for tid in [num_token_id, cat_token_id]:
            model.get_input_embeddings().weight[tid].normal_(0.0, 0.02)
            if model.get_output_embeddings() is not None:
                model.get_output_embeddings().weight[tid].normal_(0.0, 0.02)

    # 4. Update config & save (No architecture pruning)
    print(f"\n[4/5] Saving to {OUTPUT_PATH}...")

    # Update the MODEL's internal config
    model.config.vocab_size = new_vocab_size
    model.config.num_token_id = num_token_id
    model.config.cat_token_id = cat_token_id
    if hasattr(model.config, "text_config"):
        model.config.text_config.vocab_size = new_vocab_size

    model.save_pretrained(OUTPUT_PATH)

    # Also update standalone config
    config.vocab_size = new_vocab_size
    config.num_token_id = num_token_id
    config.cat_token_id = cat_token_id
    if hasattr(config, "text_config"):
        config.text_config.vocab_size = new_vocab_size
    config.save_pretrained(OUTPUT_PATH)

    # Copy auxiliary files from original model
    for fname in COPY_FILES:
        src = ORIGINAL_MODEL_PATH / fname
        if src.exists():
            shutil.copy2(src, OUTPUT_PATH / fname)
            print(f"  Copied {fname}")

    # 5. Save manifest
    print("\n[5/5] Saving manifest...")
    manifest = {
        "source": str(ORIGINAL_MODEL_PATH),
        "original_vocab_size": tokenizer.vocab_size,
        "final_vocab_size": new_vocab_size,
        "num_token_id": num_token_id,
        "cat_token_id": cat_token_id,
        "vocab_pruned": False,
        "architectural_pruning": False,
    }
    with open(OUTPUT_PATH / "prune_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Summary
    total_params = sum(p.numel() for p in model.parameters())
    embed_params = model.get_input_embeddings().weight.numel()
    print("\n" + "=" * 60)
    print("Injection complete!")
    print(f"  Vocab:       248077 active tokens (Embeddings padded to {new_vocab_size})")
    print(f"  Embed:       {embed_params/1e6:.1f}M params ({embed_params * 2 / 1e6:.1f} MB bf16)")
    print(f"  Total:       {total_params/1e6:.1f}M params")
    print(f"  Output:      {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
