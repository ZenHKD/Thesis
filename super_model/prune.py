"""
SpatialVLM Super Pruning Pipeline (No External Pruning Libs)
- Vision: Keep ALL 12 ViT blocks (no pruning — needed for DPT multi-layer features)
- Decoder: Keep all 24 layers (single pass)
- Vocab: FULL original vocabulary (248,320) + 4 special tokens appended
- Add <|mcq|>   token (ID = 248077) for MCQ Head
- Add <|lr|>    token (ID = 248078) for Left-Right Head
- Add <|dist|>  token (ID = 248079) for Distance Head
- Add <|count|> token (ID = 248080) for Count Head
- Context length: 768
"""

import json
import shutil
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, AutoModelForImageTextToText

# ============== CONFIG ==============
ORIGINAL_MODEL_PATH = Path(__file__).parent.parent / "model" / "qwen3.5-0.8b"
OUTPUT_PATH = Path(__file__).parent / "qwen3.5-super"



MCQ_TOKEN   = "<|mcq|>"
LR_TOKEN    = "<|lr|>"
DIST_TOKEN  = "<|dist|>"
COUNT_TOKEN = "<|count|>"

# Copy these auxiliary files from original model
COPY_FILES = ["preprocessor_config.json", "video_preprocessor_config.json", "chat_template.jinja"]


# ============== TOKENIZER (no pruning) ==============

def setup_tokenizer_with_special_tokens(tokenizer):
    """Copy original tokenizer files and append 4 special tokens.
    
    No vocabulary pruning — keeps all 248,320 original tokens.
    Adds <|mcq|>   (ID=248077) for MCQ classification
         <|lr|>    (ID=248078) for Left-Right classification
         <|dist|>  (ID=248079) for Distance regression
         <|count|> (ID=248080) for Count regression
    
    Returns new_vocab_size, mcq_token_id, lr_token_id, dist_token_id, count_token_id.
    """
    # Copy ALL tokenizer files from original (fully intact)
    tok_files = ["vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json"]
    for fname in tok_files:
        src = ORIGINAL_MODEL_PATH / fname
        if src.exists():
            shutil.copy2(src, OUTPUT_PATH / fname)

    # len(tokenizer) is 248077 (base vocab + special tokens)
    # The original embedding matrix is size 248320 (padded for tensor cores)
    mcq_token_id   = 248077  # <|mcq|>   for MCQ Head
    lr_token_id    = 248078  # <|lr|>    for Left-Right Head
    dist_token_id  = 248079  # <|dist|>  for Distance Head
    count_token_id = 248080  # <|count|> for Count Head
    new_vocab_size = 248320  # the physical size of the embeddings

    # Define special tokens to add
    special_tokens = [
        (mcq_token_id,   MCQ_TOKEN),
        (lr_token_id,    LR_TOKEN),
        (dist_token_id,  DIST_TOKEN),
        (count_token_id, COUNT_TOKEN),
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
        print(f"  Added <|mcq|>, <|lr|>, <|dist|>, <|count|> to tokenizer_config.json")

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
    print(f"  <|mcq|>   token ID: {mcq_token_id}")
    print(f"  <|lr|>    token ID: {lr_token_id}")
    print(f"  <|dist|>  token ID: {dist_token_id}")
    print(f"  <|count|> token ID: {count_token_id}")
    print(f"  Final vocab size: {new_vocab_size}")

    return new_vocab_size, mcq_token_id, lr_token_id, dist_token_id, count_token_id


# ============== MAIN ==============

def main():
    print("=" * 60)
    print("SpatialVLM Super Pruning Pipeline")
    print("  Vision: ALL 12 blocks (no pruning)")
    print("  Context: 768")
    print("=" * 60)
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # 1. Load original model
    print("\n[1/6] Loading original model...")
    tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL_PATH)
    config = AutoConfig.from_pretrained(ORIGINAL_MODEL_PATH)
    model = AutoModelForImageTextToText.from_pretrained(
        ORIGINAL_MODEL_PATH, dtype=torch.bfloat16, device_map="cpu"
    )
    print(f"  Loaded: {ORIGINAL_MODEL_PATH}")

    # 2. Setup tokenizer with 4 special tokens (no vocab pruning)
    print("\n[2/6] Setting up tokenizer (full vocab + <|mcq|> + <|lr|> + <|dist|> + <|count|>)...")
    new_vocab_size, mcq_token_id, lr_token_id, dist_token_id, count_token_id = \
        setup_tokenizer_with_special_tokens(tokenizer)

    print(f"\n[3/6] Initializing special token embeddings...")
    # No need to resize, 248320 is already large enough to hold 248077-248080
    with torch.no_grad():
        for tid in [mcq_token_id, lr_token_id, dist_token_id, count_token_id]:
            model.get_input_embeddings().weight[tid].normal_(0.0, 0.02)
            if model.get_output_embeddings() is not None:
                model.get_output_embeddings().weight[tid].normal_(0.0, 0.02)

    # 4. No architecture pruning — all 12 ViT blocks + 24 decoder layers kept
    print("\n[4/6] Architecture check (no pruning)...")
    print("  Vision: ALL 12 blocks kept")
    print("  Decoder: ALL 24 layers kept")
    n_vis_blocks = len(list(model.model.visual.blocks))
    n_dec_layers = len(list(model.model.language_model.layers))
    print(f"  Model: {n_vis_blocks} vision blocks, {n_dec_layers} decoder layers")

    # 5. Update config & save
    print(f"\n[5/6] Saving to {OUTPUT_PATH}...")

    # Update the MODEL's internal config
    model.config.vocab_size = new_vocab_size
    model.config.mcq_token_id   = mcq_token_id
    model.config.lr_token_id    = lr_token_id
    model.config.dist_token_id  = dist_token_id
    model.config.count_token_id = count_token_id
    model.config.max_position_embeddings = 768
    if hasattr(model.config, "text_config"):
        model.config.text_config.num_hidden_layers = n_dec_layers
        model.config.text_config.vocab_size = new_vocab_size
        model.config.text_config.max_position_embeddings = 768
        if hasattr(model.config.text_config, "layer_types") and model.config.text_config.layer_types:
            pass  # No pruning, keep original layer_types
    # Vision config: depth stays at 12 (full)
    if hasattr(model.config, "vision_config"):
        model.config.vision_config.depth = n_vis_blocks

    model.save_pretrained(OUTPUT_PATH)

    # Also update standalone config
    config.vocab_size = new_vocab_size
    config.mcq_token_id   = mcq_token_id
    config.lr_token_id    = lr_token_id
    config.dist_token_id  = dist_token_id
    config.count_token_id = count_token_id
    config.max_position_embeddings = 768
    if hasattr(config, "text_config"):
        config.text_config.num_hidden_layers = n_dec_layers
        config.text_config.vocab_size = new_vocab_size
        config.text_config.max_position_embeddings = 768
        if hasattr(config.text_config, "layer_types") and config.text_config.layer_types:
            pass  # No pruning, keep original layer_types
    if hasattr(config, "vision_config"):
        config.vision_config.depth = n_vis_blocks
    config.save_pretrained(OUTPUT_PATH)

    # Copy auxiliary files from original model
    for fname in COPY_FILES:
        src = ORIGINAL_MODEL_PATH / fname
        if src.exists():
            shutil.copy2(src, OUTPUT_PATH / fname)
            print(f"  Copied {fname}")

    # 6. Save manifest
    print("\n[6/6] Saving manifest...")
    manifest = {
        "source": str(ORIGINAL_MODEL_PATH),
        "original_vocab_size": tokenizer.vocab_size,
        "final_vocab_size": new_vocab_size,
        "mcq_token_id": mcq_token_id,
        "lr_token_id": lr_token_id,
        "dist_token_id": dist_token_id,
        "count_token_id": count_token_id,
        "vocab_pruned": False,
        "vision_pruned": False,
        "kept_vision_blocks": list(range(n_vis_blocks)),
        "kept_decoder_layers": list(range(n_dec_layers)),
        "max_position_embeddings": 768,
    }
    with open(OUTPUT_PATH / "prune_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Summary
    total_params = sum(p.numel() for p in model.parameters())
    embed_params = model.get_input_embeddings().weight.numel()
    print("\n" + "=" * 60)
    print("Pruning complete!")
    print(f"  Vocab:       248077 active tokens + 4 special (Embeddings padded to {new_vocab_size})")
    print(f"  Special:     <|mcq|>={mcq_token_id}, <|lr|>={lr_token_id}, "
          f"<|dist|>={dist_token_id}, <|count|>={count_token_id}")
    print(f"  Vision:      {n_vis_blocks} blocks (ALL kept, no pruning)")
    print(f"  Decoder:     {n_dec_layers} layers (ALL kept, no pruning)")
    print(f"  Context:     768 max positions")
    print(f"  Embed:       {embed_params/1e6:.1f}M params ({embed_params * 2 / 1e6:.1f} MB bf16)")
    print(f"  Total:       {total_params/1e6:.1f}M params")
    print(f"  Output:      {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
