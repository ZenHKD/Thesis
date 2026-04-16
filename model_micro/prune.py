"""
SpatialVLM Micro Pruning Pipeline (No External Pruning Libs)
- Vision: Keep ViT blocks [8,9,10,11] -> renumber [0,1,2,3]
- Decoder: Keep all 24 layers (single pass)
- Vocab: FULL original vocabulary (248,320) + <|num|> token appended
- Add <|num|> token at end of vocab (ID = 248,320)
"""

import json
import shutil
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, AutoModelForImageTextToText

# ============== CONFIG ==============
ORIGINAL_MODEL_PATH = Path(__file__).parent.parent / "model" / "qwen3.5-0.8b"
OUTPUT_PATH = Path(__file__).parent / "qwen3.5-micro"

KEEP_VISION_BLOCKS = [8, 9, 10, 11]
KEEP_DECODER_LAYERS = list(range(24))
NUM_TOKEN = "<|num|>"

# Copy these auxiliary files from original model
COPY_FILES = ["preprocessor_config.json", "video_preprocessor_config.json", "chat_template.jinja"]


# ============== TOKENIZER (no pruning) ==============

def setup_tokenizer_with_num(tokenizer):
    """Copy original tokenizer files and append <|num|> token.
    
    No vocabulary pruning — keeps all 248,320 original tokens.
    Only adds <|num|> at the end (ID = 248,320).
    
    Returns new_vocab_size.
    """
    # Copy ALL tokenizer files from original (fully intact)
    tok_files = ["vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json"]
    for fname in tok_files:
        src = ORIGINAL_MODEL_PATH / fname
        if src.exists():
            shutil.copy2(src, OUTPUT_PATH / fname)

    # len(tokenizer) is 248077 (base vocab + special tokens)
    # The original embedding matrix is size 248320 (padded for tensor cores)
    num_token_id = 248077  # <|num|> gets the next contiguous ID (DO NOT USE 248044, IT OVERWRITES <|endoftext|>)
    new_vocab_size = 248320  # the physical size of the embeddings

    # Patch tokenizer_config.json: add <|num|> as added token
    tc_path = OUTPUT_PATH / "tokenizer_config.json"
    if tc_path.exists():
        with open(tc_path, "r", encoding="utf-8") as f:
            tc = json.load(f)
        
        if "added_tokens_decoder" not in tc:
            tc["added_tokens_decoder"] = {}
        if "added_tokens_encoder" not in tc:
            tc["added_tokens_encoder"] = {}
        
        tc["added_tokens_decoder"][str(num_token_id)] = {
            "content": NUM_TOKEN,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "single_word": False,
            "special": True
        }
        tc["added_tokens_encoder"][NUM_TOKEN] = num_token_id
    
        with open(tc_path, "w", encoding="utf-8") as f:
            json.dump(tc, f, indent=2, ensure_ascii=False)
        print(f"  Added <|num|> to tokenizer_config.json")

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
    print(f"  Final vocab size: {new_vocab_size}")

    return new_vocab_size, num_token_id


# ============== ARCHITECTURE PRUNING ==============

def prune_vision(sd: dict) -> dict:
    print(f"  Vision: blocks {KEEP_VISION_BLOCKS} -> [0..3]")
    new = {}
    for k, v in sd.items():
        if "model.visual.blocks." in k:
            parts = k.split(".")
            # model.visual.blocks.N.xxx -> index of "blocks" + 1
            bi = parts.index("blocks")
            idx = int(parts[bi + 1])
            if idx in KEEP_VISION_BLOCKS:
                parts[bi + 1] = str(KEEP_VISION_BLOCKS.index(idx))
                new[".".join(parts)] = v
        else:
            new[k] = v
    return new


def prune_decoder(sd: dict) -> dict:
    print(f"  Decoder: layers {KEEP_DECODER_LAYERS} (all 24 layers)")
    new = {}
    for k, v in sd.items():
        if "model.language_model.layers." in k:
            parts = k.split(".")
            li = parts.index("layers")
            idx = int(parts[li + 1])
            if idx in KEEP_DECODER_LAYERS:
                new[k] = v
        else:
            new[k] = v
    return new

# ============== MAIN ==============

def main():
    print("=" * 60)
    print("SpatialVLM Micro Pruning Pipeline")
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

    # 2. Setup tokenizer with <|num|> token (no vocab pruning)
    print("\n[2/6] Setting up tokenizer (full vocab + <|num|>)...")
    new_vocab_size, num_token_id = setup_tokenizer_with_num(tokenizer)

    print(f"\n[3/6] Initializing <|num|> token embedding at index {num_token_id}...")
    # No need to resize, 248320 is already large enough to hold 248077
    with torch.no_grad():
        model.get_input_embeddings().weight[num_token_id].normal_(0.0, 0.02)
        if model.get_output_embeddings() is not None:
            model.get_output_embeddings().weight[num_token_id].normal_(0.0, 0.02)

    # 4. Prune architecture (vision + decoder only)
    print("\n[4/6] Pruning model architecture...")
    sd = prune_vision(model.state_dict())
    sd = prune_decoder(sd)
    model.load_state_dict(sd, strict=False)

    # Physically remove extra layers/blocks from model
    import torch.nn as nn
    n_keep_dec = len(KEEP_DECODER_LAYERS)
    n_keep_vis = len(KEEP_VISION_BLOCKS)
    model.model.language_model.layers = nn.ModuleList(
        list(model.model.language_model.layers)[:n_keep_dec]
    )
    model.model.visual.blocks = nn.ModuleList(
        list(model.model.visual.blocks)[:n_keep_vis]
    )
    print(f"  Trimmed model: {n_keep_vis} vision blocks, {n_keep_dec} decoder layers")

    # 5. Update config & save
    print(f"\n[5/6] Saving to {OUTPUT_PATH}...")

    # Update the MODEL's internal config
    model.config.vocab_size = new_vocab_size
    model.config.num_token_id = num_token_id
    model.config.max_position_embeddings = 512
    if hasattr(model.config, "text_config"):
        model.config.text_config.num_hidden_layers = len(KEEP_DECODER_LAYERS)
        model.config.text_config.vocab_size = new_vocab_size
        model.config.text_config.max_position_embeddings = 512
        if hasattr(model.config.text_config, "layer_types") and model.config.text_config.layer_types:
            model.config.text_config.layer_types = [
                model.config.text_config.layer_types[i] for i in KEEP_DECODER_LAYERS
            ]
    if hasattr(model.config, "vision_config"):
        model.config.vision_config.depth = len(KEEP_VISION_BLOCKS)

    model.save_pretrained(OUTPUT_PATH)

    # Also update standalone config
    config.vocab_size = new_vocab_size
    config.num_token_id = num_token_id
    config.max_position_embeddings = 512
    if hasattr(config, "text_config"):
        config.text_config.num_hidden_layers = len(KEEP_DECODER_LAYERS)
        config.text_config.vocab_size = new_vocab_size
        config.text_config.max_position_embeddings = 512
        if hasattr(config.text_config, "layer_types") and config.text_config.layer_types:
            config.text_config.layer_types = [
                config.text_config.layer_types[i] for i in KEEP_DECODER_LAYERS
            ]
    if hasattr(config, "vision_config"):
        config.vision_config.depth = len(KEEP_VISION_BLOCKS)
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
        "num_token_id": num_token_id,
        "vocab_pruned": False,
        "kept_vision_blocks": KEEP_VISION_BLOCKS,
        "kept_decoder_layers": KEEP_DECODER_LAYERS,
    }
    with open(OUTPUT_PATH / "prune_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Summary
    total_params = sum(p.numel() for p in model.parameters())
    embed_params = model.get_input_embeddings().weight.numel()
    print("\n" + "=" * 60)
    print("Pruning complete!")
    print(f"  Vocab:       248077 active tokens (Embeddings padded to {new_vocab_size})")
    print(f"  Vision:      12 blocks -> {len(KEEP_VISION_BLOCKS)}")
    print(f"  Decoder:     24 layers -> {len(KEEP_DECODER_LAYERS)}")
    print(f"  Embed:       {embed_params/1e6:.1f}M params ({embed_params * 2 / 1e6:.1f} MB bf16)")
    print(f"  Total:       {total_params/1e6:.1f}M params")
    print(f"  Output:      {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()