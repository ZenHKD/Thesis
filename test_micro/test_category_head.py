"""
test_category_head.py
======================
Integration test for CategoryHead — verifies:

  1. TOKEN POSITION CHECK
     - <|cat|> token position is correctly found in input_ids
     - <mask> positions are correctly found
     - Positions map to valid indices in hidden states

  2. HIDDEN STATE EXTRACTION
     - h_cat is extracted from the correct position (at <|cat|> token)
     - h_masks are extracted from the correct positions (at <mask> tokens)
     - h_cat contains full causal context (it's AFTER question + reasoning)
     - h_masks contain RTI-injected features

  3. CATEGORY HEAD FORWARD
     - Bilinear attention produces valid scores [N_masks]
     - Scores are differentiable (gradient flows)
     - Output shape matches number of masks

  4. LOSS COMPUTATION
     - CrossEntropy loss is computed correctly for categorical samples
     - target_cat_index maps to the correct mask

  5. MULTI-SAMPLE COVERAGE
     - Tests both MCQ and Left/Right samples

Usage:
    python test_micro/test_category_head.py
    python test_micro/test_category_head.py --no-model    # Token checks only
    python test_micro/test_category_head.py --category mcq
"""

import sys
import os
import argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataloader.dataloader import SpatialVLMDataset, get_dataloader


def decode_token(tokenizer, tok_id: int) -> str:
    if tok_id == -100:
        return "<IGNORED>"
    text = tokenizer.decode([tok_id], skip_special_tokens=False)
    text = text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    if not text.strip():
        text = repr(text)
    return text[:40]


# =========================================================================
# Section 1: Token Position Checks
# =========================================================================

def check_token_positions(batch, tokenizer, num_token_id, cat_token_id):
    """Verify <|cat|> and <mask> positions are correct."""
    print(f"\n{'='*70}")
    print("SECTION 1: TOKEN POSITION CHECK")
    print("=" * 70)

    input_ids = batch["input_ids"][0].tolist()
    cat_pos = batch.get("cat_token_positions", [None])[0]
    mask_positions = batch["mask_positions"][0]
    category = batch["categories"][0]
    is_categorical = batch["is_categorical"][0].item()
    target_cat = batch.get("target_cat_index", torch.tensor([-1]))[0].item()

    L = len(input_ids)
    print(f"  Sequence length:   {L}")
    print(f"  Category:          {category}")
    print(f"  is_categorical:    {is_categorical}")
    print(f"  target_cat_index:  {target_cat}")
    print(f"  n_masks:           {len(mask_positions)}")
    print(f"  cat_token_pos:     {cat_pos}")

    all_ok = True

    # Check 1a: <|cat|> position
    print(f"\n  --- <|cat|> Token ---")
    if is_categorical:
        if cat_pos is None or cat_pos < 0:
            print(f"  [FAIL] is_categorical=True but cat_token_pos={cat_pos}")
            all_ok = False
        else:
            actual_id = input_ids[cat_pos]
            if actual_id == cat_token_id:
                print(f"  [OK] Position {cat_pos} has token ID {actual_id} = <|cat|>")
            else:
                print(f"  [FAIL] Position {cat_pos} has token ID {actual_id} "
                      f"(decoded: '{decode_token(tokenizer, actual_id)}'), "
                      f"expected {cat_token_id} (<|cat|>)")
                all_ok = False

            # Check that <|cat|> is in the ANSWER portion (after question)
            labels = batch["labels"][0].tolist()
            answer_start = next((i for i, v in enumerate(labels) if v != -100), L)
            if cat_pos >= answer_start:
                print(f"  [OK] <|cat|> at pos {cat_pos} is in answer portion (starts at {answer_start})")
            else:
                print(f"  [WARN] <|cat|> at pos {cat_pos} is BEFORE answer start {answer_start}")
    else:
        if cat_pos is None or cat_pos < 0:
            print(f"  [OK] Not categorical, cat_token_pos={cat_pos} (correctly skipped)")
        else:
            print(f"  [WARN] Not categorical but cat_token_pos={cat_pos}")

    # Check 1b: <mask> positions
    print(f"\n  --- <mask> Positions ---")
    mask_token_len = len(tokenizer.encode("<mask>", add_special_tokens=False))
    print(f"  mask_token_len (BPE): {mask_token_len}")

    for i, mp in enumerate(mask_positions):
        if 0 <= mp < L:
            # Show tokens at and around mask position
            ctx_start = max(0, mp - 2)
            ctx_end = min(L, mp + mask_token_len + 2)
            ctx_tokens = [f"{decode_token(tokenizer, input_ids[j])}" for j in range(ctx_start, ctx_end)]
            marker_idx = mp - ctx_start
            ctx_str = " | ".join(ctx_tokens)
            print(f"  Mask {i}: pos={mp}  context: [{ctx_str}]")
        else:
            print(f"  [FAIL] Mask {i}: pos={mp} is OUT OF RANGE [0, {L})")
            all_ok = False

    # Check 1c: Mask positions are in QUESTION portion (before answer)
    labels = batch["labels"][0].tolist()
    answer_start = next((i for i, v in enumerate(labels) if v != -100), L)
    for i, mp in enumerate(mask_positions):
        if mp >= answer_start:
            print(f"  [WARN] Mask {i} at pos {mp} is AFTER answer_start {answer_start}")

    # Check 1d: Causal ordering — masks should be BEFORE <|cat|>
    if is_categorical and cat_pos is not None and cat_pos >= 0:
        for i, mp in enumerate(mask_positions):
            if mp >= cat_pos:
                print(f"  [FAIL] Mask {i} at pos {mp} is AFTER <|cat|> at pos {cat_pos}!")
                all_ok = False
            else:
                print(f"  [OK] Mask {i} at pos {mp} < <|cat|> at pos {cat_pos} (causal order ✓)")

    # ─── Section 1f: DETAILED TOKEN TABLE for CategoryHead inputs ───
    print(f"\n  {'='*70}")
    print("  SECTION 1b: CATEGORY HEAD INPUT TOKEN TABLE")
    print(f"  {'='*70}")

    labels = batch["labels"][0].tolist()
    answer_start = next((i for i, v in enumerate(labels) if v != -100), L)

    # Print each mask's 3-token span
    print(f"\n  ── h_masks: {len(mask_positions)} masks × {mask_token_len} tokens = "
          f"{len(mask_positions) * mask_token_len} hidden states ──")
    print(f"  {'Mask':<6} {'Offset':<8} {'Pos':<6} {'TokenID':>8}  {'Decoded':<25} {'Role':<15} {'Label':>6}")
    print(f"  {'─'*85}")

    rti_roles = ["region_rgb", "region_depth", "region_geo"]
    for i, mp in enumerate(mask_positions):
        for offset in range(mask_token_len):
            pos = mp + offset
            if 0 <= pos < L:
                tok_id = input_ids[pos]
                decoded = decode_token(tokenizer, tok_id)
                role = rti_roles[offset] if offset < len(rti_roles) else f"rti_{offset}"
                lbl = labels[pos]
                lbl_str = str(lbl) if lbl != -100 else "─"
                marker = "  ← KEY" if offset == 0 else ""
                print(f"  {i:<6} {offset:<8} {pos:<6} {tok_id:>8}  {decoded:<25} {role:<15} {lbl_str:>6}{marker}")
        if i < len(mask_positions) - 1:
            print(f"  {'·'*85}")

    # Print <|cat|> token context (5 tokens before + <|cat|> + 2 after)
    print(f"\n  ── h_cat: query token ──")
    if is_categorical and cat_pos is not None and cat_pos >= 0:
        ctx_start = max(0, cat_pos - 5)
        ctx_end = min(L, cat_pos + 3)
        print(f"  {'Pos':<6} {'TokenID':>8}  {'Decoded':<25} {'Section':<15} {'Label':>6}")
        print(f"  {'─'*70}")
        for pos in range(ctx_start, ctx_end):
            tok_id = input_ids[pos]
            decoded = decode_token(tokenizer, tok_id)
            lbl = labels[pos]
            lbl_str = str(lbl) if lbl != -100 else "─"
            if pos == cat_pos:
                section = "<<< h_cat QUERY"
            elif pos < answer_start:
                section = "prompt"
            else:
                section = "answer"
            print(f"  {pos:<6} {tok_id:>8}  {decoded:<25} {section:<15} {lbl_str:>6}")
    else:
        print("  [SKIP] No <|cat|> token")

    # Summary diagram
    print(f"\n  ── CategoryHead Data Flow ──")
    print(f"  ┌─────────────────────────────────────────────────────────┐")
    print(f"  │  h_masks: {len(mask_positions)} masks × concat(3 tokens) = "
          f"[{len(mask_positions)}, {mask_token_len * 1024}]       │")
    if is_categorical and cat_pos is not None and cat_pos >= 0:
        print(f"  │  h_cat:   1 token at pos {cat_pos:<4} = [{1024}]"
              f"                          │")
        print(f"  │                                                         │")
        print(f"  │  query = W_q(h_cat)        → [256]                      │")
        print(f"  │  keys  = W_k(h_masks)      → [{len(mask_positions)}, 256]"
              f"                      │")
        print(f"  │  scores = dot(query, keys)  → [{len(mask_positions)}]"
              f"                           │")
        print(f"  │  target = mask {target_cat}"
              f"                                            │")
    print(f"  └─────────────────────────────────────────────────────────┘")

    return all_ok


# =========================================================================
# Section 2: Hidden State Extraction
# =========================================================================

def check_hidden_states(pipeline, batch):
    """Verify hidden states at mask and <|cat|> positions."""
    print(f"\n{'='*70}")
    print("SECTION 2: HIDDEN STATE EXTRACTION")
    print("=" * 70)

    dev = pipeline.device
    dtype = next(pipeline.qwen.parameters()).dtype

    pixel_values = batch["pixel_values"].to(device=dev, dtype=dtype)
    pixel_values_rgb = batch["pixel_values_rgb"].to(device=dev, dtype=dtype)
    image_grid_thw = batch["image_grid_thw"].to(device=dev)
    depth_maps = batch["depth_maps"].to(device=dev, dtype=dtype)
    input_ids = batch["input_ids"].to(device=dev)
    attention_mask = batch["attention_mask"].to(device=dev)

    cat_pos = batch.get("cat_token_positions", [None])[0]
    mask_positions = batch["mask_positions"][0]

    with torch.no_grad():
        # Step 0: Get ORIGINAL text embeddings (BEFORE RTI) for comparison
        embed_layer = pipeline.qwen.model.language_model.embed_tokens
        orig_text_embeds = embed_layer(input_ids).clone()  # [B, L, D]

        # Step 1: Build inputs_embeds (WITH RTI injection)
        inputs_embeds, n_visual = pipeline._build_inputs_embeds(
            pixel_values, pixel_values_rgb, image_grid_thw, depth_maps, input_ids,
            rle_list=batch["rle_list"],
            mask_token_positions=batch["mask_positions"],
            decoded_masks=batch["decoded_masks"],
        )

        # Step 1b: Verify RTI injection actually modified mask embeddings
        print(f"\n  --- RTI Injection Verification ---")
        mask_token_len = len(pipeline.processor.tokenizer.encode("<mask>", add_special_tokens=False))
        for i, mp in enumerate(mask_positions):
            for offset in range(mask_token_len):
                pos = mp + offset
                if pos < orig_text_embeds.shape[1] and pos < inputs_embeds.shape[1]:
                    orig_emb = orig_text_embeds[0, pos, :]
                    rti_emb = inputs_embeds[0, pos, :]  # n_visual=0, so same index
                    cos = F.cosine_similarity(orig_emb.float().unsqueeze(0),
                                              rti_emb.float().unsqueeze(0)).item()
                    diff_norm = (rti_emb - orig_emb).float().norm().item()
                    rti_names = ["region_rgb", "region_depth", "region_geo"]
                    name = rti_names[offset] if offset < len(rti_names) else f"tok_{offset}"
                    if diff_norm < 1e-6:
                        print(f"  [FAIL] Mask {i} [{name}] pos={pos}: "
                              f"embedding NOT modified by RTI! (diff_norm={diff_norm:.6f})")
                        all_ok = False
                    else:
                        print(f"  [OK]   Mask {i} [{name}] pos={pos}: "
                              f"RTI injected (diff_norm={diff_norm:.2f}, cos_sim={cos:.4f})")

        # Step 2: Run backbone
        full_attn = attention_mask
        hidden = pipeline._backbone_forward(inputs_embeds, attention_mask=full_attn)

        # Step 3: Normalize
        h_normed = pipeline.qwen.model.language_model.norm(hidden)

    B, T, D = h_normed.shape
    print(f"\n  h_normed shape: [{B}, {T}, {D}]")
    print(f"  n_visual offset: {n_visual}")

    all_ok = True

    # Check 2a: h_cat extraction
    print(f"\n  --- h_cat (query context) ---")
    if cat_pos is not None and cat_pos >= 0:
        adj_cat = n_visual + cat_pos
        if 0 <= adj_cat < T:
            h_cat = h_normed[0, adj_cat, :]
            norm_cat = h_cat.float().norm().item()
            mean_cat = h_cat.float().mean().item()
            std_cat = h_cat.float().std().item()
            print(f"  [OK] h_cat at adj_pos={adj_cat}: norm={norm_cat:.4f}, mean={mean_cat:.6f}, std={std_cat:.6f}")

            # Sanity: h_cat should not be all zeros
            if norm_cat < 1e-6:
                print(f"  [FAIL] h_cat is all zeros!")
                all_ok = False
        else:
            print(f"  [FAIL] adj_cat_pos={adj_cat} out of range [0, {T})")
            all_ok = False
    else:
        print(f"  [SKIP] No <|cat|> token (not categorical)")

    # Check 2b: h_masks extraction (concat 3 RTI tokens per mask)
    print(f"\n  --- h_masks (candidate keys, 3-token concat) ---")
    mask_token_len = len(pipeline.processor.tokenizer.encode("<mask>", add_special_tokens=False))
    mask_hiddens = []
    for i, mp in enumerate(mask_positions):
        token_hiddens = []
        for offset in range(mask_token_len):
            adj_mp = n_visual + mp + offset
            if 0 <= adj_mp < T:
                token_hiddens.append(h_normed[0, adj_mp, :])
        if token_hiddens:
            h_mask = torch.cat(token_hiddens, dim=0)  # [3072]
            norm_m = h_mask.float().norm().item()
            print(f"  Mask {i}: concat {len(token_hiddens)} tokens (pos {mp}..{mp+mask_token_len-1}), "
                  f"dim={h_mask.shape[0]}, norm={norm_m:.4f}")
            mask_hiddens.append(h_mask)

            if norm_m < 1e-6:
                print(f"  [FAIL] h_mask[{i}] is all zeros!")
                all_ok = False
        else:
            print(f"  [FAIL] Mask {i}: no valid positions in range [0, {T})")
            all_ok = False

    # Check 2c: Are h_masks distinguishable from each other?
    if len(mask_hiddens) >= 2:
        print(f"\n  --- Mask Pairwise Cosine Similarity ---")
        for i in range(len(mask_hiddens)):
            for j in range(i + 1, len(mask_hiddens)):
                cos_sim = F.cosine_similarity(
                    mask_hiddens[i].float().unsqueeze(0),
                    mask_hiddens[j].float().unsqueeze(0),
                ).item()
                status = "[WARN: very similar]" if cos_sim > 0.99 else "[OK: distinguishable]"
                print(f"  cos(mask_{i}, mask_{j}) = {cos_sim:.6f}  {status}")

    # Check 2d: h_cat (1024) vs h_masks (3072) — skipped, different dimensions after concat

    return all_ok, h_normed, n_visual


# =========================================================================
# Section 3: CategoryHead Forward
# =========================================================================

def check_cat_head_forward(pipeline, h_normed, n_visual, batch):
    """Run CategoryHead and verify output."""
    print(f"\n{'='*70}")
    print("SECTION 3: CATEGORY HEAD FORWARD")
    print("=" * 70)

    cat_pos = batch.get("cat_token_positions", [None])[0]
    mask_positions = batch["mask_positions"][0]
    target_cat = batch.get("target_cat_index", torch.tensor([-1]))[0].item()
    category = batch["categories"][0]

    if cat_pos is None or cat_pos < 0:
        print("  [SKIP] Not a categorical sample")
        return True

    all_ok = True
    T = h_normed.shape[1]

    # Extract h_cat and h_masks (3-token concat per mask)
    adj_cat = n_visual + cat_pos
    h_cat = h_normed[0, adj_cat, :]

    mask_token_len = 3
    mask_hiddens = []
    for mp in mask_positions:
        token_hiddens = []
        for offset in range(mask_token_len):
            adj_mp = n_visual + mp + offset
            if 0 <= adj_mp < T:
                token_hiddens.append(h_normed[0, adj_mp, :])
        if token_hiddens:
            mask_hiddens.append(torch.cat(token_hiddens, dim=0))

    if not mask_hiddens:
        print("  [FAIL] No valid mask hiddens!")
        return False

    h_masks = torch.stack(mask_hiddens, dim=0)  # [N_masks, 1024]
    N_masks = h_masks.shape[0]

    print(f"  h_masks shape: [{N_masks}, {h_masks.shape[1]}]")
    print(f"  h_cat shape:   [{h_cat.shape[0]}]")
    print(f"  Category:      {category}")
    print(f"  target_index:  {target_cat}")

    # Run CategoryHead
    # Need to enable grad for this test
    h_masks_grad = h_masks.detach().clone().requires_grad_(True)
    h_cat_grad = h_cat.detach().clone().requires_grad_(True)

    scores = pipeline.cat_head(h_masks_grad, h_cat_grad)

    print(f"\n  --- CategoryHead Output ---")
    print(f"  scores shape:  {list(scores.shape)}  (expected: [{N_masks}])")
    if list(scores.shape) != [N_masks]:
        print(f"  [FAIL] Shape mismatch!")
        all_ok = False
    else:
        print(f"  [OK] Shape correct")

    # Print raw scores
    probs = F.softmax(scores.float(), dim=0)
    print(f"\n  {'Mask':<6} {'Score':>10} {'Prob':>10} {'Target?':>10}")
    print(f"  {'─'*40}")
    for i in range(N_masks):
        is_target = " ← TARGET" if i == target_cat else ""
        print(f"  {i:<6} {scores[i].item():>10.4f} {probs[i].item():>10.4f} {is_target}")

    pred_idx = scores.argmax().item()
    print(f"\n  Predicted: mask {pred_idx}")
    print(f"  Target:    mask {target_cat}")
    print(f"  Correct:   {'YES ✓' if pred_idx == target_cat else 'NO ✗'}")

    # Check gradient flow
    if target_cat >= 0 and target_cat < N_masks:
        loss = F.cross_entropy(
            scores.float().unsqueeze(0),
            torch.tensor([target_cat], device=scores.device),
        )
        loss.backward()
        
        has_grad_masks = h_masks_grad.grad is not None and h_masks_grad.grad.abs().sum() > 0
        has_grad_cat = h_cat_grad.grad is not None and h_cat_grad.grad.abs().sum() > 0
        
        print(f"\n  --- Gradient Check ---")
        print(f"  CE Loss:            {loss.item():.6f}")
        print(f"  Grad flows to h_masks: {'[OK]' if has_grad_masks else '[FAIL]'}")
        print(f"  Grad flows to h_cat:   {'[OK]' if has_grad_cat else '[FAIL]'}")
        
        if has_grad_masks:
            grad_norms = h_masks_grad.grad.float().norm(dim=1)
            for i in range(N_masks):
                print(f"    grad_norm(mask_{i}) = {grad_norms[i].item():.6f}")
        
        if not has_grad_masks or not has_grad_cat:
            all_ok = False

    return all_ok


# =========================================================================
# Section 4: Full Pipeline Forward (end-to-end)
# =========================================================================

def check_full_forward(pipeline, batch, criterion):
    """Run the full pipeline.forward() and verify cat_logits in output."""
    print(f"\n{'='*70}")
    print("SECTION 4: FULL PIPELINE FORWARD + LOSS")
    print("=" * 70)

    dev = pipeline.device
    dtype = next(pipeline.qwen.parameters()).dtype
    
    pixel_values = batch["pixel_values"].to(device=dev, dtype=dtype)
    pixel_values_rgb = batch["pixel_values_rgb"].to(device=dev, dtype=dtype)
    image_grid_thw = batch["image_grid_thw"].to(device=dev)
    depth_maps = batch["depth_maps"].to(device=dev, dtype=dtype)
    input_ids = batch["input_ids"].to(device=dev)
    labels = batch["labels"].to(device=dev)
    attention_mask = batch["attention_mask"].to(device=dev)

    pipeline.eval()
    with torch.no_grad():
        output = pipeline(
            pixel_values=pixel_values,
            pixel_values_rgb=pixel_values_rgb,
            image_grid_thw=image_grid_thw,
            depth_maps=depth_maps,
            input_ids=input_ids,
            rle_list=batch["rle_list"],
            mask_token_positions=batch["mask_positions"],
            decoded_masks=batch["decoded_masks"],
            num_token_positions=batch.get("num_token_positions"),
            cat_token_positions=batch.get("cat_token_positions"),
            attention_mask=attention_mask,
        )

    cat_logits = output.get("cat_logits", None)
    category = batch["categories"][0]
    is_cat = batch["is_categorical"][0].item()

    print(f"  Category:       {category}")
    print(f"  is_categorical: {is_cat}")

    all_ok = True

    if is_cat:
        if cat_logits is None or len(cat_logits) == 0:
            print(f"  [FAIL] is_categorical=True but cat_logits is empty!")
            all_ok = False
        elif cat_logits[0] is None:
            print(f"  [FAIL] cat_logits[0] is None!")
            all_ok = False
        else:
            scores = cat_logits[0]
            n_masks = len(batch["mask_positions"][0])
            print(f"  cat_logits[0] shape: {list(scores.shape)} (expected [{n_masks}])")
            if list(scores.shape) != [n_masks]:
                print(f"  [FAIL] Shape mismatch!")
                all_ok = False
            else:
                print(f"  [OK] cat_logits shape correct")

            # Compute loss
            target_cat = batch.get("target_cat_index", torch.tensor([-1]))[0]
            loss, components = criterion(
                output["logits"], labels,
                output["num_pred"], batch["target_num"].to(dev),
                batch["is_numeric"].to(dev),
                cat_logits=cat_logits,
                cat_targets=target_cat.unsqueeze(0).to(dev),
                is_categorical=batch["is_categorical"].to(dev),
                return_components=True,
            )
            print(f"\n  --- Loss Components ---")
            print(f"  total_loss:  {loss.item():.6f}")
            print(f"  CE:          {components['ce']:.6f}")
            print(f"  SL1:         {components['sl1']:.6f}")
            print(f"  Cat CE:      {components['cat_ce']:.6f}")
            
            if components['cat_ce'] == 0.0 and is_cat:
                print(f"  [FAIL] Cat CE is 0 for a categorical sample!")
                all_ok = False
            else:
                print(f"  [OK] Cat CE is non-zero")
    else:
        if cat_logits is None or len(cat_logits) == 0 or cat_logits[0] is None:
            print(f"  [OK] Not categorical, cat_logits correctly empty/None")
        else:
            print(f"  [WARN] Not categorical but cat_logits has values")

    return all_ok


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="CategoryHead Integration Test")
    parser.add_argument("--split",      default="train_sample",
                        choices=["train", "val", "test", "train_sample"])
    parser.add_argument("--category",   default=None,
                        choices=["mcq", "left_right", "distance", "count"])
    parser.add_argument("--resolution", default="320p",
                        choices=["1080p", "720p", "540p", "450p", "320p"])
    parser.add_argument("--device",     default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype",      default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--attn-impl",  default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--no-model",   action="store_true",
                        help="Token position checks only (no GPU needed)")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    target_size = {"1080p": None, "720p": (1280, 720),
                   "540p": (960, 540), "450p": (800, 450),
                   "320p": (512, 320)}[args.resolution]

    # Load model or processor
    if args.no_model:
        from transformers import AutoProcessor
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "model_micro", "qwen3.5-micro")
        print("Loading processor only (--no-model)...")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        pipeline = None
    else:
        from model_micro.pipeline import SpatialVLM, print_vram_usage
        print("=" * 70)
        print("LOADING MODEL")
        print("=" * 70)
        pipeline = SpatialVLM(dtype=dtype, device_map=args.device,
                              attn_implementation=args.attn_impl)
        processor = pipeline.processor
        print_vram_usage("after model load")

    tokenizer = processor.tokenizer

    # Load dataset
    dataset = SpatialVLMDataset(args.split, processor=processor, target_size=target_size)
    loader = get_dataloader(dataset, batch_size=1, shuffle=False,
                            num_workers=0, pin_memory=False)

    # Get special token IDs
    from model_micro.pipeline import NUM_TOKEN_ID, CAT_TOKEN_ID
    print(f"\n  <|num|> token ID: {NUM_TOKEN_ID}")
    print(f"  <|cat|> token ID: {CAT_TOKEN_ID}")

    # Test categories
    categories_to_test = [args.category] if args.category else ["mcq", "left_right"]
    
    overall_ok = True
    from model_micro.loss import SpatialLoss
    criterion = SpatialLoss(alpha=0.1, gamma=1.0)

    for cat_name in categories_to_test:
        print(f"\n\n{'#'*70}")
        print(f"# TESTING CATEGORY: {cat_name.upper()}")
        print(f"{'#'*70}")

        # Find a sample of this category
        batch = None
        for b in loader:
            if b["categories"][0] == cat_name:
                batch = b
                break

        if batch is None:
            print(f"  [SKIP] No {cat_name} sample found in {args.split}")
            continue

        print(f"  Image:    {batch['image_names'][0]}")
        print(f"  Category: {batch['categories'][0]}")
        print(f"  Answer:   {batch['answers'][0]}")
        print(f"  n_masks:  {len(batch['mask_positions'][0])}")

        # Section 1: Token positions
        ok1 = check_token_positions(batch, tokenizer, NUM_TOKEN_ID, CAT_TOKEN_ID)

        if args.no_model:
            print("\n  [--no-model] Skipping Sections 2-4.")
            overall_ok = overall_ok and ok1
            continue

        # Section 2: Hidden states
        ok2, h_normed, n_visual = check_hidden_states(pipeline, batch)

        # Section 3: CategoryHead forward
        ok3 = check_cat_head_forward(pipeline, h_normed, n_visual, batch)

        # Section 4: Full pipeline forward + loss
        ok4 = check_full_forward(pipeline, batch, criterion)

        overall_ok = overall_ok and ok1 and ok2 and ok3 and ok4

    # Summary
    print(f"\n\n{'='*70}")
    print(f"OVERALL RESULT: {'ALL PASSED ✓' if overall_ok else 'SOME FAILED ✗'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
