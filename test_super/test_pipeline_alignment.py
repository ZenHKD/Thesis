"""
test_pipeline_alignment.py (Super)
===================================
Integration test that loads REAL data and verifies:

  1. TOKEN TABLE
     Prints every token with position, ID, decoded text, label, active status.
     Shows where the answer starts.

  2. VOCAB + EMBEDDING CHECK
     Verifies input_ids are valid for the model vocab.
     Checks embeddings are trainable.

  3. FORWARD PASS + LABEL ALIGNMENT
     Runs pipeline.forward() with real batch.
     Shows per-token CE breakdown.

  4. LOSS CHECK
     Computes SpatialLoss (Standard CE + SmoothL1).

  5. INFERENCE
     Runs pipeline.generate() with dataloader tensors.

Usage:
    python test_super/test_pipeline_alignment.py
    python test_super/test_pipeline_alignment.py --resolution 320p
    python test_super/test_pipeline_alignment.py --no-model
"""

import sys
import os
import re
import json
import math
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from super_model.dataloader import SpatialVLMDataset, get_dataloader
from super_model.loss import SpatialLoss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def decode_token(tokenizer, tok_id: int) -> str:
    """Decode a single token ID to a printable string."""
    if tok_id == -100:
        return "<IGNORED>"
    text = tokenizer.decode([tok_id], skip_special_tokens=False)
    text = text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    if not text.strip():
        text = repr(text)
    return text[:30]


# ---------------------------------------------------------------------------
# Section 1: Token Table
# ---------------------------------------------------------------------------

def print_token_table(input_ids: torch.Tensor, labels: torch.Tensor,
                      tokenizer):
    """Print per-token table: position | token_id | decoded | label | active."""
    ids  = input_ids[0].tolist()
    lbls = labels[0].tolist()
    L    = len(ids)

    answer_start = next((i for i, v in enumerate(lbls) if v != -100), L)
    n_active = sum(1 for v in lbls if v != -100)

    print(f"\n  Sequence length L = {L}")
    print(f"  Prompt tokens (ignored): {answer_start}")
    print(f"  Answer tokens (active):  {n_active}")
    print(f"  Answer starts at position: {answer_start}")

    for pos in range(L):
        tok_id  = ids[pos]
        lbl     = lbls[pos]
        decoded = decode_token(tokenizer, tok_id)
        lbl_str = str(lbl) if lbl != -100 else "-100"
        active  = "  YES  <--" if lbl != -100 else ""
        marker  = " <<< ANSWER START" if pos == answer_start else ""
        print(f"  {pos:>5}  {tok_id:>7}  {decoded:<32}  {lbl_str:>8}  {active}{marker}")

    print(f"  {'─'*72}")
    print(f"  Answer tokens decoded:")
    answer_toks = [ids[i] for i in range(answer_start, L) if lbls[i] != -100]
    print(f"    {tokenizer.decode(answer_toks, skip_special_tokens=False)!r}")
    print()


# ---------------------------------------------------------------------------
# Section 2: Vocab + Embedding Check (replaces old remapping check)
# ---------------------------------------------------------------------------

def check_vocab_and_embedding(pipeline, input_ids, labels):
    """Verify input IDs are valid for full vocab embed_tokens."""
    embed_size = pipeline.qwen.model.language_model.embed_tokens.weight.shape[0]
    max_id = input_ids.max().item()
    min_id = input_ids.min().item()
    print(f"\n  input_ids range: [{min_id}, {max_id}]")
    print(f"  embed_tokens rows: {embed_size}")
    print(f"  All in range: {'[OK]' if max_id < embed_size else '[FAIL - INDEX OUT OF RANGE]'}")

    # Check labels
    active_labels = labels[labels != -100]
    if len(active_labels) > 0:
        max_label = active_labels.max().item()
        print(f"  Active labels range: [{active_labels.min().item()}, {max_label}]")
        print(f"  All labels in vocab: {'[OK]' if max_label < embed_size else '[FAIL]'}")

    # Check embeddings are trainable (full fine-tuning)
    embed_trainable = pipeline.qwen.model.language_model.embed_tokens.weight.requires_grad
    print(f"  Embeddings trainable: {'[OK]' if embed_trainable else '[FAIL - should be trainable]'}")

    return max_id < embed_size and embed_trainable


# ---------------------------------------------------------------------------
# Section 3: Forward alignment
# ---------------------------------------------------------------------------

def print_forward_alignment(logits, labels, tokenizer):
    """Show per-position alignment after RTI shortening + label trim + shift."""
    L_orig  = labels.shape[1]
    L_prime = logits.shape[1]
    diff    = L_orig - L_prime

    print(f"\n  Original labels length  L  = {L_orig}")
    print(f"  Logits text length      L' = {L_prime}")
    print(f"  RTI diff (n_masks)         = {diff}  (tokens dropped from FRONT of labels)")

    # Labels are already in native Qwen IDs (no remapping needed)
    if diff > 0:
        trimmed_labels = labels[:, diff:]
    else:
        trimmed_labels = labels

    lbls_t = trimmed_labels[0].tolist()
    n_active = sum(1 for v in lbls_t[1:] if v != -100)
    print(f"  Active targets after shift: {n_active}")

    active_positions = [(t, lbls_t[t+1]) for t in range(len(lbls_t)-1) if lbls_t[t+1] != -100]

    # Per-token breakdown
    vocab_size = logits.shape[2]
    print(f"  Vocab size: {vocab_size}")
    print("\n    Pos   TokenID         Decoded      Pred         DecPred   P(target)    CE Loss  Match?")
    print("  " + "─" * 120)

    logits_cpu = logits.cpu().float()
    per_token_losses = []

    for t, token_id in active_positions:
        logit_vec = logits_cpu[0, t]
        log_probs = torch.log_softmax(logit_vec, dim=0)
        ce_loss = -log_probs[token_id].item()
        per_token_losses.append(ce_loss)
        p_target = math.exp(-ce_loss) if math.isfinite(ce_loss) else 0.0

        pred_id = logit_vec.argmax().item()
        # Decode directly (native Qwen IDs, no remapping)
        dec_target = decode_token(tokenizer, token_id)
        dec_pred = decode_token(tokenizer, pred_id)
        match_str = "[OK]" if token_id == pred_id else "[FAIL]"
        
        # Optionally, limit print to last N to avoid huge terminal spam
        if len(active_positions) < 60 or (t > active_positions[-60][0]):
            print(f"  {t+1:>5}  {token_id:>8}  {dec_target:>14}  {pred_id:>8}  {dec_pred:>14}  {p_target:>8.6f}  {ce_loss:>9.4f}  {match_str}")

    print("  " + "─" * 120)

    if per_token_losses:
        avg = sum(per_token_losses) / len(per_token_losses)
        print(f"  Average CE loss (raw)      = {avg:.6f}")
        print(f"  Expected (untrained) ≈ log({logits.size(-1)}) = {math.log(logits.size(-1)):.2f}")

    return per_token_losses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pipeline alignment test (Super)")
    parser.add_argument("--split",      default="train_sample",
                        choices=["train", "val", "test", "train_sample"])
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--category",       default=None,
                        choices=["mcq", "left_right", "distance", "count"],
                        help="Override sample-idx by filtering for a specific category")
    parser.add_argument("--resolution", default="320p",
                        choices=["1080p", "720p", "540p", "450p", "320p"])
    parser.add_argument("--device",     default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype",      default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--attn-impl",  default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--no-model",   action="store_true",
                        help="Skip model loading — token table + remap only")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    target_size = {"1080p": None, "720p": (1280, 720),
                   "540p": (960, 540), "450p": (800, 450),
                   "320p": (512, 320)}[args.resolution]

    # ------------------------------------------------------------------ #
    # Load model or processor only
    # ------------------------------------------------------------------ #
    if args.no_model:
        from transformers import AutoProcessor
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "super_model", "qwen3.5-super")
        print("Loading processor only (--no-model)...")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        pipeline  = None
    else:
        from super_model.pipeline import SpatialVLM, print_vram_usage
        print("=" * 70)
        print("LOADING MODEL (Super)")
        print("=" * 70)
        pipeline = SpatialVLM(dtype=dtype, device_map=args.device,
                              attn_implementation=args.attn_impl)
        processor = pipeline.processor
        print_vram_usage("after model load")

    tokenizer = processor.tokenizer

    # ------------------------------------------------------------------ #
    # Load dataset sample
    # ------------------------------------------------------------------ #
    print(f"\n{'='*70}")
    print(f"LOADING SAMPLE  (split={args.split}, idx={args.sample_idx}, res={args.resolution})")
    print("=" * 70)

    dataset = SpatialVLMDataset(args.split, processor=processor, target_size=target_size)
    loader  = get_dataloader(dataset, batch_size=1, shuffle=False,
                             num_workers=0, pin_memory=False)

    batch = None
    if args.category:
        for b in loader:
            if b["categories"][0] == args.category:
                batch = b
                break
        if batch is None:
            print(f"[!] No sample found with category={args.category}")
            return
    else:
        for i, b in enumerate(loader):
            if i == args.sample_idx:
                batch = b
                break
        if batch is None:
            print(f"[!] sample_idx={args.sample_idx} out of range")
            return

    print(f"  Image:    {batch['image_names'][0]}")
    print(f"  Category: {batch['categories'][0]}")
    print(f"  Answer:   {batch['answers'][0]}")
    print(f"  n_masks:  {len(batch['mask_positions'][0])}")
    print(f"  is_numeric: {batch['is_numeric'][0].item()}")
    print(f"  target_num: {batch['target_num'][0].item()}")

    # ------------------------------------------------------------------ #
    # SECTION 1: Token Table
    # ------------------------------------------------------------------ #
    print(f"\n{'='*70}")
    print("SECTION 1: TOKEN-LEVEL LABEL TABLE")
    print("=" * 70)

    print_token_table(batch["input_ids"], batch["labels"], tokenizer)

    if args.no_model:
        print("  [--no-model] Skipping forward pass and inference.")
        return

    # ------------------------------------------------------------------ #
    # SECTION 1b: Full Backbone Sequence Map
    # ------------------------------------------------------------------ #
    from super_model.pipeline import print_vram_usage, find_mask_positions

    print(f"\n{'='*70}")
    print("SECTION 1b: FULL BACKBONE SEQUENCE MAP")
    print("=" * 70)

    dev = pipeline.device
    dtype_model = next(pipeline.qwen.parameters()).dtype

    pixel_values_1b   = batch["pixel_values"].to(device=dev, dtype=dtype_model)
    image_grid_thw_1b = batch["image_grid_thw"].to(device=dev)
    depth_maps_1b     = batch["depth_maps"].to(device=dev, dtype=dtype_model)
    input_ids_1b      = batch["input_ids"].to(device=dev)

    # Build inputs_embeds (visual + text with RTI)
    with torch.no_grad():
        inputs_embeds, _region_tokens = pipeline._build_inputs_embeds(
            pixel_values_1b, image_grid_thw_1b, depth_maps_1b, input_ids_1b,
            rle_list=batch["rle_list"],
            mask_token_positions=batch["mask_positions"],
            decoded_masks=batch["decoded_masks"],
        )

    B, total_seq, D = inputs_embeds.shape
    n_text = input_ids_1b.shape[1]
    n_masks = len(batch["mask_positions"][0]) if batch["mask_positions"] else 0

    # Get vision patch grid info
    t_grid, h_grid, w_grid = [int(x) for x in image_grid_thw_1b[0].tolist()]
    h_vis, w_vis = h_grid // 2, w_grid // 2  # after 2x2 merger

    # Define the width
    WIDTH = 64

    def box_line(text=""):
        """Pad text and wrap with box borders."""
        # -4 accounts for "  │ " and " │" (the spaces and borders)
        inner_width = WIDTH - 4
        padded = text.ljust(inner_width)
        return f"  │ {padded} │"

    print(f"\n  ┌{'─' * (WIDTH - 2)}┐")
    print(box_line(f"BACKBONE INPUT:  inputs_embeds = [{B}, {total_seq}, {D}]"))
    print(box_line())
    print(box_line("Vision Encoder + Merger:"))
    print(box_line(f"  pixel_values: {list(pixel_values_1b.shape)}"))
    print(box_line(f"  image_grid_thw: [{t_grid}, {h_grid}, {w_grid}]"))
    print(box_line(f"  after merger (2×2): [{t_grid}, {h_vis}, {w_vis}]"))
    print(box_line(f"  -> visual_tokens: inline (replaced <|image_pad|>)"))
    print(box_line())
    print(box_line("Text Embeddings + RTI:"))
    print(box_line(f"  input_ids: [{B}, {n_text}]"))
    print(box_line(f"  n_masks (RTI 3->1 replace): {n_masks}"))
    print(box_line())
    print(box_line("Inline Padding Fusion:"))
    print(box_line("  Replaced <|image_pad|> tokens with visual embeddings"))
    print(box_line(f"  inputs_embeds = [{B}, {total_seq}, {D}]"))
    print(f"  └{'─' * (WIDTH - 2)}┘")

    # Now print the full sequence map
    ids  = input_ids_1b[0].tolist()
    lbls = batch["labels"][0].tolist()
    mask_positions_flat = batch["mask_positions"][0] if batch["mask_positions"] else []

    mask_token_len = len(tokenizer.encode("<mask>", add_special_tokens=False))
    rti_positions = set()
    for mp in mask_positions_flat:
        for offset in range(mask_token_len):
            if mp + offset < n_text:
                rti_positions.add(mp + offset)

    print(f"\n  Full Sequence Map (backbone sees {total_seq} positions):")
    print(f"  {'─'*80}")
    print(f"  {'Backbone':>8}  {'Source':>8}  {'Type':<20}  {'Content':<25}  {'Label':>6}  Active?")
    print(f"  {'Pos':>8}  {'Pos':>8}")
    print(f"  {'─'*80}")

    # Part 2: Text tokens (with RTI markers)
    answer_start = next((i for i, v in enumerate(lbls) if v != -100), n_text)

    # Show every text token with its backbone position
    mask_region_idx = 0
    i = 0
    
    # In Super, 3 tokens (<, mask, >) are replaced by 3 tokens (mask_rgb, mask_depth, mask_gdep)
    # Therefore backbone position is exactly 1:1 with input_ids

    while i < n_text:
        backbone_pos = i

        tok_id  = ids[i]
        lbl     = lbls[i]
        decoded = decode_token(tokenizer, tok_id)

        # Determine type
        if i in rti_positions:
            is_first = (i in [mp for mp in mask_positions_flat])
            if is_first:
                mask_region_idx += 1

            for mp in mask_positions_flat:
                if mp <= i < mp + mask_token_len:
                    offset = i - mp
                    break
            else:
                offset = 0

            if offset == 0:
                rti_label = "mask_rgb"
            elif offset == 1:
                rti_label = "mask_depth"
            else:
                rti_label = "mask_gdep"
                
            src_type = f"[RTI] Region {mask_region_idx}"
            content  = f"[{rti_label}] (was: {decoded})"
            lbl_str = str(lbl) if lbl != -100 else "─"
            active  = "  YES  ←" if lbl != -100 else ""
            print(f"  {backbone_pos:>8}  {i:>8}  {src_type:<20}  {content:<25}  {lbl_str:>6}  {active}")
        else:
            if i == answer_start:
                src_type = ">> ANSWER START"
                content  = decoded
            elif i < answer_start:
                src_type = "TEXT (prompt)"
                content  = decoded
            else:
                src_type = "TEXT (answer)"
                content  = decoded

            lbl_str = str(lbl) if lbl != -100 else "─"
            active  = "  YES  ←" if lbl != -100 else ""

            print(f"  {backbone_pos:>8}  {i:>8}  {src_type:<20}  {content:<25}  {lbl_str:>6}  {active}")
            
        i += 1

    print(f"  {'─'*80}")

    # Summary
    print(f"\n  Summary:")
    summary_vis_tok = int(h_vis * w_vis)
    print(f"    Visual padding:    {summary_vis_tok} tokens inline over <|image_pad|>")
    print(f"    Text tokens:       {n_text} tokens")
    print(f"    RTI replacements:  {n_masks} × 3 tokens = {n_masks*3} positions replaced")
    print(f"        [<] [mask] [>]  -> [mask_rgb] [mask_depth] [mask_gdep]")
    print(f"    Total backbone:    {total_seq} positions")
    print(f"    Depth map:         {list(depth_maps_1b.shape)}")
    print()

    # ------------------------------------------------------------------ #
    # SECTION 2: Token Remapping Check
    # ------------------------------------------------------------------ #
    print(f"\n{'='*70}")
    print("SECTION 2: VOCAB + EMBEDDING CHECK")
    print("=" * 70)

    dev = pipeline.device
    input_ids = batch["input_ids"].to(device=dev)
    labels    = batch["labels"].to(device=dev)

    vocab_ok = check_vocab_and_embedding(pipeline, input_ids, labels)

    # ------------------------------------------------------------------ #
    # SECTION 3: Forward pass + alignment
    # ------------------------------------------------------------------ #
    from super_model.pipeline import print_vram_usage

    print(f"\n{'='*70}")
    print("SECTION 3: FORWARD PASS LABEL ALIGNMENT")
    print("=" * 70)

    pixel_values   = batch["pixel_values"].to(device=dev, dtype=dtype)
    image_grid_thw = batch["image_grid_thw"].to(device=dev)
    depth_maps     = batch["depth_maps"].to(device=dev, dtype=dtype)

    pipeline.eval()
    with torch.no_grad():
        output = pipeline(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            depth_maps=depth_maps,
            input_ids=input_ids,
            rle_list=batch["rle_list"],
            mask_token_positions=batch["mask_positions"],
            decoded_masks=batch["decoded_masks"],
            mcq_token_positions=batch.get("mcq_token_positions"),
            lr_token_positions=batch.get("lr_token_positions"),
            dist_token_positions=batch.get("dist_token_positions"),
            count_token_positions=batch.get("count_token_positions"),
        )

    logits = output["logits"]
    dist_pred = output["dist_pred"]
    count_pred = output["count_pred"]
    mcq_logits = output.get("mcq_logits", None)
    lr_logits = output.get("lr_logits", None)

    print(f"  logits: {list(logits.shape)}")
    print(f"  dist_pred: {dist_pred.tolist()}")
    print(f"  count_pred: {count_pred.tolist()}")
    print(f"  mcq_logits: {[cl.shape if cl is not None else None for cl in mcq_logits] if mcq_logits else None}")
    print(f"  lr_logits: {[cl.shape if cl is not None else None for cl in lr_logits] if lr_logits else None}")

    per_token_losses = print_forward_alignment(logits, labels.cpu(), tokenizer)

    # ------------------------------------------------------------------ #
    # SECTION 4: Loss check
    # ------------------------------------------------------------------ #
    print(f"\n{'='*70}")
    print("SECTION 4: LOSS CHECK (Standard CE + SmoothL1)")
    print("=" * 70)

    # Disable label smoothing so we can perfectly match mathematical CE base calculation
    criterion = SpatialLoss(label_smoothing=0.0)
    device = logits.device
    
    categories = batch["categories"]
    B = len(categories)
    is_distance = torch.tensor([c == "distance" for c in categories], dtype=torch.bool, device=device)
    is_count    = torch.tensor([c == "count" for c in categories], dtype=torch.bool, device=device)
    is_mcq      = torch.tensor([c == "mcq" for c in categories], dtype=torch.bool, device=device)
    is_lr       = torch.tensor([c == "left_right" for c in categories], dtype=torch.bool, device=device)

    official_loss, components = criterion(
        logits,
        labels.to(device),
        dist_pred=dist_pred,
        dist_gt=batch["target_num"].to(device),
        is_distance=is_distance,
        count_pred=count_pred,
        count_gt=batch["target_num"].to(device),
        is_count=is_count,
        mcq_logits=mcq_logits,
        mcq_targets=batch.get("target_cat_index", torch.zeros(B, dtype=torch.long)).to(device),
        is_mcq=is_mcq,
        lr_logits=lr_logits,
        lr_targets=batch.get("target_cat_index", torch.zeros(B, dtype=torch.long)).to(device),
        is_lr=is_lr,
        return_components=True,
    )

    print(f"\n  SpatialLoss output: {official_loss.item():.6f}")
    print(f"  Components: ce={components['ce']:.6f}, dist={components.get('dist', 0.0):.6f}, count={components.get('count', 0.0):.6f}, mcq={components.get('mcq', 0.0):.6f}, lr={components.get('lr', 0.0):.6f}")
    
    avg_raw = sum(per_token_losses) / len(per_token_losses) if per_token_losses else 0.0
    print(f"  Manual CE (raw):      {avg_raw:.8f}")
    
    diff_check = abs(avg_raw - official_loss.item())
    if is_distance.any() or is_count.any():
        print(f"  (Contains Regression component — diff expected)")
    elif is_mcq.any() or is_lr.any():
        print(f"  (Contains Classification component — diff expected)")
    else:
        print(f"  Manual vs official diff: {diff_check:.8f}  "
              f"{'[MATCH]' if diff_check < 0.01 else '[MISMATCH!]'}")

    is_finite = math.isfinite(official_loss.item())
    print(f"  Finite: {is_finite}")
    print(f"  Full vocab: {logits.shape[2]}")

    # ------------------------------------------------------------------ #
    # SECTION 5: Inference
    # ------------------------------------------------------------------ #
    from super_model.pipeline import find_mask_positions

    print(f"\n{'='*70}")
    print("SECTION 5: INFERENCE (pipeline.generate)")
    print("=" * 70)

    question = batch["_question"][0] if "_question" in batch and len(batch["_question"]) > 0 else "Dummy Question <mask>"

    import re
    mask_idx = [0]
    def replace_mask(m):
        i = mask_idx[0]
        mask_idx[0] += 1
        return f"[Region {i}]: <|object_ref_start|>{m.group(1)}<|object_ref_end|>"
    question = re.sub(r'(<mask.*?>)', replace_mask, question)

    # RGB tokens (first image)
    h_p_rgb, w_p_rgb = image_grid_thw[0, 1].item(), image_grid_thw[0, 2].item()
    num_visual_rgb = int((h_p_rgb // 2) * (w_p_rgb // 2))
    # Depth tokens (second image)
    h_p_dep, w_p_dep = image_grid_thw[1, 1].item(), image_grid_thw[1, 2].item()
    num_visual_dep = int((h_p_dep // 2) * (w_p_dep // 2))
    
    vision_str_1 = "Picture 1 (RGB): <|vision_start|>" + "<|image_pad|>" * num_visual_rgb + "<|vision_end|>\n"
    vision_str_2 = "Picture 2 (Depth): <|vision_start|>" + "<|image_pad|>" * num_visual_dep + "<|vision_end|>\n"
    user_str = f"<|im_start|>user\n{vision_str_1}{vision_str_2}{question}<|im_end|>\n"
    eval_prompt = f"<|im_start|>assistant\n"
    
    full_prompt = user_str + eval_prompt

    # Build generation-format input_ids
    gen_input_ids = pipeline.processor.tokenizer(
        full_prompt, return_tensors="pt", padding=False
    ).input_ids.to(device=dev)

    # Find <mask> positions
    mask_positions = find_mask_positions(gen_input_ids, pipeline.processor.tokenizer)

    rle_list      = batch["rle_list"]
    decoded_masks = batch["decoded_masks"]
    n = min(len(mask_positions), len(rle_list[0]))
    mask_positions = mask_positions[:n]

    print(f"  Question:     {question}")
    print(f"  GT answer:    {batch['answers'][0]}")
    print(f"  n_masks:      {n}")

    pipeline.eval()
    with torch.no_grad():
        output_ids = pipeline.generate(
            pixel_values, image_grid_thw, depth_maps, gen_input_ids,
            rle_list=[rle_list[0][:n]] if n > 0 else None,
            mask_token_positions=[mask_positions] if n > 0 else None,
            decoded_masks=[decoded_masks[0][:n]] if n > 0 else None,
            max_new_tokens=20,
        )

    raw_output = pipeline.processor.tokenizer.decode(
        output_ids[0], skip_special_tokens=False
    ).replace("<|endoftext|>", "").replace("<|im_end|>", "").strip()
    parsed = pipeline.parse_output(raw_output)

    print(f"\n  Raw output:    {raw_output!r}")
    print(f"  Parsed cat:    {parsed.get('category')!r}")
    print(f"  Parsed answer: {parsed.get('answer')!r}")
    print(f"  GT answer:     {batch['answers'][0]!r}")

    print_vram_usage("after inference")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print(f"\n{'='*70}")
    print("ALL SECTIONS COMPLETE")
    print("=" * 70)

    checks = [
        (vocab_ok, "Token IDs valid for embed + embeddings trainable"),
        (is_finite, f"Loss is finite ({official_loss.item():.4f})"),
        (logits.shape[2] == pipeline.qwen.model.language_model.embed_tokens.weight.shape[0],
         f"Logits vocab matches model ({logits.shape[2]} == {pipeline.qwen.model.language_model.embed_tokens.weight.shape[0]})"),
    ]
    for ok, msg in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {msg}")


if __name__ == "__main__":
    main()
