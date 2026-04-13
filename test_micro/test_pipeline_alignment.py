"""
test_pipeline_alignment.py (Micro)
===================================
Integration test that loads REAL data and verifies:

  1. TOKEN TABLE
     Prints every token with position, ID, decoded text, label, active status.
     Shows where the answer starts and <num> token position.

  2. VOCAB + EMBEDDING CHECK
     Verifies input_ids are valid for the model vocab.
     Checks embeddings are trainable.

  3. FORWARD PASS + LABEL ALIGNMENT
     Runs pipeline.forward() with real batch.
     Shows per-token CE breakdown (uses final loop step logits).

  4. LOSS CHECK
     Computes SpatialLoss (LoopLM per-step CE + entropy + SmoothL1).

  5. INFERENCE
     Runs pipeline.generate() with dataloader tensors.

Usage:
    python test_micro/test_pipeline_alignment.py
    python test_micro/test_pipeline_alignment.py --resolution 320p
    python test_micro/test_pipeline_alignment.py --no-model
"""

import sys
import os
import re
import json
import math
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataloader.dataloader_new import SpatialVLMDataset, get_dataloader
from model_micro.loss import SpatialLoss


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

    head_rows   = list(range(min(5, L)))
    pre_answer  = list(range(max(0, answer_start - 3), answer_start))
    answer_rows = list(range(answer_start, min(answer_start + 20, L)))
    tail_rows   = list(range(max(answer_start + 20, L - 3), L))

    rows_to_show = sorted(set(head_rows + pre_answer + answer_rows + tail_rows))

    header = f"  {'Pos':>5}  {'TokID':>7}  {'Decoded Token':<32}  {'Label':>8}  Active?"
    print(f"\n{header}")
    print(f"  {'─'*72}")

    prev = -1
    for pos in rows_to_show:
        if pos - prev > 1:
            print(f"  {'...':>5}  {'...':>7}  {'...':^32}  {'...':>8}  ...")
        tok_id  = ids[pos]
        lbl     = lbls[pos]
        decoded = decode_token(tokenizer, tok_id)
        lbl_str = str(lbl) if lbl != -100 else "-100"
        active  = "  YES  <--" if lbl != -100 else ""
        marker  = " <<< ANSWER START" if pos == answer_start else ""
        print(f"  {pos:>5}  {tok_id:>7}  {decoded:<32}  {lbl_str:>8}  {active}{marker}")
        prev = pos

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

def print_forward_alignment(logits, labels, tokenizer, pipeline):
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

    print(f"\n  {'Pos':>5}  {'TokenID':>8}  {'Decoded':>14}  "
          f"{'Pred':>8}  {'P(target)':>10}  {'CE Loss':>9}  Match?")
    print(f"  {'─'*85}")

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
        target_text = decode_token(tokenizer, token_id)
        match = "[OK]" if pred_id == token_id else "[FAIL]"

        print(f"  {t:>5}  {token_id:>8}  {target_text:>14}  "
              f"{pred_id:>8}  {p_target:>10.6f}  {ce_loss:>9.4f}  {match}")

    if per_token_losses:
        avg = sum(per_token_losses) / len(per_token_losses)
        print(f"  {'─'*85}")
        print(f"  Average CE loss = {avg:.6f}")
        print(f"  Expected (untrained) ≈ log({vocab_size}) = {math.log(vocab_size):.2f}")

    return per_token_losses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pipeline alignment test (Micro)")
    parser.add_argument("--split",      default="train_sample",
                        choices=["train", "val", "test", "train_sample"])
    parser.add_argument("--sample-idx", type=int, default=0)
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
                                  "model_micro", "qwen3.5-micro")
        print("Loading processor only (--no-model)...")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        pipeline  = None
    else:
        from model_micro.pipeline import SpatialVLM, print_vram_usage
        print("=" * 70)
        print("LOADING MODEL (Micro)")
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
    from model_micro.pipeline import SpatialVLM, print_vram_usage, find_mask_positions

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
        inputs_embeds, n_visual = pipeline._build_inputs_embeds(
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

    print(f"\n  ┌─────────────────────────────────────────────────────────────┐")
    print(f"  │  BACKBONE INPUT:  inputs_embeds = [{B}, {total_seq}, {D}]")
    print(f"  │")
    print(f"  │  Vision Encoder + Merger + GSA:")
    print(f"  │    pixel_values: {list(pixel_values_1b.shape)}")
    print(f"  │    image_grid_thw: [{t_grid}, {h_grid}, {w_grid}]")
    print(f"  │    after merger (2×2): [{t_grid}, {h_vis}, {w_vis}]")
    print(f"  │    → visual_tokens: [{B}, {n_visual}, {D}]")
    print(f"  │")
    print(f"  │  Text Embeddings + RTI:")
    print(f"  │    input_ids: [{B}, {n_text}]")
    print(f"  │    n_masks (RTI 3→3 replace): {n_masks}")
    print(f"  │    → text_embeds: [{B}, {n_text}, {D}]  (length unchanged by RTI)")
    print(f"  │")
    print(f"  │  Concat Fusion:")
    print(f"  │    inputs_embeds = [visual_tokens | text_embeds]")
    print(f"  │                  = [{B}, {n_visual}+{n_text}, {D}]")
    print(f"  │                  = [{B}, {total_seq}, {D}]")
    print(f"  └─────────────────────────────────────────────────────────────┘")

    # Now print the full sequence map
    ids  = input_ids_1b[0].tolist()
    lbls = batch["labels"][0].tolist()
    mask_positions_flat = batch["mask_positions"][0] if batch["mask_positions"] else []

    # Identify which text positions are RTI-replaced <mask> tokens
    # find_mask_positions returns start positions; each <mask> = 3 tokens: [<, mask, >]
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

    # Part 1: Visual tokens block (summarized)
    print(f"  {0:>8}  {'─':>8}  {'VISUAL':<20}  {'[CLS/patch embed]':<25}  {'─':>6}")
    print(f"  {'...':>8}  {'─':>8}  {'VISUAL':<20}  {f'({n_visual} patch tokens)':<25}  {'─':>6}  (no labels)")
    print(f"  {n_visual-1:>8}  {'─':>8}  {'VISUAL':<20}  {'[last patch token]':<25}  {'─':>6}")
    print(f"  {'─'*80}")

    # Part 2: Text tokens (with RTI markers)
    answer_start = next((i for i, v in enumerate(lbls) if v != -100), n_text)

    # Show every text token with its backbone position
    prev_pos = n_visual - 1
    mask_region_idx = 0
    i = 0
    shown = 0
    max_show = 80  # limit output

    while i < n_text and shown < max_show:
        backbone_pos = n_visual + i
        tok_id  = ids[i]
        lbl     = lbls[i]
        decoded = decode_token(tokenizer, tok_id)

        # Determine type
        if i in rti_positions:
            # This token was replaced by RTI
            # Determine which of the 3 replacement tokens this is
            is_first = (i in [mp for mp in mask_positions_flat])
            if is_first:
                mask_region_idx += 1

            # Figure out position within the 3-token replacement
            for mp in mask_positions_flat:
                if mp <= i < mp + mask_token_len:
                    offset = i - mp
                    break
            else:
                offset = 0

            rti_names = ["region_rgb", "region_depth", "space"]
            if offset < len(rti_names):
                rti_label = rti_names[offset]
            else:
                rti_label = f"rti_tok_{offset}"

            src_type = f"[RTI] Region {mask_region_idx}"
            content  = f"[{rti_label}] (was: {decoded})"
        elif i == answer_start:
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

        # Condense consecutive prompt tokens
        if src_type == "TEXT (prompt)" and i > 4 and i < answer_start - 3:
            if prev_pos == backbone_pos - 1 and shown > 5:
                if i == 5:
                    print(f"  {'...':>8}  {'...':>8}  {'TEXT (prompt)':<20}  {'...':<25}  {'─':>6}")
                    shown += 1
                i += 1
                prev_pos = backbone_pos
                continue

        print(f"  {backbone_pos:>8}  {i:>8}  {src_type:<20}  {content:<25}  {lbl_str:>6}  {active}")
        prev_pos = backbone_pos
        shown += 1
        i += 1

    if i < n_text:
        print(f"  {'...':>8}  {'...':>8}  {'...':<20}  {'(truncated)':<25}")

    print(f"  {'─'*80}")

    # Summary
    print(f"\n  Summary:")
    print(f"    Visual tokens:     positions [0 .. {n_visual-1}]  ({n_visual} tokens)")
    print(f"    Text tokens:       positions [{n_visual} .. {n_visual+n_text-1}]  ({n_text} tokens)")
    print(f"    RTI replacements:  {n_masks} × 3 tokens = {n_masks*3} positions replaced")
    print(f"      Original:  [<] [mask] [>]  → 3 text tokens")
    print(f"      Replaced:  [region_rgb] [region_depth] [space]  → 3 RTI embeddings")
    print(f"    Total backbone:    {total_seq} positions = {n_visual} visual + {n_text} text")
    print(f"    Labels:            offset by n_visual={n_visual} (logits[:, n_visual:, :] aligns with labels)")
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
    from model_micro.pipeline import print_vram_usage

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
            num_token_positions=batch.get("num_token_positions"),
        )

    logits_per_step = output["logits_per_step"]
    num_pred = output["num_pred"]

    # Use final step logits for alignment analysis
    logits = logits_per_step[-1]  # Last step = deepest

    print(f"  logits_per_step: {len(logits_per_step)} steps, each {list(logits_per_step[0].shape)}")
    print(f"  num_pred: {num_pred.tolist()}")

    per_token_losses = print_forward_alignment(logits, labels.cpu(), tokenizer, pipeline)

    # ------------------------------------------------------------------ #
    # SECTION 4: Loss check
    # ------------------------------------------------------------------ #
    print(f"\n{'='*70}")
    print("SECTION 4: LOSS CHECK (LoopLM: uniform CE + SmoothL1)")
    print("=" * 70)

    criterion = SpatialLoss(alpha=0.1)
    device = logits_per_step[0].device
    official_loss, components = criterion(
        logits_per_step,
        labels.to(device),
        num_pred.to(device).float(), batch["target_num"].to(device),
        batch["is_numeric"].to(device),
        return_components=True,
    )

    print(f"\n  SpatialLoss output: {official_loss.item():.6f}")
    print(f"  Components: ce={components['ce']:.6f}, sl1={components['sl1']:.6f}")
    print(f"  CE per step: {components.get('ce_per_step', [])}")
    if per_token_losses:
        avg = sum(per_token_losses) / len(per_token_losses)
        diff_check = abs(avg - official_loss.item())
        # CE won't exactly match if SmoothL1 is nonzero
        if batch["is_numeric"][0].item():
            print(f"  (Contains SmoothL1 component — diff expected)")
        else:
            print(f"  Manual vs official diff: {diff_check:.8f}  "
                  f"{'[MATCH]' if diff_check < 0.001 else '[MISMATCH!]'}")

    is_finite = math.isfinite(official_loss.item())
    print(f"  Finite: {is_finite}")
    print(f"  Full vocab: {logits.shape[2]}")

    # ------------------------------------------------------------------ #
    # SECTION 5: Inference
    # ------------------------------------------------------------------ #
    from model_micro.pipeline import find_mask_positions
    from PIL import Image

    print(f"\n{'='*70}")
    print("SECTION 5: INFERENCE (pipeline.generate)")
    print("=" * 70)

    ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "nvidia_warehouse_dataset")
    json_configs = {
        "train_sample": os.path.join(ROOT, "train_sample", "train_sample.json"),
        "train":        os.path.join(ROOT, "train.json"),
        "val":          os.path.join(ROOT, "val.json"),
    }
    with open(json_configs[args.split]) as f:
        raw_data = json.load(f)

    entry    = raw_data[args.sample_idx]
    question = entry["conversations"][0]["value"].replace("<image>\n","").replace("<image>","").strip()

    img_name   = batch["image_names"][0]
    image_path = os.path.join(ROOT,
                              {"train_sample":"train_sample"}.get(args.split, args.split),
                              "images", img_name)
    pil_image  = Image.open(image_path).convert("RGB")
    if target_size:
        pil_image = pil_image.resize(target_size, Image.LANCZOS)

    # Build generation-format input_ids (direct tokenization, no chat template)
    gen_input_ids = pipeline.processor.tokenizer(
        question, return_tensors="pt", padding=False
    ).input_ids.to(device=dev)

    # Find <mask> positions
    mask_positions = find_mask_positions(gen_input_ids, pipeline.processor.tokenizer)

    rle_list      = batch["rle_list"]
    decoded_masks = batch["decoded_masks"]
    n = min(len(mask_positions), len(rle_list[0]))
    mask_positions = mask_positions[:n]

    # GT thinking (chain-of-thought reasoning from dataset)
    gt_thinking = entry["conversations"][1]["value"] if len(entry.get("conversations", [])) > 1 else "(none)"

    print(f"  Question:     {question}")
    print(f"  GT thinking:  {gt_thinking}")
    print(f"  GT answer:    {batch['answers'][0]}")
    print(f"  n_masks:      {n}")

    pipeline.eval()
    with torch.no_grad():
        output_ids = pipeline.generate(
            pixel_values, image_grid_thw, depth_maps, gen_input_ids,
            rle_list=[rle_list[0][:n]] if n > 0 else None,
            mask_token_positions=[mask_positions] if n > 0 else None,
            decoded_masks=[decoded_masks[0][:n]] if n > 0 else None,
            max_new_tokens=150,
        )

    raw_output = pipeline.processor.tokenizer.decode(
        output_ids[0], skip_special_tokens=True
    ).strip()
    raw_output = re.sub(r'<think>.*?</think>\s*', '', raw_output, flags=re.DOTALL).strip()
    parsed = pipeline.parse_output(raw_output)

    print(f"\n  Raw output:    {raw_output!r}")
    print(f"  Parsed cat:    {parsed['category']!r}")
    print(f"  Parsed answer: {parsed['answer']!r}")
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
