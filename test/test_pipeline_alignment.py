"""
test_pipeline_alignment.py
==========================
Integration test that loads REAL data from the dataloader and verifies:

  1. TOKEN TABLE
     Prints every token in the sequence with:
       - position index
       - token ID
       - decoded text  (what the token actually is)
       - label value   (-100 = ignored prompt, else = answer token ID)
       - active?       (YES for answer tokens, no for prompt)
     This lets you see EXACTLY where the answer starts and whether it
     aligns with the right sequence position.

  2. FORWARD PASS + LABEL ALIGNMENT
     Runs pipeline.forward() with the real batch.
     Shows:
       - L  (original labels length from dataloader)
       - L' (logits length after RTI injects region tokens)
       - How many tokens were trimmed from the END (n_masks)
       - Per-position table: logit[t] predicts labels[t+1]
         Highlights which logit positions are actually trained on.

  3. LOSS CHECK
     Computes SpatialVLMLoss on the real logits and labels.
     Verifies the loss is finite and not NaN.

  4. INFERENCE
     Runs pipeline.predict() on the same sample.
     Compares the generated answer to the ground truth.

Usage:
    python test/test_pipeline_alignment.py
    python test/test_pipeline_alignment.py --resolution 450p --split train_sample
    python test/test_pipeline_alignment.py --no-model    # token table only, no GPU needed
    python test/test_pipeline_alignment.py --attn-impl sdpa
"""

import sys
import os
import math
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataloader.dataloader import SpatialVLMDataset, get_dataloader
from model.loss import SpatialVLMLoss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def decode_token(tokenizer, tok_id: int) -> str:
    """Decode a single token ID to a printable string."""
    if tok_id == -100:
        return "<IGNORED>"
    text = tokenizer.decode([tok_id], skip_special_tokens=False)
    # Make whitespace visible
    text = text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    if not text.strip():
        text = repr(text)
    return text[:30]   # cap width for table


# ---------------------------------------------------------------------------
# Section 1: Token Table
# ---------------------------------------------------------------------------

def print_token_table(input_ids: torch.Tensor, labels: torch.Tensor,
                      tokenizer, max_rows: int = 30):
    """Print a per-token table: position | token_id | decoded | label | active.

    Shows full prompt context (first 10 rows) + a window around where
    the answer starts + the last few rows.
    """
    ids  = input_ids[0].tolist()    # [L]
    lbls = labels[0].tolist()       # [L]
    L    = len(ids)

    # Find where active labels begin
    answer_start = next((i for i, v in enumerate(lbls) if v != -100), L)
    n_active = sum(1 for v in lbls if v != -100)

    print(f"\n  Sequence length L = {L}")
    print(f"  Prompt tokens (ignored): {answer_start}")
    print(f"  Answer tokens (active):  {n_active}")
    print(f"  Answer starts at position: {answer_start}")

    # Decide which rows to print
    head_rows  = list(range(min(5, L)))
    # a few rows before + all rows at/after answer_start
    pre_answer = list(range(max(0, answer_start - 3), answer_start))
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
        # Mark the exact answer start
        marker  = " <<< ANSWER START" if pos == answer_start else ""
        print(f"  {pos:>5}  {tok_id:>7}  {decoded:<32}  {lbl_str:>8}  {active}{marker}")
        prev = pos

    print(f"  {'─'*72}")
    print(f"  Answer tokens decoded:")
    answer_toks = [ids[i] for i in range(answer_start, L) if lbls[i] != -100]
    print(f"    {tokenizer.decode(answer_toks, skip_special_tokens=False)!r}")
    print()


# ---------------------------------------------------------------------------
# Section 2: Forward pass alignment
# ---------------------------------------------------------------------------

def print_forward_alignment(logits: torch.Tensor, labels: torch.Tensor,
                             tokenizer, n_visual: int):
    """Show per-position alignment after RTI shortening + label trim + shift."""
    L_orig  = labels.shape[1]
    L_prime = logits.shape[1]          # text tokens only (visual already removed)
    diff    = L_orig - L_prime

    print(f"\n  Original labels length  L  = {L_orig}")
    print(f"  Logits text length      L' = {L_prime}")
    print(f"  RTI diff (n_masks)         = {diff}  (tokens dropped from FRONT of labels)")

    # Apply the same trim as loss.py (from FRONT)
    # RTI shortens the PROMPT, shifting answer LEFT by n_masks positions.
    if diff > 0:
        trimmed_labels = labels[:, diff:]   # [1, L']
    else:
        trimmed_labels = labels

    lbls_t = trimmed_labels[0].tolist()
    # shift_labels[t] = trimmed_labels[t+1]  (what logit[t] is trained to predict)

    n_active = sum(1 for v in lbls_t[1:] if v != -100)
    print(f"  Active targets after shift: {n_active}")

    # Find where active targets start in shift_labels
    # shift_labels[t] = lbls_t[t+1]
    active_positions = [(t, lbls_t[t+1]) for t in range(len(lbls_t)-1) if lbls_t[t+1] != -100]

    print(f"\n  SHIFT ALIGNMENT (logit[t] --predicts--> label[t+1]):")
    print(f"  {'LogitPos':>8}  {'TargetTokID':>12}  {'Decoded Target':<30}")
    print(f"  {'─'*58}")
    for t, tok_id in active_positions:
        decoded = decode_token(tokenizer, tok_id)
        print(f"  {t:>8}  {tok_id:>12}  {decoded:<30}")

    if not active_positions:
        print("  (no active positions -- all labels are -100)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Token-level label alignment + loss + inference test")
    parser.add_argument("--split",       default="train_sample",
                        choices=["train", "val", "test", "train_sample"])
    parser.add_argument("--sample-idx",  type=int, default=0,
                        help="Which dataset sample index to test (0-based)")
    parser.add_argument("--resolution",  default="450p",
                        choices=["1080p", "720p", "540p", "450p"])
    parser.add_argument("--device",      default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype",       default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--attn-impl",   default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--no-model",    action="store_true",
                        help="Skip model loading -- only print the token table (no GPU needed)")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    target_size = {"1080p": None, "720p": (1280, 720),
                   "540p": (960, 540), "450p": (800, 450)}[args.resolution]

    # ------------------------------------------------------------------ #
    # Load processor (always needed for tokenization)
    # ------------------------------------------------------------------ #
    if args.no_model:
        from transformers import AutoProcessor
        model_name = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "model", "qwen3.5-0.8b"
        )
        print("Loading processor only (--no-model)...")
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        pipeline  = None
    else:
        from model.pipeline import SpatialVLM, print_vram_usage
        print("=" * 70)
        print("LOADING MODEL")
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

    dataset = SpatialVLMDataset(args.split, processor=processor,
                                target_size=target_size)
    loader  = get_dataloader(dataset, batch_size=1, shuffle=False,
                             num_workers=0, pin_memory=False)

    # Get the specific sample
    batch = None
    for i, b in enumerate(loader):
        if i == args.sample_idx:
            batch = b
            break
    if batch is None:
        print(f"[!] sample_idx={args.sample_idx} out of range (dataset size={len(dataset)})")
        return

    print(f"  Image:    {batch['image_names'][0]}")
    print(f"  Category: {batch['categories'][0]}")
    print(f"  Answer:   {batch['answers'][0]}")
    print(f"  n_masks:  {len(batch['mask_positions'][0])}")

    # ------------------------------------------------------------------ #
    # SECTION 1: Token Table
    # ------------------------------------------------------------------ #
    print(f"\n{'='*70}")
    print("SECTION 1: TOKEN-LEVEL LABEL TABLE")
    print("=" * 70)
    print("  Each row: position | token_id | decoded text | label | active?")
    print("  'active' means the model is TRAINED on this position (label != -100)")

    print_token_table(batch["input_ids"], batch["labels"], tokenizer)

    if args.no_model:
        print("  [--no-model] Skipping forward pass and inference.")
        return

    # ------------------------------------------------------------------ #
    # SECTION 2: Forward pass + alignment
    # ------------------------------------------------------------------ #
    from model.pipeline import print_vram_usage

    print(f"\n{'='*70}")
    print("SECTION 2: FORWARD PASS LABEL ALIGNMENT")
    print("=" * 70)

    dev = pipeline.device

    pixel_values   = batch["pixel_values"].to(device=dev, dtype=dtype)
    image_grid_thw = batch["image_grid_thw"].to(device=dev)
    depth_maps     = batch["depth_maps"].to(device=dev, dtype=dtype)
    input_ids      = batch["input_ids"].to(device=dev)
    labels         = batch["labels"].to(device=dev)

    pipeline.eval()
    with torch.no_grad():
        output = pipeline(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            depth_maps=depth_maps,
            input_ids=input_ids,
            rle_list=batch["rle_list"][0],
            mask_token_positions=batch["mask_positions"][0],
            decoded_masks=batch["decoded_masks"][0],
        )

    logits   = output["logits"]    # [1, L', vocab]
    n_visual = pixel_values.shape[0] // 1  # patches for this image

    print(f"  logits shape: {list(logits.shape)}  (text tokens only, visual stripped)")

    # Compute n_visual from pipeline internals
    # logits = hidden[:, n_visual:, :] so n_visual = total_seq - L'
    # We can infer it: total input_embeds = n_visual + L, logits = L'
    L_orig = input_ids.shape[1]
    L_prime = logits.shape[1]

    print_forward_alignment(logits, labels.cpu(), tokenizer, n_visual=L_orig - L_prime)

    # ------------------------------------------------------------------ #
    # SECTION 3: Loss check (detailed per-token breakdown)
    # ------------------------------------------------------------------ #
    import math as _math

    print(f"\n{'='*70}")
    print("SECTION 3: LOSS CHECK (per-token breakdown)")
    print("=" * 70)

    # Replicate what loss.py does step by step
    logits_cpu = logits.cpu().float()
    labels_cpu = labels.cpu()

    # Step 1: Trim labels from FRONT
    L_labels = labels_cpu.shape[1]
    L_logits = logits_cpu.shape[1]
    diff = L_labels - L_logits
    if diff > 0:
        trimmed_labels = labels_cpu[:, diff:]
    else:
        trimmed_labels = labels_cpu

    # Step 2: Shift
    shift_logits = logits_cpu[:, :-1, :]       # [1, L'-1, vocab]
    shift_labels = trimmed_labels[:, 1:]       # [1, L'-1]

    # Step 3: Find active positions
    active_mask = (shift_labels[0] != -100)
    active_positions = active_mask.nonzero(as_tuple=True)[0].tolist()
    n_active = len(active_positions)

    vocab_size = shift_logits.shape[2]
    print(f"\n  Trim: labels[:, {diff}:] (dropped {diff} from front)")
    print(f"  Shift: logits[:, :-1]  labels[:, 1:]")
    print(f"  Vocab size: {vocab_size:,}")
    print(f"  Active targets: {n_active}")

    # Step 4: Per-token breakdown
    print(f"\n  {'Pos':>5}  {'Target':>8}  {'TargetText':<14}  "
          f"{'Pred':>8}  {'PredText':<14}  "
          f"{'P(target)':>10}  {'CE Loss':>9}  Match?")
    print(f"  {'─'*95}")

    per_token_losses = []
    for t in active_positions:
        target_id = shift_labels[0, t].item()
        logit_vec = shift_logits[0, t]           # [vocab]

        # log_softmax for numerically stable CE (matches F.cross_entropy internals)
        log_probs = torch.log_softmax(logit_vec, dim=0)
        ce_loss = -log_probs[target_id].item()   # -log P(target)
        per_token_losses.append(ce_loss)

        # Probability for display
        p_target = _math.exp(-ce_loss)

        # Top-1 prediction
        pred_id = logit_vec.argmax().item()
        pred_text = decode_token(tokenizer, pred_id)
        target_text = decode_token(tokenizer, target_id)

        match = "[OK]" if pred_id == target_id else "[FAIL]"
        print(f"  {t:>5}  {target_id:>8}  {target_text:<14}  "
              f"{pred_id:>8}  {pred_text:<14}  "
              f"{p_target:>10.6f}  {ce_loss:>9.4f}  {match}")

    # Step 5: Summary
    if n_active > 0:
        avg_loss = sum(per_token_losses) / n_active
        print(f"  {'─'*95}")
        print(f"  Average CE loss = sum({' + '.join(f'{l:.4f}' for l in per_token_losses)}) / {n_active}")
        print(f"                  = {sum(per_token_losses):.4f} / {n_active}")
        print(f"                  = {avg_loss:.6f}")

    # Cross-check with SpatialVLMLoss
    criterion = SpatialVLMLoss()
    official_loss = criterion(logits_cpu, labels_cpu)

    print(f"\n  SpatialVLMLoss output: {official_loss.item():.6f}")
    if n_active > 0:
        diff_check = abs(avg_loss - official_loss.item())
        print(f"  Manual vs official diff: {diff_check:.8f}  "
              f"{'[MATCH]' if diff_check < 0.001 else '[MISMATCH!]'}")

    is_finite = _math.isfinite(official_loss.item())
    print(f"\n  Finite: {is_finite}")
    print(f"  (Untrained model: expected loss ≈ log(vocab) ≈ {_math.log(vocab_size):.2f})")

    # Top-3 predictions for each active position
    print(f"\n  TOP-3 PREDICTIONS per active position:")
    print(f"  {'Pos':>5}  {'Target':<14}  "
          f"{'#1':<20}  {'#2':<20}  {'#3':<20}")
    print(f"  {'─'*85}")
    for t in active_positions:
        target_id = shift_labels[0, t].item()
        target_text = decode_token(tokenizer, target_id)
        logit_vec = shift_logits[0, t]
        probs = torch.softmax(logit_vec, dim=0)

        topk = torch.topk(probs, k=3)
        top3 = []
        for rank in range(3):
            tid = topk.indices[rank].item()
            tp  = topk.values[rank].item()
            txt = decode_token(tokenizer, tid)
            marker = " *" if tid == target_id else ""
            top3.append(f"{txt}({tp:.4f}){marker}")

        print(f"  {t:>5}  {target_text:<14}  "
              f"{top3[0]:<20}  {top3[1]:<20}  {top3[2]:<20}")

    print_vram_usage("after forward pass")

    # ------------------------------------------------------------------ #
    # SECTION 4: Inference
    # ------------------------------------------------------------------ #
    from model.pipeline import find_mask_positions, SYSTEM_PROMPT
    import re
    import json
    from PIL import Image

    print(f"\n{'='*70}")
    print("SECTION 4: INFERENCE  (pipeline.generate with dataloader tensors)")
    print("=" * 70)
    print("  (Uses decoded_masks from dataloader -- same resolution as depth_map)")

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

    # Load image at correct resolution for the processor chat template
    img_name   = batch["image_names"][0]
    image_path = os.path.join(ROOT, {"train_sample":"train_sample"}.get(args.split, args.split),
                              "images", img_name)
    pil_image  = Image.open(image_path).convert("RGB")
    if target_size:
        pil_image = pil_image.resize(target_size, Image.LANCZOS)

    # Build generation-format input_ids (prompt-only, no answer)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": [
            {"type": "image", "image": pil_image},
            {"type": "text",  "text": question},
        ]},
    ]
    text = pipeline.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    inputs      = pipeline.processor(text=[text], images=[pil_image],
                                     return_tensors="pt", padding=False)
    gen_input_ids = inputs["input_ids"].to(device=dev)

    # Find <mask> positions in the inference input_ids
    mask_positions = find_mask_positions(gen_input_ids, pipeline.processor.tokenizer)

    # Use pre-decoded masks from the dataloader (already at the right resolution)
    rle_list      = batch["rle_list"][0]
    decoded_masks = batch["decoded_masks"][0]
    n = min(len(mask_positions), len(rle_list))
    mask_positions = mask_positions[:n]
    rle_list       = rle_list[:n]
    decoded_masks  = decoded_masks[:n]

    print(f"  Question: {question[:100]}...")
    print(f"  GT answer: {batch['answers'][0]}")
    print(f"  n_masks:   {n}")

    pipeline.eval()
    with torch.no_grad():
        output_ids = pipeline.generate(
            pixel_values, image_grid_thw, depth_maps, gen_input_ids,
            rle_list=rle_list if n > 0 else None,
            mask_token_positions=mask_positions if n > 0 else None,
            decoded_masks=decoded_masks if n > 0 else None,
            max_new_tokens=40,
        )

    raw_output = pipeline.processor.tokenizer.decode(
        output_ids[0], skip_special_tokens=True
    ).strip()
    # Strip <think>...</think> tags (Qwen3.5 default generation behavior)
    raw_output = re.sub(r'<think>.*?</think>\s*', '', raw_output, flags=re.DOTALL).strip()
    parsed = pipeline.parse_output(raw_output)

    print(f"\n  Raw output:    {raw_output!r}")
    print(f"  Parsed cat:    {parsed['category']!r}")
    print(f"  Parsed answer: {parsed['answer']!r}")
    print(f"  GT answer:     {batch['answers'][0]!r}")
    match = str(parsed.get("answer","")).strip('"') == str(batch["answers"][0]).strip('"')
    print(f"  Match: {'[OK]' if match else '[FAIL]'}")

    print_vram_usage("after inference")

    print(f"\n{'='*70}")
    print("ALL SECTIONS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
