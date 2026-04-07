"""
test_pipeline_alignment.py (Micro)
===================================
Integration test that loads REAL data and verifies:

  1. TOKEN TABLE
     Prints every token with position, ID, decoded text, label, active status.
     Shows where the answer starts and [NUM] token position.

  2. TOKEN REMAPPING CHECK
     Verifies old -> new ID remapping produces valid indices for embed_tokens.

  3. FORWARD PASS + LABEL ALIGNMENT
     Runs pipeline.forward() with real batch.
     Shows label trimming, shift alignment, per-token CE breakdown.

  4. LOSS CHECK
     Computes SpatialLoss (CE + MSE) on real logits and labels.

  5. INFERENCE
     Runs pipeline.generate() with dataloader tensors.

Usage:
    python test_micro/test_pipeline_alignment.py
    python test_micro/test_pipeline_alignment.py --resolution 450p
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
                      tokenizer, max_rows: int = 30):
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
# Section 2: Token Remapping Check
# ---------------------------------------------------------------------------

def check_remapping(pipeline, input_ids, labels):
    """Verify old -> new remapping produces valid indices."""
    print(f"\n  input_ids range: [{input_ids.min().item()}, {input_ids.max().item()}]")

    new_ids = pipeline.remap_to_new(input_ids)
    max_new = new_ids.max().item()
    embed_size = pipeline.qwen.model.language_model.embed_tokens.weight.shape[0]
    print(f"  new_ids range:   [{new_ids.min().item()}, {max_new}]")
    print(f"  embed_tokens rows: {embed_size}")
    print(f"  All in range: {'[OK]' if max_new < embed_size else '[FAIL - INDEX OUT OF RANGE]'}")

    # Check labels remapping (preserves -100)
    new_labels = pipeline.remap_to_new(labels)
    n_ignore = (new_labels == -100).sum().item()
    n_orig_ignore = (labels == -100).sum().item()
    print(f"  Labels -100 preserved: {n_ignore == n_orig_ignore} "
          f"({n_orig_ignore} -> {n_ignore})")

    active_labels = new_labels[new_labels != -100]
    if len(active_labels) > 0:
        max_label = active_labels.max().item()
        print(f"  Active labels range: [{active_labels.min().item()}, {max_label}]")
        print(f"  All labels in vocab: {'[OK]' if max_label < embed_size else '[FAIL]'}")

    return max_new < embed_size


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

    # Remap labels to new vocab
    new_labels = pipeline.remap_to_new(labels)

    if diff > 0:
        trimmed_labels = new_labels[:, diff:]
    else:
        trimmed_labels = new_labels

    lbls_t = trimmed_labels[0].tolist()
    n_active = sum(1 for v in lbls_t[1:] if v != -100)
    print(f"  Active targets after shift: {n_active}")

    active_positions = [(t, lbls_t[t+1]) for t in range(len(lbls_t)-1) if lbls_t[t+1] != -100]

    # Per-token breakdown
    vocab_size = logits.shape[2]
    print(f"  Vocab size: {vocab_size}")

    print(f"\n  {'Pos':>5}  {'NewID':>8}  {'OldID':>8}  {'Decoded':>14}  "
          f"{'Pred':>8}  {'P(target)':>10}  {'CE Loss':>9}  Match?")
    print(f"  {'─'*100}")

    logits_cpu = logits.cpu().float()
    per_token_losses = []

    for t, new_id in active_positions:
        logit_vec = logits_cpu[0, t]
        log_probs = torch.log_softmax(logit_vec, dim=0)
        ce_loss = -log_probs[new_id].item()
        per_token_losses.append(ce_loss)
        p_target = math.exp(-ce_loss) if math.isfinite(ce_loss) else 0.0

        pred_id = logit_vec.argmax().item()
        # Remap back to old IDs for display
        old_target = pipeline.remap_to_old(torch.tensor([new_id])).item()
        old_pred = pipeline.remap_to_old(torch.tensor([pred_id])).item()
        target_text = decode_token(tokenizer, old_target)
        match = "[OK]" if pred_id == new_id else "[FAIL]"

        print(f"  {t:>5}  {new_id:>8}  {old_target:>8}  {target_text:>14}  "
              f"{pred_id:>8}  {p_target:>10.6f}  {ce_loss:>9.4f}  {match}")

    if per_token_losses:
        avg = sum(per_token_losses) / len(per_token_losses)
        print(f"  {'─'*100}")
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
    parser.add_argument("--resolution", default="450p",
                        choices=["1080p", "720p", "540p", "450p"])
    parser.add_argument("--device",     default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype",      default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--attn-impl",  default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--no-model",   action="store_true",
                        help="Skip model loading — token table + remap only")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    target_size = {"1080p": None, "720p": (1280, 720),
                   "540p": (960, 540), "450p": (800, 450)}[args.resolution]

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
    # SECTION 2: Token Remapping Check
    # ------------------------------------------------------------------ #
    print(f"\n{'='*70}")
    print("SECTION 2: TOKEN ID REMAPPING CHECK")
    print("=" * 70)

    dev = pipeline.device
    input_ids = batch["input_ids"].to(device=dev)
    labels    = batch["labels"].to(device=dev)

    remap_ok = check_remapping(pipeline, input_ids, labels)

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

    logits   = output["logits"]
    num_pred = output["num_pred"]

    print(f"  logits shape: {list(logits.shape)}")
    print(f"  num_pred: {num_pred.tolist()}")

    per_token_losses = print_forward_alignment(logits, labels.cpu(), tokenizer, pipeline)

    # ------------------------------------------------------------------ #
    # SECTION 4: Loss check
    # ------------------------------------------------------------------ #
    print(f"\n{'='*70}")
    print("SECTION 4: LOSS CHECK (SpatialLoss: CE + MSE)")
    print("=" * 70)

    criterion = SpatialLoss(alpha=1.0, remap_fn=pipeline.remap_to_new)
    official_loss = criterion(
        logits.cpu().float(), labels.cpu(),
        num_pred.cpu().float(), batch["target_num"],
        batch["is_numeric"],
    )

    print(f"\n  SpatialLoss output: {official_loss.item():.6f}")
    if per_token_losses:
        avg = sum(per_token_losses) / len(per_token_losses)
        diff_check = abs(avg - official_loss.item())
        # CE won't exactly match if MSE is nonzero
        if batch["is_numeric"][0].item():
            print(f"  (Contains MSE component — diff expected)")
        else:
            print(f"  Manual vs official diff: {diff_check:.8f}  "
                  f"{'[MATCH]' if diff_check < 0.001 else '[MISMATCH!]'}")

    is_finite = math.isfinite(official_loss.item())
    print(f"  Finite: {is_finite}")
    print(f"  Micro vocab: {logits.shape[2]}")
    print(f"  Expected loss ≈ log({logits.shape[2]}) ≈ {math.log(logits.shape[2]):.2f}")

    # ------------------------------------------------------------------ #
    # SECTION 5: Inference
    # ------------------------------------------------------------------ #
    from model_micro.pipeline import find_mask_positions, SYSTEM_PROMPT
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

    # Build generation-format input_ids (prompt only, no answer)
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
    inputs = pipeline.processor(text=[text], images=[pil_image],
                                return_tensors="pt", padding=False)
    gen_input_ids = inputs["input_ids"].to(device=dev)

    # Find <mask> positions
    mask_positions = find_mask_positions(gen_input_ids, pipeline.processor.tokenizer)

    rle_list      = batch["rle_list"]
    decoded_masks = batch["decoded_masks"]
    n = min(len(mask_positions), len(rle_list[0]))
    mask_positions = mask_positions[:n]

    print(f"  Question: {question[:100]}...")
    print(f"  GT answer: {batch['answers'][0]}")
    print(f"  n_masks:   {n}")

    pipeline.eval()
    with torch.no_grad():
        output_ids = pipeline.generate(
            pixel_values, image_grid_thw, depth_maps, gen_input_ids,
            rle_list=[rle_list[0][:n]] if n > 0 else None,
            mask_token_positions=[mask_positions] if n > 0 else None,
            decoded_masks=[decoded_masks[0][:n]] if n > 0 else None,
            max_new_tokens=40,
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
        (remap_ok, "Token remapping produces valid embed indices"),
        (is_finite, f"Loss is finite ({official_loss.item():.4f})"),
        (logits.shape[2] == pipeline.micro_vocab_size,
         f"Logits vocab matches micro_vocab ({logits.shape[2]} == {pipeline.micro_vocab_size})"),
    ]
    for ok, msg in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {msg}")


if __name__ == "__main__":
    main()
