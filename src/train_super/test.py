"""
Generate predictions for the test set.

Usage:
    python src/train_super/test.py --checkpoint checkpoints/super/stage3/epoch_X --split test
"""

import sys
import os
import json
import argparse
import torch
import re
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from super_model.pipeline import (
    SpatialVLM, print_vram_usage, find_mask_positions,
    MCQ_TOKEN_ID, LR_TOKEN_ID, DIST_TOKEN_ID, COUNT_TOKEN_ID,
)
from super_model.dataloader import SpatialVLMDataset

# Paths
ROOT     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _fix_checkpoint_state_dict(state_dict: dict) -> dict:
    cleaned = {}
    for k, v in state_dict.items():
        new_k = k.replace("._orig_mod.", ".").replace("_orig_mod.", "")
        cleaned[new_k] = v
    embed_key = "qwen.model.language_model.embed_tokens.weight"
    lm_head_key = "qwen.lm_head.weight"
    if embed_key in cleaned and lm_head_key not in cleaned:
        cleaned[lm_head_key] = cleaned[embed_key]
    return cleaned

def load_checkpoint_weights(pipeline, path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint directory not found: {path}")

    print(f"  Loading checkpoint from: {path}")
    
    model_path = os.path.join(path, "model.safetensors")
    if os.path.exists(model_path):
        from safetensors.torch import load_file
        model_state = load_file(model_path)
        model_state = _fix_checkpoint_state_dict(model_state)
        pipeline.load_state_dict(model_state, strict=False)
    else:
        ckpt_path = os.path.join(path, "checkpoint.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found in {path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model_state = ckpt["model_state_dict"]
        model_state = _fix_checkpoint_state_dict(model_state)
        pipeline.load_state_dict(model_state, strict=False)

def run_inference_batch(pipeline, batch_samples: list, max_new_tokens: int = 1) -> list:
    dev = pipeline.device
    dtype = next(pipeline.qwen.parameters()).dtype

    B = len(batch_samples)
    
    pixel_values = torch.cat([s["pixel_values"] for s in batch_samples], dim=0).to(device=dev, dtype=dtype)
    image_grid_thw = torch.cat([s["image_grid_thw"] for s in batch_samples], dim=0).to(device=dev)
    depth_maps = torch.stack([s["depth_map"] for s in batch_samples]).to(device=dev, dtype=dtype)

    prompts = []
    for s in batch_samples:
        question = s["_question"]
        mask_idx = [0]
        def replace_mask(m):
            i = mask_idx[0]
            mask_idx[0] += 1
            return f"[Region {i}]: <|object_ref_start|>{m.group(1)}<|object_ref_end|>"
        question = re.sub(r'(<mask.*?>)', replace_mask, question)

        h_p_rgb, w_p_rgb = s["image_grid_thw"][0, 1].item(), s["image_grid_thw"][0, 2].item()
        num_visual_rgb = int((h_p_rgb // 2) * (w_p_rgb // 2))
        h_p_dep, w_p_dep = s["image_grid_thw"][1, 1].item(), s["image_grid_thw"][1, 2].item()
        num_visual_dep = int((h_p_dep // 2) * (w_p_dep // 2))
        
        vision_str_1 = "Picture 1 (RGB): <|vision_start|>" + "<|image_pad|>" * num_visual_rgb + "<|vision_end|>\n"
        vision_str_2 = "Picture 2 (Depth): <|vision_start|>" + "<|image_pad|>" * num_visual_dep + "<|vision_end|>\n"
        user_str = f"<|im_start|>user\n{vision_str_1}{vision_str_2}{question}<|im_end|>\n"
        eval_prompt = f"<|im_start|>assistant\n"
        
        prompts.append(user_str + eval_prompt)

    tokenizer = pipeline.processor.tokenizer
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device=dev)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    mask_positions_list = []
    rle_lists = []
    decoded_masks_lists = []

    for b in range(B):
        pos = find_mask_positions(input_ids[b].unsqueeze(0), tokenizer)
        s = batch_samples[b]
        rle = s["rle_list"]
        dec = s["decoded_masks"]
        n = min(len(pos), len(rle))
        mask_positions_list.append(pos[:n])
        rle_lists.append(rle[:n])
        decoded_masks_lists.append(dec[:n])

    with torch.amp.autocast("cuda", dtype=dtype):
        output_ids = pipeline.generate(
            pixel_values, image_grid_thw, depth_maps, input_ids,
            attention_mask=attention_mask,
            rle_list=rle_lists,
            mask_token_positions=mask_positions_list,
            decoded_masks=decoded_masks_lists,
            max_new_tokens=max_new_tokens,
        )

    results = []

    for b in range(B):
        raw_full = tokenizer.decode(output_ids[b], skip_special_tokens=False).replace("<|endoftext|>", "").replace("<|im_end|>", "").strip()
        raw_output = raw_full.strip()
        parsed = pipeline.parse_output(raw_output)

        gen_ids_list = output_ids[b].tolist()
        if MCQ_TOKEN_ID in gen_ids_list:
            pred_cat = "mcq"
        elif LR_TOKEN_ID in gen_ids_list:
            pred_cat = "left_right"
        elif DIST_TOKEN_ID in gen_ids_list:
            pred_cat = "distance"
        elif COUNT_TOKEN_ID in gen_ids_list:
            pred_cat = "count"
        else:
            # Fallback based on question heuristics if token generation fails
            q = batch_samples[b]["_question"].lower()
            if "how many" in q or "count" in q:
                pred_cat = "count"
            elif "distance" in q or "distant" in q:
                pred_cat = "distance"
            elif "left" in q and "right" in q:
                pred_cat = "left_right"
            else:
                pred_cat = "mcq"

        dist_pred_val = None
        count_pred_val = None
        mcq_pred_idx = None
        lr_pred_idx = None

        non_pad_mask = attention_mask[b].bool()
        clean_input_ids = input_ids[b][non_pad_mask]
        pad_offset = (~non_pad_mask).sum().item()
        clean_mask_positions = [p - pad_offset for p in mask_positions_list[b]]

        s_pv = batch_samples[b]["pixel_values"].to(device=dev, dtype=dtype)
        s_grid = batch_samples[b]["image_grid_thw"].to(device=dev)
        s_depth = batch_samples[b]["depth_map"].unsqueeze(0).to(device=dev, dtype=dtype)

        has_dist  = DIST_TOKEN_ID in gen_ids_list
        has_count = COUNT_TOKEN_ID in gen_ids_list
        has_mcq   = MCQ_TOKEN_ID in gen_ids_list
        has_lr    = LR_TOKEN_ID in gen_ids_list

        full_generated_ids = torch.cat([clean_input_ids.unsqueeze(0), output_ids[b].unsqueeze(0)], dim=1)
        full_ids_list = full_generated_ids[0].tolist()

        def _find_last_pos(token_id):
            for idx_pos in range(len(full_ids_list) - 1, -1, -1):
                if full_ids_list[idx_pos] == token_id:
                    return idx_pos
            return -1

        if pred_cat == "distance" and has_dist:
            pos = _find_last_pos(DIST_TOKEN_ID)
            if pos >= 0:
                out = pipeline(
                    pixel_values=s_pv,
                    image_grid_thw=s_grid, depth_maps=s_depth,
                    input_ids=full_generated_ids,
                    attention_mask=torch.ones_like(full_generated_ids),
                    rle_list=[rle_lists[b]],
                    mask_token_positions=[clean_mask_positions],
                    decoded_masks=[decoded_masks_lists[b]],
                    dist_token_positions=[pos],
                )
                if out.get("dist_pred") is not None:
                    dist_pred_val = out["dist_pred"][0].item()

        elif pred_cat == "count" and has_count:
            pos = _find_last_pos(COUNT_TOKEN_ID)
            if pos >= 0:
                out = pipeline(
                    pixel_values=s_pv,
                    image_grid_thw=s_grid, depth_maps=s_depth,
                    input_ids=full_generated_ids,
                    attention_mask=torch.ones_like(full_generated_ids),
                    rle_list=[rle_lists[b]],
                    mask_token_positions=[clean_mask_positions],
                    decoded_masks=[decoded_masks_lists[b]],
                    count_token_positions=[pos],
                )
                if out.get("count_pred") is not None:
                    count_pred_val = out["count_pred"][0].item()

        elif pred_cat == "mcq" and has_mcq:
            pos = _find_last_pos(MCQ_TOKEN_ID)
            if pos >= 0:
                out = pipeline(
                    pixel_values=s_pv,
                    image_grid_thw=s_grid, depth_maps=s_depth,
                    input_ids=full_generated_ids,
                    attention_mask=torch.ones_like(full_generated_ids),
                    rle_list=[rle_lists[b]],
                    mask_token_positions=[clean_mask_positions],
                    decoded_masks=[decoded_masks_lists[b]],
                    mcq_token_positions=[pos],
                )
                mcq_logits_out = out.get("mcq_logits")
                if mcq_logits_out and mcq_logits_out[0] is not None:
                    mcq_pred_idx = mcq_logits_out[0].argmax().item()

        elif pred_cat == "left_right" and has_lr:
            pos = _find_last_pos(LR_TOKEN_ID)
            if pos >= 0:
                out = pipeline(
                    pixel_values=s_pv,
                    image_grid_thw=s_grid, depth_maps=s_depth,
                    input_ids=full_generated_ids,
                    attention_mask=torch.ones_like(full_generated_ids),
                    rle_list=[rle_lists[b]],
                    mask_token_positions=[clean_mask_positions],
                    decoded_masks=[decoded_masks_lists[b]],
                    lr_token_positions=[pos],
                )
                lr_logits_out = out.get("lr_logits")
                if lr_logits_out and lr_logits_out[0] is not None:
                    lr_pred_idx = lr_logits_out[0].argmax().item()

        results.append({
            "category":   pred_cat,
            "answer":     parsed.get("answer"),
            "dist_pred":  dist_pred_val,
            "count_pred": count_pred_val,
            "mcq_pred":   mcq_pred_idx,
            "lr_pred":    lr_pred_idx,
            "raw":        raw_output,
        })
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Test SpatialVLM Super")
    parser.add_argument("--checkpoint",     type=str, default=None, required=True)
    parser.add_argument("--split",          default="test", choices=["val", "test"])
    parser.add_argument("--device",         default="cuda", choices=["cuda"])
    parser.add_argument("--attn-impl",      default="flash_attention_2")
    parser.add_argument("--resolution",     default="320p")
    parser.add_argument("--batch-size",     type=int, default=16)
    parser.add_argument("--compile",        action="store_true")
    parser.add_argument("--dist-head-version", type=int, default=2)
    args = parser.parse_args()

    print("=" * 70)
    print("TEST INFERENCE: SpatialVLM Super")
    print("=" * 70)

    target_size = {"1080p": None, "720p": (1280, 720),
                   "540p": (960, 540), "450p": (800, 450),
                   "320p": (512, 320)}[args.resolution]

    pipeline = SpatialVLM(
        dtype=torch.bfloat16,
        device_map=args.device,
        attn_implementation=args.attn_impl,
        dist_head_version=args.dist_head_version,
    )
    
    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint {ckpt_path} not found.")
        sys.exit(1)

    load_checkpoint_weights(pipeline, ckpt_path)
    
    if args.compile:
        pipeline.qwen = torch.compile(pipeline.qwen)
        pipeline.mcq_head = torch.compile(pipeline.mcq_head)
        pipeline.lr_head = torch.compile(pipeline.lr_head)
        pipeline.dist_head = torch.compile(pipeline.dist_head)
        pipeline.count_head = torch.compile(pipeline.count_head)
        
    pipeline.eval()

    from torch.utils.data import DataLoader
    
    dataset = SpatialVLMDataset(
        args.split,
        processor=pipeline.processor,
        target_size=target_size,
    )
    
    N = len(dataset)
    print(f"  Total samples to predict: {N}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=lambda x: x,
        prefetch_factor=2,
        pin_memory=False
    )
    
    predictions = []
    fallback_count = 0

    for batch_samples in tqdm(loader, desc="Inference Progress"):
        try:
            with torch.no_grad():
                batch_results = run_inference_batch(pipeline, batch_samples)
        except Exception as e:
            print(f"  [ERROR] Inference failed on a batch: {e}")
            batch_results = [{"category": "error", "answer": None, "raw": str(e),
                              "dist_pred": None, "count_pred": None,
                              "mcq_pred": None, "lr_pred": None} for _ in batch_samples]

        for i, s in enumerate(batch_samples):
            res = batch_results[i]
            cat = res["category"]
            
            final_ans = None
            
            if cat == "count":
                if res.get("count_pred") is not None:
                    final_ans = int(round(float(res["count_pred"])))
                else:
                    fallback_count += 1
                    final_ans = 0
            
            elif cat == "distance":
                if res.get("dist_pred") is not None:
                    final_ans = round(float(res["dist_pred"]), 2)
                else:
                    fallback_count += 1
                    final_ans = 0.0
                        
            elif cat == "mcq":
                if res.get("mcq_pred") is not None:
                    final_ans = str(int(res["mcq_pred"]))
                else:
                    fallback_count += 1
                    final_ans = "0"
                        
            elif cat == "left_right":
                if res.get("lr_pred") is not None:
                    final_ans = "left" if res["lr_pred"] == 0 else "right"
                else:
                    fallback_count += 1
                    final_ans = "left" # default fallback
            else:
                # Fallback for unknown category
                fallback_count += 1
                final_ans = 0

            predictions.append({
                "id": s["_id"],
                "normalized_answer": final_ans
            })

    output_path = "predicions.json"
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)
        
    print(f"\n  Done! Saved {len(predictions)} predictions to {output_path}")
    print(f"  Total fallbacks (no routing token / unknown): {fallback_count} / {N}")

if __name__ == "__main__":
    main()
