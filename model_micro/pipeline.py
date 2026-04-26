"""
MODULE: Full Pipeline — SpatialVLM Micro (Training)

Architecture (~797M total, ~797M trainable):
    1. Qwen 3.5 Vision Encoder (pruned: 4 ViT blocks, 44M)
       4 ViT blocks (768-dim) + merger (VL Projector, 768->1024)
    2. RTI: Region-Level Token Injection (batched) (~0.07M)
       Each <mask> -> [mask_rgb | mask_depth | mask_geo] (3 -> 3 tokens × 1024-dim)
       Independent of Vision Encoder.
    3. Concat Fusion: [visual_tokens | text+region_tokens]
    4. Qwen 3.5 Backbone (full: 24 layers, single pass, 498M)
    5. Dual Heads:
       - LM Head (tied w/ embed, TRAINABLE): category + text answer
       - Number Head (xVal): distance/count regression (~0.26M)

<|num|> token ID read from config (set by prune.py).

Output format (with chain-of-thought):
    <think>GPT reasoning</think>left_right | "left"
    <think>GPT reasoning</think>mcq | "2"
    <think>GPT reasoning</think>distance | <|num|>
    <think>GPT reasoning</think>count | <|num|>
"""

import re
import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoConfig
from transformers.masking_utils import create_causal_mask

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_micro.rti import RTE
from model_micro.num_head import NumberHead
from model_micro.cat_head import CategoryHead


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _top_k_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Zero out all logits except the top-k highest values."""
    if top_k <= 0:
        return logits
    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    threshold = values[:, -1].unsqueeze(-1)
    return logits.masked_fill(logits < threshold, float("-inf"))


def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus (top-p) filtering: zero out logits below the cumulative p mass."""
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens whose cumulative probability exceeds top_p (shift right by 1)
    sorted_remove = cum_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
    sorted_logits[sorted_remove] = float("-inf")
    # Re-scatter back to original order
    return sorted_logits.scatter(1, sorted_idx, sorted_logits)

# Default model path
MODEL_NAME = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "qwen3.5-micro"
)

# Special token IDs
def _read_special_token_ids():
    config_path = os.path.join(MODEL_NAME, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        return cfg.get("num_token_id", 248077), cfg.get("cat_token_id", 248078)
    return 248077, 248078

NUM_TOKEN_ID, CAT_TOKEN_ID = _read_special_token_ids()

# Regex for structured output parsing
# Format: category | answer
_OUTPUT_RE = re.compile(
    r'(?P<category>left_right|mcq|distance|count)\s*\|\s*'
    r'(?P<answer><\|num\|>|<\|cat\|>)',
    re.IGNORECASE,
)


def find_mask_positions(input_ids: torch.Tensor, tokenizer) -> list[int]:
    """Find token positions of <mask> in input_ids.
    Handles BPE punctuation merging (e.g., '>' + ',' -> '>,').
    Caches per tokenizer instance to avoid stale values.
    """
    tok_id = id(tokenizer)
    if not hasattr(find_mask_positions, '_cache'):
        find_mask_positions._cache = {}
    if tok_id not in find_mask_positions._cache:
        mask_id = tokenizer.encode("mask", add_special_tokens=False)[0]
        lt_ids = set()
        for test in [" <", "  <", "<"]:
            enc = tokenizer.encode(test, add_special_tokens=False)
            if len(enc) == 1:
                lt_ids.add(enc[0])
        find_mask_positions._cache[tok_id] = (mask_id, lt_ids)

    mask_id, lt_ids = find_mask_positions._cache[tok_id]

    ids = input_ids[0].tolist() if input_ids.dim() == 2 else input_ids.tolist()
    positions = []
    i = 0
    while i < len(ids) - 2:
        if ids[i] in lt_ids and ids[i+1] == mask_id:
            # Robust check: BPE merges '>' with punctuation (e.g., '>,' )
            decoded_gt = tokenizer.decode([ids[i+2]])
            if decoded_gt.startswith(">"):
                positions.append(i)
                i += 3
                continue
        i += 1
    return positions


def print_vram_usage(label: str = ""):
    """Print current VRAM usage."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  VRAM [{label}]: {alloc:.2f} / {total:.2f} GB ({100*alloc/total:.0f}%)")


class SpatialVLM(nn.Module):
    """Full pipeline: Qwen 3.5 VLM (vision pruned) + RTI + NumberHead + CategoryHead.

    Custom modules:
        self.region_token_extractor    - RegionTokenExtractor (~0.07M)
        self.num_head                  - NumberHead (xVal)    (~0.66M)
        self.cat_head                  - CategoryHead (MCQ/LR)(~0.66M)

    Qwen built-in (from Qwen 3.5 0.8B):
        self.qwen.model.visual         - Vision Encoder (4 blocks)
        self.qwen.model.language_model - 24-layer backbone (single pass)
        self.qwen.lm_head              - Vocab projection
    """

    def __init__(
        self,
        model_name:              str   = MODEL_NAME,
        dropout:                 float = 0.1,
        dtype                          = torch.bfloat16,
        device_map:              str   = "auto",
        attn_implementation:     str   = "sdpa",
    ):
        super().__init__()

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        num_token_id = getattr(config, 'num_token_id', None)
        cat_token_id = getattr(config, 'cat_token_id', None)
        if num_token_id is None or cat_token_id is None:
            config_path = os.path.join(model_name, "config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    raw_config = json.load(f)
                num_token_id = num_token_id or raw_config.get("num_token_id", 248077)
                cat_token_id = cat_token_id or raw_config.get("cat_token_id", 248078)
            else:
                num_token_id = num_token_id or 248077
                cat_token_id = cat_token_id or 248078
        self.num_token_id = num_token_id
        self.cat_token_id = cat_token_id

        # Update module-level for dataloader access
        global NUM_TOKEN_ID, CAT_TOKEN_ID
        NUM_TOKEN_ID = self.num_token_id
        CAT_TOKEN_ID = self.cat_token_id

        print(f"Loading {model_name}...")
        self.qwen = AutoModelForImageTextToText.from_pretrained(
            model_name,
            config=config,
            dtype=dtype,
            attn_implementation=attn_implementation,
            device_map=device_map,
            trust_remote_code=True,
        )
        print(f"  attn_implementation: {attn_implementation}")

        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )

        # Custom Modules
        self.region_token_extractor = RTE(hidden_dim=1024)
        self.num_head = NumberHead(hidden_dim=1024)
        self.cat_head = CategoryHead(hidden_dim=1024)

        self.decoder_dropout = nn.Dropout(dropout)

        # Move custom modules to match Qwen device/dtype
        qwen_device = next(self.qwen.parameters()).device
        qwen_dtype  = next(self.qwen.parameters()).dtype
        self.region_token_extractor = self.region_token_extractor.to(
            device=qwen_device, dtype=qwen_dtype
        )
        self.num_head = self.num_head.to(device=qwen_device, dtype=qwen_dtype)
        self.cat_head = self.cat_head.to(device=qwen_device, dtype=qwen_dtype)
        print(f"  Custom modules (RTI + NumHead + CatHead) -> {qwen_device} ({qwen_dtype})")

        embed = self.qwen.model.language_model.embed_tokens
        embed.weight.requires_grad = True
        print(f"  Embeddings: TRAINABLE ({embed.weight.shape[0]} tokens, requires_grad=True)")
        print(f"  <|num|> token ID: {self.num_token_id}")
        print(f"  <|cat|> token ID: {self.cat_token_id}")

        n_layers = len(list(self.qwen.model.language_model.layers))
        print(f"  Decoder: {n_layers} layers (single pass)")


    @property
    def device(self):
        return next(self.qwen.parameters()).device

    # ---- Vision Encoder ----

    def _get_visual_tokens(
        self,
        pixel_values:   torch.Tensor,
        image_grid_thw: torch.Tensor,
        vision_requires_grad: bool = False,
    ) -> torch.Tensor:
        """Run Qwen's Vision Encoder + Merger -> [B, N, 1024]."""
        visual = self.qwen.model.visual
        ctx = torch.enable_grad() if vision_requires_grad else torch.no_grad()
        with ctx:
            visual_out = visual(pixel_values, grid_thw=image_grid_thw)

        if isinstance(visual_out, torch.Tensor):
            hidden = visual_out
        elif hasattr(visual_out, "last_hidden_state"):
            hidden = visual_out.last_hidden_state
        elif isinstance(visual_out, tuple):
            hidden = visual_out[0]
        else:
            hidden = visual_out

        B = image_grid_thw.shape[0]
        patches_per_image = [
            int(image_grid_thw[i, 0] * image_grid_thw[i, 1] * image_grid_thw[i, 2])
            for i in range(B)
        ]

        if hidden.dim() == 2:
            hidden_list = hidden.split(patches_per_image, dim=0)
        else:
            hidden_list = [hidden[i] for i in range(B)]

        if hidden_list[0].shape[-1] == 1024:
            max_n = max(h.shape[0] for h in hidden_list)
            stacked = torch.stack([
                F.pad(h, (0, 0, 0, max_n - h.shape[0])) for h in hidden_list
            ])
            return stacked

        # Pre-merger: apply merger per-image
        ms = 2
        merged = []
        for i in range(B):
            h_i = hidden_list[i].unsqueeze(0)
            t, h, w = [int(x) for x in image_grid_thw[i].tolist()]
            C = h_i.shape[-1]

            h_i = visual.merger.norm(h_i)
            h_i = h_i.view(1, t, h, w, C)
            h_i = h_i.view(1, t, h // ms, ms, w // ms, ms, C)
            h_i = h_i.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
            h_i = h_i.view(1, -1, ms * ms * C)

            h_i = visual.merger.linear_fc1(h_i)
            h_i = F.gelu(h_i)
            h_i = visual.merger.linear_fc2(h_i)

            merged.append(h_i)

        return torch.cat(merged, dim=0)

    # ---- Build inputs embeds ----

    def _build_inputs_embeds(
        self,
        pixel_values:         torch.Tensor,
        pixel_values_rgb:     torch.Tensor,   # Raw RGB for RTI [B, 3, H, W]
        image_grid_thw:       torch.Tensor,
        depth_maps:           torch.Tensor,
        input_ids:            torch.Tensor,   # [B, L]
        rle_list:             list = None,    # [B][num_masks]
        mask_token_positions: list = None,    # [B][num_masks]
        decoded_masks:        list = None,    # [B][num_masks]
        vision_requires_grad: bool = False,
    ) -> tuple:
        """Build [B, T, 1024] inputs_embeds for the backbone.

        RTI uses 3 -> 3 replacement: sequence length is UNCHANGED.

        Returns:
            inputs_embeds: [B, T, 1024]
            n_visual:      int (0, since visual tokens are inline padded)
        """
        # Step 1: Vision Encoder + Merger -> [B, N, 1024]
        visual_tokens = self._get_visual_tokens(
            pixel_values, image_grid_thw,
            vision_requires_grad=vision_requires_grad,
        )
        n_visual = visual_tokens.shape[1]

        # Step 2: Text embeddings
        embed = self.qwen.model.language_model.embed_tokens
        text_embeds = embed(input_ids)

        # Step 3: RTI (Independent of Vision Encoder)
        if (rle_list is not None and mask_token_positions is not None
                and any(len(rl) > 0 for rl in rle_list)):
            region_tokens = self.region_token_extractor(
                pixel_values_rgb, depth_maps, rle_list, image_grid_thw,
                decoded_masks=decoded_masks,
            )
            mask_token_len = len(self.processor.tokenizer.encode(
                "<mask>", add_special_tokens=False
            ))

            text_embeds = self.region_token_extractor.inject_into_text_embeds(
                text_embeds, mask_token_positions, region_tokens,
                mask_token_len=mask_token_len,
            )

        # Step 4: Inline Pad Replacement Fusion
        img_pad_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        B = text_embeds.shape[0]
        
        for b in range(B):
            pad_indices = (input_ids[b] == img_pad_id).nonzero(as_tuple=True)[0]
            if len(pad_indices) > 0:
                n_vis = min(len(pad_indices), visual_tokens.shape[1])
                text_embeds[b, pad_indices[:n_vis]] = visual_tokens[b, :n_vis]

        inputs_embeds = text_embeds
        n_visual_offset = 0

        return inputs_embeds, n_visual_offset

    # ---- Backbone forward ----

    def _backbone_forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor = None,
        past_key_values=None,
        cache_position: torch.Tensor = None,
        use_gradient_checkpointing: bool = False,
    ):
        """Run standard backbone: all layers, single pass.

        Returns:
            hidden: [B, T, D] — final hidden state before norm
        """
        B, seq_len, _ = inputs_embeds.shape
        lm = self.qwen.model.language_model

        if cache_position is not None:
            position_ids = cache_position.unsqueeze(0).expand(B, -1)
        else:
            position_ids = torch.arange(
                seq_len, device=inputs_embeds.device
            ).unsqueeze(0).expand(B, -1)

        position_embeddings = None
        if hasattr(lm, "rotary_emb"):
            position_embeddings = lm.rotary_emb(inputs_embeds, position_ids)

        causal_mask = None
        linear_mask = None

        if attention_mask is not None:
            causal_mask = create_causal_mask(
                config=lm.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position if cache_position is not None
                    else torch.arange(seq_len, device=inputs_embeds.device),
                past_key_values=past_key_values,
            )

            if cache_position is not None and cache_position[0] > 0:
                linear_mask = None
            elif torch.all(attention_mask == 1):
                linear_mask = None
            else:
                linear_mask = attention_mask

        hidden = inputs_embeds

        for layer in lm.layers:
            if hasattr(layer, 'layer_type'):
                layer_mask = linear_mask if layer.layer_type == "linear_attention" else causal_mask
            else:
                layer_mask = causal_mask

            kwargs = {
                "position_ids": position_ids,
                "attention_mask": layer_mask,
            }
            if position_embeddings is not None:
                kwargs["position_embeddings"] = position_embeddings
            if past_key_values is not None:
                kwargs["past_key_values"] = past_key_values
                kwargs["cache_position"] = cache_position

            if use_gradient_checkpointing and self.training:
                def _layer_fn(h, _layer=layer, _kwargs=kwargs):
                    try:
                        out = _layer(h, **_kwargs)
                    except TypeError:
                        out = _layer(h)
                    return out[0] if isinstance(out, tuple) else out
                hidden = grad_checkpoint(_layer_fn, hidden, use_reentrant=False)
            else:
                try:
                    layer_out = layer(hidden, **kwargs)
                except TypeError:
                    layer_out = layer(hidden)
                hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out

            hidden = self.decoder_dropout(hidden)

        return hidden


    # ---- Forward (training) ----

    def forward(
        self,
        pixel_values:         torch.Tensor,
        pixel_values_rgb:     torch.Tensor,
        image_grid_thw:       torch.Tensor,
        depth_maps:           torch.Tensor,
        input_ids:            torch.Tensor,
        rle_list:             list = None,
        mask_token_positions: list = None,
        decoded_masks:        list = None,
        num_token_positions:  list = None,
        cat_token_positions:  list = None,
        attention_mask:       torch.Tensor = None,
        use_gradient_checkpointing: bool = False,
        vision_requires_grad: bool = False,
    ) -> dict:
        """Training forward pass.

        Returns:
            dict with:
                'logits':   [B, L, V] — text logits
                'num_pred': [B]       — Number Head predictions
        """
        inputs_embeds, n_visual = self._build_inputs_embeds(
            pixel_values, pixel_values_rgb, image_grid_thw, depth_maps, input_ids,
            rle_list, mask_token_positions, decoded_masks,
            vision_requires_grad=vision_requires_grad,
        )

        full_attention_mask = None
        if attention_mask is not None:
            B_mask = attention_mask.shape[0]
            vis_mask = torch.ones(B_mask, n_visual, dtype=attention_mask.dtype,
                                  device=attention_mask.device)
            full_attention_mask = torch.cat([vis_mask, attention_mask], dim=1)

        hidden = self._backbone_forward(
            inputs_embeds,
            attention_mask=full_attention_mask,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        lm_norm = self.qwen.model.language_model.norm
        h_normed = lm_norm(hidden)

        # Text logits
        text_h = h_normed[:, n_visual:, :]
        logits = self.qwen.lm_head(text_h)

        # Number Head
        B = input_ids.shape[0]
        num_pred = torch.zeros(B, device=h_normed.device, dtype=h_normed.dtype)

        if num_token_positions is not None:
            num_hidden_list = []
            num_indices = []
            for b, pos in enumerate(num_token_positions):
                if pos is not None and pos >= 0:
                    adjusted_pos = n_visual + pos
                    if 0 <= adjusted_pos < h_normed.shape[1]:
                        num_hidden_list.append(h_normed[b, adjusted_pos, :])
                        num_indices.append(b)

            if num_hidden_list:
                h_num = torch.stack(num_hidden_list, dim=0)
                preds = self.num_head(h_num)
                for k, b in enumerate(num_indices):
                    num_pred[b] = preds[k]

        # Category Head (MCQ / Left-Right)
        cat_logits_list = []  # list of [N_masks] tensors
        if cat_token_positions is not None and mask_token_positions is not None:
            for b, cat_pos in enumerate(cat_token_positions):
                if cat_pos is not None and cat_pos >= 0:
                    adj_cat_pos = n_visual + cat_pos
                    if 0 <= adj_cat_pos < h_normed.shape[1]:
                        h_cat = h_normed[b, adj_cat_pos, :].detach()
                    else:
                        cat_logits_list.append(None)
                        continue

                    # Get hidden states at ALL mask positions for this sample
                    # Concat all 3 RTI tokens per mask: [region_rgb, region_depth, region_geo]
                    mask_pos_b = mask_token_positions[b]
                    mask_token_len = 3  # <mask> = 3 BPE tokens: <, mask, >
                    mask_hiddens = []
                    for mp in mask_pos_b:
                        token_hiddens = []
                        for offset in range(mask_token_len):
                            adj_mp = n_visual + mp + offset
                            if 0 <= adj_mp < h_normed.shape[1]:
                                token_hiddens.append(h_normed[b, adj_mp, :].detach())
                        if token_hiddens:
                            # Concat: [3, 1024] -> [3072]
                            mask_hiddens.append(torch.cat(token_hiddens, dim=0))
                    if mask_hiddens:
                        h_masks = torch.stack(mask_hiddens, dim=0)  # [N_masks, 3072]
                        scores = self.cat_head(h_masks, h_cat)  # [N_masks]
                        cat_logits_list.append(scores)
                    else:
                        cat_logits_list.append(None)
                else:
                    cat_logits_list.append(None)

        return {
            "logits": logits,
            "num_pred": num_pred,
            "cat_logits": cat_logits_list,
        }

    # ---- Generate (inference) ----

    @torch.no_grad()
    def generate(
        self,
        pixel_values:         torch.Tensor,
        pixel_values_rgb:     torch.Tensor,
        image_grid_thw:       torch.Tensor,
        depth_maps:           torch.Tensor,
        input_ids:            torch.Tensor,
        rle_list:             list = None,
        mask_token_positions: list = None,
        max_new_tokens:       int  = 150,
        do_sample:            bool = False,
        temperature:          float = 1.0,
        top_p:                float = 0.9,
        top_k:                int   = 50,
        repetition_penalty:   float = 1.2,
        decoded_masks:        list = None,
        **gen_kwargs,
    ) -> torch.Tensor:
        """Autoregressive generation with top-p/top-k sampling and repetition penalty."""
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache

        inputs_embeds, n_visual = self._build_inputs_embeds(
            pixel_values, pixel_values_rgb, image_grid_thw, depth_maps, input_ids,
            rle_list, mask_token_positions, decoded_masks,
        )

        lm = self.qwen.model.language_model
        embed = lm.embed_tokens
        B, T, _ = inputs_embeds.shape
        dev = inputs_embeds.device

        eos_id = self.processor.tokenizer.eos_token_id
        cache = Qwen3_5DynamicCache(config=lm.config)
        attn_mask = torch.ones(B, T, dtype=torch.long, device=dev)

        cache_position = torch.arange(T, device=dev)
        hidden = self._backbone_forward(
            inputs_embeds, attention_mask=attn_mask,
            past_key_values=cache, cache_position=cache_position,
        )

        hidden_norm = lm.norm(hidden[:, -1:, :])
        logits = self.qwen.lm_head(hidden_norm)

        if repetition_penalty != 1.0:
            for b in range(B):
                for tok_id in input_ids[b].unique():
                    if logits[b, -1, tok_id] > 0:
                        logits[b, -1, tok_id] /= repetition_penalty
                    else:
                        logits[b, -1, tok_id] *= repetition_penalty

        if do_sample and temperature > 0:
            logits_s = logits[:, -1, :] / temperature
            logits_s = _top_k_filter(logits_s, top_k)
            logits_s = _top_p_filter(logits_s, top_p)
            probs = torch.softmax(logits_s, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
        else:
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        generated = [next_tok]
        all_generated = next_tok.clone()

        for step in range(max_new_tokens - 1):
            if eos_id is not None and (next_tok == eos_id).all():
                break

            tok_embed = embed(next_tok)
            step_cache_pos = torch.tensor([T + step], device=dev)

            hidden = self._backbone_forward(
                tok_embed, past_key_values=cache, cache_position=step_cache_pos,
            )
            hidden_norm = lm.norm(hidden)
            logits = self.qwen.lm_head(hidden_norm)

            if repetition_penalty != 1.0:
                for b in range(B):
                    for tok_id in all_generated[b].unique():
                        if logits[b, -1, tok_id] > 0:
                            logits[b, -1, tok_id] /= repetition_penalty
                        else:
                            logits[b, -1, tok_id] *= repetition_penalty

            if do_sample and temperature > 0:
                logits_s = logits[:, -1, :] / temperature
                logits_s = _top_k_filter(logits_s, top_k)
                logits_s = _top_p_filter(logits_s, top_p)
                probs = torch.softmax(logits_s, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            else:
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)

            generated.append(next_tok)
            all_generated = torch.cat([all_generated, next_tok], dim=1)

        output_ids = torch.cat(generated, dim=1)
        return output_ids

    @staticmethod
    def parse_output(text: str) -> dict:
        """Parse structured LM output -> {category, answer}.

        Expected format: <think>reasoning</think>category | value
        Strips <think>...</think> before parsing.
        """
        # Strip chain-of-thought reasoning
        clean = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()
        m = _OUTPUT_RE.search(clean)
        if m:
            category = m.group("category").strip().lower()
            answer   = m.group("answer").strip()
            return {"category": category, "answer": answer}
        return {"category": "unknown", "answer": None}

    # ---- Full inference ----

    @torch.no_grad()
    def predict(
        self,
        image_rgb_tensor,               # [1, 3, H, W] 0-1 for RTI
        image_processor_output,         # Dict from image_processor
        question: str,
        depth_map: torch.Tensor,        # [H, W] raw
        rle_list: list = None,
        max_new_tokens: int = 150,
    ) -> dict:
        """Single-shot inference: image + question -> {category, answer, raw}."""
        # Tokenize question directly — no chat template, no system prompt
        dev   = self.device
        dtype = next(self.qwen.parameters()).dtype

        input_ids = self.processor.tokenizer(
            question, return_tensors="pt", padding=False
        ).input_ids.to(dev)

        pixel_values   = image_processor_output["pixel_values"].to(device=dev, dtype=dtype)
        image_grid_thw = image_processor_output["image_grid_thw"].to(device=dev)
        
        pixel_values_rgb = image_rgb_tensor.to(device=dev, dtype=dtype)
        depth_batch      = depth_map.unsqueeze(0).to(device=dev, dtype=dtype)

        # Auto-find <mask> positions
        mask_positions = find_mask_positions(input_ids, self.processor.tokenizer)

        if rle_list is not None and len(rle_list) > 0:
            n = min(len(mask_positions), len(rle_list))
            mask_positions = mask_positions[:n]
            rle_list = rle_list[:n]
            rle_list_batched = [rle_list]
            mask_positions_batched = [mask_positions]
        else:
            rle_list_batched = None
            mask_positions_batched = None

        output_ids = self.generate(
            pixel_values, pixel_values_rgb, image_grid_thw, depth_batch, input_ids,
            rle_list=rle_list_batched,
            mask_token_positions=mask_positions_batched,
            max_new_tokens=max_new_tokens,
        )
        raw_output = self.processor.tokenizer.decode(
            output_ids[0], skip_special_tokens=False
        ).replace("<|endoftext|>", "").replace("<|im_end|>", "").strip()

        parsed = self.parse_output(raw_output)
        return {
            "category": parsed["category"],
            "answer":   parsed["answer"],
            "raw":      raw_output,
        }

def count_parameters(model: nn.Module) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",   default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype",    default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--attn-impl", default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"],
                        help="Attention implementation (default: sdpa)")
    parser.add_argument("--model", default=None,
                        help="Model path (default: model_micro/qwen3.5-micro)")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    # Default to pruned Micro checkpoint
    model_path = args.model or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "qwen3.5-micro"
    )

    print("=" * 70)
    print("MODULE: SpatialVLM Micro")
    print("=" * 70)

    pipeline = SpatialVLM(
        model_name=model_path,
        dtype=dtype,
        device_map=args.device,
        attn_implementation=args.attn_impl,
    )
    print_vram_usage("after model load")

    # Parameter Breakdown
    print(f"\n{'='*70}")
    print("PARAMETER BREAKDOWN")
    print(f"{'='*70}")

    components = {
        "Qwen Visual (encoder+merger)":    pipeline.qwen.model.visual,
        "Qwen Embeddings (+ LM Head)":     pipeline.qwen.model.language_model.embed_tokens,
        "Qwen Backbone (layers)":          pipeline.qwen.model.language_model.layers,
        "Qwen Final Norm":                 pipeline.qwen.model.language_model.norm,
        "Qwen LM Head (tied->Embed)":      pipeline.qwen.lm_head,
        "RTI (Region Token Injector)":     pipeline.region_token_extractor,
        "Number Head (xVal regression)":   pipeline.num_head,
        "Category Head (MCQ/LR class.)":   pipeline.cat_head,
    }
    custom_names = {
        "RTI (Region Token Injector)",
        "Number Head (xVal regression)",
        "Category Head (MCQ/LR class.)",
    }
    tied_names = {"Qwen LM Head (tied->Embed)"}

    total_custom, total_qwen = 0, 0
    for name, module in components.items():
        p = count_parameters(module)
        tag = "[*] CUSTOM" if name in custom_names else "    Qwen  "
        tied_note = "  <- shared, not counted" if name in tied_names else ""
        print(f"  {tag} {name:42s}: {p['total']:>12,} ({p['total']/1e6:.4f}M){tied_note}")
        if name in tied_names:
            continue
        if name in custom_names:
            total_custom += p["total"]
        else:
            total_qwen += p["total"]

    print(f"\n  {'-'*70}")
    print(f"  Qwen Micro:       {total_qwen:>12,} ({total_qwen/1e6:.4f}M)")
    print(f"  Custom modules:   {total_custom:>12,} ({total_custom/1e6:.4f}M)")
    print(f"  Total unique:     {total_qwen + total_custom:>12,} ({(total_qwen + total_custom)/1e6:.4f}M)")
    print(f"\n  Vocab: {pipeline.qwen.model.language_model.embed_tokens.weight.shape[0]} (TRAINABLE)")
    print(f"  <|num|> token ID: {pipeline.num_token_id}")
    print(f"  <|cat|> token ID: {pipeline.cat_token_id}")
    print(f"  Trainable params: {sum(p.numel() for p in pipeline.parameters() if p.requires_grad)/1e6:.2f}M")
    n_layers = len(list(pipeline.qwen.model.language_model.layers))
    print(f"  Decoder: {n_layers} layers")
    print(f"  ViT blocks: {len(list(pipeline.qwen.model.visual.blocks))}")

    print_vram_usage("final")
