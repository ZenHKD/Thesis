"""
MODULE: Full Pipeline — SpatialVLM Micro (Training)

Architecture (~216M total, ~216M trainable):
    1. Qwen 3.5 Vision Encoder (pruned: 4 ViT blocks, 44M)
       4 ViT blocks (768-dim) + merger (VL Projector, 768->1024)
    2. GSA: Geometry Self-Attention — DFormerv2 Full_GSA CVPR 2025
       Position: after merger, before concat fusion (~16.9M)
    3. RTI: Region-Level Token Injection (batched) (~0.032M)
       Each <mask> -> [mask_rgb | mask_depth | space] (3 -> 3 tokens × 1024-dim)
    4. Concat Fusion: [visual_tokens | text+region_tokens]
    5. Qwen 3.5 Backbone (pruned: 4 layers, looped T_max=4× via LoopLM)
       LoopLM: per-step LM head loss + exit gate + entropy regularization
       Based on "Scaling Latent Reasoning via Looped Language Models" (Ouro)
    6. Dual Heads:
       - LM Head (tied w/ embed, TRAINABLE): category + text answer (per-step)
       - Number Head (xVal): distance/count regression (~0.26M)
    7. Exit Gate: Learned adaptive exit for each loop step (~1K)

<num> token ID read from config (set by prune.py).

Output format (with chain-of-thought):
    <think>GPT reasoning</think>left_right | "left"
    <think>GPT reasoning</think>mcq | "2"
    <think>GPT reasoning</think>distance | <num>
    <think>GPT reasoning</think>count | <num>
"""

import re
import os
import sys
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoConfig
from transformers.masking_utils import create_causal_mask

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_micro.gsa import GSA
from model_micro.rti import RTE

from model_micro.num_head import NumberHead

# Default model path — pruned Micro checkpoint (from prune.py)
MODEL_NAME = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "qwen3.5-micro"
)

# NUM_TOKEN_ID: <num> token ID read from config.json at module level (set by prune.py)
# This must be available at import time for the dataloader
def _read_num_token_id():
    config_path = os.path.join(MODEL_NAME, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        return cfg.get("num_token_id", 248044)
    return 248044  # fallback for old checkpoints

NUM_TOKEN_ID = _read_num_token_id()

# Regex for structured output parsing
# Format: category | answer
_OUTPUT_RE = re.compile(
    r'(?P<category>left_right|mcq|distance|count)\s*\|\s*'
    r'(?P<answer>"left"|"right"|"\d+"|<num>|\d+\.\d+|\d+)',
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
        for test in [" <", "  <"]:
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
    """Full Micro pipeline: Qwen 3.5 VLM (pruned) + GSA + RTI + Number Head.

    Custom modules:
        self.gsa                       - GeometrySelfAttention (~16.9M)
        self.region_token_extractor    - RegionTokenExtractor  (~0.032M)
        self.num_head                  - NumberHead (xVal)     (~0.26M)


    Qwen built-in (pruned):
        self.qwen.model.visual         - Vision Encoder + Merger (4 blocks)
        self.qwen.model.language_model - 4-layer backbone (looped T_max×)
        self.qwen.lm_head              - Vocab projection

    LoopLM decoder (from Ouro paper, arXiv:2510.25741):
        4 physical layers × T_max iterations = effective depth 16 (default)
        Exit gate predicts per-step exit probability λ_t.
        Training: per-step LM head loss weighted by exit distribution + entropy.
        Inference: early exit when cumulative exit prob exceeds threshold q.
    """

    def __init__(
        self,
        model_name:              str   = MODEL_NAME,
        gsa_heads:               int   = 8,
        gsa_ffn_dim:             int   = 2048,
        dropout:                 float = 0.1,
        dtype                          = torch.bfloat16,
        device_map:              str   = "auto",
        attn_implementation:     str   = "sdpa",
        num_loops:               int   = None,  # None = read from config (T_max)
    ):
        super().__init__()

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        # Read num_loops (T_max) from config (set by prune.py), or use default
        if num_loops is None:
            num_loops = getattr(config, 'num_loops', None)
            if num_loops is None:
                config_path = os.path.join(model_name, "config.json")
                if os.path.exists(config_path):
                    with open(config_path) as f:
                        raw_config = json.load(f)
                    num_loops = raw_config.get("num_loops", 4)
                else:
                    num_loops = 4
        self.num_loops = num_loops  # T_max

        # Read <num> token ID from config
        num_token_id = getattr(config, 'num_token_id', None)
        if num_token_id is None:
            config_path = os.path.join(model_name, "config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    raw_config = json.load(f)
                num_token_id = raw_config.get("num_token_id", 248044)
            else:
                num_token_id = 248044
        self.num_token_id = num_token_id

        # Update module-level for dataloader access
        global NUM_TOKEN_ID
        NUM_TOKEN_ID = self.num_token_id

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

        # Custom Module 1: GSA
        self.gsa = GSA(
            hidden_dim=1024,
            num_heads=gsa_heads,
            ffn_dim=gsa_ffn_dim,
            dropout=dropout,
            num_blocks=2,
        )

        # Custom Module 2: RTI
        self.region_token_extractor = RTE(hidden_dim=1024)

        # Custom Module 3: Number Head
        self.num_head = NumberHead(hidden_dim=1024)



        # Decoder dropout (applied after each backbone layer)
        self.decoder_dropout = nn.Dropout(dropout)

        # Move custom modules to match Qwen device/dtype
        qwen_device = next(self.qwen.parameters()).device
        qwen_dtype  = next(self.qwen.parameters()).dtype
        self.gsa = self.gsa.to(device=qwen_device, dtype=qwen_dtype)
        self.region_token_extractor = self.region_token_extractor.to(
            device=qwen_device, dtype=qwen_dtype
        )
        self.num_head = self.num_head.to(device=qwen_device, dtype=qwen_dtype)
        print(f"  Custom modules (GSA + RTI + NumHead) -> {qwen_device} ({qwen_dtype})")

        # --- Embeddings are TRAINABLE ---
        embed = self.qwen.model.language_model.embed_tokens
        embed.weight.requires_grad = True
        print(f"  Embeddings: TRAINABLE ({embed.weight.shape[0]} tokens, requires_grad=True)")
        print(f"  <num> token ID: {self.num_token_id}")
        n_layers = len(list(self.qwen.model.language_model.layers))
        print(f"  LoopLM: {n_layers} layers × T_max={self.num_loops} = max depth {n_layers * self.num_loops}")

        # Cache space token embedding for RTI 3 -> 3 injection
        # Try several encodings since BPE may handle whitespace differently
        self._space_token_id = None
        for test_str in [" ", "  ", " .", ". "]:
            ids = self.processor.tokenizer.encode(test_str, add_special_tokens=False)
            if ids:
                self._space_token_id = ids[0]
                break
        if self._space_token_id is None:
            # Fallback: use token ID 0 (usually a byte token)
            self._space_token_id = 0
            print(f"  [WARN] Could not find space token, using ID 0 as fallback")
        else:
            decoded = self.processor.tokenizer.decode([self._space_token_id])
            print(f"  Space token ID: {self._space_token_id} -> '{decoded}'")

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
        """Run Qwen's Vision Encoder + Merger -> [B, N', 1024]."""
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
            n_visual:      int (number of visual tokens)
        """
        # Step 1: Vision Encoder + Merger -> [B, N, 1024]
        visual_tokens = self._get_visual_tokens(
            pixel_values, image_grid_thw,
            vision_requires_grad=vision_requires_grad,
        )
        n_visual = visual_tokens.shape[1]

        # Patch grid from image_grid_thw (after 2x2 merger)
        t, h, w = [int(x) for x in image_grid_thw[0].tolist()]
        h_vis, w_vis = h // 2, w // 2

        # Step 2: GSA -- depth-aware attention on visual tokens
        visual_tokens = self.gsa(
            visual_tokens, depth_maps, h_patches=h_vis, w_patches=w_vis
        )

        # Step 3: Text embeddings (direct — pruned vocab, trainable)
        embed = self.qwen.model.language_model.embed_tokens
        text_embeds = embed(input_ids)  # [B, L, 1024]

        # Step 4: RTI - inject region tokens at <mask> positions (3 -> 3)
        if (rle_list is not None and mask_token_positions is not None
                and any(len(rl) > 0 for rl in rle_list)):
            region_tokens = self.region_token_extractor(
                visual_tokens, depth_maps, rle_list, image_grid_thw,
                decoded_masks=decoded_masks,
            )
            mask_token_len = len(self.processor.tokenizer.encode(
                "<mask>", add_special_tokens=False
            ))

            # Get space embedding for 3 -> 3 RTI padding
            space_embed = embed(
                torch.tensor([self._space_token_id], device=embed.weight.device)
            ).squeeze(0).detach()  # [1024] detached (frozen space token)

            text_embeds = self.region_token_extractor.inject_into_text_embeds(
                text_embeds, mask_token_positions, region_tokens,
                mask_token_len=mask_token_len, space_embed=space_embed,
            )

        # Step 5: Concat Fusion -- [visual | text+region]
        inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)

        return inputs_embeds, n_visual

    # ---- Backbone: single loop step ----

    def _run_one_loop(
        self,
        hidden: torch.Tensor,
        lm,
        position_ids: torch.Tensor,
        position_embeddings,
        causal_mask,
        linear_mask,
        past_key_values=None,
        cache_position: torch.Tensor = None,
        use_gradient_checkpointing: bool = False,
        loop_offset: int = 0,
    ) -> torch.Tensor:
        """Run one iteration of the N physical layers."""
        for layer in lm.layers:
            orig_idx = getattr(layer, "layer_idx", None)
            if orig_idx is not None and past_key_values is not None:
                layer.layer_idx = orig_idx + loop_offset

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

            if orig_idx is not None and past_key_values is not None:
                layer.layer_idx = orig_idx

            hidden = self.decoder_dropout(hidden)

        return hidden

    # ---- Backbone forward (LoopLM) ----

    def _backbone_forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor = None,
        past_key_values=None,
        cache_position: torch.Tensor = None,
        use_gradient_checkpointing: bool = False,
    ):
        """Run LoopLM backbone: N layers × T_max loop steps.

        Returns:
            hidden_per_step: list of T_max tensors [B, T, D] — hidden states
                             after each loop step (before final norm).
            exit_lambdas:    list of T_max tensors [B] — per-step exit probs.
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
        hidden_per_step = []

        # === LoopLM: apply N layers × T_max times ===
        for loop_idx in range(self.num_loops):
            hidden = self._run_one_loop(
                hidden, lm, position_ids, position_embeddings,
                causal_mask, linear_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_gradient_checkpointing=use_gradient_checkpointing,
                loop_offset=loop_idx * len(lm.layers),
            )
            hidden_per_step.append(hidden)

        return hidden_per_step


    # ---- Forward (training) ----

    def forward(
        self,
        pixel_values:         torch.Tensor,
        image_grid_thw:       torch.Tensor,
        depth_maps:           torch.Tensor,
        input_ids:            torch.Tensor,   # [B, L]
        rle_list:             list = None,    # [B][num_masks]
        mask_token_positions: list = None,    # [B][num_masks]
        decoded_masks:        list = None,    # [B][num_masks]
        num_token_positions:  list = None,    # [B] position of <num> in input_ids
        attention_mask:       torch.Tensor = None,  # [B, L]
        use_gradient_checkpointing: bool = False,
        vision_requires_grad: bool = False,
    ) -> dict:
        """Training forward pass with LoopLM per-step outputs.

        Returns:
            dict with:
                'logits_per_step': list of T_max tensors [B, L, V] — text logits per loop
                'num_pred':        [B] — Number Head predictions (from final step)
        """
        inputs_embeds, n_visual = self._build_inputs_embeds(
            pixel_values, image_grid_thw, depth_maps, input_ids,
            rle_list, mask_token_positions, decoded_masks,
            vision_requires_grad=vision_requires_grad,
        )

        # Build full attention_mask: [B, n_visual + text_len]
        full_attention_mask = None
        if attention_mask is not None:
            B_mask = attention_mask.shape[0]
            vis_mask = torch.ones(B_mask, n_visual, dtype=attention_mask.dtype,
                                  device=attention_mask.device)
            full_attention_mask = torch.cat([vis_mask, attention_mask], dim=1)

        # Backbone -- LoopLM: per-step hidden states (always T_max steps)
        hidden_per_step = self._backbone_forward(
            inputs_embeds,
            attention_mask=full_attention_mask,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        # Per-step LM logits (only on text tokens)
        lm_norm = self.qwen.model.language_model.norm
        logits_per_step = []
        for h in hidden_per_step:
            h_normed = lm_norm(h)
            text_h = h_normed[:, n_visual:, :]
            logits_per_step.append(self.qwen.lm_head(text_h))

        # Number Head -- uses FINAL step hidden state
        final_hidden = lm_norm(hidden_per_step[-1])
        B = input_ids.shape[0]
        num_pred = torch.zeros(B, device=final_hidden.device, dtype=final_hidden.dtype)

        if num_token_positions is not None:
            num_hidden_list = []
            num_indices = []
            for b, pos in enumerate(num_token_positions):
                if pos is not None and pos >= 0:
                    adjusted_pos = n_visual + pos
                    if 0 <= adjusted_pos < final_hidden.shape[1]:
                        num_hidden_list.append(final_hidden[b, adjusted_pos, :])
                        num_indices.append(b)

            if num_hidden_list:
                h_num = torch.stack(num_hidden_list, dim=0)
                preds = self.num_head(h_num)
                for k, b in enumerate(num_indices):
                    num_pred[b] = preds[k]

        return {
            "logits_per_step": logits_per_step,
            "num_pred": num_pred,
        }

    # ---- Generate (inference) ----

    @torch.no_grad()
    def generate(
        self,
        pixel_values:         torch.Tensor,
        image_grid_thw:       torch.Tensor,
        depth_maps:           torch.Tensor,
        input_ids:            torch.Tensor,
        rle_list:             list = None,
        mask_token_positions: list = None,
        max_new_tokens:       int  = 150,
        do_sample:            bool = False,
        temperature:          float = 1.0,
        repetition_penalty:   float = 1.2,
        decoded_masks:        list = None,
        **gen_kwargs,
    ) -> torch.Tensor:
        """Autoregressive generation with LoopLM (always T_max loops).

        At each token generation step, the backbone loops T_max times.
        Uses the FINAL loop step's logits for token selection.

        Args:
            repetition_penalty: Penalize repeated tokens. >1.0 reduces repetition.
        Returns:
            output_ids: [B, generated_len] newly generated token ids
        """
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache

        inputs_embeds, n_visual = self._build_inputs_embeds(
            pixel_values, image_grid_thw, depth_maps, input_ids,
            rle_list, mask_token_positions, decoded_masks,
        )

        lm = self.qwen.model.language_model
        embed = lm.embed_tokens
        B, T, _ = inputs_embeds.shape
        dev = inputs_embeds.device

        eos_id = self.processor.tokenizer.eos_token_id
        cache = Qwen3_5DynamicCache(config=lm.config)
        attn_mask = torch.ones(B, T, dtype=torch.long, device=dev)

        # Prefill — run full T_max loops
        cache_position = torch.arange(T, device=dev)
        hidden_per_step = self._backbone_forward(
            inputs_embeds, attention_mask=attn_mask,
            past_key_values=cache, cache_position=cache_position,
        )
        # Use final step for first token
        hidden = lm.norm(hidden_per_step[-1][:, -1:, :])
        logits = self.qwen.lm_head(hidden)

        if repetition_penalty != 1.0:
            for b in range(B):
                for tok_id in input_ids[b].unique():
                    if logits[b, -1, tok_id] > 0:
                        logits[b, -1, tok_id] /= repetition_penalty
                    else:
                        logits[b, -1, tok_id] *= repetition_penalty

        if do_sample and temperature > 0:
            probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
        else:
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        generated = [next_tok]
        all_generated = next_tok.clone()

        # Decode with LoopLM early exit
        for step in range(max_new_tokens - 1):
            if eos_id is not None and (next_tok == eos_id).all():
                break

            tok_embed = embed(next_tok)
            step_cache_pos = torch.tensor([T + step], device=dev)

            # LoopLM decode: always run full T_max loops
            hidden_per_step = self._backbone_forward(
                tok_embed, past_key_values=cache, cache_position=step_cache_pos,
            )
            hidden = lm.norm(hidden_per_step[-1])
            logits = self.qwen.lm_head(hidden)

            if repetition_penalty != 1.0:
                for b in range(B):
                    for tok_id in all_generated[b].unique():
                        if logits[b, -1, tok_id] > 0:
                            logits[b, -1, tok_id] /= repetition_penalty
                        else:
                            logits[b, -1, tok_id] *= repetition_penalty

            if do_sample and temperature > 0:
                probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            else:
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)

            generated.append(next_tok)
            all_generated = torch.cat([all_generated, next_tok], dim=1)

        output_ids = torch.cat(generated, dim=1)  # [B, gen_len]
        return output_ids

    # ---- Output parsing ----

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
        image,                          # PIL.Image
        question: str,
        depth_map: torch.Tensor,        # [H, W] raw depth tensor
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

        # Process image separately
        image_inputs = self.processor.image_processor(
            images=image, return_tensors="pt"
        )
        pixel_values   = image_inputs["pixel_values"].to(device=dev, dtype=dtype)
        image_grid_thw = image_inputs["image_grid_thw"].to(device=dev)
        depth_batch    = depth_map.unsqueeze(0).to(device=dev, dtype=dtype)

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
            pixel_values, image_grid_thw, depth_batch, input_ids,
            rle_list=rle_list_batched,
            mask_token_positions=mask_positions_batched,
            max_new_tokens=max_new_tokens,
        )
        raw_output = self.processor.tokenizer.decode(
            output_ids[0], skip_special_tokens=False
        ).replace("<|endoftext|>", "").replace("<|im_end|>", "").replace("<|num|>", "<num>").strip()

        parsed = self.parse_output(raw_output)

        return {
            "category": parsed["category"],
            "answer":   parsed["answer"],
            "raw":      raw_output,
        }


# Parameter counting util

def count_parameters(model: nn.Module) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


# -----------------------------------------------------------------------------
# Standalone demo:   python model_micro/pipeline.py
# -----------------------------------------------------------------------------

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
        "GSA (DFormerv2 Full_GSA x2)":     pipeline.gsa,
        "RTI (Region Token Injector)":     pipeline.region_token_extractor,
        "Number Head (xVal regression)":   pipeline.num_head,
    }
    custom_names = {
        "GSA (DFormerv2 Full_GSA x2)",
        "RTI (Region Token Injector)",
        "Number Head (xVal regression)",
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
    print(f"  <num> token ID: {pipeline.num_token_id}")
    print(f"  Trainable params: {sum(p.numel() for p in pipeline.parameters() if p.requires_grad)/1e6:.2f}M")
    n_layers = len(list(pipeline.qwen.model.language_model.layers))
    print(f"  LoopLM: {n_layers} layers × T_max={pipeline.num_loops} = max depth {n_layers * pipeline.num_loops}")
    print(f"  ViT blocks: {len(list(pipeline.qwen.model.visual.blocks))}")

    print_vram_usage("final")
