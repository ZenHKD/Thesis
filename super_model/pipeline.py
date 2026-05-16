"""
MODULE: Full Pipeline — SpatialVLM Super (Training)

Architecture:
    1. Qwen 3.5 Vision Encoder (pruned: 4 ViT blocks)
       4 ViT blocks (768-dim) + merger (VL Projector, 768->1024)
    2. RTI: Region-Level Token Injection (batched)
       Each <mask> -> [mask_rgb | mask_depth | global_depth] (3 tokens x 1024-dim)
    3. Concat Fusion: [visual_tokens | text+region_tokens]
    4. Qwen 3.5 Backbone (full: 24 layers, single pass)
    5. Five Heads:
       - LM Head (tied w/ embed): category text
       - MCQ Head: Tri-Source + Visual Context Scoring
       - LeftRight Head: Binary Tri-Source + Visual Context Scoring
       - Distance Head: Tri-Source + Visual Context -> Regression
       - Count Head: Mask-Centric Tri-Source Regression

4 special token IDs read from config (set by prune.py).

Output format (direct, no chain-of-thought):
    mcq | <|mcq|>
    left_right | <|lr|>
    distance | <|dist|>
    count | <|count|>
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

from super_model.rti import RTE
from super_model.heads import SharedVisualFuser, MCQHead, LeftRightHead, DistanceHead, CountHead


# Default model path
MODEL_NAME = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "qwen3.5-super"
)

# Special token IDs
def _read_special_token_ids():
    config_path = os.path.join(MODEL_NAME, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        return (
            cfg.get("mcq_token_id", 248077),
            cfg.get("lr_token_id", 248078),
            cfg.get("dist_token_id", 248079),
            cfg.get("count_token_id", 248080),
        )
    return 248077, 248078, 248079, 248080

MCQ_TOKEN_ID, LR_TOKEN_ID, DIST_TOKEN_ID, COUNT_TOKEN_ID = _read_special_token_ids()

# Regex for structured output parsing
# Format: category | answer
_OUTPUT_RE = re.compile(
    r'(?P<category>left_right|mcq|distance|count)\s*\|\s*'
    r'(?P<answer><\|mcq\|>|<\|lr\|>|<\|dist\|>|<\|count\|>)',
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
    """Full pipeline: Qwen 3.5 VLM (vision pruned) + RTI + 4 Dedicated Heads.

    Custom modules:
        self.region_token_extractor  - RTE (Tri-Source U-Net)
        self.mcq_head                - MCQHead
        self.lr_head                 - LeftRightHead
        self.dist_head               - DistanceHead
        self.count_head              - CountHead

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

        # Read 4 special token IDs from config
        config_path_json = os.path.join(model_name, "config.json")
        raw_cfg = {}
        if os.path.exists(config_path_json):
            with open(config_path_json) as f:
                raw_cfg = json.load(f)

        self.mcq_token_id   = getattr(config, 'mcq_token_id', None) or raw_cfg.get("mcq_token_id", 248077)
        self.lr_token_id    = getattr(config, 'lr_token_id', None) or raw_cfg.get("lr_token_id", 248078)
        self.dist_token_id  = getattr(config, 'dist_token_id', None) or raw_cfg.get("dist_token_id", 248079)
        self.count_token_id = getattr(config, 'count_token_id', None) or raw_cfg.get("count_token_id", 248080)

        # Update module-level for dataloader access
        global MCQ_TOKEN_ID, LR_TOKEN_ID, DIST_TOKEN_ID, COUNT_TOKEN_ID
        MCQ_TOKEN_ID   = self.mcq_token_id
        LR_TOKEN_ID    = self.lr_token_id
        DIST_TOKEN_ID  = self.dist_token_id
        COUNT_TOKEN_ID = self.count_token_id

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

        # Custom Modules — shared visual fuser + 4 dedicated heads
        self.region_token_extractor = RTE(hidden_dim=1024, vit_dim=768)
        self.visual_fuser = SharedVisualFuser(hidden_dim=1024)
        self.mcq_head   = MCQHead(hidden_dim=1024)
        self.lr_head    = LeftRightHead(hidden_dim=1024)
        self.dist_head  = DistanceHead(hidden_dim=1024)
        self.count_head = CountHead(hidden_dim=1024)

        self.decoder_dropout = nn.Dropout(dropout)

        # Move custom modules to match Qwen device/dtype
        qwen_device = next(self.qwen.parameters()).device
        qwen_dtype  = next(self.qwen.parameters()).dtype
        self.region_token_extractor = self.region_token_extractor.to(device=qwen_device, dtype=qwen_dtype)
        self.visual_fuser = self.visual_fuser.to(device=qwen_device, dtype=qwen_dtype)
        self.mcq_head   = self.mcq_head.to(device=qwen_device, dtype=qwen_dtype)
        self.lr_head    = self.lr_head.to(device=qwen_device, dtype=qwen_dtype)
        self.dist_head  = self.dist_head.to(device=qwen_device, dtype=qwen_dtype)
        self.count_head = self.count_head.to(device=qwen_device, dtype=qwen_dtype)
        print(f"  Custom modules (RTI + Fuser + Heads) -> {qwen_device} ({qwen_dtype})")

        embed = self.qwen.model.language_model.embed_tokens
        embed.weight.requires_grad = True
        print(f"  Embeddings: TRAINABLE ({embed.weight.shape[0]} tokens, requires_grad=True)")
        print(f"  <|mcq|>={self.mcq_token_id}  <|lr|>={self.lr_token_id}  <|dist|>={self.dist_token_id}  <|count|>={self.count_token_id}")

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
    ) -> tuple:
        """Run Qwen's Vision Encoder + Merger.
        Returns:
            stacked_visual_tokens: [B, N, 1024]
            stacked_vit_feat:      [B, 768, H, W] (pre-merger unflattened feature map)
        """
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
            return stacked, None

        # Pre-merger: apply merger per-image
        ms = 2
        merged = []
        vit_feats = []
        for i in range(B):
            h_i = hidden_list[i].unsqueeze(0)
            t, h, w = [int(x) for x in image_grid_thw[i].tolist()]
            C = h_i.shape[-1]

            h_i = visual.merger.norm(h_i)
            
            # Extract spatial ViT feature map [1, C, h, w]
            vit_spatial = h_i.view(1, t, h, w, C).squeeze(1).permute(0, 3, 1, 2).contiguous()
            vit_feats.append(vit_spatial)

            h_i = h_i.view(1, t, h, w, C)
            h_i = h_i.view(1, t, h // ms, ms, w // ms, ms, C)
            h_i = h_i.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
            h_i = h_i.view(1, -1, ms * ms * C)

            h_i = visual.merger.linear_fc1(h_i)
            h_i = F.gelu(h_i)
            h_i = visual.merger.linear_fc2(h_i)

            merged.append(h_i)

        return torch.cat(merged, dim=0), torch.cat(vit_feats, dim=0)

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
            inputs_embeds:  [B, T, 1024]
            n_visual:       int (0, since visual tokens are inline padded)
            region_tokens:  list[list[tuple]] — RTI projected tokens per sample/mask
                            region_tokens[b][m] = (rgb [1,1024], dep [1,1024], gdep [1,1024])
                            None if no RTI was computed.
        """
        # Step 1: Vision Encoder + Merger -> [B, N, 1024]
        visual_tokens, vit_feat = self._get_visual_tokens(
            pixel_values, image_grid_thw,
            vision_requires_grad=vision_requires_grad,
        )
        n_visual = visual_tokens.shape[1]

        # Step 2: Text embeddings
        embed = self.qwen.model.language_model.embed_tokens
        text_embeds = embed(input_ids)

        # Step 3: RTI (Independent of Vision Encoder)
        region_tokens = None
        if (rle_list is not None and mask_token_positions is not None
                and any(len(rl) > 0 for rl in rle_list)):
            region_tokens = self.region_token_extractor(
                pixel_values_rgb, depth_maps, rle_list, image_grid_thw,
                decoded_masks=decoded_masks, vit_feat=vit_feat,
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

        return inputs_embeds, n_visual_offset, region_tokens

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
        mcq_token_positions:   list = None,
        lr_token_positions:    list = None,
        dist_token_positions:  list = None,
        count_token_positions: list = None,
        attention_mask:       torch.Tensor = None,
        use_gradient_checkpointing: bool = False,
        vision_requires_grad: bool = False,
    ) -> dict:
        """Training forward pass.

        Returns:
            dict with:
                'logits':       [B, L, V] — text logits
                'dist_pred':    [B]       — Distance Head predictions
                'count_pred':   [B]       — Count Head predictions
                'mcq_logits':   list of [N_masks] tensors
                'lr_logits':    list of [2] tensors
        """
        inputs_embeds, n_visual, region_tokens = self._build_inputs_embeds(
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

        # Build per-modality RTI features (kept separate, no concat)
        B = input_ids.shape[0]
        rgb_batch  = []  # list of [N_masks, 1024] or None
        dep_batch  = []  # list of [N_masks, 1024] or None
        gdep_batch = []  # list of [N_masks, 1024] or None
        if region_tokens is not None:
            for b in range(B):
                if b < len(region_tokens) and region_tokens[b]:
                    rgb_list = []
                    dep_list = []
                    gdep_list = []
                    for rgb, dep, gdep in region_tokens[b]:
                        rgb_list.append(rgb.squeeze(0))   # [1024]
                        dep_list.append(dep.squeeze(0))   # [1024]
                        gdep_list.append(gdep.squeeze(0)) # [1024]
                    rgb_batch.append(torch.stack(rgb_list))   # [N_masks, 1024]
                    dep_batch.append(torch.stack(dep_list))   # [N_masks, 1024]
                    gdep_batch.append(torch.stack(gdep_list)) # [N_masks, 1024]
                else:
                    rgb_batch.append(None)
                    dep_batch.append(None)
                    gdep_batch.append(None)
        else:
            rgb_batch  = [None] * B
            dep_batch  = [None] * B
            gdep_batch = [None] * B

        # Extract visual tokens (post-merger, 160 tokens) for head context
        # Visual tokens are embedded at image_pad positions in h_normed
        img_pad_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        vis_batch = []  # list of [N_vis, 1024]
        for b in range(B):
            pad_indices = (input_ids[b] == img_pad_id).nonzero(as_tuple=True)[0]
            if len(pad_indices) > 0:
                vis_tokens_b = h_normed[b, n_visual + pad_indices, :]  # [N_vis, 1024]
                vis_batch.append(vis_tokens_b)
            else:
                vis_batch.append(None)

        # ---- Distance Head ----
        dist_pred = torch.zeros(B, device=h_normed.device, dtype=h_normed.dtype)
        if dist_token_positions is not None:
            dist_h_list, dist_rgb, dist_dep, dist_gdep, dist_vis = [], [], [], [], []
            dist_indices = []
            for b, pos in enumerate(dist_token_positions):
                if pos is not None and pos >= 0:
                    adj = n_visual + pos
                    if 0 <= adj < h_normed.shape[1] and rgb_batch[b] is not None:
                        dist_h_list.append(h_normed[b, adj, :])
                        dist_rgb.append(rgb_batch[b])
                        dist_dep.append(dep_batch[b])
                        dist_gdep.append(gdep_batch[b])
                        dist_vis.append(vis_batch[b])
                        dist_indices.append(b)
            if dist_h_list:
                h_dist = torch.stack(dist_h_list, dim=0)
                # Compute fused queries via shared fuser
                dist_q_list = []
                for k in range(len(dist_h_list)):
                    q, _ = self.visual_fuser(dist_h_list[k], dist_vis[k])
                    dist_q_list.append(q)
                preds = self.dist_head(h_dist, dist_rgb, dist_dep, dist_gdep, dist_q_list)
                for k, b in enumerate(dist_indices):
                    dist_pred[b] = preds[k]

        # ---- Count Head ----
        count_pred = torch.zeros(B, device=h_normed.device, dtype=h_normed.dtype)
        if count_token_positions is not None:
            cnt_h_list, cnt_rgb, cnt_dep, cnt_gdep, cnt_vis = [], [], [], [], []
            cnt_indices = []
            for b, pos in enumerate(count_token_positions):
                if pos is not None and pos >= 0:
                    adj = n_visual + pos
                    if 0 <= adj < h_normed.shape[1] and rgb_batch[b] is not None:
                        cnt_h_list.append(h_normed[b, adj, :])
                        cnt_rgb.append(rgb_batch[b])
                        cnt_dep.append(dep_batch[b])
                        cnt_gdep.append(gdep_batch[b])
                        cnt_vis.append(vis_batch[b])
                        cnt_indices.append(b)
            if cnt_h_list:
                h_cnt = torch.stack(cnt_h_list, dim=0)
                # Compute fused queries + scene contexts via shared fuser
                cnt_q_list, cnt_scene_list = [], []
                for k in range(len(cnt_h_list)):
                    q, sc = self.visual_fuser(cnt_h_list[k], cnt_vis[k])
                    cnt_q_list.append(q)
                    cnt_scene_list.append(sc)
                preds = self.count_head(h_cnt, cnt_rgb, cnt_dep, cnt_gdep, cnt_q_list, cnt_scene_list)
                for k, b in enumerate(cnt_indices):
                    count_pred[b] = preds[k]

        # ---- MCQ Head ----
        mcq_logits_list = []
        if mcq_token_positions is not None:
            for b, pos in enumerate(mcq_token_positions):
                if pos is not None and pos >= 0:
                    adj = n_visual + pos
                    if 0 <= adj < h_normed.shape[1] and rgb_batch[b] is not None:
                        h_mcq = h_normed[b, adj, :]
                        q, _ = self.visual_fuser(h_mcq, vis_batch[b])
                        scores = self.mcq_head(rgb_batch[b], dep_batch[b], gdep_batch[b], q)
                        mcq_logits_list.append(scores)
                    else:
                        mcq_logits_list.append(None)
                else:
                    mcq_logits_list.append(None)

        # ---- LeftRight Head ----
        lr_logits_list = []
        if lr_token_positions is not None:
            for b, pos in enumerate(lr_token_positions):
                if pos is not None and pos >= 0:
                    adj = n_visual + pos
                    if 0 <= adj < h_normed.shape[1] and rgb_batch[b] is not None:
                        h_lr = h_normed[b, adj, :]
                        q, _ = self.visual_fuser(h_lr, vis_batch[b])
                        scores = self.lr_head(rgb_batch[b], dep_batch[b], gdep_batch[b], q)
                        lr_logits_list.append(scores)
                    else:
                        lr_logits_list.append(None)
                else:
                    lr_logits_list.append(None)

        return {
            "logits": logits,
            "dist_pred": dist_pred,
            "count_pred": count_pred,
            "mcq_logits": mcq_logits_list,
            "lr_logits": lr_logits_list,
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
        max_new_tokens:       int  = 20,
        repetition_penalty:   float = 1.2,
        decoded_masks:        list = None,
        **gen_kwargs,
    ) -> torch.Tensor:
        """Autoregressive generation with repetition penalty."""
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache

        inputs_embeds, n_visual, _region_tokens = self._build_inputs_embeds(
            pixel_values, pixel_values_rgb, image_grid_thw, depth_maps, input_ids,
            rle_list, mask_token_positions, decoded_masks,
        )

        lm = self.qwen.model.language_model
        embed = lm.embed_tokens
        B, T, _ = inputs_embeds.shape
        dev = inputs_embeds.device

        eos_id = self.processor.tokenizer.eos_token_id
        cache = Qwen3_5DynamicCache(config=lm.config)
        attn_mask = gen_kwargs.get("attention_mask", torch.ones(B, T, dtype=torch.long, device=dev))

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

        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        generated = [next_tok]
        all_generated = next_tok.clone()

        for step in range(max_new_tokens - 1):
            if eos_id is not None and (next_tok == eos_id).all():
                break

            tok_embed = embed(next_tok)
            step_cache_pos = torch.tensor([T + step], device=dev)
            
            attn_mask = torch.cat([attn_mask, torch.ones(B, 1, dtype=torch.long, device=dev)], dim=1)

            hidden = self._backbone_forward(
                tok_embed, past_key_values=cache, cache_position=step_cache_pos,
                attention_mask=attn_mask,
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

            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)

            generated.append(next_tok)
            all_generated = torch.cat([all_generated, next_tok], dim=1)

        output_ids = torch.cat(generated, dim=1)
        return output_ids

    @staticmethod
    def parse_output(text: str) -> dict:
        """Parse structured LM output -> {category, answer}.

        Expected format: category | value
        """
        clean = text.strip()
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
        os.path.dirname(os.path.abspath(__file__)), "qwen3.5-super"
    )

    print("=" * 70)
    print("MODULE: SpatialVLM Super")
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
        "MCQ Head":                        pipeline.mcq_head,
        "LeftRight Head":                  pipeline.lr_head,
        "Distance Head":                   pipeline.dist_head,
        "Count Head":                      pipeline.count_head,
    }
    custom_names = {
        "RTI (Region Token Injector)",
        "MCQ Head",
        "LeftRight Head",
        "Distance Head",
        "Count Head",
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
    print(f"  Qwen base:        {total_qwen:>12,} ({total_qwen/1e6:.4f}M)")
    print(f"  Custom modules:   {total_custom:>12,} ({total_custom/1e6:.4f}M)")
    print(f"  Total unique:     {total_qwen + total_custom:>12,} ({(total_qwen + total_custom)/1e6:.4f}M)")
    print(f"\n  Vocab: {pipeline.qwen.model.language_model.embed_tokens.weight.shape[0]} (TRAINABLE)")
    print(f"  <|mcq|>={pipeline.mcq_token_id}  <|lr|>={pipeline.lr_token_id}  <|dist|>={pipeline.dist_token_id}  <|count|>={pipeline.count_token_id}")
    print(f"  Trainable params: {sum(p.numel() for p in pipeline.parameters() if p.requires_grad)/1e6:.2f}M")
    n_layers = len(list(pipeline.qwen.model.language_model.layers))
    print(f"  Decoder: {n_layers} layers")
    print(f"  ViT blocks: {len(list(pipeline.qwen.model.visual.blocks))}")

    print_vram_usage("final")
