"""
MODULE: Full Pipeline -- SpatialVLM (Test Inference only)

Architecture:
    1. Qwen 3.5 Vision Encoder (pretrained, 100.59M)
       12 ViT blocks (768-dim) + merger (VL Projector, 768->1024)
    2. GSA: Geometry Self-Attention -- DFormerv2 Full_GSA CVPR 2025
       Position: after merger, before concat fusion
    3. RTI: Region-Level Token Injection
       Each <mask> -> [mask_rgb | mask_depth] (2 tokens x 1024-dim)
    4. Concat Fusion: [visual_tokens | text+region_tokens]   [B, T, 1024]
    5. Qwen 3.5 Backbone 24 layers (pretrained, 498M) + LoRA in Phase 2
    6. Qwen LM Head (built-in, tied with embeddings) -> structured text

Output format:
    CATEGORY: <left_right|mcq|distance|count> | ANSWER: <value> | FREE_ANSWER: <explanation>

Qwen 3.5 0.8B model hierarchy (verified from weights):
    model.visual                          -> Vision Encoder (100.59M)
    model.visual.patch_embed              -> Conv3D [768, 3, 2, 16, 16]
    model.visual.blocks[0..11]            -> 12 ViT blocks (Attn + GELU MLP, 768-dim)
    model.visual.merger                   -> VL Projector (3072->1024)
    model.language_model.embed_tokens     -> Token Embeddings (248320x1024, tied)
    model.language_model.layers[0..23]    -> 24 backbone layers
    model.language_model.norm             -> Final RMSNorm
    lm_head                               -> Tied with embed_tokens
"""

import re
import sys
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoConfig
import pycocotools.mask as mask_utils

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.gsa import GSA
from model.rti import RTE

# Default: local clone (download once via `huggingface-cli download`)
MODEL_NAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen3.5-0.8b")

# System Prompt (prepended to every input, train & inference)
# Format: CATEGORY first -> ANSWER (type-committed) -> FREE_ANSWER (coherent explanation)
# Approach 1: reversed order ensures FREE_ANSWER is always consistent with ANSWER.
SYSTEM_PROMPT = (
    "You are a spatial reasoning assistant. "
    "You MUST always respond in the following exact format:\n"
    "CATEGORY: <left_right|mcq|distance|count> | "
    "ANSWER: <value> | "
    "FREE_ANSWER: <natural language explanation>\n"
    "For left_right: ANSWER is \"left\" or \"right\". "
    "For mcq: ANSWER is a quoted integer e.g. \"1\". "
    "For distance: ANSWER is a float e.g. 5.2. "
    "For count: ANSWER is an integer e.g. 3. "
    "Do not add any text before or after this format."
)

# Regex for structured output parsing
# Format: CATEGORY -> ANSWER -> FREE_ANSWER  (Approach 1: answer-first, explanation last)
# Flat regex: no lookbehind - cross-field type check done in parse_output() instead.
# ANSWER alternatives ordered float-before-int so "5.2" always matches as float.
# FREE_ANSWER uses .+ safely - it is the final field, nothing follows it.
_OUTPUT_RE = re.compile(
    r'CATEGORY:\s*(?P<category>left_right|mcq|distance|count)\s*\|\s*'
    r'ANSWER:\s*(?P<answer>"left"|"right"|"\d+"|\d+\.\d+|\d+)\s*\|\s*'
    r'FREE_ANSWER:\s*(?P<free_answer>.+)',
    re.IGNORECASE,
)

# Approach 2: category -> expected answer type (post-parse validation + coercion)
# Enforces CATEGORY <-> ANSWER consistency that regex alone cannot guarantee.
_ANSWER_TYPE = {
    "left_right": lambda a: a.strip('"') in ("left", "right"),
    "mcq":        lambda a: a.startswith('"') and a.strip('"').isdigit(),
    "distance":   lambda a: '.' in a and not a.startswith('"'),
    "count":      lambda a: a.isdigit(),
}


class SpatialVLM(nn.Module):
    """Full pipeline: Qwen 3.5 VLM + GSA + RTI + structured LM output.

    Custom modules:
        self.gsa                       - GeometrySelfAttention
        self.region_token_extractor    - RegionTokenExtractor

    Qwen built-in:
        self.qwen.model.visual         - Vision Encoder + Merger
        self.qwen.model.language_model - 24-layer backbone
        self.qwen.lm_head              - Tied vocab projection (output)
    """

    def __init__(
        self,
        model_name:              str   = MODEL_NAME,
        gsa_heads:               int   = 8,
        gsa_ffn_dim:             int   = 2048,
        dropout:                 float = 0.1,
        dtype                          = torch.bfloat16,
        device_map:              str   = "auto",
        attn_implementation:     str   = "sdpa",  # "flash_attention_2", "sdpa", or "eager"
    ):
        super().__init__()

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

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
        self.region_token_extractor = RTE(
            hidden_dim=1024,
        )

        # Move custom modules to match Qwen device/dtype
        qwen_device = next(self.qwen.parameters()).device
        qwen_dtype  = next(self.qwen.parameters()).dtype
        self.gsa                    = self.gsa.to(device=qwen_device, dtype=qwen_dtype)
        self.region_token_extractor = self.region_token_extractor.to(
            device=qwen_device, dtype=qwen_dtype
        )
        print(f"  Custom modules (GSA + RTI) -> {qwen_device} ({qwen_dtype})")

    # Properties
    @property
    def device(self):
        return next(self.qwen.parameters()).device

    # Internal helpers

    def _get_visual_tokens(
        self,
        pixel_values:   torch.Tensor,   # from Qwen processor
        image_grid_thw: torch.Tensor,   # [num_images, 3]
    ) -> torch.Tensor:
        """Run Qwen's Vision Encoder + Merger -> [B, N', 1024].

        model.visual() returns 768-dim pre-merger hidden states.
        Apply the merger (LN -> 2x2 merge -> MLP 3072->1024) manually.
        """
        visual = self.qwen.model.visual
        visual_out = visual(pixel_values, grid_thw=image_grid_thw)

        # Unpack output
        if isinstance(visual_out, torch.Tensor):
            hidden = visual_out
        elif hasattr(visual_out, "last_hidden_state"):
            hidden = visual_out.last_hidden_state
        elif isinstance(visual_out, tuple):
            hidden = visual_out[0]
        else:
            hidden = visual_out

        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)

        if hidden.shape[-1] == 1024:
            return hidden

        # Pre-merger: [B, N, 768] -> Apply merger manually
        B, N, C = hidden.shape
        t, h, w = [int(x) for x in image_grid_thw[0].tolist()]

        # Step 1: LayerNorm
        hidden = visual.merger.norm(hidden)

        # Step 2: 2x2 spatial merge -> [B, N/4, 4xC=3072]
        ms = 2
        hidden = hidden.view(B, t, h, w, C)
        hidden = hidden.view(B, t, h // ms, ms, w // ms, ms, C)
        hidden = hidden.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        hidden = hidden.view(B, -1, ms * ms * C)   # [B, N/4, 3072]

        # Step 3: MLP 3072 -> 1024
        hidden = visual.merger.linear_fc1(hidden)
        hidden = F.gelu(hidden)
        hidden = visual.merger.linear_fc2(hidden)  # [B, N/4, 1024]
        return hidden

    def _build_inputs_embeds(
        self,
        pixel_values:         torch.Tensor,   # from processor
        image_grid_thw:       torch.Tensor,   # [num_images, 3]
        depth_maps:           torch.Tensor,   # [B, H, W]
        input_ids:            torch.Tensor,   # [B, L]
        rle_list:             list = None,    # list[dict] -- one per <mask>
        mask_token_positions: list = None,    # sorted token indices of <mask>
    ) -> tuple:
        """Build [B, T, 1024] inputs_embeds for the backbone.

        Returns:
            inputs_embeds: [B, T, 1024]  (visual + text + region tokens)
            n_visual:       int           (number of visual tokens)
        """
        # Step 1: Vision Encoder + Merger -> [B, N, 1024]
        visual_tokens = self._get_visual_tokens(pixel_values, image_grid_thw)
        n_visual = visual_tokens.shape[1]

        # Patch grid from image_grid_thw (after 2x2 merger)
        t, h, w = [int(x) for x in image_grid_thw[0].tolist()]
        h_vis, w_vis = h // 2, w // 2

        # Step 2: GSA -- depth-aware attention on visual tokens
        visual_tokens = self.gsa(
            visual_tokens, depth_maps, h_patches=h_vis, w_patches=w_vis
        )  # [B, N, 1024]

        # Step 3: Text embeddings
        embed = self.qwen.model.language_model.embed_tokens
        text_embeds = embed(input_ids)  # [B, L, 1024]

        # Step 4: RTI - inject region tokens at <mask> positions
        if rle_list is not None and mask_token_positions is not None and len(rle_list) > 0:
            region_tokens = self.region_token_extractor(
                visual_tokens, depth_maps, rle_list, image_grid_thw
            )
            # <mask> may be multi-token (Qwen: '<','mask','>' = 3 tokens)
            mask_token_len = len(self.processor.tokenizer.encode("<mask>", add_special_tokens=False))
            text_embeds = self.region_token_extractor.inject_into_text_embeds(
                text_embeds, mask_token_positions, region_tokens,
                mask_token_len=mask_token_len,
            )  # [B, L', 1024]

        # Step 5: Concat Fusion -- [visual | text+region]
        inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)  # [B, T, 1024]

        return inputs_embeds, n_visual

    def _backbone_forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values=None,
        cache_position: torch.Tensor = None,
    ):
        """Run 24 Qwen backbone layers on inputs_embeds -> [B, T, 1024].

        Optionally supports KV-cache: pass a Qwen3_5DynamicCache as
        past_key_values and the corresponding cache_position.
        The cache is updated **in-place** by each layer.

        Returns:
            hidden: [B, T, 1024]
        """
        B, seq_len, _ = inputs_embeds.shape
        lm = self.qwen.model.language_model

        # Position ids -- use cache_position if available (handles offset for decode steps)
        if cache_position is not None:
            position_ids = cache_position.unsqueeze(0).expand(B, -1)
        else:
            position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0).expand(B, -1)

        position_embeddings = None
        if hasattr(lm, "rotary_emb"):
            position_embeddings = lm.rotary_emb(inputs_embeds, position_ids)

        hidden = inputs_embeds
        for layer in lm.layers:
            kwargs = {"position_ids": position_ids}
            if position_embeddings is not None:
                kwargs["position_embeddings"] = position_embeddings
            if past_key_values is not None:
                kwargs["past_key_values"] = past_key_values
                kwargs["cache_position"] = cache_position
            try:
                layer_out = layer(hidden, **kwargs)
            except TypeError:
                # Fallback: some layer types (e.g. DeltaNet) may not accept position kwargs
                layer_out = layer(hidden)
            hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        return hidden

    def forward(
        self,
        pixel_values:         torch.Tensor,
        image_grid_thw:       torch.Tensor,
        depth_maps:           torch.Tensor,
        input_ids:            torch.Tensor,
        rle_list:             list = None,
        mask_token_positions: list = None,
    ) -> dict:
        """Inference forward pass.

        Args:
            pixel_values:         [B*patches, C*ph*pw] from Qwen processor
            image_grid_thw:       [num_images, 3]
            depth_maps:           [B, H, W] raw depth
            input_ids:            [B, L] tokenized question
            rle_list:             list[dict] RLE per <mask>, or None
            mask_token_positions: sorted token indices of <mask> in input_ids, or None
        Returns:
            dict with 'logits': [B, T, vocab_size]
        """
        inputs_embeds, n_visual = self._build_inputs_embeds(
            pixel_values, image_grid_thw, depth_maps, input_ids,
            rle_list, mask_token_positions,
        )

        # Backbone
        hidden = self._backbone_forward(inputs_embeds)          # [B, T, 1024]
        hidden = self.qwen.model.language_model.norm(hidden)    # [B, T, 1024]
        logits = self.qwen.lm_head(hidden)                      # [B, T, vocab]

        return {"logits": logits}

    @torch.no_grad()
    def generate(
        self,
        pixel_values:         torch.Tensor,
        image_grid_thw:       torch.Tensor,
        depth_maps:           torch.Tensor,
        input_ids:            torch.Tensor,
        rle_list:             list = None,
        mask_token_positions: list = None,
        max_new_tokens:       int  = 80,
        do_sample:            bool = False,
        temperature:          float = 1.0,
        **gen_kwargs,
    ) -> torch.Tensor:
        """Autoregressive generation with KV-cache.

        Uses _backbone_forward with Qwen3_5DynamicCache passed directly to
        each layer (bypasses lm.forward() which creates causal_mask that
        crashes flash_attention_2 with custom inputs_embeds).

        Complexity: O(n + m) where n = prefill length, m = max_new_tokens.

        Returns:
            output_ids: [B, generated_len] newly generated token ids
        """
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache

        inputs_embeds, _ = self._build_inputs_embeds(
            pixel_values, image_grid_thw, depth_maps, input_ids,
            rle_list, mask_token_positions,
        )

        lm = self.qwen.model.language_model
        embed = lm.embed_tokens
        eos_id = self.processor.tokenizer.eos_token_id
        B, T, _ = inputs_embeds.shape
        dev = inputs_embeds.device

        # Create KV-cache (handles both GatedAttn KV-cache and DeltaNet recurrent states)
        cache = Qwen3_5DynamicCache(config=lm.config)

        # --- Prefill: run full sequence, populate cache ---
        cache_position = torch.arange(T, device=dev)
        hidden = self._backbone_forward(inputs_embeds, past_key_values=cache, cache_position=cache_position)
        hidden = lm.norm(hidden[:, -1:, :])                                # [B, 1, 1024]
        logits = self.qwen.lm_head(hidden)                                 # [B, 1, vocab]

        if do_sample and temperature > 0:
            probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
        else:
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        generated = [next_tok]

        # --- Decode: 1 token at a time with KV-cache ---
        for step in range(max_new_tokens - 1):
            if eos_id is not None and (next_tok == eos_id).all():
                break

            tok_embed = embed(next_tok)                                    # [B, 1, 1024]
            step_cache_pos = torch.tensor([T + step], device=dev)

            hidden = self._backbone_forward(tok_embed, past_key_values=cache, cache_position=step_cache_pos)
            hidden = lm.norm(hidden)                                       # [B, 1, 1024]
            logits = self.qwen.lm_head(hidden)                             # [B, 1, vocab]

            if do_sample and temperature > 0:
                probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            else:
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)

            generated.append(next_tok)

        return torch.cat(generated, dim=1)  # [B, generated_len]

    # Output parsing

    @staticmethod
    def parse_output(text: str) -> dict:
        """Parse structured LM output -> {free_answer, category, answer, type_ok}.

        Expected format (Approach 1 -- answer-first):
            CATEGORY: <task> | ANSWER: <value> | FREE_ANSWER: <explanation>

        Approach 2 -- cross-field type validation after parse:
            Checks that ANSWER type matches CATEGORY (e.g. distance must be float).
            'type_ok' = False signals a format-consistency failure.

        Returns:
            dict with keys: 'free_answer', 'category', 'answer', 'type_ok'
            On parse failure: category='unknown', answer=None, type_ok=False.
        """
        m = _OUTPUT_RE.search(text)
        if m:
            category = m.group("category").strip().lower()
            answer   = m.group("answer").strip()
            type_ok  = _ANSWER_TYPE.get(category, lambda _: False)(answer)
            return {
                "free_answer": m.group("free_answer").strip(),
                "category":    category,
                "answer":      answer,
                "type_ok":     type_ok,   # Approach 2: CATEGORY<->ANSWER consistency flag
            }
        return {"free_answer": text.strip(), "category": "unknown", "answer": None, "type_ok": False}

    # Full inference convenience method

    @torch.no_grad()
    def predict(
        self,
        image,                          # PIL.Image
        question: str,
        depth_map: torch.Tensor,        # [H, W] raw depth tensor
        rle_list: list = None,
        mask_token_positions: list = None,
        max_new_tokens: int = 100,
    ) -> dict:
        """End-to-end: image + question -> {free_answer, category, answer, type_ok}.

        Adds SYSTEM_PROMPT automatically.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": question},
            ]},
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")

        dev   = self.device
        dtype = next(self.qwen.parameters()).dtype
        pixel_values   = inputs["pixel_values"].to(device=dev, dtype=dtype)
        image_grid_thw = inputs["image_grid_thw"].to(device=dev)
        input_ids      = inputs["input_ids"].to(device=dev)
        depth_batch    = depth_map.unsqueeze(0).to(device=dev, dtype=dtype)

        output_ids = self.generate(
            pixel_values, image_grid_thw, depth_batch, input_ids,
            rle_list=rle_list, mask_token_positions=mask_token_positions,
            max_new_tokens=max_new_tokens,
        )
        decoded = self.processor.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )
        return self.parse_output(decoded[0])


# Parameter counting util

def count_parameters(model: nn.Module) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


# VRAM monitoring util

def print_vram_usage(label: str = ""):
    """Print current GPU VRAM usage."""
    if not torch.cuda.is_available():
        print(f"  [!] VRAM [{label}]: CUDA not available")
        return
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved  = torch.cuda.memory_reserved()  / 1024**3
    max_alloc = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  [^] VRAM [{label}]: allocated={allocated:.2f}GB  reserved={reserved:.2f}GB  peak={max_alloc:.2f}GB")


# -----------------------------------------------------------------------------
# Standalone demo:   python model/pipeline.py
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import numpy as np
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--device",   default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype",    default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--attn-impl", default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"],
                        help="Attention implementation (default: flash_attention_2)")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    print("=" * 70)
    print("MODULE: SpatialVLM -- Inference Only")
    print("=" * 70)

    pipeline = SpatialVLM(
        model_name=MODEL_NAME,
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
        "Qwen Visual (encoder+merger)":   pipeline.qwen.model.visual,
        "Qwen Embeddings (tied->LM Head)": pipeline.qwen.model.language_model.embed_tokens,
        "Qwen Backbone (24 layers)":      pipeline.qwen.model.language_model.layers,
        "Qwen Final Norm":                pipeline.qwen.model.language_model.norm,
        "Qwen LM Head (tied->Embeddings)": pipeline.qwen.lm_head,
        "GSA (DFormerv2 Full_GSA x2)":    pipeline.gsa,
        "RTI (Region Token Injector)":    pipeline.region_token_extractor,
    }
    custom_names = {"GSA (DFormerv2 Full_GSA x2)", "RTI (Region Token Injector)"}
    tied_names   = {"Qwen LM Head (tied->Embeddings)"}  # skip from total (shared weights)

    total_custom, total_qwen = 0, 0
    for name, module in components.items():
        p = count_parameters(module)
        tag = "[*] CUSTOM" if name in custom_names else "    Qwen  "
        tied_note = "  <- shared, not counted" if name in tied_names else ""
        print(f"  {tag} {name:40s}: {p['total']:>12,} ({p['total']/1e6:.4f}M){tied_note}")
        if name in tied_names:
            continue  # don't double-count tied weights
        if name in custom_names:
            total_custom += p["total"]
        else:
            total_qwen += p["total"]

    print(f"\n  {'-'*70}")
    print(f"  Qwen 3.5 unique:  {total_qwen:>12,} ({total_qwen/1e6:.4f}M)")
    print(f"  Custom modules:   {total_custom:>12,} ({total_custom/1e6:.4f}M)")
    print(f"  Total unique:     {total_qwen + total_custom:>12,} ({(total_qwen + total_custom)/1e6:.4f}M)")

    # parse_output() demo
    print(f"\n{'='*70}")
    print("parse_output() DEMO")
    print(f"{'='*70}")
    examples = [
        'CATEGORY: distance | ANSWER: 5.2 | FREE_ANSWER: The distance between the two pallets is about 5 meters.',
        'CATEGORY: left_right | ANSWER: "left" | FREE_ANSWER: The object on the left appears closer.',
        'CATEGORY: count | ANSWER: 3 | FREE_ANSWER: There are 3 pallets in the buffer zone.',
        'CATEGORY: mcq | ANSWER: "2" | FREE_ANSWER: Pallet index 2 is optimal for the transporter.',
        'CATEGORY: distance | ANSWER: 3 | FREE_ANSWER: Type mismatch -- integer instead of float.',
        'Some unexpected output that does not match.',
    ]
    for ex in examples:
        parsed = SpatialVLM.parse_output(ex)
        print(f"  Input:    {ex[:70]}...")
        type_flag = "[OK]" if parsed['type_ok'] else "[FAIL]"
        print(f"  Parsed:   category={parsed['category']!r}  answer={parsed['answer']!r}  type_ok={type_flag}")
        print()

    # Forward pass test
    print(f"{'='*70}")
    print("FORWARD PASS TEST (FullHD 1920x1080)")
    print(f"{'='*70}")

    dev   = pipeline.device
    B     = 1
    H, W  = 1080, 1920

    dummy_image = Image.fromarray(
        np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    )
    dummy_depth = torch.rand(B, H, W, device=dev, dtype=dtype) * 10.0

    question = (
        "Using the buffer masks <mask> <mask> and pallet masks <mask> <mask>, "
        "how many pallets are in the buffer zone?"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": [
            {"type": "image", "image": dummy_image},
            {"type": "text",  "text": question},
        ]},
    ]
    text   = pipeline.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    inputs = pipeline.processor(
        text=[text], images=[dummy_image], return_tensors="pt", padding=True
    )

    pixel_values   = inputs["pixel_values"].to(device=dev, dtype=dtype)
    image_grid_thw = inputs["image_grid_thw"].to(device=dev)
    input_ids      = inputs["input_ids"].to(device=dev)

    # Dummy RLE: 200x400 rectangle (encoded via pycocotools)
    mask_np = np.zeros((H, W), dtype=np.uint8, order="F")
    mask_np[200:400, 300:700] = 1
    dummy_rle = mask_utils.encode(mask_np)
    dummy_rle["counts"] = dummy_rle["counts"].decode("utf-8")
    dummy_rle_list       = [dummy_rle] * 4   # 4 <mask> in question
    dummy_mask_positions = [5, 10, 15, 20]   # fake positions

    print(f"\n  pixel_values:   {list(pixel_values.shape)}")
    print(f"  image_grid_thw: {image_grid_thw.tolist()}")
    print(f"  depth_map:      {list(dummy_depth.shape)}")
    print(f"  input_ids:      {list(input_ids.shape)}")
    print(f"  rle_list:       {len(dummy_rle_list)} masks")

    # Reset peak stats to isolate inference-only peak
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        out = pipeline(
            pixel_values, image_grid_thw, dummy_depth, input_ids,
            rle_list=dummy_rle_list,
            mask_token_positions=dummy_mask_positions,
        )

    print(f"\n  logits: {list(out['logits'].shape)}")
    print_vram_usage("after forward pass")

    print(f"\n{'='*70}")
    print("  Pipeline OK: Qwen VE -> GSA -> RTI -> Concat -> Backbone -> LM Head")
    print(f"   Custom: GSA + RTI |  Output: structured text")
    print(f"{'='*70}")
    print_vram_usage("final summary")
