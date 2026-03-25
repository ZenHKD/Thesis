# SpatialVLM Architecture — AI City Challenge 2025 Track 3

## Dataset 

| Split | QA Pairs | RGB-D pairs |
|-------|----------|-------------|
| Train | **499k** | ~95k |
| Test  | 19k | — |
| Val   | 1.9k | — |

**4 Task Categories:**
| Category | `normalized_answer` | Description |
|----------|---------------------|-------------|
| `left_right` | `"left"` or `"right"` | Spatial relation between 2 regions |
| `mcq` | `"0"`, `"1"`, ...  | Region index (which object to pick) |
| `distance` | `9.81` (float, meters) | Distance between 2 regions |
| `count` | `2` (int) | Count of objects in buffer zone |

> **Depth**: ~95k RGB-D pairs — real depth sensor data available \
> **Regions**: encoded as `<mask>` in question text, with per-region **RLE** in JSON

---

> **Focus**: Simplicity & Efficiency & End-to-End — **Single structured LM output, zero mismatch** \
> **Backbone**: Qwen 3.5 0.8B — Native VLM (built-in Vision Encoder)

---

## Qwen 3.5 0.8B Architecture

| Spec | Value |
|------|-------|
| Type | Causal LM **with Vision Encoder** |
| LLM Hidden Dim | 1024 |
| Layers | 24 |
| Layer Layout | 6 × (3 × DeltaNet-FFN → 1 × GatedAttn-FFN) |
| DeltaNet | `linear_attn`: QKV + gate z + delta params (a,b) + conv1d + A_log |
| GatedAttn | `self_attn`: 8Q+2KV heads, dim=256, RoPE, QK-norm |
| FFN (all layers) | **SwiGLU**: gate_proj + up_proj + down_proj, dim=3584 |
| Vocab / Embedding | 248,320 (tied with LM Head) |
| Context Length | 262,144 |

#### Parameter Breakdown (verified)

| Component | Params | % | Detail |
|-----------|--------|---|--------|
| Vision Encoder | 100.59M | 11.79% | 12 ViT blocks (768-dim) + **merger** (VL projector) |
| Token Embeddings (tied w/ LM Head) | 254.28M | 29.81% | 248,320 × 1024 |
| Text Decoder (24 layers + Norm) | 498.11M | 58.40% | 18 DeltaNet + 6 GatedAttn layers |
| **Total** | **852.99M** | 100% | |

#### Vision Encoder Flow (verified)

```
Image [B, 3, H, W]  — Dataset: 1920×1080 FullHD
  → patch_embed (Conv3D) → [B, N, 768]        (N patches, 16px each)
  → pos_embed (learned)
  → 12 ViT blocks (Attn + GELU MLP, 768-dim)
  → merger (VL Projector):
      LayerNorm(768)
      → spatial merge 2×2 patches  → [B, N/4, 4×768=3072]
      → MLP(3072→3072, GELU, 3072→1024)
  → visual tokens [B, N/4, 1024]
```

**Dynamic resolution examples (from image_grid_thw):**

| Input | ViT patches | Post-merger tokens | Grid layout |
|-------|------------|-------------------|-------------|
| 448×448 (square test) | 28×28 = 784 | 196 | 14×14 (square) |
| 1920×1080 (FullHD dataset) | ~60×34 = ~2040 | ~510 | ~30×17 (rectangular) |

> **Key**: Grid is **rectangular** for FullHD. `isqrt(N)` crashes — use `image_grid_thw` from processor. \
> `model.visual()` returns **pre-merger** (768-dim). Merger is called manually.

---

## Pipeline Overview

### Main Pipeline

```mermaid
flowchart TB
    subgraph INPUT["📥 INPUT"]
        Q["Spatial Question\n(contains mask placeholders)"]
        RGB["RGB Image"]
        D["Depth Map\n(from RGB-D sensor)"]
        RLE["RLE Masks\n(per mask in question)"]
    end

    subgraph QWEN_VE["Qwen 3.5 Vision Encoder"]
        VE["12 ViT Blocks\nhidden=768, GELU MLP"]
        MG["Merger (VL Projector)\n2×2 merge → 3072→1024\n784→196 tokens"]
        VE --> MG
    end

    subgraph GSA["GSA: Geometry Self-Attention ×2 blocks — DFormerv2 Full_GSA (CUSTOM)"]
        GP1["Block 1: GeoPriorGen\n(depth only → RoPE sin/cos + decay mask)"]
        FGSA1["Block 1: CNN_pos → Full_GSA + FFN\n(geo prior as side-input)"]
        GP2["Block 2: GeoPriorGen\n(depth only → RoPE sin/cos + decay mask)"]
        FGSA2["Block 2: CNN_pos → Full_GSA + FFN\n(geo prior as side-input)"]
        GSA_OUT["Depth-aware Tokens [B, N, 1024]"]
        FGSA1 --> FGSA2 --> GSA_OUT
        GP1 --> FGSA1
        GP2 --> FGSA2
    end

    subgraph RTI["RTI: Region-Level Token Injection (CUSTOM)"]
        RTI_IN["RLE → soft coverage mask (DB-style σ)"]
        RGB_TOK["mask_rgb: Gated Attention Pool\n(log-domain soft mask + rgb_gate Linear→softmax)\n→ [B, 1024]"]
        DEP_TOK["mask_depth: [mean_d, std_d, cx, cy, 24-ray radial]\n→ Linear(28→1024) → [B, 1024]"]
        RTI_OUT["Replace mask → mask_rgb + mask_depth\nin text embedding sequence"]
        RTI_IN --> RGB_TOK & DEP_TOK --> RTI_OUT
    end

    subgraph QWEN_TOK["Qwen 3.5 Tokenizer"]
        TOK["Question → Text Embeds [B, L', 1024]\n(L' = L + num_masks, injected region tokens)"]
    end

    subgraph FUSION["Concat Fusion"]
        CONCAT["[IMG] v1..vN [QUE] ...q_i [mask_rgb_i][mask_depth_i]... [ANS]"]
    end

    subgraph BACKBONE["Qwen 3.5 Backbone (24 layers)"]
        LAYERS["6 × (3 × DeltaNet-FFN\n→ 1 × GatedAttn-FFN)\n+ LoRA rank=32"]
    end

    subgraph OUT["Output — LM Head (Qwen built-in)"]
        ANS["Structured Text Output\nCATEGORY: distance | ANSWER: 5.2 | FREE_ANSWER: ...\n↓ regex parse → category + normalized_answer"]
    end

    RGB --> VE
    MG --> FGSA1
    D --> GP1
    D --> GP2
    RLE --> RTI_IN
    GSA_OUT --> RTI_IN
    GSA_OUT --> RGB_TOK
    D --> DEP_TOK
    Q --> TOK
    RTI_OUT --> TOK

    GSA_OUT --> CONCAT
    TOK --> CONCAT

    CONCAT --> BACKBONE
    LAYERS --> ANS

    style INPUT fill:#1a1a2e,stroke:#e94560,color:#fff
    style QWEN_VE fill:#16213e,stroke:#0f3460,color:#fff
    style GSA fill:#0f3460,stroke:#533483,color:#fff
    style RTI fill:#0d3320,stroke:#00b894,color:#fff
    style QWEN_TOK fill:#16213e,stroke:#0f3460,color:#fff
    style FUSION fill:#1a1a2e,stroke:#e94560,color:#fff
    style BACKBONE fill:#e94560,stroke:#fff,color:#fff
    style OUT fill:#533483,stroke:#00b894,color:#fff
```

## Custom Modules (Implementation Files)

Qwen 3.5 provides the full VLM pipeline out-of-the-box. We add only **2 custom modules**:

| Custom Module | File | Position | Function | Params |
|--------------|------|----------|----------|--------|
| **GSA** (2 blocks) | `gsa.py` | After Merger, before Concat | Inject depth geometry into visual tokens | **~16.9M** |
| **RTI** | `region_token.py` | After GSA, before Backbone | Decode RLE → `<mask_rgb><mask_depth>` token injection | **~0.032M** |
| **Full Pipeline** | `pipeline.py` | — | Integrates Qwen 3.5 + all custom modules | — |

---

### GSA Detail — DFormerv2 Full_GSA (CVPR 2025, 2 blocks × 8.46M)

| Sub-module | Params/block | Notes |
|------------|-------------|-------|
| `GeoPriorGen` | ~0.000M (2 params) | RoPE + exponential depth/pos decay, learnable blend weight |
| `cnn_pos_encode` DWConv2d(1024, 3, 1, 1) | 0.010M | Local spatial context before attention |
| `norm1` LayerNorm(1024) | 0.002M | Pre-attention norm |
| `Full_GSA` (Q/K/V/O + lepe DWConv5×5) | 4.225M | Full N×N attention + geometry prior + RoPE |
| `norm2` LayerNorm(1024) | 0.002M | Pre-FFN norm |
| `FeedForwardNetwork` (FC→GELU→DWConv→FC) | 4.218M | FFN with subconv local context |
| **Per block** | **~8.457M** | |
| **× 2 blocks** | **~16.91M** | Verified ✅ |

> **FullHD example**: 1920×1080 → merger → h_vis=18, w_vis=30 → N=540 tokens (non-square ✅) \
> `image_grid_thw` always passed explicitly — never use `isqrt(N)` on non-square grids.

---

### RTI Detail (Region-Level Token Injection)

**Problem**: `<mask>` in the question is an empty token — the model has no information about where that region is in the image.

**Solution**: Each `<mask>` → 2 consecutive tokens `<mask_rgb><mask_depth>`:

```
<image> From this viewpoint, does the pallet <mask> appear on the right-hand side of the pallet <mask>?
→
<image> From this viewpoint, does the pallet <mask_rgb><mask_depth> appear on the right-hand side of the pallet <mask_rgb><mask_depth>?
```

#### RTI Data Flow

```
RLE dict  ("size":[1080,1920], "counts":"...")
    ↓
decode → binary mask [1080, 1920]
    ↓
resize → patch coverage [h_vis, w_vis]  (adaptive_avg_pool → float [0,1])
    ↓ DB-style soft binarization
soft_mask = σ(K × (coverage − θ))  K=50, θ=0.3  [h_vis, w_vis] ∈(0,1)
         ↙                                  ↘
 visual_tokens (post-GSA)              depth_map [B, H, W]
 [B, N, 1024]                               ↓
     ↓                            pixel-level masked vals → [B, M]
 Gated Attention Pool (soft)                 ↓
   score_i = Linear(token_i)      mean_d, std_d, cx_soft, cy_soft → [B, 4]
   log_soft_i = log(soft_mask_i)  8-bin soft histogram            → [B, 8]
   weights = softmax(score+log_soft)  concat → stats [B, 12]
   mask_rgb = Σ wᵢ · tokenᵢ           ↓
     ↓                        Linear(10→1024) + LayerNorm
 [B, 1024]                         [B, 1024]
 <mask_rgb>                        <mask_depth>
         ↘                         ↙
   inject into text embedding sequence at <mask> positions
```

#### RTI Sub-modules

| Sub-module | Params | Details |
|------------|--------|---------|
| RGB gate (Gated Attention Pool) | `Linear(1024, 1, bias=False)` = 1,024 | Learned importance score; log-domain DB-style soft mask |
| Depth projector | `Linear(28, 1024) + LayerNorm` = 31,744 | [mean_d, std_d, cx_soft, cy_soft, r0..r23] → 1024-dim depth token |
| **Total RTI** | **~0.032M** (32,768 params) | |

#### RTI — mask_rgb: DB-style Soft Masking

Instead of hard binarization (`coverage ≥ 0.5`), we use **differentiable binarization** (as in DBNet++):

```
soft_mask = σ(K × (coverage − θ)),   K=50, θ=0.3
```

Gated Attention Pool with soft mask:
```
score_i    = Linear(token_i)           [B, N]
log_soft_i = log(σ(K(c_i − θ)))        [N]   — from patch coverage fraction
weights    = softmax(score + log_soft)  [B, N]
mask_rgb   = Σ weights_i × token_i     [B, 1024]
```

#### RTI — mask_depth:

```
Stats: [mean_d, std_d, cx_soft, cy_soft, r0..r23]  — 28 dims, all ✅ gradient
cx_soft = Σᵢ(col_i × soft_i) / Σᵢ(soft_i)  — soft-weighted centroid X
24-ray radial depth profile (F.grid_sample bilinear, differentiable):
  r_k = soft-weighted avg depth along ray θ_k = k×15°, k=0..23
  Cast from (cx,cy), weighting by soft_mask — distinguishes overlapping masks
```

| Stat | Gradient | Meaning |
|------|----------|---------|
| `mean_d` | ✅ Non-zero | Average depth of the region |
| `std_d` | ✅ Non-zero | Depth uniformity within the region |
| `cx_soft` | ✅ Non-zero | Centroid X (soft-weighted from DB soft_mask) |
| `cy_soft` | ✅ Non-zero | Centroid Y (soft-weighted from DB soft_mask) |
| `r0..r23` (24 rays, k×15°) | ✅ Non-zero | Radial depth profile — distinguishes overlapping masks & complex shapes |

### LM Head — Structured Output Format

Instead of separate custom heads, the **LM Head (Qwen built-in)** is fine-tuned to produce structured text output directly:

```
CATEGORY: distance | ANSWER: 5.2 | FREE_ANSWER: The pallet on the left is closer to the camera.
```

| Field | Meaning | Example |
|-------|---------|---------|
| `FREE_ANSWER` | Natural language explanation (free-form) | `"The pallet appears on the left side"` |
| `CATEGORY` | Task type | `distance` / `count` / `left_right` / `mcq` |
| `ANSWER` | Exact `normalized_answer` | `5.2`, `left`, `3`, `0` |

#### Target string during training (preprocessed from JSON):

```python
target = f"CATEGORY: {category} | ANSWER: {normalized_answer} | FREE_ANSWER: {free_answer}"
# Examples:
# "CATEGORY: distance | ANSWER: 5.2 | FREE_ANSWER: The distance between the two pallets is about 5 meters."
# "CATEGORY: left_right | ANSWER: \"left\" | FREE_ANSWER: The object on the left appears closer."
# "CATEGORY: count | ANSWER: 3 | FREE_ANSWER: There are 3 objects in the buffer zone."
# "CATEGORY: mcq | ANSWER: \"2\" | FREE_ANSWER: Pallet index 2 is optimal for the transporter."
```

> **Why this order?** (Approach 1)
> Committing to `CATEGORY` and `ANSWER` **before** `FREE_ANSWER` forces the model to
> reason: *"I know the answer is 5.2 — now I explain why."* This eliminates the coherence bug
> where FREE_ANSWER says "right side" but ANSWER is forced to `"left"` by the FSM.

#### Format Enforcement — System Prompt

A **fixed system prompt** is prepended to every input so that Qwen always follows the format:

```python
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
```

Prepend system prompt to every input during **both training and inference** (using Qwen chat template):

```python
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user",   "content": question_with_image},
]
# Qwen tokenizer handles <|im_start|>system ... <|im_end|> automatically
input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")
```

> ✅ Same format at train and inference → model learns the correct distribution \
> ✅ System prompt adds no parameters — it is only a prefix token sequence

#### Format Enforcement — Constrained Decoding

At inference, use **prefix-forcing** to hard-constrain LM output to the correct format:

**Step 1 — Force prefix `"CATEGORY: "`** immediately after `<|im_start|>assistant`:

```python
# Force output to begin with "CATEGORY: "
forced_prefix = tokenizer.encode("CATEGORY: ", add_special_tokens=False)
# Use `force_words_ids` or `prefix_allowed_tokens_fn` in model.generate()
```

**Step 2 — Regex FSM (Finite State Machine)** constrains the entire output:

```python
from outlines import models, generate  # or use lm-format-enforcer

# Approach 1: CATEGORY first → type-committed ANSWER → FREE_ANSWER (coherent, unconstrained)
# FREE_ANSWER uses .+ safely — it is the last field, nothing follows it.
PATTERN = (
    r"CATEGORY: (left_right|mcq|distance|count) \| "
    r'ANSWER: ("left"|"right"|"[0-9]+"|[0-9]+\.[0-9]+|[0-9]+) \| '
    #          ↑ float before int — order matters for FSM disambiguation
    r"FREE_ANSWER: .+"
)

# Approach 2: post-parse cross-field type validation (in parse_output())
# Regex alone cannot enforce CATEGORY↔ANSWER consistency; validate after match:
_ANSWER_TYPE = {
    "left_right": lambda a: a.strip('"') in ("left", "right"),
    "mcq":        lambda a: a.startswith('"') and a.strip('"').isdigit(),
    "distance":   lambda a: '.' in a and not a.startswith('"'),
    "count":      lambda a: a.isdigit(),
}
# type_ok = _ANSWER_TYPE[category](answer)  →  False = format inconsistency detected

# With lm-format-enforcer (compatible with HuggingFace):
from lm_format_enforcer import RegexParser
from lm_format_enforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

parser = RegexParser(PATTERN)
prefix_fn = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)

output = model.generate(
    input_ids,
    prefix_allowed_tokens_fn=prefix_fn,
    max_new_tokens=100,
)
```

| Mechanism | Effect | Library |
|-----------|--------|---------|
| **System Prompt** | Model learns format during training → reduces hallucination | Qwen chat template |
| **Prefix Forcing** | Forces `CATEGORY:` from the very first token | `force_words_ids` or manual logit masking |
| **Regex FSM** | Blocks any token that violates the pattern at each decode step | `lm-format-enforcer` or `outlines` |
| **Post-parse Type Check** | Catches CATEGORY↔ANSWER mismatch that FSM cannot prevent | `_ANSWER_TYPE` dict in `parse_output()` |

> ✅ Approach 1: `CATEGORY → ANSWER → FREE_ANSWER` — model commits to answer before explaining; FREE_ANSWER is always coherent \
> ✅ Approach 2 (inference): `_ANSWER_TYPE` post-parse validation catches residual type mismatches \
> ✅ Approach 2 (training): `type_ok` flag logged per sample → used to monitor format consistency during training \
> ✅ `lm-format-enforcer` is lightweight, requires no model changes \
> ✅ FREE_ANSWER uses `.+` safely — it is the **last** field, no greedy `|` conflict

---

### Training Loss

```
L_total = L_lm + λ_fmt · L_fmt
```

**Two losses** — both are autoregressive CrossEntropy-based, no separate heads:

| Loss | Formula | Notes |
|------|---------|-------|
| `L_lm` | CrossEntropy(lm_logits, target_tokens) | Target = full structured string over entire sequence |
| `L_fmt` | `0` if `type_ok`, else `1` (binary format penalty) | Approach 2: penalizes CATEGORY↔ANSWER type mismatch |

- `λ_fmt ≈ 0.1–0.3` — small auxiliary weight, does not distort main LM objective
- **`type_ok`** evaluated by `_ANSWER_TYPE[category](answer)` after each forward pass
- During **Phase 1** (Qwen frozen): `L_fmt` helps GSA+RTI produce representations that encourage correct ANSWER types
- During **Phase 2** (LoRA): `L_fmt` signal propagates through LoRA adapters → backbone learns type-safe generation

> No `L_classifier`, no MSE, no EDL — structure and type safety come from output format + soft penalty

---

## Training Strategy (2 Phases)

### Phase 1: GSA + RTI Warmup (3 epochs)

| Component | Status | LR |
|-----------|--------|-----|
| Qwen 3.5 (vision + backbone) | ❄️ Frozen | — |
| GSA (2 blocks) | ✅ Trainable | 1e-4 |
| RTI | ✅ Trainable | 1e-4 |

**Loss**: `L = L_lm + λ_fmt · L_fmt` — CrossEntropy on structured target + format consistency penalty \
**Goal**: Teach GSA/RTI to produce geometry-aware representations; LM Head learns CATEGORY→ANSWER→FREE_ANSWER format on frozen Qwen backbone. `L_fmt` starts applying from Phase 1 to encourage type-safe ANSWER generation early.

### Phase 2: Full Fine-tuning (5 epochs)

| Component | Status | LR |
|-----------|--------|-----|
| Qwen 3.5 Vision Encoder | LoRA (rank=32) | 5e-5 |
| Qwen 3.5 Backbone | LoRA (rank=64) | 2e-5 |
| GSA + RTI | ✅ Trainable | 5e-5 |

**Loss**: `L = L_lm + λ_fmt · L_fmt` — same formula as Phase 1; LoRA adapters receive both LM and format-consistency gradients → backbone fine-tunes toward type-safe spatial reasoning.

---

## Parameter Budget (Simplified)

| Component | Params | LoRA Rank | Trainable | QLoRA size (train) |
|-----------|--------|-----------|-----------|---------------------|
| Qwen 3.5 Vision Encoder (12 ViT blocks) | 100.59M | r=32 (Q,K,V,O) | ~2.4M | NF4 (~50MB) |
| Qwen 3.5 Vision Merger | (in above) | r=32 | ~0.3M | NF4 (in above) |
| Qwen 3.5 Token Embeddings (tied w/ LM Head) | 254.28M | frozen | 0 | NF4 (~63MB) |
| Qwen 3.5 Text Decoder (24 layers) | 498.11M | r=64 (Q,K,V,O,Gate,Up,Down) | ~14.5M | NF4 (~125MB) |
| GSA (2 blocks, Full_GSA DFormerv2) | **~16.9M** | — (full train) | 16.9M | bfloat16 (34MB) |
| RTI (Region Token Injector) | **~0.032M** | — (full train) | 0.032M | bfloat16 (~0.06MB) |
| **Total** | **~870M** | | **~34.1M** | **~272MB VRAM** |

> **NF4** = 4-bit NormalFloat (QLoRA default) — `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")` \
> Applied **uniformly to all linear layers** of Qwen (vision + backbone), without distinction. \
> GSA + RTI are **not quantized** — trained at full precision bfloat16.

> ✅ **~34M trainable** (LoRA ~17.2M + GSA ~16.9M + RTI ~0.009M) → RTX 3060 feasible \
> ✅ **~271MB VRAM** during training with QLoRA (INT4 Qwen base + bfloat16 GSA/RTI/LoRA adapters) \
> ✅ **2 custom modules only** (GSA + RTI) — LM Head handles all output, **zero mismatch**

---

## Quantization Strategy

### QLoRA (used during training — Phase 1 & 2)

```
Base Qwen loaded NF4 (BitsAndBytes)    ← saves VRAM during training
LoRA adapters                          ← trained in bfloat16 (A, B matrices)
GSA + RTI                              ← trained in bfloat16
```

**Output after Phase 2 — unmerged**, because they cannot be added directly:
```
W_merged = W_base(NF4) + B·A(bfloat16)   ← type mismatch, invalid
```
The runtime forward pass dequantizes temporarily (`NF4 → bfloat16`) on-the-fly — not saved back.
Only disk storage: `adapter_model.bin` (LoRA A/B, bfloat16) + pointer to Qwen Hub.

### Phase 3 (Post-Training Quantization)

The base model must be reloaded at full precision before merging:

```
Phase 2 output:  NF4 base (Qwen Hub) + LoRA adapters (bfloat16, ~68MB)
    ↓  reload base at bfloat16 + merge_and_unload()
Merged bfloat16 model   (~1.7GB)
    ↓  GPTQ / AWQ / GGUF (llama.cpp)
Quantized for deploy  (~500MB INT4)
```

| Phase | Purpose | Tool |
|-------|---------|------|
| Phase 1–2 train | QLoRA — fit VRAM | `bitsandbytes` (NF4 base + bfloat16 adapters) |
| Phase 3 (optional) | Merge + re-quantize for deployment | `AutoAWQ` / `GPTQ` / `llama.cpp` |

> **Accuracy risk on small models (0.8B):** Quantizing to INT4 on a small model may reduce accuracy,
> especially for the `distance` task (float output) and `count` task (requires precise counting) — fine weight details can be lost.
> **Recommendation:** After merging, test the bfloat16 version (~1.7GB) first. Only go to INT4 if hardware strictly requires it.

> **Custom module compatibility:** GPTQ/AWQ only scan standard `nn.Linear` — GSA has `DWConv2d`, `GeoPriorGen` (non-linear)
> which may fail conversion. **Quantize only the Qwen backbone, exclude GSA + RTI:**
> ```python
> # AutoAWQ example — exclude custom module prefixes
> quantize_config = {"modules_to_not_convert": ["gsa", "region_token_extractor"]}
> ```
> GSA + RTI remain in bfloat16 (~35MB) — negligible compared to the total ~500MB.

> **Calibration data for AWQ/GPTQ:** Requires 128–512 samples to measure activation distribution.
> Using plain text (C4/WikiText) will **not reflect** the activation distribution of GSA/RTI.
> Must use samples from the **AI City Challenge dataset** (with images + depth maps):
> ```python
> # Calibration sample: forward pass with real image + depth + question
> calib_samples = load_aicity_samples(n=256, split="train")  # RGB-D + question
> ```
> → AWQ/GPTQ will calibrate correctly to the activation distribution of the fine-tuned model.
