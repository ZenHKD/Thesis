# SpatialVLM Architecture — Qwen 3.5 Micro

## Origin

Surgically pruned from **Qwen 3.5 0.8B (853M)** — a ~85% parameter reduction.
Pretrained weights are transferred (not random init), then fine-tuned on the 499K warehouse dataset.

| Cut | Original | Micro | Savings |
|-----|----------|-------|---------|
| Vision ViT blocks | 12 | **4** | 56.7M |
| Decoder layers | 24 (6 groups) | **8 (2 groups, LoopLM T_max=3)** | 332M |
| Vocab / Embedding | 248,320 | **248,321 (full + <num>, TRAINABLE)** | 0 |
| Context length | 262,144 | **512** | VRAM only |
| Number Head | — | **+0.26M** (NEW, softplus) | — |
| **Total** | **853M** | **~465M (~465M trainable)** | **~388M (45%)** |

> **Key**: `hidden_dim = 1024` is **unchanged** — all tensor shapes between modules stay identical.

---

## Dataset

| Split | QA Pairs | RGB-D pairs |
|-------|----------|-------------|
| Train | **499K** | ~78K |
| Test  | 19K | — |
| Val   | 1.9K | — |

**4 Task Categories:**
| Category | `normalized_answer` | Description |
|----------|---------------------|-------------|
| `left_right` | `"left"` or `"right"` | Spatial relation between 2 regions |
| `mcq` | `"0"`, `"1"`, ...  | Region index (which object to pick) |
| `distance` | `9.81` (float, meters) | Distance between 2 regions |
| `count` | `2` (int) | Count of objects in buffer zone |

> **Depth**: ~78K RGB-D pairs — real depth sensor data available \
> **Regions**: encoded as `<mask>` in question text, with per-region **RLE** in JSON \
> **Vocab**: Full original vocabulary (248,321 = 248,320 + <num> token) \
> **CoT**: GPT reasoning wrapped in `<think>...</think>` as chain-of-thought training signal \
> **Labels**: Question & answer tokenized separately then concatenated (BPE-safe boundary)

---

## Qwen 3.5 Micro Architecture

| Spec | Original 0.8B | **Micro** |
|------|-----------|-----------|
| Type | VLM (Vision + Language) | Same |
| LLM Hidden Dim | 1024 | **1024** (unchanged) |
| Vision Blocks | 12 | **4** |
| Vision Hidden Dim | 768 | **768** (unchanged) |
| Decoder Layers | 24 | **8 (LoopLM T_max=3 = effective depth 24)** |
| Layer Layout | 6 × (3 DeltaNet + 1 GatedAttn) | **2 × (3 DeltaNet + 1 GatedAttn) × T_max loops** |
| FFN (SwiGLU) dim | 3584 | **3584** (unchanged) |
| Vocab / Embedding | 248,320 | **248,321 (full + <num>)** |
| Context Length | 262,144 | **512** |
| **Number Head** | — | **Linear(1024->256->1)** |

### Parameter Breakdown

| Component | Params | Trainable | Detail |
|-----------|--------|-----------|--------|
| Vision Encoder | 44.10M | ✅ Yes | 4 ViT blocks (768-dim) + merger |
| Token Embeddings (tied w/ LM Head) | 254M | ✅ **Trainable** | 248,321 × 1024 (full vocab) |
| Text Decoder (8 layers × T_max=3) | ~166M | ✅ Yes | 2 × (3 DeltaNet + 1 GatedAttn), effective depth 24 |
| Number Head | 0.26M | ✅ Yes | xVal-style regression (softplus) |
| **Total (Qwen Micro)** | **~465M** | **~465M** | All trainable |
| GSA (2 blocks, custom) | +16.91M | ✅ Yes | Custom module |
| RTI (region tokens, custom) | +0.032M | ✅ Yes | Custom module |
| **Grand Total** | **~482M** | **~482M trainable** | |

### VRAM Estimate (BF16 training)

| | Original 0.8B | **Micro ~482M** |
|---|---|---|
| Model weights | ~1.7 GB | **~0.96 GB** |
| KV cache (inference) | ~2 GB | **~0.04 GB** |
| Gradients + optimizer | ~5 GB | **~2.9 GB** |
| Logits [B=8, L=80, V] | ~640 MB (248K) | **~640 MB (248K)** |
| Batch data headroom (12GB GPU) | ~3 GB | **~7.5 GB** |
| **Estimated batch size** | 1–2 | **6–10** |

---

## Pruning Strategy — Which Weights to Keep

### Vision: 12 -> 4 ViT Blocks

**Keep the LAST 4 blocks: [8, 9, 10, 11] -> renumber to [0, 1, 2, 3]**

Later ViT blocks encode higher-level semantic features (object identity, spatial layout).
Early blocks (0–7) do low-level edge/texture processing. With fine-tuning, the remaining
4 blocks can compensate. The merger (VL projector) is kept as-is.

```
Original:  block.0  block.1  block.2  ...  block.7  [block.8  block.9  block.10  block.11]
                                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Micro:                                               [block.0  block.1  block.2   block.3 ]
```

**Kept components:**
- `patch_embed` (Conv3D): unchanged — 1.18M
- `pos_embed`: unchanged — 1.77M
- `blocks.8->0, 9->1, 10->2, 11->3`: 4 × 7.09M = 28.36M
- `merger`: unchanged — 12.59M

### Decoder: 24 -> 8 Layers (LoopLM, T_max=3)

**Keep groups [0,1] -> 8 layers, loop T_max=3 at runtime = effective depth 24**

Based on "Scaling Latent Reasoning via Looped Language Models" (Ouro, arXiv:2510.25741).

- Group 0 (layers 0–3): **early** — question parsing, token-level understanding
- Group 1 (layers 4–7): **mid** — deeper reasoning, spatial relation processing
- LoopLM: iterative refinement with **fixed T_max=3 loops** (always full depth)

```
Original:  [Group 0]  [Group 1]  Group 2  Group 3  Group 4  Group 5
           layers 0-3  layers 4-7                            layers 20-23
              ↓
Micro:     [Group 0 + Group 1] × T_max = effective depth 24 (always 3 loops)
           layers 0-7  →  layers 0-7  →  layers 0-7
           loop 1         loop 2         loop 3
```

**Per group** = 3 × DeltaNet (~21.55M each) + 1 × GatedAttn (~18.35M) = ~83.0M
**2 groups × T_max=3** = ~166M params, effective depth 24 (same as original!)

**Layer type pattern**: `[linear, linear, linear, full, linear, linear, linear, full]`

**Training**: uniform-weighted CE loss across all loop steps: `L = (1/T) · Σ_t CE^(t)`.
All steps receive equal gradient, ensuring deeper loops are properly trained.
**Inference**: always runs full T_max=3 loops, uses final step's logits.

### Vocabulary: 248,320 -> 248,321 (Full Original + <num>)

**No vocabulary pruning.** The full original Qwen 3.5 vocabulary (248,320 tokens) is kept intact.
Only the `<num>` token is appended at the end (ID = 248,320) for numeric regression tasks.

This preserves the pretrained BPE tokenizer completely — all byte-level tokens and merge rules
are intact, ensuring correct encoding/decoding of all text.

**<num> token**: Placed at the **end of vocab** (ID = 248,320).
Random init, stored in config as `num_token_id`.

**TRAINABLE**: `embed_tokens.weight.requires_grad = True`. LM head is tied → also trainable.
With 248K tokens, embedding is ~254M params — the largest component of the model.



### Context: 262,144 -> 512

Config-only change. RoPE is computed dynamically — no weight modification.

At 320p input resolution with 2×2 spatial merge:
- Visual tokens: 160
- Question text: ~25–35 tokens
- `<think>` GPT reasoning: ~30–80 tokens
- Structured answer: ~8 tokens
- **Total: ~230–290 tokens** — fits in 512

---

## Pipeline Overview

```mermaid
flowchart TB
    subgraph INPUT["INPUT"]
        Q["Question Text\n(with <mask> placeholders)"]
        RGB["RGB Image (450p)"]
        D["Depth Map"]
        RLE["RLE Masks"]
    end

    subgraph VISION["Vision Encoder (44M)"]
        PE["patch_embed (Conv3D)\n-> [B, N, 768]"]
        POS["pos_embed (learned)"]
        VIT["4 ViT Blocks\n(768-dim, blocks 8-11 from original)"]
        MG["Merger (VL Projector)\n2×2 merge -> MLP(3072->1024)\n-> visual tokens [B, N/4, 1024]"]
        PE --> POS --> VIT --> MG
    end

    subgraph GSA["GSA: Geometry Self-Attention ×2 (16.9M)"]
        GP["GeoPriorGen\n(depth -> RoPE + decay)"]
        FGSA["Full_GSA + FFN\n(2 blocks)"]
        GP --> FGSA
    end

    subgraph RTI["RTI: Region Token Injection (0.032M)"]
        SOFT["RLE -> soft coverage mask"]
        MRGB["mask_rgb: Gated Attn Pool\n-> [B, 1024]"]
        MDEP["mask_depth: depth stats\n-> Linear(28->1024) -> [B, 1024]"]
        SOFT --> MRGB & MDEP
    end

    subgraph DECODER["Text Decoder (166M, LoopLM T_max=3)"] 
        EMB["Token Embeddings\n248K vocab x 1024\nTRAINABLE"]
        DEC["8 Layers x T_max=3 loops (LoopLM)\n2 x (3 DeltaNet + 1 GatedAttn)\nhidden=1024, FFN=3584"]
        EMB --> DEC
    end

    subgraph HEADS["Dual Output Heads"]
        LM["LM Head (tied w/ embed)\n-> category + text answer\n(count, mcq, left_right)"]
        NUM["Number Head (xVal)\nLinear(1024->256->1)\n-> continuous distance"]
    end

    RGB --> PE
    D --> GP
    D --> MDEP
    RLE --> SOFT
    MG --> FGSA
    FGSA --> MRGB
    Q --> EMB
    MRGB --> DEC
    MDEP --> DEC

    DEC --> LM
    DEC -->|"hidden @ <num>"| NUM

    style INPUT fill:#1a1a2e,stroke:#e94560,color:#fff
    style VISION fill:#16213e,stroke:#0f3460,color:#fff
    style GSA fill:#0f3460,stroke:#533483,color:#fff
    style RTI fill:#0d3320,stroke:#00b894,color:#fff
    style DECODER fill:#16213e,stroke:#3b82f6,color:#fff
    style HEADS fill:#533483,stroke:#a855f7,color:#fff
```

---

## Custom Modules

| Custom Module | File | Position | Function | Params |
|--------------|------|----------|----------|--------|
| **GSA** (2 blocks) | `gsa.py` | After Merger, before Decoder | Inject depth geometry into visual tokens | **~16.9M** |
| **RTI** | `rti.py` | After GSA, before Decoder | Decode RLE -> `[mask_rgb, mask_depth, space]` 3→3 injection | **~0.032M** |
| **Number Head** | `num_head.py` | After Decoder | xVal-style distance regression | **~0.26M** |

### GSA Detail — DFormerv2 Full_GSA (2 blocks × ~8.46M)

Operates on visual tokens [B, N, 1024] with depth prior.

| Sub-module | Params/block |
|------------|-------------|
| GeoPriorGen | ~0.000M |
| cnn_pos_encode | 0.010M |
| Full_GSA (Q/K/V/O + lepe) | 4.225M |
| FeedForwardNetwork | 4.218M |
| **× 2 blocks** | **~16.91M** |

### RTI Detail — Region-Level Token Injection (3→3)

Each `<mask>` (3 tokens: `<`, `mask`, `>`) → `[mask_rgb, mask_depth, space_embed]` (3 tokens).

**3→3 in-place replacement**: sequence length is **UNCHANGED**. No offset calculations,
no front-trimming of labels, no padding needed. This avoids alignment complexity.

| Sub-module | Params |
|------------|--------|
| RGB gate | 1,024 |
| Depth projector | 31,744 |
| **Total** | **~0.032M** |

| Token | Source |
|-------|--------|
| `mask_rgb` | Gated attention pooling over visual tokens [1, 1024] |
| `mask_depth` | Depth statistics → Linear(28→1024) [1, 1024] |
| `space_embed` | Frozen embedding of `" "` (space) token [1, 1024] |

### RTI Batching Strategy (enables batch_size > 1)

The RTI supports `batch_size > 1` via in-place per-sample replacement (no output length changes).

Implementation: `model_micro/rti.py` + `src/dataloader/dataloader_new.py`

| Aspect | Detail |
|--------|-------|
| `forward()` input | `rle_list: list[list[dict]]` (batched) |
| `inject_into_text_embeds()` | **3→3 in-place** (no length change) |
| `collate_fn()` | Pads to max length, nests masks as `list[list]` |
| Effective batch | **8–16** (true batching) |

### Number Head — xVal-style Regression

Implementation: `model_micro/num_head.py`

`LayerNorm(1024) -> Linear(1024, 256) -> GELU -> Linear(256, 1) -> softplus()`

Params: ~262K. Takes hidden state at `<num>` position, outputs non-negative scalar.
`softplus(x) = log(1 + exp(x))` — smooth everywhere, always positive (no gradient discontinuity).

**How it handles distance and count:**

- **Prediction**: Instead of generating multiple digit tokens (e.g., 4 tokens for `4.52`), a single `<num>` token outputs the scalar value.
- **Loss**: SmoothL1 loss provides a proportional, bounded gradient based on numerical error.
- **Evaluation**: Training MSE translates directly to evaluation RMSE.

---

## Output Format — Chain-of-Thought with `<think>`

The model generates GPT reasoning inside `<think>...</think>` before the structured answer.
This is chain-of-thought distillation: the model learns spatial reasoning from GPT explanations.

### Training target format

| Task | Training target |
|------|-----------------|
| mcq | `<think>The transporter [Region 1] is not transporting... pallet [Region 5] is closest.</think>mcq \| "5"` |
| distance | `<think>The pallet [Region 0] and pallet [Region 1] are 9.81 meters apart.</think>distance \| <num>` |
| count | `<think>The buffer region [Region 0] is filled with pallets [Region 3] [Region 9]... has 2 pallets.</think>count \| <num>` |
| left_right | `<think>The pallet [Region 0] is situated on the right of pallet [Region 1].</think>left_right \| "right"` |

### Dual output heads

| Head | What it learns | Active for |
|------|---------------|------------|
| **LM Head** | `<think>` reasoning + category + format | All tasks |
| **Number Head** | Scalar from hidden state @ <num> position | distance, count |

---

## Loss Function — Uniform Per-Step CE

Implementation: `model_micro/loss.py`

Based on "Scaling Latent Reasoning via Looped Language Models" (Ouro, arXiv:2510.25741).

`L = (1/T) · Σ_t CE^(t) + α · L_SmoothL1`

Where:
```
L = (1/T) · Σ_t CE^(t)(label_smoothing=0.1)  +  α · L_SmoothL1
```

- **CE^(t)**: Per-step cross-entropy at loop step t (LM head applied at every step)
- **T**: Number of loop steps (T_max = 4). All steps receive equal gradient weight.
- **L_SmoothL1**: SmoothL1 (Huber) on Number Head output vs ground truth (distance + count)
  - Bounded gradients (max 1.0 per sample) — eliminates gradient spikes from large targets
  - β=1.0: quadratic for |error| < 1, linear for |error| ≥ 1
- **No label trimming**: 3→3 RTI preserves sequence length — labels and logits always align

**Training regime**: Single-stage joint training. All loop steps are trained equally.

---

## Tokenization & Label Strategy (BPE-safe)

Question and answer are **tokenized separately**, then their token IDs are **concatenated**.
This guarantees exact label boundaries regardless of BPE context-dependent merging.

```
q_ids   = tokenizer.encode(question)          # [q1, q2, ..., qN]
sep_ids = tokenizer.encode("\n")              # [sep]
a_ids   = tokenizer.encode(full_answer)       # [a1, a2, ..., aM]

input_ids = q_ids + sep_ids + a_ids           # exact concatenation
labels    = [-100]*len(q_ids+sep_ids) + a_ids  # -100 for question, active for answer
```

**Why this matters**: BPE tokenizers are context-dependent. Tokenizing `question + "\n" + answer`
as one string can produce different token boundaries at the `\n` than tokenizing parts separately.
Separate tokenization + concatenation guarantees the label boundary is always exact.

---

## Training Strategy — Full Fine-tuning

**All components trainable**. With pruned ~300 vocab, the embedding
layer is negligible (~0.3M params), allowing nearly all capacity for the decoder.

| Component | Status | LR |
|-----------|--------|-----|
| Vision Encoder (4 ViT blocks) | ✅ **Trainable** | 5e-5 |
| Token Embeddings (248K, tied w/ LM Head) | ✅ **Trainable** | 1e-5 |
| Text Decoder (8 layers × T_max=3) | ✅ **Trainable** | 1e-5 |
| GSA (2 blocks) | ✅ **Trainable** | 5e-5 |
| RTI | ✅ **Trainable** | 5e-5 |
| Number Head | ✅ **Trainable** | 5e-4 |

**Loss**: `L = (1/T) · Σ_t CE^(t)(label_smoothing=0.1) + α·L_SmoothL1` (α=0.1)

**Optimizer**: AdamW with cosine LR scheduler, warmup steps = 500

---

## Visual Token Count at Different Resolutions

| Input | ViT patches | Post-merger tokens | Fits 512 ctx? |
|-------|------------|-------------------|----------------|
| **512×320 (320p)** | **32×20 = 640** | **160** | ✅ **~260 total (default)** |
| 640×384 (384p) | 40×24 = 960 | 240 | ✅ ~340 total |
| 800×448 (450p) | 50×28 = 1400 | 350 | ✅ ~450 total |
| 960×544 (540p) | 60×34 = 2040 | 510 | ✅ ~610 total |
| 1280×720 (720p) | 80×44 = 3520 | 880 | ⚠️ ~980 total |

> Default training resolution: **320p (512×320)** — 160 visual tokens.
> Total sequence (visual + CoT text): ~230–290 tokens. Fits in 512.
