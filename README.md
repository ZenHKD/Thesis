# SpatialVLM

**Spatial Reasoning Vision-Language Model with Geometry Self-Attention and Region Token Injection**

> AI City Challenge 2025 Track 3 -- Spatial understanding in warehouse environments using RGB-D data.

## Architecture

Built on **Qwen 3.5 0.8B** (native VLM) with 2 custom modules:

```
RGB Image --> [Qwen Vision Encoder] --> [Merger] --> visual tokens [B, N, 1024]
                                                          |
Depth Map ------------------------------------------------+
                                                          v
                                                  [GSA] Geometry Self-Attention
                                                   (DFormerv2 Full_GSA x2 blocks)
                                                          |
                                                          v
RLE Masks ----> [RTI] Region Token Injection -----> token injection
                  (mask_rgb + mask_depth)                  |
                                                          v
Question -----> [Tokenizer] --> text embeds ------> [Concat Fusion]
                                                          |
                                                          v
                                                  [Qwen Backbone] 24 layers
                                                   (DeltaNet + GatedAttn)
                                                          |
                                                          v
                                                  [LM Head] --> Structured Output
```

### Custom Modules

| Module | File | Params | Description |
|--------|------|--------|-------------|
| **GSA** | `model/gsa.py` | ~16.9M | Depth-aware attention via geometry priors (RoPE + exponential decay) |
| **RTI** | `model/rti.py` | ~0.032M | Extracts region tokens from RLE masks via gated attention pooling |
| **Pipeline** | `model/pipeline.py` | -- | Full inference pipeline integrating Qwen + GSA + RTI |

### Parameter Budget

| Component | Params | Trainable (LoRA) |
|-----------|--------|------------------|
| Qwen Vision Encoder | 100.59M | ~2.4M (r=32) |
| Qwen Backbone (24 layers) | 498.11M | ~14.5M (r=64) |
| Qwen Embeddings (tied) | 254.28M | Frozen |
| GSA (2 blocks) | 16.9M | 16.9M (full) |
| RTI | 0.032M | 0.032M (full) |
| **Total** | **~870M** | **~34.1M** |

## Dataset

[nvidia/PhysicalAI-Spatial-Intelligence-Warehouse](https://huggingface.co/datasets/nvidia/PhysicalAI-Spatial-Intelligence-Warehouse)

**4 Task Categories**: `left_right`, `mcq`, `distance`, `count`

## Project Structure

```
Thesis/
├── model/
│   ├── pipeline.py          # Full SpatialVLM pipeline (inference)
│   ├── gsa.py               # Geometry Self-Attention (DFormerv2)
│   ├── rti.py               # Region Token Injection
│   ├── architecture.md      # Detailed architecture documentation (in progress)
│   └── qwen3.5-0.8b/        # Local model weights (gitignored)
├── src/
├── test/
│   └── test_inference.py    # Inference test with real samples
├── data/                    # Dataset directory (gitignored)
├── setup_nvidia_dataset.py  # Dataset download script
├── .env                     # HF_TOKEN (gitignored)
└── README.md
```

## Setup

### Prerequisites


### Installation

```bash
# Clone the repo
git clone git@github.com:ZenHKD/Thesis.git
cd Thesis

# Download Qwen 3.5 0.8B locally
huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir model/qwen3.5-0.8b

# Setup HF token for dataset access
echo "HF_TOKEN=hf_your_token_here" > .env

# Download the dataset
python setup_nvidia_dataset.py
```

### Quick Test

```bash
# Run inference on real samples (default: 5 samples)
python test/test_inference.py
```

## Output Format

The model produces structured text output:

```
CATEGORY: <left_right|mcq|distance|count> | ANSWER: <value> | FREE_ANSWER: <explanation>
```

## Training Strategy



## References

- **Qwen 3.5**: [Qwen Team](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- **DFormerv2**: [CVPR 2025](https://arxiv.org/abs/2504.04701) -- Geometry Self-Attention
- **DBNet++**: [TPAMI 2022](https://arxiv.org/abs/2202.10304) -- Differentiable Binarization (soft mask in RTI)
- **Gated Attention MIL**: [ICML 2018](https://arxiv.org/abs/1802.04712) -- Attention-based pooling
