# SpatialVLM

**Spatial Reasoning Vision-Language Model with Geometry Self-Attention and Region Token Injection**

> AI City Challenge 2025 Track 3 -- Spatial understanding in warehouse environments using RGB-D data.

## Architecture 

### Original model, see `model/`

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
                  (mask_rgb + mask_depth)                 |
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

### New Micro model, see `model_micro/`

Pruned from **Qwen 3.5 0.8B** (853M): Vision Encoder (12 → 4 blocks), Backbone (full 24 layers, **single-pass** — no looping). **Full original vocabulary** (248,076 + `<num>` = 248,077 tokens -> remain 248,320 tokens (padded)). Adds **Number Head** for direct numeric regression on `distance` and `count` tasks. Two parallel input streams: visual (Vision Encoder → 160 tokens) and region (RTI, independent of Vision Encoder → 3 learned tokens per `<mask>`). Full fine-tuning (~797M trainable parameters).

```
RGB Image -----> [Vision Encoder] --> [Merger] --> visual tokens [B, 160, 1024]
                  (4 ViT blocks)                         |
                                                         |
                +----------------------------------------+
                |                                        |
                v                                        |
RGB Image --> [RTI: Region Token Injection]              |
Depth Map  --> (Independent of Vision Encoder)           |
RLE Masks  --> (mask_rgb + mask_depth + mask_geo)        |
                |  3 learned tokens per <mask>           |
                v                                        v
Question --> [Embed] (Full 248,321 vocab) --> [Inject] --> [Concat Fusion]
                                                                |
                                                                v
                                               [Qwen Backbone] 24 layers (single pass)
                                               (6 × [3 DeltaNet + 1 GatedAttn])
                                                                |
                                              +-----------------+-----------------+
                                              v                                   v
                                          [LM Head]                       [Number Head] (xVal)
                                              |                                   |
                                              v                                   v
                                       Structured Output                  Numeric Prediction
                                  <think>reasoning</think>              distance (m) / count (n)
                                       category | answer
                                              |                                   |
                                              v                                   v
                                    L_CE (label_smooth=0.1)             L_SmoothL1 (regression)
                                              |                                   |
                                              +----------- L_total ---------------+
                                                    L = L_CE + α·L_SmoothL1 (α=0.1)
```

**Key differences from original model:**
| Component | Original | Micro |
|-----------|----------|----------|
| Vision Encoder | 12 ViT blocks | **4 ViT blocks** (last 4 kept) |
| Decoder | 24 layers | **24 layers, single pass** (no loop) |
| GSA | DFormerv2 ×2 blocks | **Removed** |
| RTI tokens | mask_rgb + mask_depth | **mask_rgb + mask_depth + mask_geo** |
| RTI coupling | Depends on Vision Encoder | **Independent (raw RGB+Depth+RLE)** |
| Output format | Direct answer | **`<think>CoT</think>` + structured answer** |
| Trainable params | ~482M | **~797M** |


## Dataset

[nvidia/PhysicalAI-Spatial-Intelligence-Warehouse](https://huggingface.co/datasets/nvidia/PhysicalAI-Spatial-Intelligence-Warehouse)

**4 Task Categories**: `left_right`, `mcq`, `distance`, `count`

| Split | QA Pairs | RGB-D pairs |
|-------|----------|-------------|
| Train | **499K** | ~78K |
| Val   | 1.9K | — |
| Test  | 19K | — |

## Project Structure

```
Thesis/
├── model/
│   ├── pipeline.py                 # Full SpatialVLM pipeline
│   ├── gsa.py                      # Geometry Self-Attention (DFormerv2)
│   ├── rti.py                      # Region Token Injection (original, batch_size=1)
│   ├── architecture.md             # Detailed architecture documentation
│   └── qwen3.5-0.8b/               # Local model weights (gitignored)
├── model_micro/
│   ├── pipeline.py                 # Micro pipeline (single-pass, no GSA/LoopLM)
│   ├── rti.py                      # RTI (independent of Vision Encoder, batched)
│   ├── num_head.py                 # Number Head (xVal-style softplus regression)
│   ├── loss.py                     # Combined CE + SmoothL1 loss
│   ├── prune.py                    # Pruning script (Qwen 3.5 0.8B → Micro)
│   ├── architecture_micro.md       # Detailed Micro architecture documentation
│   └── qwen3.5-micro/              # Micro model weights (gitignored)
├── notebooks/
│   ├── 00_EDA.ipynb                # Exploratory Data Analysis
│   ├── 01_RLE_Mask.ipynb           # Mask Analysis
│   ├── 02_Error_Image.ipynb        # Error Image Analysis
│   └── 03_Mask_Features.ipynb      # RTI feature visualization (RGB/Depth/Geo)
├── analysis/                       # Question type analysis (train + val + test)
├── src/
│   ├── dataloader/                 # Dataset loader (batched RTI, decoded masks)
│   ├── train_micro/                # Micro training + validation (train.py, val.py)
│   ├── train_phase1/               # Phase 1 training (original model)
│   └── train_phase2/               # Phase 2 training (original model)
├── test/
│   ├── test_inference.py           # Inference test (original model)
│   ├── test_backprop_1.py          # Backprop test phase 1 (original)
│   ├── test_backprop_2.py          # Backprop test phase 2 (original)
│   ├── test_dataloader.py          # Dataloader test (old RTI, batch_size=1)
│   └── test_pipeline_alignment.py  # Pipeline alignment test (original model)
├── test_micro/
│   ├── test_inference.py           # Inference test (Micro, cuda only)
│   ├── test_backprop.py            # Backprop test (Micro, all components)
│   ├── test_dataloader_new.py      # Dataloader test (batched RTI, decoded masks)
│   └── test_pipeline_alignment.py  # Pipeline alignment test (Micro)
├── checkpoints/micro/              # Training checkpoints + training.csv (gitignored)
├── data/nvidia_warehouse_dataset/  # Dataset directory (gitignored)
├── count_qwen3_5_params.py         # Parameter counting script
├── Qwen3.5-0.8B.txt                # Parameter breakdown output
├── setup_nvidia_dataset.py         # Dataset download script
├── .env                            # HF_TOKEN (gitignored)
└── README.md
```

## Setup

### Prerequisites


### Installation

```bash
# Clone the repo
git clone https://github.com/ZenHKD/Thesis.git
cd Thesis

# Download Qwen 3.5 0.8B locally
hf download Qwen/Qwen3.5-0.8B --local-dir model/qwen3.5-0.8b

# Setup HF token for dataset access
echo "HF_TOKEN=hf_your_token_here" > .env

# Download the dataset
python setup_nvidia_dataset.py
```

## Output Format

The model produces structured text output:

```
<left_right|mcq|distance|count> | <value>
```

## References

- **SmolRGPT**: [arXiv 2509.15490](https://arxiv.org/abs/2509.15490) -- Region-level spatial reasoning for warehouse environments, submitted to ICCVW (primary inspiration for RTI)
- **RegionGPT**: [CVPR 2024](https://arxiv.org/abs/2403.02330) -- Region understanding VLM with `<region>` token injection (foundation for RTI design)
- **Qwen 3.5**: [Qwen Team](https://huggingface.co/Qwen/Qwen3.5-0.8B) -- Base VLM backbone
- **DBNet++**: [TPAMI 2022](https://arxiv.org/abs/2202.10304) -- Differentiable Binarization (soft mask in RTI)
- **Gated Attention MIL**: [ICML 2018](https://arxiv.org/abs/1802.04712) -- Attention-based pooling (RTI mask_rgb)
- **xVal**: [NeurIPS 2023](https://arxiv.org/abs/2310.02989) -- A Continuous Numerical Tokenization (Number Head)
