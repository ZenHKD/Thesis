# SpatialVLM

**Spatial Reasoning Vision-Language Model with Region Token Injection**

> AI City Challenge 2025 Track 3 -- Spatial understanding in warehouse environments using RGB-D data.

## Architecture

### Original model, see `model/`

**Qwen 3.5 0.8B** (native VLM)

### Micro model, see `model_micro/`

Pruned from **Qwen 3.5 0.8B** (853M): Vision Encoder (12 -> 4 blocks), Backbone (full 24 layers, **single-pass** — no looping). **Full original vocabulary** (248,076 + `<num>` = 248,077 tokens -> remain 248,320 tokens (padded)). Adds **Number Head** for direct numeric regression and **Category Head** (Bilinear) for MCQ/left-right classification. Two parallel input streams: visual (Vision Encoder -> 160 tokens) and region (RTI, independent of Vision Encoder -> 3 learned tokens per `<mask>`). 2-stage fine-tuning (~797M parameters).

```text
Stream 1 (Visual)
RGB Image --------> [Vision Encoder] --> [Merger] ======> visual tokens [160, 1024]
                     (4 ViT blocks)                              ||
                                                                 ||
Stream 2 (Region)                                                ||
RGB Image --------> [RTI: Region Token Injection]                ||
Depth Map -------->  (Independent of Vision Encoder)             ||
RLE Masks -------->  (mask_rgb + mask_depth + mask_geo)          ||
                                                    |            ||
                                                    v            ||
Question* --------> [Token Embedding] ==========> [Inject] ==> [Concat Fusion]
(*w/ <mask..>)    (Full 248,320 vocab)       (3->3 Replace)     ||
                                                                 vv
                                                   [Qwen Backbone] 24 layers (single pass)
                                                   (6 × [3 DeltaNet + 1 GatedAttn])
                                                                 ||
                                +==================+=============++==============+
                                v                  v                             v
                            [LM Head]        [Category Head]               [Number Head]
                                |            (Bilinear Attn)                     |
                                v                  v                             v
                           Structured          MCQ/LR pred               Numeric Prediction
                     <think>reasoning</think>      |                      distance / count
                        category | answer          |                             |
                                 |                 |                             |
                                 v                 v                             v
                               L_CE             L_Focal                   L_SmoothL1 (dynamic β)
                                 |                 |                             |
                                 +------------------L_total ---------------------+
                                            L = L_CE + α·L_SmoothL1 + γ·L_Focal
```

**Key differences from original model:**
| Component | Original | Micro |
|-----------|----------|----------|
| Vision Encoder | 12 ViT blocks | **4 ViT blocks** (last 4 kept) |
| Decoder | 24 layers | **24 layers, single pass** |
| RTI tokens | - | **mask_rgb + mask_depth + mask_geo** |
| RTI coupling | - | **Independent (raw RGB+Depth+RLE)** |
| Category Head | - | **Bilinear attention (MCQ + left_right)** |
| Number Head | - | **xVal regression (distance + count)** |
| Output format | Direct answer | **`<think>CoT</think>` + structured answer** |
| Trainable params | ~853M | **~797M** |

## Dataset

[nvidia/PhysicalAI-Spatial-Intelligence-Warehouse](https://huggingface.co/datasets/nvidia/PhysicalAI-Spatial-Intelligence-Warehouse)

**4 Task Categories**: `left_right`, `mcq`, `distance`, `count`

| Split | QA Pairs | RGB-D pairs |
|-------|----------|-------------|
| Train (original) | **499K** | ~78K |
| Train (balanced) | **204K** | ~72K |
| Val   | 1.9K | — |
| Test  | 19K | — |

> Balanced data addresses severe within-category imbalance (count=3: 47%, MCQ idx 0/1/2: 92%)
> via stratified down/upsampling. See `data/balance_train_data.py`.

## Project Structure

```
Thesis/
├── model/
│   └── qwen3.5-0.8b/               # Local model weights (gitignored)
├── model_micro/
│   ├── pipeline.py                 # Micro pipeline (single-pass, no GSA/LoopLM)
│   ├── rti.py                      # RTI (independent of Vision Encoder, batched)
│   ├── num_head.py                 # Number Head (xVal-style softplus regression)
│   ├── cat_head.py                 # Category Head (Bilinear attention)
│   ├── loss.py                     # Combined CE + SmoothL1 + Focal loss
│   ├── prune.py                    # Pruning script (Qwen 3.5 0.8B -> Micro)
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
│   └── train_micro/                # Training + validation + evaluation
│       ├── train.py                # 2-stage training (Stage 1: frozen decoder, Stage 2: full)
│       ├── val.py                  # Validation (teacher-forced metrics)
│       └── evaluation.py           # Inference evaluation (autoregressive, multi-threshold)
├── test_micro/
│   ├── test_inference.py           # Inference test (per-sample debug output)
│   ├── test_backprop.py            # Backprop test (all components)
│   ├── test_dataloader.py          # Dataloader test (batched RTI, decoded masks)
│   └── test_pipeline_alignment.py  # Pipeline alignment test
├── data/
│   ├── balance_train_data.py       # Data balancing script (stratified down/upsampling)
│   └── nvidia_warehouse_dataset/   # Dataset directory (gitignored)
├── checkpoints/micro/              # Training checkpoints + training.csv (gitignored)
└── README.md
```

## Setup

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

## References

- **SmolRGPT**: [arXiv 2509.15490](https://arxiv.org/abs/2509.15490) -- Region-level spatial reasoning for warehouse environments, submitted to ICCVW (primary inspiration for RTI)
- **RegionGPT**: [CVPR 2024](https://arxiv.org/abs/2403.02330) -- Region understanding VLM with `<region>` token injection (foundation for RTI design)
- **Qwen 3.5**: [Qwen Team](https://huggingface.co/Qwen/Qwen3.5-0.8B) -- Base VLM backbone
- **Qwen-VL**: [arXiv 2308.12966](https://arxiv.org/abs/2308.12966) -- Visual grounding token boundary mapping (`<|object_ref_start|>`)
- **DBNet++**: [TPAMI 2022](https://arxiv.org/abs/2202.10304) -- Differentiable Binarization (soft mask in RTI)
- **xVal**: [NeurIPS 2023](https://arxiv.org/abs/2310.02989) -- A Continuous Numerical Tokenization (Number Head)
- **Fast R-CNN**: [ICCV 2015](https://arxiv.org/abs/1504.08083) -- SmoothL1 evaluation metric for bounded geometric distance regression
