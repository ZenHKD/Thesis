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

Pruned from Qwen 3.5 0.8B: Vision Encoder (12 -> 4 blocks), Backbone (24 -> 8 layers), Vocabulary (248K -> 319 tokens). Adds **Number Head** for direct numeric regression on `distance` and `count` tasks. Full fine-tuning from scratch (~211M parameters).

```
RGB Image --> [Qwen Vision Encoder] --> [Merger] --> visual tokens [B, N, 1024]
               (4 ViT blocks)                             |
                                                          |
Depth Map ------------------------------------------------+
                                                          v
                                                  [GSA] Geometry Self-Attention
                                                   (DFormerv2 Full_GSA x2 blocks)
                                                          |
                                                          v
RLE Masks ----> [RTI] Region Token Injection -----> token injection (batched)
                  (mask_rgb + mask_depth)                  |
                                                          v
Question -----> [Tokenizer] --> old IDs --> [Remap] --> new IDs [0..318]
                                                          |
                                                   [Embed] (319 × 1024)
                                                          |
                                                          v
                                                  [Qwen Backbone] 8 layers
                                                   (DeltaNet + GatedAttn)
                                                          |
                                          +---------------+---------------+
                                          v                               v
                                   [LM Head] (319)               [Number Head] (xVal)
                                          |                               |
                                          v                               v
                                   Structured Output              Numeric Prediction
                                  category | answer            distance (m) / count (n)
                                          |                               |
                                          v                               v
                                   L_CE (CrossEntropy)             L_MSE (regression)
                                          |                               |
                                          +---------- L_total ------------+
                                                   L = L_CE + α·L_MSE
```


## Dataset

[nvidia/PhysicalAI-Spatial-Intelligence-Warehouse](https://huggingface.co/datasets/nvidia/PhysicalAI-Spatial-Intelligence-Warehouse)

**4 Task Categories**: `left_right`, `mcq`, `distance`, `count`

## Project Structure

```
Thesis/
├── model/
│   ├── pipeline.py                 # Full SpatialVLM pipeline 
│   ├── gsa.py                      # Geometry Self-Attention (DFormerv2)
│   ├── rti.py                      # Region Token Injection (original, can not be batched)
│   ├── architecture.md             # Detailed architecture documentation
│   └── qwen3.5-0.8b/               # Local model weights (gitignored)
├── model_micro/       
│   ├── pipeline.py                 # Full SpatialVLM pipeline 
│   ├── gsa.py                      # Geometry Self-Attention (same as in `model/`)
│   ├── rti.py                      # Region Token Injection (new, can be batched)
│   ├── num_head.py                 # Number Head for distance and count tasks
|   ├── train_tokenizer.py          # Scan all dataset to create minimize vocabulary for new model
│   ├── architecture.md             # Detailed architecture documentation
│   └── qwen3.5-micro/              # Local model weights (gitignored)
├── notebooks/       
│   ├── 00_EDA.ipynb                # Exploratory Data Analysis
│   ├── 01_RLE_Mask.ipynb           # Mask Analysis
│   └── 02_Error_Image.ipynb        # Error Image Analysis
├── analysis/                       # Check full question's type in whole dataset (train + val + test)
├── src/       
│   ├── dataloader/                 # Dataset loader
|   ├── train_micro/                # Micro model training
│   ├── train_phase1/               # Phase 1 training (original)
│   └── train_phase2/               # Phase 2 training (original)
├── test/       
│   ├── test_inference.py           # Inference test with real samples (untrained original model)
│   ├── test_backprop_1.py          # A backpropagation test for phase 1 training (origin)
│   ├── test_backprop_2.py          # A backpropagation test for phase 2 training (origin)
│   ├── test_dataloader.py          # A dataloader test (old RTI, batch_size = 1)
|   └── test_pipeline_alignment.py  # Pipeline alignment test (original model)
├── test_micro/  
│   ├── test_backprop.py            # A backpropagation test for micro model (full fine-tuning)
│   ├── test_dataloader_new.py      # A dataloader test for micro model (new RTI, batch_size > 1)
|   └── test_pipeline_alignment.py  # Pipeline alignment test (micro model)
├── data/nvidia_warehouse_dataset   # Dataset directory (gitignored)
├── count_qwen3_5_params.py         # Parameter counting script (outputs to Qwen3.5-0.8B.txt)
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
huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir model/qwen3.5-0.8b

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
- **DFormerv2**: [CVPR 2025](https://arxiv.org/abs/2504.04701) -- Geometry Self-Attention (GSA architecture)
- **DBNet++**: [TPAMI 2022](https://arxiv.org/abs/2202.10304) -- Differentiable Binarization (soft mask in RTI)
- **Gated Attention MIL**: [ICML 2018](https://arxiv.org/abs/1802.04712) -- Attention-based pooling (RTI mask_rgb)
- **xVal**: [NeurIPS 2023](https://arxiv.org/abs/2310.02989) -- A Continuous Numerical Tokenization (Number Head)
