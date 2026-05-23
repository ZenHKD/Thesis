# SpatialVLM

**Spatial Reasoning Vision-Language Model with Region Token Injection**

> AI City Challenge 2025 Track 3 -- Spatial understanding in warehouse environments using RGB-D data.

## Architecture

**Super Model**, see `super_model/`

The "Super" architecture is heavily optimized for spatial fidelity and boundary awareness, eliminating the parameter bloat of previous iterations.

### Core Components
1. **Vision Encoder**: Qwen 3.5 0.8B ViT (or seamlessly scalable to Qwen 2B ViT for higher patch resolution).
2. **Backbone**: Qwen 3.5 0.8B Language Model.
3. **Dual-Image Input**: Both RGB and Depth maps are processed via the Vision Encoder as separate "pictures" (`Picture 1` and `Picture 2`). This solves the global depth context leakage.
4. **Region Token Injection v2 (RTI v2)**: Replaces lossy CNN pools. Uses a DPT-based Polyphase Multiplexing Module that hooks directly into intermediate ViT layers. Extracts sub-pixel boundary geometry using Multi-Head Mask-Guided Pooling.
5. **Smart Routing via Special Tokens**: The LLM generates exactly **one** special token per spatial query (`<|dist|>`, `<|count|>`, `<|mcq|>`, or `<|lr|>`). This token acts as a dynamic router—triggering a secondary forward pass that diverts the token's hidden state directly into the specialized heads. No more parsing numbers from text.

```text
Stream 1 (Global Context)
[RGB Image]   --> [Vision Encoder] --> [Merger] ======> visual_rgb_tokens [160, 1024]
[Depth Image] --> [Vision Encoder] --> [Merger] ======> visual_dep_tokens [160, 1024]
                                                               ||
Stream 2 (Region-Level Detail)                                 ||
[Intermediate ViT Features]                                    ||
[RLE Masks] ------------------> [RTI v2 (Mask-Guided Pooling)] ||
                                                     |         ||
                                                     v         ||
Question* --------> [Token Embedding] ==========> [Inject] ==> [Concat Fusion]
(*w/ <mask..>)                               (3->3 Replace)    ||
                                                               vv
                                                [Qwen 0.8B Backbone]
                                                               ||
                        [Auto-Regressive Generation of 1 Special Token]
                                                               ||
                                 +==================+=============++==============+
                                 v                  v              v              v
                            [Dist Head]        [Count Head]   [MCQ Head]      [LR Head]
                                 |                  |              |              |
                                 v                  v              v              v
                            (float) dist       (int) count    (class) mcq     (class) lr
```

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

```text
Thesis/
├── super_model/
│   ├── pipeline.py                 # Core VLM Pipeline & Head Routing
│   ├── dataloader.py               # Multiprocessing DataLoader (Dual-Image + RLE masks)
│   └── rti.py                      # DPT-based Region Token Injection modules
├── src/
│   └── train_super/                # Training + inference infrastructure
│       ├── train.py                # Multi-stage training
│       ├── test.py                 # High-speed private test inference via DataLoader
│       └── evaluation.py           # Validation pipeline & accuracy metrics
├── data/
│   ├── balance_train_data.py       # Data balancing script
│   └── nvidia_warehouse_dataset/   # Dataset directory (gitignored)
├── checkpoints/super/              # Training checkpoints (gitignored)
└── README.md
```

## Setup & Inference

```bash
# Clone the repo
git clone https://github.com/ZenHKD/Thesis.git
cd Thesis

# Download Qwen 3.5 0.8B locally
hf download Qwen/Qwen3.5-0.8B --local-dir model/qwen3.5-0.8b

# Setup HF token for dataset access
echo "HF_TOKEN=hf_your_token_here" > .env

# Generate Private Test Submission
python src/train_super/test.py \
    --checkpoint checkpoints/super/stage3/epoch_X \
    --split test \
    --batch-size 16
```

## References
- **RegionGPT**: [CVPR 2024](https://arxiv.org/abs/2403.02330) -- Region understanding VLM with `<region>` token injection (foundation for RTI design)
- **Qwen 3.5**: [Qwen Team](https://huggingface.co/Qwen/Qwen3.5-0.8B) -- Base VLM backbone
- **Qwen-VL**: [arXiv 2308.12966](https://arxiv.org/abs/2308.12966) -- Visual grounding token boundary mapping (`<|object_ref_start|>`)


## Download Dataset
```bash
HF_TOKEN=$(grep HF_TOKEN .env | cut -d= -f2) && \
hf download nvidia/PhysicalAI-Spatial-Intelligence-Warehouse \
  --repo-type dataset \
  --local-dir ./data/nvidia_warehouse_dataset \
  --token "$HF_TOKEN"

find ./data/nvidia_warehouse_dataset -name 'chunk_*.tar.gz' -print0 | \
  xargs -0 -P 8 -I {} sh -c 'echo "[EXTRACT] $(basename "$1")"; tar --no-same-owner -xzf "$1" -C "$(dirname "$1")" && echo "[REMOVE] $(basename "$1")" && rm "$1"' _ {}
```
