#!/bin/bash
set -e

# Source conda so 'conda activate' works in scripts
eval "$(conda shell.bash hook)"

conda create -n myenv12 python=3.12 -y
conda activate myenv12

pip install uv
uv pip install --system --index-strategy unsafe-best-match -r requirements.txt
pip install causal-conv1d --no-build-isolation