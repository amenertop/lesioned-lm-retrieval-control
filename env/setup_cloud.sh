#!/usr/bin/env bash
# Create or update the 'lesion' conda env on the AutoDL instance. Safe to re-run.
# Run from the project root on the server: bash env/setup_cloud.sh
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh

conda env list | grep -q '^lesion ' || conda create -n lesion python=3.12 -y
conda activate lesion

# USTC measured ~10x faster than the preconfigured Aliyun mirror from this instance.
PIP_INDEX=https://mirrors.ustc.edu.cn/pypi/simple

# The PyPI Linux wheel of torch 2.8.0 is built against CUDA 12.8, which RTX PRO 6000 (Blackwell) needs.
pip install torch==2.8.0 -i "$PIP_INDEX"
pip install -r requirements.txt -i "$PIP_INDEX"

grep -q '# >>> lesion env >>>' ~/.bashrc || cat >> ~/.bashrc <<'EOF'
# >>> lesion env >>>
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf
# <<< lesion env <<<
EOF

python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.is_available(), '| transformers', transformers.__version__)"
