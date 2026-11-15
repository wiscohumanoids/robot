#!/usr/bin/env bash
# Create the perception env on the GPU VM: ~/hsim_envs/perception (Python 3.12, torch CUDA 12.8 wheels for Blackwell).
# Model weights download to ~/hsim_data/hf-cache on first server start (or run with --prefetch).
set -euo pipefail
UV="$HOME/.local/bin/uv"
ENV="$HOME/hsim_envs/perception"
HERE="$(cd "$(dirname "$0")" && pwd)"
export HF_HOME="$HOME/hsim_data/hf-cache"
mkdir -p "$HF_HOME" "$HOME/hsim_envs"
[ -d "$ENV" ] || "$UV" venv --python 3.12 "$ENV"
VIRTUAL_ENV="$ENV" "$UV" pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
VIRTUAL_ENV="$ENV" "$UV" pip install -r "$HERE/requirements.txt"
env -u LD_LIBRARY_PATH "$ENV/bin/python" -c "import torch, transformers, torch.nn.functional as F; F.conv2d(torch.zeros(1,1,8,8,device='cuda'), torch.zeros(1,1,3,3,device='cuda')); print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('transformers', transformers.__version__)"
if [ "${1:-}" = "--prefetch" ]; then
  "$ENV/bin/python" - <<'PY'
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, Sam2Model, Sam2Processor
for m in ("IDEA-Research/grounding-dino-base", "IDEA-Research/grounding-dino-tiny"):
    AutoProcessor.from_pretrained(m); AutoModelForZeroShotObjectDetection.from_pretrained(m)
for m in ("facebook/sam2.1-hiera-base-plus", "facebook/sam2.1-hiera-small"):
    Sam2Processor.from_pretrained(m); Sam2Model.from_pretrained(m)
print("weights cached in", __import__("os").environ["HF_HOME"])
PY
fi
