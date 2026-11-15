#!/usr/bin/env bash
# One-time setup for the GPU VM (g7e.2xlarge, Ubuntu 24.04, AWS Deep Learning Base GPU AMI).
# Safe to re-run. Run from the repo root: infra/aws/gpu_vm/setup.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

echo "== GPU + driver =="
if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi not found. Use the AWS Deep Learning Base GPU AMI (Ubuntu 24.04) or install driver >= 580." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
DRIVER_MAJOR="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
if (( DRIVER_MAJOR < 580 )); then
  echo "Driver $DRIVER_MAJOR is too old: Blackwell + mjlab need >= 580." >&2
  exit 1
fi

echo "== System packages (headless rendering + video) =="
sudo apt-get update -qq
sudo apt-get install -y -qq libegl1 libgl1 libglfw3 ffmpeg git rsync tmux

echo "== uv =="
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

echo "== Python env (mjlab) =="
cd "$REPO_ROOT/training/locomotion"
uv sync

echo "== torch / warp see the GPU =="
MUJOCO_GL=egl uv run python - <<'PY'
import torch, warp as wp
assert torch.cuda.is_available(), "torch cannot see the GPU"
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print(f"torch {torch.__version__}  cuda {torch.version.cuda}  {name}  sm_{cap[0]}{cap[1]}")
x = torch.randn(1024, 1024, device="cuda"); (x @ x).sum().item()  # fails here if no kernels for this GPU
wp.init()
print("warp", wp.__version__, "devices:", [d.alias for d in wp.get_cuda_devices()])
import mjlab; print("mjlab import OK")
PY

grep -q 'MUJOCO_GL=egl' ~/.bashrc || echo 'export MUJOCO_GL=egl' >> ~/.bashrc
echo "Setup OK. Next: training/locomotion/scripts/smoke_test.sh"
