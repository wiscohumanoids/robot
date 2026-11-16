#!/usr/bin/env bash
# Copy the repo to a VM (code only, no build outputs, logs, or weights).
# Usage: tools/sync_to_vm.sh <ssh-host>     e.g. hsim-gpu
set -euo pipefail
HOST="${1:?usage: sync_to_vm.sh <ssh-host>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
rsync -az --delete \
  --exclude .git --exclude .venv --exclude .env --exclude __pycache__ \
  --exclude 'ros2_ws/build' --exclude 'ros2_ws/install' --exclude 'ros2_ws/log' \
  --exclude 'training/*/logs' --exclude 'training/*/data' --exclude wandb \
  --exclude '*.onnx' --exclude '*.pt' \
  "$ROOT/" "$HOST:~/humanoid-sim-stack/"
echo "Synced to $HOST:~/humanoid-sim-stack"
