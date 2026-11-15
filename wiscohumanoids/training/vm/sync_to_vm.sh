#!/usr/bin/env bash
# Copy wiscohumanoids/ (robot model + training code) to the GPU VM. Nothing else on the VM is touched:
# no --delete, and only this folder. Training outputs stay on the VM under training/locomotion/logs/.
# Usage (from anywhere on the Mac):  wiscohumanoids/training/vm/sync_to_vm.sh [ssh-host, default hsim-gpu]
set -euo pipefail
HOST="${1:-hsim-gpu}"
SRC="$(cd "$(dirname "$0")/../.." && pwd)"          # .../wiscohumanoids
rsync -az --exclude __pycache__ --exclude .DS_Store --exclude 'training/phase_a/results/*.png' \
  "$SRC/" "$HOST:~/humanoid-sim-stack/wiscohumanoids/"
echo "Synced $SRC -> $HOST:~/humanoid-sim-stack/wiscohumanoids/"
