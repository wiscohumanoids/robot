#!/usr/bin/env bash
# Copy the Wisco V1 folders (robot model + locomotion and tracking training code) to the GPU VM checkout, at the
# same paths as in this repo. Nothing else on the VM is touched: no --delete, only these three folders. Training
# outputs stay on the VM (training/locomotion/logs/, ~/hsim_runs/). To sync the whole repo use tools/sync_to_vm.sh.
# Usage (from anywhere on the Mac):  training/locomotion/wisco_v1/vm/sync_to_vm.sh [ssh-host, default hsim-gpu]
#   VM_REPO=<dir under ~ on the VM> overrides the checkout directory (default humanoid-sim-stack, as tools/sync_to_vm.sh).
set -euo pipefail
HOST="${1:-hsim-gpu}"
VM_REPO="${VM_REPO:-humanoid-sim-stack}"
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"   # repo root
for d in robots/wisco_v1 training/locomotion/wisco_v1 training/tracking/wisco_v1; do
  ssh "$HOST" "mkdir -p ~/$VM_REPO/$d"
  rsync -az --exclude __pycache__ --exclude .DS_Store --exclude 'checks/results/*.png' \
    "$ROOT/$d/" "$HOST:~/$VM_REPO/$d/"
  echo "Synced $d -> $HOST:~/$VM_REPO/$d/"
done
