#!/usr/bin/env bash
# Run on the laptop or CPU VM. Builds the manifest on the GPU VM, then copies ONNX + manifest here.
# Usage: tools/pull_policy.sh <run-dir> <policy-id> [ssh-host]
#   run-dir: path on the GPU VM relative to training/locomotion, e.g. logs/rsl_rl/g1_velocity/2026-11-08_10-00-00
set -euo pipefail
RUN="${1:?usage: pull_policy.sh <run-dir> <policy-id> [host]}"
ID="${2:?policy-id, e.g. g1_velocity_all_joints}"
HOST="${3:-hsim-gpu}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ssh "$HOST" "cd ~/humanoid-sim-stack/training/locomotion && ~/.local/bin/uv run python scripts/export_manifest.py '$RUN' --policy-id '$ID'"
mkdir -p "$ROOT/models/locomotion/$ID"
rsync -az "$HOST:~/humanoid-sim-stack/models/locomotion/$ID/" "$ROOT/models/locomotion/$ID/"
echo "Pulled to models/locomotion/$ID/"
ls -la "$ROOT/models/locomotion/$ID"
