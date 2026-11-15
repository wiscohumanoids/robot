#!/usr/bin/env bash
# Convert one LAFAN1 CSV to mjlab's NPZ (50 Hz, with forward kinematics) and upload it to your W&B "Motions" registry.
# Usage: scripts/upload_motion.sh <clip-name>     e.g. walk1_subject1
# Needs: `uv run wandb login` and the Motions registry created in the W&B UI.
set -euo pipefail
cd "$(dirname "$0")/.."
CLIP="${1:?usage: upload_motion.sh <clip-name>}"
CSV="data/lafan1/g1/${CLIP}.csv"
[[ -f "$CSV" ]] || { echo "Missing $CSV (run scripts/fetch_lafan1.sh)"; exit 1; }
export MUJOCO_GL=egl
uv run -m mjlab.scripts.csv_to_npz \
  --input-file "$CSV" \
  --output-name "$CLIP" \
  --input-fps 30 \
  --output-fps 50 \
  --render True
