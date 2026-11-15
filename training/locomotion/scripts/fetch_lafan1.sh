#!/usr/bin/env bash
# Download Unitree's LAFAN1 motions retargeted to the G1 (CSV, 30 fps).
# The official repo has gone offline before; set LAFAN1_REPO to a mirror if this fails,
# e.g. LAFAN1_REPO=Luca-liang/LAFAN1_Retargeting_Dataset
# License: CC BY-NC-ND 4.0 (non-commercial).
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="${LAFAN1_REPO:-unitreerobotics/LAFAN1_Retargeting_Dataset}"
uvx --from huggingface_hub hf download "$REPO" \
  --repo-type dataset --include "g1/*" --local-dir data/lafan1
ls data/lafan1/g1 | head -50
