#!/usr/bin/env bash
# G1 motion tracking (BeyondMimic-style, mjlab "Mjlab-Tracking-Flat-Unitree-G1") on one LAFAN1 clip.
# One policy per clip (for example dance1_subject2, jumps1_subject1, fallAndGetUp1_subject1).
#
# Usage (on the VM):  training/tracking/train_g1_tracking.sh <clip> [--log-root DIR] [extra mjlab train flags...]
#   e.g.  training/tracking/train_g1_tracking.sh dance1_subject2 --log-root ~/hsim_runs/g1_track_dance1
# Env overrides: NUM_ENVS (4096), ITERS (15000), SAVE_EVERY (250), LAFAN1_DIR (~/hsim_data/lafan1/g1),
#                MOTION_DIR (~/hsim_data/motions/g1)
# Data: Unitree's G1-retargeted LAFAN1 CSV (mirror Luca-liang/LAFAN1_Retargeting_Dataset, CC BY-NC-ND 4.0).
set -euo pipefail
CLIP="${1:?usage: train_g1_tracking.sh <clip> [--log-root DIR] [extra flags]}"; shift
LOG_ROOT="$HOME/hsim_runs/g1_tracking_${CLIP}"
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"
LAFAN1_DIR="${LAFAN1_DIR:-$HOME/hsim_data/lafan1/g1}"
MOTION_DIR="${MOTION_DIR:-$HOME/hsim_data/motions/g1}"
CSV="$LAFAN1_DIR/$CLIP.csv"
NPZ="$MOTION_DIR/$CLIP.npz"
export MUJOCO_GL=egl WANDB_MODE=offline
# The DLAMI puts the system CUDA 13.2 libs on LD_LIBRARY_PATH; torch then loads a mismatched cuDNN sublibrary and every
# cuDNN op fails (CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED; tracking's motion sampling uses conv1d). torch ships its own.
unset LD_LIBRARY_PATH
cd "$REPO/training/locomotion"

if [[ ! -f "$NPZ" ]]; then
  [[ -f "$CSV" ]] || { echo "missing $CSV (download: hf download Luca-liang/LAFAN1_Retargeting_Dataset g1/$CLIP.csv --repo-type dataset --local-dir $(dirname "$LAFAN1_DIR"))"; exit 1; }
  echo "[tracking] converting $CSV -> $NPZ"
  "$UV" run python "$REPO/training/tracking/motion_to_npz.py" --robot g1 --input "$CSV" --output "$NPZ" --device cuda:0
fi

mkdir -p "$LOG_ROOT"
echo "[tracking] training G1 tracking on $CLIP -> $LOG_ROOT"
exec "$UV" run train Mjlab-Tracking-Flat-Unitree-G1 \
  --env.commands.motion.motion-file "$NPZ" \
  --env.scene.num-envs "${NUM_ENVS:-4096}" \
  --agent.max-iterations "${ITERS:-15000}" \
  --agent.save-interval "${SAVE_EVERY:-250}" \
  --agent.run-name "$CLIP" \
  --agent.logger tensorboard \
  --log-root "$LOG_ROOT" \
  "${EXTRA[@]}"
