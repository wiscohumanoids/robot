#!/usr/bin/env bash
# Wisco V1 motion tracking (task Wisco-V1-Tracking-Flat, whole body, 29 joints) on one retargeted LAFAN1 clip.
# The clip CSV comes from retarget_lafan1.py (written to training/tracking/wisco_v1/motions/<clip>.csv, not kept in the repo);
# it is converted to the motion npz on first use (GPU forward kinematics in the Wisco training model).
#
# Usage (on the VM):  training/tracking/wisco_v1/train_wisco_tracking.sh <clip> [--log-root DIR] [extra mjlab flags...]
#   e.g.  training/tracking/wisco_v1/train_wisco_tracking.sh dance1_subject2 --log-root ~/hsim_runs/wisco_track_dance1
# Env overrides: NUM_ENVS (4096), ITERS (15000), SAVE_EVERY (250), MOTION_DIR (~/hsim_data/motions/wisco)
set -euo pipefail
CLIP="${1:?usage: train_wisco_tracking.sh <clip> [--log-root DIR] [extra flags]}"; shift
LOG_ROOT="$HOME/hsim_runs/wisco_tracking_${CLIP}"
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"
MOTION_DIR="${MOTION_DIR:-$HOME/hsim_data/motions/wisco}"
CSV="$HERE/motions/$CLIP.csv"
NPZ="$MOTION_DIR/$CLIP.npz"
export MUJOCO_GL=egl WANDB_MODE=offline
# The DLAMI puts the system CUDA 13.2 libs on LD_LIBRARY_PATH; torch then loads a mismatched cuDNN sublibrary and every
# cuDNN op fails (CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED; tracking's motion sampling uses conv1d). torch ships its own.
unset LD_LIBRARY_PATH
cd "$REPO/training/locomotion"

if [[ ! -f "$NPZ" || "$CSV" -nt "$NPZ" ]]; then
  [[ -f "$CSV" ]] || { echo "missing $CSV (make it with retarget_lafan1.py)"; exit 1; }
  echo "[tracking] converting $CSV -> $NPZ"
  "$UV" run python "$REPO/training/tracking/motion_to_npz.py" --robot wisco --input "$CSV" --output "$NPZ" --device cuda:0
fi

mkdir -p "$LOG_ROOT"
echo "[tracking] training Wisco tracking on $CLIP -> $LOG_ROOT"
exec "$UV" run python "$HERE/train.py" Wisco-V1-Tracking-Flat \
  --env.commands.motion.motion-file "$NPZ" \
  --env.scene.num-envs "${NUM_ENVS:-4096}" \
  --agent.max-iterations "${ITERS:-15000}" \
  --agent.save-interval "${SAVE_EVERY:-250}" \
  --agent.run-name "$CLIP" \
  --agent.logger tensorboard \
  --log-root "$LOG_ROOT" \
  "${EXTRA[@]}"
