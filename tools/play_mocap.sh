#!/usr/bin/env bash
# Watch a motion-tracking (mocap) policy in an empty MuJoCo world on the Mac.
#   tools/play_mocap.sh                      # list the available models
#   tools/play_mocap.sh g1_dance1            # live viewer (close the window to stop)
#   tools/play_mocap.sh wisco_dance1 --video results/videos/wisco_dance1.mp4 --seconds 30   # record instead
# Models live in results/tracking_models/<name>/ (policy.onnx + motion.npz), one per robot and clip.
# Runs on the CPU with the mjlab env in .venv-mjlab; the live viewer needs mjpython on macOS.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="$ROOT/results/tracking_models"
NAME="${1:-}"
if [[ -z "$NAME" || ! -d "$MODELS/$NAME" ]]; then
  echo "usage: tools/play_mocap.sh <model> [--video out.mp4 --seconds N]"
  echo "models:"; for d in "$MODELS"/*/; do echo "  $(basename "$d")"; done
  exit 1
fi
shift
case "$NAME" in
  wisco_*) TASK="Wisco-V2-Tracking-Flat" ;;
  g1_*)    TASK="Mjlab-Tracking-Flat-Unitree-G1" ;;
  *) echo "don't know the robot for $NAME (expected g1_* or wisco_*)"; exit 1 ;;
esac
ARGS=(--task "$TASK" --policy "$MODELS/$NAME/policy.onnx" --motion-file "$MODELS/$NAME/motion.npz")
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=
if [[ " $* " == *" --video "* ]]; then
  exec "$ROOT/.venv-mjlab/bin/python" tools/showcase_play.py "${ARGS[@]}" "$@"
fi
exec "$ROOT/.venv-mjlab/bin/mjpython" tools/showcase_play.py "${ARGS[@]}" --viewer --seconds "${SECONDS_TO_PLAY:-600}" "$@"
