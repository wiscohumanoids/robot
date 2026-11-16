#!/usr/bin/env bash
# Watch an arms-free walking (velocity) policy in an empty MuJoCo world on the Mac, at a fixed command.
#   tools/play_walk.sh wisco                 # forward 0.5 m/s, live viewer
#   tools/play_walk.sh g1 0,0.4,0            # sidestep 0.4 m/s   (vx,vy,wz in m/s, m/s, rad/s)
#   tools/play_walk.sh g1 0,0,0.8 --video results/videos/g1_turn.mp4 --seconds 20   # record instead
# Models: the newest .onnx in results/velocity_models/<g1|wisco>/. The arms move to random targets, as in training.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ROBOT="${1:-}"
case "$ROBOT" in
  wisco) TASK="Wisco-V1-Velocity-Flat" ;;
  g1)    TASK="Hsim-G1-Velocity-Flat-ArmsFree" ;;
  *) echo "usage: tools/play_walk.sh <g1|wisco> [vx,vy,wz] [--video out.mp4 --seconds N]"; exit 1 ;;
esac
shift
CMD="0.5,0,0"
if [[ $# -gt 0 && "$1" != --* ]]; then CMD="$1"; shift; fi
POLICY=$(ls -t "$ROOT/results/velocity_models/$ROBOT"/*.onnx | head -1)
ARGS=(--task "$TASK" --policy "$POLICY" --cmd "$CMD")
echo "playing $(basename "$POLICY") at command $CMD"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=
if [[ " $* " == *" --video "* ]]; then
  exec "$ROOT/.venv-mjlab/bin/python" tools/showcase_play.py "${ARGS[@]}" "$@"
fi
exec "$ROOT/.venv-mjlab/bin/mjpython" tools/showcase_play.py "${ARGS[@]}" --viewer --seconds "${SECONDS_TO_PLAY:-600}" "$@"
