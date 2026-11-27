#!/usr/bin/env bash
# Watch a locomotion policy walk in the MuJoCo viewer on the Mac.
#   tools/sim/play.sh                                   # snapshot policy
#   tools/sim/play.sh --policy models/locomotion/<id>   # another policy
#   tools/sim/play.sh --headless                        # scripted test, no window
# First run creates .venv/ with MuJoCo 3.11.0 + onnxruntime.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$ROOT/.venv"
if [[ ! -x "$VENV/bin/python" ]] || ! "$VENV/bin/python" -c 'import mujoco, onnxruntime, numpy, yaml' >/dev/null 2>&1; then
  echo "Creating $VENV ..."
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q -r "$ROOT/tools/dev/requirements-mac.txt"
fi
cd "$ROOT"
if [[ " $* " == *" --headless "* ]]; then
  exec "$VENV/bin/python" tools/sim/sim2sim_play.py "$@"
fi
# macOS needs mjpython for the interactive MuJoCo viewer; Linux uses the normal interpreter.
if [[ "$(uname -s)" == "Darwin" ]]; then
  exec "$VENV/bin/mjpython" tools/sim/sim2sim_play.py "$@"
fi
exec "$VENV/bin/python" tools/sim/sim2sim_play.py "$@"
