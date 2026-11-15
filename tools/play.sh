#!/usr/bin/env bash
# Watch a locomotion policy walk in the MuJoCo viewer on the Mac.
#   tools/play.sh                                   # snapshot policy
#   tools/play.sh --policy models/locomotion/<id>   # another policy
#   tools/play.sh --headless                        # scripted test, no window
# First run creates .venv/ with MuJoCo 3.11.0 + onnxruntime.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv"
if [[ ! -x "$VENV/bin/mjpython" ]]; then
  echo "Creating $VENV ..."
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q -r "$ROOT/tools/requirements-mac.txt"
fi
cd "$ROOT"
if [[ " $* " == *" --headless "* ]]; then
  exec "$VENV/bin/python" tools/sim2sim_play.py "$@"
fi
# macOS: MuJoCo's interactive viewer has to run under mjpython.
exec "$VENV/bin/mjpython" tools/sim2sim_play.py "$@"
