#!/usr/bin/env bash
# Native MuJoCo window for the ROS stack: shows the ROS sim and drives it with arrow keys.
# Start the stack first (tools/ros.sh shell -> ros2 launch hsim_bringup sim_stack.launch.py).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$ROOT/.venv"
if [[ ! -x "$VENV/bin/python" ]] || ! "$VENV/bin/python" -c 'import mujoco, onnxruntime, numpy, yaml' >/dev/null 2>&1; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q -r "$ROOT/tools/dev/requirements-mac.txt"
fi
cd "$ROOT"
if [[ "$(uname -s)" == "Darwin" ]]; then
  exec "$VENV/bin/mjpython" tools/sim/ros_viewer.py "$@"
fi
exec "$VENV/bin/python" tools/sim/ros_viewer.py "$@"
