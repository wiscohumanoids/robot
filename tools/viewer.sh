#!/usr/bin/env bash
# Native MuJoCo window for the ROS stack: shows the ROS sim and drives it with arrow keys.
# Start the stack first (tools/ros.sh shell -> ros2 launch hsim_bringup sim_stack.launch.py).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv"
if [[ ! -x "$VENV/bin/mjpython" ]]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q -r "$ROOT/tools/requirements-mac.txt"
fi
cd "$ROOT"
exec "$VENV/bin/mjpython" tools/ros_viewer.py "$@"
