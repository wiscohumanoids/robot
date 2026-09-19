#!/usr/bin/env bash
# Build the workspace natively (after ./scripts/setup_ubuntu.sh).
#   ./scripts/build.sh                    # full build
#   ./scripts/build.sh --packages-select cmd_vel_mux nav_stub   # extra args go to colcon
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set +u; source /opt/ros/humble/setup.bash; set -u

export MUJOCO_DIR="${MUJOCO_DIR:-$HOME/.local/mujoco/mujoco-${MUJOCO_VERSION:-3.2.7}}"
if [[ ! -d "$MUJOCO_DIR" ]]; then
  echo "MuJoCo not found at $MUJOCO_DIR, run ./scripts/setup_ubuntu.sh first (or set MUJOCO_DIR)." >&2
  exit 1
fi

cd "$REPO_ROOT"
colcon build --symlink-install \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DMUJOCO_HEADLESS_OSMESA=ON "$@"
echo
echo "Built. Now:  source install/setup.bash"
