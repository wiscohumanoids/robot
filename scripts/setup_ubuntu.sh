#!/usr/bin/env bash
# One-time native setup for this workspace on Ubuntu 22.04 + ROS 2 Humble.
# No Docker needed. Safe to re-run.
#
#   ./scripts/setup_ubuntu.sh        # installs apt/pip deps, MuJoCo, and rosdep deps
#   ./scripts/build.sh               # builds the workspace
#
# You must already have ROS 2 Humble installed (this script does not install ROS
# itself): https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html
# Uses sudo for apt and rosdep.
#
# Env overrides: MUJOCO_VERSION (default 3.2.7), MUJOCO_PREFIX (default ~/.local/mujoco)
# NOTE: the docker/Dockerfile does the same steps; keep the two in step.
set -euo pipefail

MUJOCO_VERSION="${MUJOCO_VERSION:-3.2.7}"
MUJOCO_PREFIX="${MUJOCO_PREFIX:-$HOME/.local/mujoco}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$(uname -s)" != "Linux" ]] || ! grep -q 'VERSION_ID="22.04"' /etc/os-release 2>/dev/null; then
  echo "This script targets Ubuntu 22.04 (ROS 2 Humble's supported platform)." >&2
  echo "On macOS/Windows use ./docker/build.sh + ./docker/run.sh, or an Ubuntu 22.04 VM/WSL2." >&2
  exit 1
fi
if [[ ! -f /opt/ros/humble/setup.bash ]]; then
  echo "ROS 2 Humble not found at /opt/ros/humble. Install it first:" >&2
  echo "  https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html" >&2
  exit 1
fi

echo "==> apt packages"
sudo apt-get update
sudo apt-get install -y \
  python3-pip python3-pytest wget \
  libglfw3-dev libosmesa6-dev libeigen3-dev libyaml-cpp-dev \
  ros-humble-control-toolbox ros-humble-effort-controllers \
  ros-humble-joint-state-broadcaster ros-humble-robot-state-publisher \
  ros-humble-xacro ros-humble-nav2-msgs ros-humble-tf2-ros-py \
  python3-yaml python3-colcon-common-extensions python3-rosdep

echo "==> python packages"
pip3 install --user xacro onnxruntime

echo "==> MuJoCo ${MUJOCO_VERSION} -> ${MUJOCO_PREFIX}"
case "$(uname -m)" in
  x86_64) MJ_ARCH=x86_64 ;;
  aarch64) MJ_ARCH=aarch64 ;;
  *) echo "unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac
MUJOCO_DIR="${MUJOCO_PREFIX}/mujoco-${MUJOCO_VERSION}"
if [[ -f "${MUJOCO_DIR}/lib/libmujoco.so" ]]; then
  echo "already present"
else
  mkdir -p "${MUJOCO_PREFIX}"
  tmp="$(mktemp -d)"
  wget -q -P "$tmp" "https://github.com/google-deepmind/mujoco/releases/download/${MUJOCO_VERSION}/mujoco-${MUJOCO_VERSION}-linux-${MJ_ARCH}.tar.gz"
  tar -xzf "$tmp/mujoco-${MUJOCO_VERSION}-linux-${MJ_ARCH}.tar.gz" -C "${MUJOCO_PREFIX}"
  rm -rf "$tmp"
fi

echo "==> rosdep"
set +u; source /opt/ros/humble/setup.bash; set -u
[[ -f /etc/ros/rosdep/sources.list.d/20-default.list ]] || sudo rosdep init
rosdep update
rosdep install --from-paths "${REPO_ROOT}/src" --ignore-src -y

cat <<MSG

Done. Next:
  ./scripts/build.sh
  source install/setup.bash
  export LD_LIBRARY_PATH="${MUJOCO_DIR}/lib:\${LD_LIBRARY_PATH:-}"   # so the sim can find libmujoco (add to ~/.bashrc)
  ros2 launch bringup full_stack.launch.py sim:=false               # stubs only
MSG
