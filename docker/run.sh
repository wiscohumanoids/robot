#!/usr/bin/env bash
# Run the robot workspace dev container with YOUR working copy mounted, so edits
# on the host show up in the container immediately.
#
#   ./docker/run.sh                    # interactive shell
#   ./docker/run.sh <command...>       # run one command, e.g.
#   ./docker/run.sh ros2 launch bringup full_stack.launch.py sim:=false
#
# Inside the container:
#   - Python edits are live (the image was built with --symlink-install).
#   - After changing C++, a .msg/.action, or package.xml:  colcon build --symlink-install
#   - build/ install/ log/ live in named Docker volumes so they persist between runs.
#     Reset them (e.g. after rebuilding the image) with:
#         docker volume rm robot_build robot_install robot_log
#
# GUI: on Linux with an X server, the MuJoCo viewer and rviz2 open on your
# display. On macOS (Docker Desktop) there is no X11 forwarding: run the sim
# headless:  ros2 launch bringup full_stack.launch.py headless:=true
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"

ARGS=(-it --rm
  -v "${REPO_ROOT}/src:/home/ros2_ws/src"
  -v robot_build:/home/ros2_ws/build
  -v robot_install:/home/ros2_ws/install
  -v robot_log:/home/ros2_ws/log)

case "$(uname -s)" in
  Linux)
    ARGS+=(--network host)
    if [[ -n "${DISPLAY:-}" ]]; then
      xhost +local:docker >/dev/null 2>&1 || true
      ARGS+=(-e DISPLAY="${DISPLAY}" -v /tmp/.X11-unix:/tmp/.X11-unix:rw)
    else
      echo "note: no \$DISPLAY set, use headless:=true for the sim" >&2
    fi
    ;;
  Darwin)
    echo "note: macOS has no X11 forwarding here, use headless:=true for the sim" >&2
    ;;
esac

if [[ $# -gt 0 ]]; then
  exec docker run "${ARGS[@]}" wiscohumanoids/robot:latest "$@"
fi
exec docker run "${ARGS[@]}" wiscohumanoids/robot:latest
