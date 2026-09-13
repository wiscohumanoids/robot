#!/usr/bin/env bash
# Run the robot workspace dev container with GUI (MuJoCo viewer, rviz2)
# forwarded to the host X server. Run from the repo root:
#   ./docker/run.sh
set -euo pipefail
xhost +local:docker >/dev/null 2>&1 || true
docker run -it --rm \
  --network host \
  -e DISPLAY="${DISPLAY:-}" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  wiscohumanoids/robot:latest
