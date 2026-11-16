#!/usr/bin/env bash
# Talk to the robot from the Mac:  tools/ask.sh "walk to the table"     tools/ask.sh stop
# (the autonomy stack must be running: ros2 launch hsim_bringup autonomy.launch.py env:=...)
set -euo pipefail
# -t when run from a terminal, so Ctrl-C reaches the process in the container
[[ $# -ge 1 ]] || { echo 'usage: tools/ask.sh "<request>"   |   tools/ask.sh stop'; exit 1; }
docker exec -i $([[ -t 0 ]] && echo -t) hsim-ros bash -lc "source /opt/ros/jazzy/setup.bash; source /hsim/ros2_ws/install/setup.bash; ros2 run hsim_planner ask $(printf '%q ' "$@")"
