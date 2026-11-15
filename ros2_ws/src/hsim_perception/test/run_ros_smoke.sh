#!/usr/bin/env bash
# Build hsim_perception in a scratch workspace inside a throwaway container and run ros_smoke.py.
#   docker run --rm -v "$PWD:/src:ro" hsim/ros-jazzy:dev bash /src/ros2_ws/src/hsim_perception/test/run_ros_smoke.sh
set -eo pipefail
source /opt/ros/jazzy/setup.bash
mkdir -p /tmp/ws/src && cp -r /src/ros2_ws/src/hsim_perception /tmp/ws/src/
cd /tmp/ws && colcon build --packages-select hsim_perception > /tmp/build.log 2>&1 || { tail -30 /tmp/build.log; exit 1; }
source /tmp/ws/install/setup.bash
K=/src/assets/envs/robocasa/kitchen_l1_s1/build
python3 /tmp/ws/src/hsim_perception/test/ros_smoke.py "$K/world.mjb" "$K/objects.yaml"
