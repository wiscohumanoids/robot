#!/bin/bash
# Sources ROS and the workspace for EVERY way of running the container
# (interactive shell, `docker run image ros2 ...`, CI's `bash -c`). A .bashrc
# alone isn't enough: Ubuntu's .bashrc returns early for non-interactive shells.
set -e
source /opt/ros/${ROS_DISTRO}/setup.bash
if [ -f /home/ros2_ws/install/setup.bash ]; then
  source /home/ros2_ws/install/setup.bash
fi
exec "$@"
