"""Puts every Python ROS package under ros2_ws/src/ on sys.path, so their pure-Python logic modules
(policy_contract, prompt, vla_client, server_link, ...) can be unit-tested with plain pytest, no ROS
installation needed:

    python -m pytest tests ros2_ws/src/*/test -q

Test modules must not import rclpy; tests that need a running stack live in tools/ (stack_smoke_test.sh,
autonomy_test.sh) and run inside the ROS container."""
import glob
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
for setup_py in glob.glob(os.path.join(ROOT, 'ros2_ws', 'src', '*', 'setup.py')):
    sys.path.insert(0, os.path.dirname(setup_py))
