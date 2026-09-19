"""Puts every ROS package directory under src/ on sys.path so the pure-Python
logic modules (mux_logic, kinematics, nav_logic, planner_logic, bt_logic, ...)
can be imported and unit-tested with plain pytest, no ROS installation needed.
Modules that import rclpy are never imported by the tests."""
import glob
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
for setup_py in glob.glob(os.path.join(ROOT, 'src', '*', 'setup.py')):
    sys.path.insert(0, os.path.dirname(setup_py))
