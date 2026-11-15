"""Bring up arm01, the 7-DOF test arm, on EtherCAT or mock hardware.

Same steps as test_bench.launch.py (it is included with arm01 defaults):
description urdf/arm01/arm01.ros2_control.xacro, joint table
config/arms/arm01.yaml (control_mode: arm, so the ArmController runs Mode P /
Mode T at 1 kHz), and the newest calibration version for arm01 if one exists.

  ros2 launch wisco_ethercat_ros2_hw arm01.launch.py
  ros2 launch wisco_ethercat_ros2_hw arm01.launch.py use_mock_hardware:=true
  ros2 launch wisco_ethercat_ros2_hw arm01.launch.py calibration_file:=/path/to/2026-11-14_v3.yaml
"""

import glob
import os
import re

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

PACKAGE = 'wisco_ethercat_ros2_hw'


def latest_calibration(share):
    files = glob.glob(os.path.join(share, 'config', 'calib', 'arm01', '*_v*.yaml'))
    files.sort(key=lambda p: int(re.search(r'_v(\d+)\.yaml$', p).group(1)))
    return files[-1] if files else ''


def generate_launch_description():
    share = get_package_share_directory(PACKAGE)
    passthrough = {
        'description': 'urdf/arm01/arm01.ros2_control.xacro',
        'joints_file': os.path.join(share, 'config', 'arms', 'arm01.yaml'),
        'calibration_file': latest_calibration(share),
        'control_mode': '',
        'update_rate': '1000',
        'joint_state_rate': '250',
        'command_timeout': '0.05',
        'use_mock_hardware': 'false',
    }
    args = [DeclareLaunchArgument(k, default_value=v) for k, v in passthrough.items()]
    include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'test_bench.launch.py')),
        launch_arguments={k: LaunchConfiguration(k) for k in passthrough}.items())
    return LaunchDescription(args + [include])
