"""Bring up arm01, the 7-DOF test arm, on EtherCAT or mock hardware.

Same steps as test_bench.launch.py (it is included with arm01 defaults):
description robots/arm01/urdf/arm01.ros2_control.xacro, joint table
robots/arm01/config/arm01.yaml (control_mode: arm, so the ArmController runs Mode P /
Mode T at 100 Hz by default), and the newest calibration version for arm01 if one exists.

  ros2 launch wisco_ethercat_ros2_hw arm01.launch.py
  ros2 launch wisco_ethercat_ros2_hw arm01.launch.py use_mock_hardware:=true
  ros2 launch wisco_ethercat_ros2_hw arm01.launch.py calibration_file:=/path/to/2026-11-14_v3.yaml
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from wisco_ethercat_ros2_hw.calibration_files import calibration_versions

PACKAGE = 'wisco_ethercat_ros2_hw'
DESCRIPTION = 'arm01_description'   # robots/arm01: URDF, meshes, joint configs, calibration store


def latest_calibration(description_share):
    """Newest calibration version for arm01, or '' when none exists yet.

    The joint config is resolved through realpath so a symlink install finds the calibration store
    in the source tree (robots/arm01/calibration/), where the calibration stages write it."""
    robot_dir = os.path.dirname(os.path.dirname(os.path.realpath(
        os.path.join(description_share, 'config', 'arm01.yaml'))))
    files = calibration_versions(os.path.join(robot_dir, 'calibration', 'arm01'))
    return files[-1] if files else ''


def generate_launch_description():
    share = get_package_share_directory(PACKAGE)
    description_share = get_package_share_directory(DESCRIPTION)
    passthrough = {
        'description': os.path.join(description_share, 'urdf', 'arm01.ros2_control.xacro'),
        'joints_file': os.path.join(description_share, 'config', 'arm01.yaml'),
        'calibration_file': latest_calibration(description_share),
        'control_mode': '',
        'update_rate': '100',
        'joint_state_rate': '50',
        'command_timeout': '0.05',
        'use_mock_hardware': 'false',
        'enable_motion': 'false',
    }
    args = [DeclareLaunchArgument(k, default_value=v) for k, v in passthrough.items()]
    include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'test_bench.launch.py')),
        launch_arguments={k: LaunchConfiguration(k) for k in passthrough}.items())
    return LaunchDescription(args + [include])
