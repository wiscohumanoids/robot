"""Start the HiPNUC IMU driver on the RK3588.

    ros2 launch wisco_hipnuc_imu imu.launch.py                        # config/hipnuc.yaml
    ros2 launch wisco_hipnuc_imu imu.launch.py port:=/dev/ttyUSB0 baudrate:=115200

Arguments left empty keep the config file's value. The node runs at normal priority; keep it off the isolated
real-time cores (infra/rk3588: cores 6-7), e.g. `taskset -c 0-3 ros2 launch ...`.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def driver(context):
    """The node with the config file plus any overrides given on the command line."""
    config = LaunchConfiguration('config').perform(context) or os.path.join(
        get_package_share_directory('wisco_hipnuc_imu'), 'config', 'hipnuc.yaml')
    overrides = {}
    port = LaunchConfiguration('port').perform(context).strip()
    baud = LaunchConfiguration('baudrate').perform(context).strip()
    if port:
        overrides['port'] = port
    if baud:
        overrides['baudrate'] = int(baud)
    return [Node(package='wisco_hipnuc_imu', executable='imu_node', name='imu', output='screen',
                 parameters=[config, overrides] if overrides else [config])]


def generate_launch_description():
    """Declare the arguments."""
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value='', description='parameter file (default: config/hipnuc.yaml)'),
        DeclareLaunchArgument('port', default_value='', description='serial port (default: from the config)'),
        DeclareLaunchArgument('baudrate', default_value='', description='baud rate (default: from the config)'),
        OpaqueFunction(function=driver),
    ])
