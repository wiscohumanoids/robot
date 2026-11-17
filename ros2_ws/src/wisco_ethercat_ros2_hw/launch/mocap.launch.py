"""Motion capture into ROS 2: the upstream vrpn_mocap client (OptiTrack Motive, Vicon, Qualisys and others stream VRPN).

Each rigid body defined in the mocap software appears as geometry_msgs/PoseStamped on
/vrpn_mocap/<rigid body>/pose (plus twist / accel topics when the system sends them), in the `frame_id` frame. The
arm toolkit reads it through `mocap.topic` in the arm's joint config (e.g. /vrpn_mocap/arm01_flange/pose for
`arm mocap-calib`, `mocap-repeat`, `validate`); `mocap_check.py` measures the stream before it is used.

  ros2 launch wisco_ethercat_ros2_hw mocap.launch.py                     # server from MOCAP_HOST in the repo's .env
  ros2 launch wisco_ethercat_ros2_hw mocap.launch.py server:=<host> port:=3883 frame_id:=mocap

The mocap computer's address is machine-specific, so it comes from MOCAP_HOST (the container loads .env; on other
machines source it) unless server:= is given. Enable VRPN streaming in the mocap software (Motive: Data Streaming →
VRPN, default port 3883) and use Y-up or Z-up consistently with `frame_id`.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def client(context):
    """The vrpn_mocap client with the resolved server."""
    server = LaunchConfiguration('server').perform(context).strip() or os.environ.get('MOCAP_HOST', '').strip()
    if not server:
        raise RuntimeError('no mocap server: set MOCAP_HOST in the repo .env (see .env.example) or pass server:=<host>')
    return [Node(package='vrpn_mocap', executable='client_node', name='vrpn_mocap_client_node', namespace='vrpn_mocap',
                 output='screen',
                 parameters=[{'server': server, 'port': int(LaunchConfiguration('port').perform(context)),
                              'frame_id': LaunchConfiguration('frame_id').perform(context),
                              'update_freq': float(LaunchConfiguration('update_freq').perform(context)),
                              'refresh_freq': 1.0, 'multi_sensor': False,
                              'use_vrpn_timestamps': LaunchConfiguration('use_vrpn_timestamps').perform(context)
                              == 'true'}])]


def generate_launch_description():
    """Declare the arguments; the client node is built once they are resolved."""
    return LaunchDescription([
        DeclareLaunchArgument('server', default_value='', description='VRPN server (default: MOCAP_HOST)'),
        DeclareLaunchArgument('port', default_value='3883'),
        DeclareLaunchArgument('frame_id', default_value='mocap', description='frame of the published poses'),
        DeclareLaunchArgument('update_freq', default_value='240.0',
                              description='Hz the client polls at; at least the mocap frame rate'),
        DeclareLaunchArgument('use_vrpn_timestamps', default_value='false',
                              description='stamp with the mocap clock (needs synchronized clocks)'),
        OpaqueFunction(function=client),
    ])
