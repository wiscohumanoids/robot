"""Phase 1 stack with stubs: sim, locomotion, cmd_vel mux, state estimation, localization.

    ros2 launch hsim_bringup stub_stack.launch.py [foxglove:=true]

Each stub uses the node name of the real node that replaces it, so the
interface contract stays the same as layers become real.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory('hsim_bringup'), 'config')
    joints = os.path.join(cfg, 'g1_joints.yaml')
    sim_time = {'use_sim_time': True}

    return LaunchDescription([
        DeclareLaunchArgument('foxglove', default_value='true',
                              description='start foxglove_bridge on port 8765'),

        # The sim is the clock source, so it runs on wall time.
        Node(package='hsim_sim', executable='sim_stub', name='sim', output='screen',
             parameters=[joints, {'use_sim_time': False}]),
        Node(package='hsim_locomotion', executable='locomotion_stub', name='locomotion', output='screen',
             parameters=[joints, sim_time]),
        Node(package='twist_mux', executable='twist_mux', name='twist_mux', output='screen',
             parameters=[os.path.join(cfg, 'twist_mux.yaml'), sim_time],
             remappings=[('cmd_vel_out', '/cmd_vel')]),
        Node(package='hsim_state_estimation', executable='state_estimation_stub', name='state_estimation',
             output='screen', parameters=[sim_time]),
        # Localization stub: identity map -> odom until Phase 3.
        Node(package='tf2_ros', executable='static_transform_publisher', name='localization',
             arguments=['--frame-id', 'map', '--child-frame-id', 'odom'], parameters=[sim_time]),
        Node(package='foxglove_bridge', executable='foxglove_bridge', name='foxglove_bridge',
             parameters=[{'port': 8765, 'address': '0.0.0.0', 'use_sim_time': True}],
             condition=IfCondition(LaunchConfiguration('foxglove'))),
    ])
