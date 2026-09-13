"""bringup / full_stack.launch.py

Starts the entire vertical stack from ARCHITECTURE.md, sim path only
(hardware path is deliberately not wired into this convenience launch file
-- use lowlevel_test.launch.py use_hardware:=true directly for that, since
running the full stack against an unfinished EtherCAT scaffold is not a
meaningful thing to do yet):

  teleop_input -> locomotion_runner + manipulation_runner -> wbc_stub ->
  lowlevel_control (+ mujoco_ros2_control) -> safety (orthogonal)

Accepts `policy_onnx_path` (forwarded to locomotion_runner) so a real
berkeley_humanoid policy can be dropped in without editing this file --
see locomotion_runner/README.md.
"""
import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():
    bringup_path = get_package_share_directory('bringup')

    policy_onnx_path_arg = DeclareLaunchArgument(
        'policy_onnx_path', default_value='',
        description='Path to a berkeley_humanoid ONNX export for locomotion_runner. '
                    'Empty (default) = stub gait. See locomotion_runner/README.md.')

    lowlevel_test = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_path, 'launch', 'lowlevel_test.launch.py')),
        launch_arguments={'use_hardware': 'false'}.items(),
    )

    node_teleop = Node(package='teleop_input', executable='teleop_node', output='screen')
    node_locomotion = Node(
        package='locomotion_runner', executable='locomotion_node', output='screen',
        parameters=[{'policy_onnx_path': LaunchConfiguration('policy_onnx_path')}],
    )
    node_manipulation = Node(
        package='manipulation_runner', executable='manipulation_node', output='screen')
    node_wbc = Node(package='wbc_stub', executable='wbc_node', output='screen')
    node_safety = Node(package='safety', executable='safety_node', output='screen')

    return LaunchDescription([
        policy_onnx_path_arg,
        lowlevel_test,
        node_teleop,
        node_locomotion,
        node_manipulation,
        node_wbc,
        node_safety,
    ])
