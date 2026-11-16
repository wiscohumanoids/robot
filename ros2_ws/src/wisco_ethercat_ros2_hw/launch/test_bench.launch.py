import os
from launch import LaunchDescription
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_share = get_package_share_directory('wisco_ethercat_ros2_hw')
    urdf_file = os.path.join(pkg_share, 'urdf', 'test_bench.urdf.xacro')
    robot_desc = ParameterValue(Command(['xacro ', urdf_file]), value_type=str)
    
    return LaunchDescription([
        Node(
            package='controller_manager',
            executable='ros2_control_node',
            parameters=[{'robot_description': robot_desc},
                       os.path.join(pkg_share, 'config', 'controllers.yaml')],
            output='both',
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_desc}],
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['effort_controller', '--controller-manager', '/controller_manager'],
        ),
        Node(
            package='wisco_ethercat_ros2_hw',
            executable='example_torque_command_ros_node',
            name='torque_ramper',
            output='screen',
        ),
    ])
