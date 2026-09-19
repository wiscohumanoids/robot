"""bringup / lowlevel_test.launch.py

THE "prove a command produces motion" milestone harness.

Starts EITHER the MuJoCo sim path (default) OR the EtherCAT hardware
scaffold path -- switchable via the `use_hardware` launch argument -- then
loads joint_state_broadcaster and lowlevel_control's
JointImpedanceController against whichever one came up. Both paths load the
exact same bringup/config/controllers.yaml (see ARCHITECTURE.md "Sim/
hardware symmetry").

Usage:
    # sim path (default, laptop-friendly, no GPU, no real hardware needed)
    ros2 launch bringup lowlevel_test.launch.py

    # hardware scaffold path (ethercat_bridge -- see its README for what's
    # actually implemented vs. stubbed; this will NOT move a real motor
    # until the hardware team fills in ethercat_bridge's TODOs)
    ros2 launch bringup lowlevel_test.launch.py use_hardware:=true

Then, in another terminal, publish a test command (see README.md for the
full worked example) and watch `/robot_state` respond:
    ros2 topic pub /joint_command humanoid_interfaces/msg/JointCommand "..." --rate 100
    ros2 topic echo /robot_state
"""
import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

import xacro


def generate_launch_description():
    g1_description_path = get_package_share_directory('g1_description')
    bringup_path = get_package_share_directory('bringup')

    use_hardware_arg = DeclareLaunchArgument(
        'use_hardware', default_value='false',
        description='false (default) = mujoco_ros2_control/MujocoSystem sim path. '
                    'true = ethercat_bridge/EthercatHardwareInterface scaffold path '
                    '(see ethercat_bridge/README.md for what that actually does today).')

    headless_arg = DeclareLaunchArgument(
        'headless', default_value='false',
        description='Run MuJoCo without a GUI window (Docker / CI / macOS host). Sim path only.')

    controllers_yaml = os.path.join(bringup_path, 'config', 'controllers.yaml')
    xacro_file = os.path.join(g1_description_path, 'g1_23dof.urdf.xacro')
    mujoco_model_path = os.path.join(g1_description_path, 'g1_23dof_rev_1_0.xml')

    sim_robot_description = {
        'robot_description': xacro.process_file(
            xacro_file, mappings={'hardware_plugin': 'mujoco_ros2_control/MujocoSystem'}
        ).toxml()
    }
    hw_robot_description = {
        'robot_description': xacro.process_file(
            xacro_file, mappings={'hardware_plugin': 'ethercat_bridge/EthercatHardwareInterface'}
        ).toxml()
    }

    # --- Sim path ---
    node_mujoco_ros2_control = Node(
        package='mujoco_ros2_control',
        executable='mujoco_ros2_control',
        output='screen',
        parameters=[sim_robot_description, controllers_yaml, {'mujoco_model_path': mujoco_model_path},
                    {'headless': ParameterValue(LaunchConfiguration('headless'), value_type=bool)}],
        condition=UnlessCondition(LaunchConfiguration('use_hardware')),
    )
    node_rsp_sim = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[sim_robot_description],
        condition=UnlessCondition(LaunchConfiguration('use_hardware')),
    )

    # --- Hardware (EtherCAT scaffold) path ---
    node_ros2_control = Node(
        package='controller_manager',
        executable='ros2_control_node',
        output='screen',
        parameters=[hw_robot_description, controllers_yaml],
        condition=IfCondition(LaunchConfiguration('use_hardware')),
    )
    node_rsp_hw = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[hw_robot_description],
        condition=IfCondition(LaunchConfiguration('use_hardware')),
    )

    # --- Shared controller loading (both paths' controller_manager answers to
    # the default unqualified `ros2 control` CLI target, "controller_manager") ---
    load_joint_state_broadcaster = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'joint_state_broadcaster'],
        output='screen',
    )
    load_joint_impedance_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'joint_impedance_controller'],
        output='screen',
    )
    delayed_broadcaster = TimerAction(period=3.0, actions=[load_joint_state_broadcaster])

    return LaunchDescription([
        use_hardware_arg,
        headless_arg,
        node_mujoco_ros2_control,
        node_rsp_sim,
        node_ros2_control,
        node_rsp_hw,
        delayed_broadcaster,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=load_joint_state_broadcaster,
                on_exit=[load_joint_impedance_controller],
            )
        ),
    ])
