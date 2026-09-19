"""mujoco_sim / g1_mujoco.launch.py

Brings up: robot_description (g1_description's xacro, hardware_plugin fixed
to mujoco_ros2_control/MujocoSystem) + the vendored mujoco_ros2_control node
+ robot_state_publisher, then loads joint_state_broadcaster and
lowlevel_control's JointImpedanceController.

This is the same pattern bipedal_nav's unitree_ros2_control/launch/
unitree_g1.launch.py used (Node + xacro + delayed `ros2 control
load_controller` ExecuteProcess chained via OnProcessExit), reused here,
not reinvented, adapted for our xacro'd URDF and effort-interface
controllers.yaml instead of position_pid.

Accepts `controllers_yaml` so callers can point at a different controllers
file than the shared bringup/config/controllers.yaml (its default).

NOTE: bringup/lowlevel_test.launch.py does NOT include this file; it
re-implements the same sim bring-up so it can also select the EtherCAT hardware
plugin. Keep the two in step.
"""
import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node

import xacro


def generate_launch_description():
    g1_description_path = get_package_share_directory('g1_description')
    bringup_path = get_package_share_directory('bringup')

    headless_arg = DeclareLaunchArgument(
        'headless', default_value='false',
        description='Run without a GUI window (Docker / CI / remote machines).')
    controllers_yaml_arg = DeclareLaunchArgument(
        'controllers_yaml',
        default_value=os.path.join(bringup_path, 'config', 'controllers.yaml'),
        description='Path to the ros2_control controllers YAML '
                     '(controller_manager + joint_impedance_controller params).')

    xacro_file = os.path.join(g1_description_path, 'g1_23dof.urdf.xacro')
    robot_description_xml = xacro.process_file(
        xacro_file, mappings={'hardware_plugin': 'mujoco_ros2_control/MujocoSystem'}).toxml()
    robot_description = {'robot_description': robot_description_xml}

    mujoco_model_path = os.path.join(g1_description_path, 'g1_23dof_rev_1_0.xml')

    node_mujoco_ros2_control = Node(
        package='mujoco_ros2_control',
        executable='mujoco_ros2_control',
        output='screen',
        parameters=[
            robot_description,
            LaunchConfiguration('controllers_yaml'),
            {'mujoco_model_path': mujoco_model_path},
            {'headless': LaunchConfiguration('headless')},
        ],
    )

    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description],
    )

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

    # Delay controller loading to give MuJoCo/controller_manager time to come up
    # (same 3s delay bipedal_nav's original launch file used).
    delayed_broadcaster = TimerAction(period=3.0, actions=[load_joint_state_broadcaster])

    return LaunchDescription([
        headless_arg,
        controllers_yaml_arg,
        node_mujoco_ros2_control,
        node_robot_state_publisher,
        delayed_broadcaster,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=load_joint_state_broadcaster,
                on_exit=[load_joint_impedance_controller],
            )
        ),
    ])
