"""bringup / full_stack.launch.py

Brings up the whole stack, every layer in INTERFACE_CONTRACT.md, with the
repo's stubs standing in for anything that isn't built yet. This is the "launch
the whole robot in seconds" entry point; teams then swap their own real node in
for one stub at a time.

Swapping a stub for your real node (one flag, no file edits):

    # launch everything except the stub locomotion node, then run yours:
    ros2 launch bringup full_stack.launch.py locomotion:=external
    ros2 run my_pkg my_locomotion_node        # now the only publisher of its topics

  <layer>:=stub      (default) launch this repo's stub for the layer
  <layer>:=external  launch nothing for the layer; you run the real node yourself
  Layers: planner behavior_tree perception state_estimation slam navigation
          locomotion manipulation wbc
  Real nodes take true|false: teleop (default false, see below),
          cmd_vel_mux (true), safety (true)

Other arguments:
  sim:=true|false          true (default) = MuJoCo + ros2_control + lowlevel_control
                           (via lowlevel_test.launch.py). false = stubs only, no
                           MuJoCo: for planner/perception/BT/nav work on a laptop
                           and for CI. /robot_state and /joint_states don't exist then.
  headless:=true|false     MuJoCo without a GUI window (Docker, CI, macOS host).
  policy_onnx_path:=PATH   forwarded to locomotion_runner (only with locomotion:=stub).

Keyboard teleop needs a real TTY, which `ros2 launch` does not provide, so
teleop defaults to false. Drive with the keyboard from a second terminal:
    ros2 run teleop_input teleop_node
(teleop:=true launches it here, where only a joystick on /joy works.)

The hardware path is not wired in here; use
`ros2 launch bringup lowlevel_test.launch.py use_hardware:=true` to test the
EtherCAT scaffold.
"""
import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# layer argument -> (package, executable)
STUB_LAYERS = {
    'planner': ('task_planner_stub', 'planner_node'),
    'behavior_tree': ('behavior_tree_stub', 'bt_node'),
    'perception': ('perception_stub', 'perception_node'),
    'state_estimation': ('state_estimation_stub', 'state_estimation_node'),
    'slam': ('slam_stub', 'slam_node'),
    'navigation': ('nav_stub', 'nav_node'),
    'locomotion': ('locomotion_runner', 'locomotion_node'),
    'manipulation': ('manipulation_runner', 'manipulation_node'),
    'wbc': ('wbc_stub', 'wbc_node'),
}
BOOL_NODES = {
    'teleop': ('teleop_input', 'teleop_node', 'false'),
    'cmd_vel_mux': ('cmd_vel_mux', 'mux_node', 'true'),
    'safety': ('safety', 'safety_node', 'true'),
}


def _choice(context, name, allowed):
    value = LaunchConfiguration(name).perform(context)
    if value not in allowed:
        raise RuntimeError(
            f"full_stack.launch.py: argument '{name}:={value}' is invalid; expected one of {allowed}")
    return value


def _launch_setup(context):
    bringup_path = get_package_share_directory('bringup')
    actions = []

    sim = _choice(context, 'sim', ('true', 'false')) == 'true'
    headless = _choice(context, 'headless', ('true', 'false'))

    if sim:
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(bringup_path, 'launch', 'lowlevel_test.launch.py')),
            launch_arguments={'use_hardware': 'false', 'headless': headless}.items()))
    else:
        # Stubs-only mode: still publish the URDF's static TF (base_link -> pelvis)
        # so the TF contract holds. No /joint_states here, so movable joints stay
        # untransformed: expected.
        xacro_file = os.path.join(
            get_package_share_directory('g1_description'), 'g1_23dof.urdf.xacro')
        description = xacro.process_file(xacro_file).toxml()
        actions.append(Node(
            package='robot_state_publisher', executable='robot_state_publisher',
            output='screen', parameters=[{'robot_description': description}]))

    for arg, (package, executable) in STUB_LAYERS.items():
        if _choice(context, arg, ('stub', 'external')) != 'stub':
            continue
        parameters = []
        if arg == 'locomotion':
            # value_type=str: an empty path must stay '' (not be YAML-parsed to null)
            parameters = [{'policy_onnx_path': ParameterValue(
                LaunchConfiguration('policy_onnx_path'), value_type=str)}]
        actions.append(Node(package=package, executable=executable,
                            output='screen', parameters=parameters))

    for arg, (package, executable, _default) in BOOL_NODES.items():
        if _choice(context, arg, ('true', 'false')) == 'true':
            actions.append(Node(package=package, executable=executable, output='screen'))

    return actions


def generate_launch_description():
    declared = [
        DeclareLaunchArgument('sim', default_value='true',
                              description='true = MuJoCo + ros2_control + lowlevel_control; '
                                          'false = stubs only (no MuJoCo, no /robot_state).'),
        DeclareLaunchArgument('headless', default_value='false',
                              description='Run MuJoCo without a GUI window.'),
        DeclareLaunchArgument('policy_onnx_path', default_value='',
                              description='ONNX policy for locomotion_runner. Empty = stub gait.'),
    ]
    for arg in STUB_LAYERS:
        declared.append(DeclareLaunchArgument(
            arg, default_value='stub',
            description=f"'stub' launches this repo's {arg} stub; "
                        f"'external' launches nothing (you run your real node)."))
    for arg, (_pkg, _exe, default) in BOOL_NODES.items():
        declared.append(DeclareLaunchArgument(
            arg, default_value=default, description=f'Launch {arg} (true|false).'))
    return LaunchDescription(declared + [OpaqueFunction(function=_launch_setup)])
