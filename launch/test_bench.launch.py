"""Bring up the test bench from the joint table.

Everything that depends on which drives are wired where comes from
config/joints.yaml (plus the model table and calibration file). At start-up
this launch file:

  1. expands the URDF once to read joint names and limits,
  2. validates the joint table against it and writes one slave config per
     drive (direction and zero offset folded into the scale factors),
  3. expands the URDF again with the generated hardware manifest,
  4. writes the controller parameters (joint list, limits, timeout), and
  5. starts ros2_control_node, robot_state_publisher and the spawners.

Examples:
  ros2 launch wisco_ethercat_ros2_hw test_bench.launch.py
  ros2 launch wisco_ethercat_ros2_hw test_bench.launch.py use_mock_hardware:=true demo:=true
  ros2 launch wisco_ethercat_ros2_hw test_bench.launch.py control_mode:=effort

Other robots reuse this file with a different `description` and joint table
(see arm01.launch.py).
"""

import os
import shutil
import tempfile

import xacro
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnShutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from wisco_ethercat_ros2_hw.hardware_config import HardwareConfig

PACKAGE = 'wisco_ethercat_ros2_hw'


def _truthy(value):
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


def _expand_urdf(urdf_file, manifest=''):
    doc = xacro.process_file(urdf_file, mappings={'hardware_manifest': manifest})
    return doc.toprettyxml(indent='  ')


def _setup(context):
    pkg_share = get_package_share_directory(PACKAGE)
    arg = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731

    urdf_file = arg('description')
    if not os.path.isabs(urdf_file):
        urdf_file = os.path.join(pkg_share, urdf_file)
    template_file = os.path.join(pkg_share, 'config', 'drive_pdo_template.yaml')
    use_mock = _truthy(arg('use_mock_hardware'))
    update_rate = float(arg('update_rate'))
    joint_state_rate = float(arg('joint_state_rate'))
    command_timeout = float(arg('command_timeout'))

    config = HardwareConfig(
        models_file=arg('models_file'),
        joints_file=arg('joints_file'),
        urdf_xml=_expand_urdf(urdf_file),
        calibration_file=arg('calibration_file') or None,
        control_mode=arg('control_mode') or None,
        template_file=template_file,
    )

    out_dir = tempfile.mkdtemp(prefix='wisco_hw_')
    manifest = config.write(out_dir, use_mock_hardware=use_mock, control_frequency=update_rate)
    robot_description = _expand_urdf(urdf_file, manifest)

    controllers_file = os.path.join(out_dir, 'controllers.yaml')
    with open(controllers_file, 'w') as f:
        with open(arg('joints_file')) as jf:
            table = yaml.safe_load(jf) or {}
        params = config.controllers_yaml(update_rate, joint_state_rate, command_timeout,
                                         arm_config=table,
                                         calibration_file=arg('calibration_file') or None)
        if config.control_mode == 'arm':
            # the ArmController builds its Pinocchio gravity model from this
            params[config.controller_name()]['ros__parameters']['robot_description'] =                 robot_description
        yaml.safe_dump(params, f, sort_keys=False)

    actions = [
        RegisterEventHandler(OnShutdown(on_shutdown=[
            OpaqueFunction(function=lambda _ctx: shutil.rmtree(out_dir, True) or [])])),
        LogInfo(msg='[%s] %s mode, %d drive(s) on %s, %d state-only joint(s); configs in %s' % (
            PACKAGE, config.control_mode, len(config.joints),
            'MOCK hardware' if use_mock else 'EtherCAT', len(config.mock_joints), out_dir)),
    ]
    for joint in config.joints:
        actions.append(LogInfo(msg='[%s]   bus %d -> %s (%s, direction %+d)' % (
            PACKAGE, joint.bus_position, joint.name, joint.model_name, joint.direction)))
    for warning in config.warnings():
        actions.append(LogInfo(msg='[%s] WARNING: %s' % (PACKAGE, warning)))

    actions += [
        Node(
            package='controller_manager',
            executable='ros2_control_node',
            parameters=[controllers_file, {'robot_description': robot_description}],
            # Jazzy reads the description from the topic; Humble from the parameter.
            remappings=[('~/robot_description', '/robot_description')],
            output='both',
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description}],
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        ),
    ]

    if not config.joints:
        actions.append(LogInfo(msg='[%s] joint table is empty: no command controller' % PACKAGE))
        return actions

    controller = config.controller_name()
    actions.append(Node(
        package='controller_manager',
        executable='spawner',
        arguments=[controller, '--controller-manager', '/controller_manager'],
    ))

    if _truthy(arg('demo')) and config.control_mode != 'arm':
        script = ('multi_velocity_cmd.py' if config.command_interface == 'velocity'
                  else 'multiple_act_command.py')
        actions.append(Node(
            package=PACKAGE,
            executable=script,
            output='screen',
            parameters=[{
                'controller': controller,
                'joints_file': arg('joints_file'),
                'models_file': arg('models_file'),
            }],
        ))
    return actions


def generate_launch_description():
    pkg_share = get_package_share_directory(PACKAGE)
    config_dir = os.path.join(pkg_share, 'config')
    args = [
        DeclareLaunchArgument('description', default_value='urdf/test_bench.urdf.xacro',
                              description='Robot description xacro (relative to the package '
                                          'share dir or absolute); takes hardware_manifest'),
        DeclareLaunchArgument('joints_file', default_value=os.path.join(config_dir, 'joints.yaml'),
                              description='Joint table: joint -> bus position, model, direction'),
        DeclareLaunchArgument('models_file',
                              default_value=os.path.join(config_dir, 'actuator_models.yaml'),
                              description='Per-model numbers (encoder, torque scale, limits)'),
        DeclareLaunchArgument('calibration_file',
                              default_value=os.path.join(config_dir, 'calibration.yaml'),
                              description='Per-joint zero offsets; empty for none'),
        DeclareLaunchArgument('control_mode', default_value='',
                              description='velocity (CSV), effort (CST) or arm (CSP/CST via '
                                          'the ArmController); empty uses the joint table'),
        DeclareLaunchArgument('update_rate', default_value='100',
                              description='Controller manager and EtherCAT cycle rate, Hz'),
        DeclareLaunchArgument('joint_state_rate', default_value='100',
                              description='joint_states publish rate, Hz (capped at update_rate)'),
        DeclareLaunchArgument('command_timeout', default_value='0.1',
                              description='Seconds without a command before the controller '
                                          'stops the joints'),
        DeclareLaunchArgument('use_mock_hardware', default_value='false',
                              description='Run the drives on mock_components instead of EtherCAT'),
        DeclareLaunchArgument('demo', default_value='false',
                              description='Also start the demo command script for the mode'),
    ]
    return LaunchDescription(args + [OpaqueFunction(function=_setup)])
