"""Autonomous humanoid (G1 or Wisco): the robot (sim_stack) + localization + Nav2 + perception, world model,
behavior tree, planner.

    ros2 launch hsim_bringup autonomy.launch.py env:=test/table_room [robot:=g1|wisco] [mapping:=rtabmap|slam|map]

robot:=wisco uses config/robots/wisco.yaml: stereo camera (no lidar), its policy and its env builds
(assets/envs/<env>/build_wisco/, made with tools/env_build.py <env> --robot wisco).

The robot estimates its own motion (leg odometry + IMU) and, by default, builds its map while it moves: RTAB-Map
SLAM from the head camera's color + depth, starting from an empty map every launch (mapping:=slam: slam_toolbox on
a lidar; mapping:=map: AMCL on the env's prebuilt map, for tests). Obstacles come from its own sensors (Wisco: the
stereo depth turned into a scan). Object detection runs on the camera images; the only thing it takes from the env is
the list of kinds of objects in the room (vocab_from_env), never their positions.
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, SetLaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ENVS = '/hsim/assets/envs'


def stack(context):
    """Resolve the robot profile and arguments, then return sim_stack plus navigation, perception, world model,
    behavior and planner.
    """
    mapping = LaunchConfiguration('mapping').perform(context)
    env = LaunchConfiguration('env').perform(context)
    bringup = get_package_share_directory('hsim_bringup')
    nav = get_package_share_directory('hsim_navigation')
    robot = LaunchConfiguration('robot').perform(context).strip() or 'wisco'
    prof = yaml.safe_load(open(os.path.join(bringup, 'config', 'robots', f'{robot}.yaml')))
    bdir = prof['env_build_dir']
    sensors = LaunchConfiguration('sensors').perform(context).strip() or prof['default_sensors']
    pass_through = {k: LaunchConfiguration(k) for k in ('env', 'policy', 'viewer', 'foxglove', 'render_hz', 'robot',
                                                        'backend', 'update_rate')}
    pass_through['sensors'] = sensors
    stereo = sensors == 'stereo'
    cam = {'fov_h_deg': 110.0, 'fov_v_deg': 70.0, 'max_range': 8.0} if stereo else \
          {'fov_h_deg': 87.0, 'fov_v_deg': 58.0, 'max_range': 6.0}       # G1 head camera (D435i-like) in lidar mode
    perception = LaunchConfiguration('perception').perform(context)
    # server: real detection on the head camera's images (both sensor modes have the camera). No fallback to the
    # privileged stub unless asked (perception_fallback:=stub): if the VM is unreachable the robot sees nothing.
    server = {'server_url': LaunchConfiguration('perception_url').perform(context),
              'fallback': LaunchConfiguration('perception_fallback').perform(context)} if perception == 'server' else {}
    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(bringup, 'launch', 'sim_stack.launch.py')),
            launch_arguments={**pass_through, 'localization': 'none',
                              'map_server': 'true' if mapping == 'map' else 'false'}.items()),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(nav, 'launch', 'navigation.launch.py')),
            launch_arguments={'mapping': mapping, 'env': LaunchConfiguration('env'), 'build_dir': bdir,
                              'robot_radius': str(prof['footprint_radius']), 'sensors': sensors,
                              'obstacle_range': '3.0' if stereo else ''}.items()),
        # Perception: stub = line-of-sight STUB (privileged positions, honest visibility);
        # server = Grounding DINO + SAM 2 on the GPU VM via the SSH tunnel (with perception_fallback:=stub it falls
        # back to the stub while the server is unreachable; the default, none, leaves the robot blind instead)
        Node(package='hsim_perception', executable='perception_client' if perception == 'server' else 'perception_stub',
             name='perception', output='screen',
             parameters=[{'world': os.path.join(ENVS, env, bdir, 'world.mjb'),
                          'objects': os.path.join(ENVS, env, bdir, 'objects.yaml'),
                          'torso_body': prof['bodies']['torso'],
                          'camera_mount': [float(v) for v in prof['mounts']['camera'][:3]],
                          'camera_pitch': float(prof['mounts']['camera'][3]),
                          'use_sim_time': True, **cam, **server}]),
        # real detectors: commit an object only after 2 consistent sightings (multi-view confirmation)
        Node(package='hsim_perception', executable='world_model', name='world_model', output='screen',
             parameters=[{'use_sim_time': True, 'confirm_views': 2 if perception == 'server' else 1}]),
        # Task behavior tree: runs skill plans (from the planner), checks safety every tick
        # (with SLAM the map grows: space off its edge is unknown, not a wall)
        Node(package='hsim_behavior', executable='behavior_node', name='behavior', output='screen',
             parameters=[{'use_sim_time': True, 'off_map_blocked': mapping == 'map'}]),
        # LLM task planner (Claude): sentence + object memory -> skill plan; needs ANTHROPIC_API_KEY
        Node(package='hsim_planner', executable='planner_node', name='planner', output='screen',
             parameters=[{'use_sim_time': True}],
             condition=IfCondition(LaunchConfiguration('planner'))),
    ]


def generate_launch_description():
    """Declare the launch arguments and defer the node list to `stack` (it needs the resolved values)."""
    return LaunchDescription([
        # Ctrl-C: give nodes 2 s to stop, then 2 s after SIGTERM, before force-killing (default 5 + 10 s)
        SetLaunchConfiguration('sigterm_timeout', '2'),
        SetLaunchConfiguration('sigkill_timeout', '2'),
        DeclareLaunchArgument('env', default_value='test/table_room'),
        DeclareLaunchArgument('mapping', default_value='rtabmap',
                              description='rtabmap (default: RTAB-Map SLAM from scratch, camera + lidar/depth) | slam '
                                          '(slam_toolbox, lidar) | map (AMCL on the prebuilt map: tests/debug only)'),
        DeclareLaunchArgument('robot', default_value='wisco', description='robot profile: wisco | g1 | g1_arms_free (hsim_bringup/config/robots/)'),
        DeclareLaunchArgument('sensors', default_value='',
                              description="lidar (G1-style) | stereo (ZED X-style); default: the robot profile's"),
        DeclareLaunchArgument('policy', default_value='', description="policy folder (default: the robot profile's)"),
        DeclareLaunchArgument('viewer', default_value='true'),
        DeclareLaunchArgument('foxglove', default_value='false'),
        DeclareLaunchArgument('render_hz', default_value='0.0'),
        DeclareLaunchArgument('backend', default_value='sim',
                              description='sim (the MuJoCo node) | ros2_control (the robot run by its own controllers, '
                                          'MuJoCo inside the controller manager) | ethercat_sim (also the EtherCAT '
                                          'drives simulated, under the robot computer\'s controller chain): see '
                                          'sim_stack.launch.py'),
        DeclareLaunchArgument('update_rate', default_value='200',
                              description='backend:=ros2_control / ethercat_sim: controller (bus) rate, Hz'),
        DeclareLaunchArgument('planner', default_value='true', description='false: run without the LLM planner (tests)'),
        DeclareLaunchArgument('perception', default_value='server',
                              description='server (default: Grounding DINO + SAM 2 on the GPU VM, on the head camera; '
                                          'needs ssh -N -L 8100:localhost:8100 hsim-gpu on the Mac) | stub (privileged)'),
        DeclareLaunchArgument('perception_url', default_value='http://host.docker.internal:8100'),
        DeclareLaunchArgument('perception_fallback', default_value='none',
                              description='none (realistic: VM down = robot sees nothing) | stub (privileged positions)'),
        OpaqueFunction(function=stack),
    ])
