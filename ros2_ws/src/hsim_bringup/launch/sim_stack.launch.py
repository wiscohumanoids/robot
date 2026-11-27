"""The robot: MuJoCo humanoid (G1 or Wisco) + ONNX walking policy + realistic sensors + state estimation.

    ros2 launch hsim_bringup sim_stack.launch.py [robot:=g1|wisco] [env:=test/table_room] [policy:=...]

robot:=g1|wisco        robot profile config/robots/<robot>.yaml: world, policy, body names, sensor mounts,
                       default sensors (Wisco: stereo, it has no lidar), env build dir (Wisco: build_wisco/)

Then on the Mac: tools/sim/viewer.sh [--env ...]  (native MuJoCo window: watch + drive with arrow keys)

env:=<source>/<name>   world from assets/envs/<env>/build/ (build with tools/build/env_build.py);
                       without env: the bare training world (flat floor, no map)
estimator:=ekf         leg odometry + IMU fused by robot_localization (default, realistic)
estimator:=truth       DEBUG ONLY: odometry copied from the simulator's ground truth
localization:=initial  map -> odom fixed at the env's spawn pose (an operator-given start pose;
                       no correction for drift). autonomy.launch.py sets localization:=none and
                       runs AMCL or SLAM instead.
map_server:=true       serve the env's geometry map on /map (autonomy.launch.py turns it off for SLAM)
sensors:=lidar         head lidar -> /scan (G1-style) + the G1's head RGB-D camera (RealSense D435i-like: 87 deg,
                       depth to 6 m) for perception and visual SLAM
sensors:=stereo        ZED X-style stereo camera (110 deg, depth to 12 m) -> depth -> point cloud -> /scan (front
                       only), plus color + depth images for perception and RTAB-Map (no lidar)
                       Both modes publish /camera/head/{color,depth}/image_raw (+ camera_info, points).
backend:=sim           the MuJoCo sim node (hsim_sim/mujoco_sim): realistic sensors, lockstep with the policy
backend:=ros2_control  the robot run by its own controllers instead: MuJoCo inside the controller manager (the
                       upstream mujoco_ros2_control plugin, update_rate:=200 Hz) with the robot's controllers on top:
                       JointTargetController `leg_targets` and `arm_targets` (wisco_ethercat_ros2_hw) take the policy's
                       /locomotion/joint_targets with hold, ramp-in, joint limits and a timeout, as on the RK3588, and
                       MuJoCo's position actuators apply the trained PD (MyActuator mode 5's law). Works with env:= (the
                       env's world_ros2_control/ from tools/build/env_build.py) and with the stereo camera, SLAM and
                       perception (GroundTruthPlugin feeds the simulated camera and the Mac viewer). No lockstep with
                       the policy (as on the robot), no lidar, no battery model; IMU and joint states are noise-free.
backend:=ethercat_sim  one layer further down: the robot's EtherCAT hardware simulated (hsim_mujoco_plugins'
                       VirtualRobotSystem: a CiA402 drive per motor in MyActuator mode 5, the master's EcCiA402Drive
                       logic, MuJoCo as the robot), under the robot computer's controller chain: leg_targets /
                       arm_targets send position + kp + kd to the drives, the ankles and waist roll/pitch through
                       ParallelLinkageController to two motors each (rod geometry), /joint_states from
                       ChainedJointStateBroadcaster. One bus cycle of delay each way, as on the robot. Same worlds,
                       sensors and limits as backend:=ros2_control; the drive states on /drive_states/*.
Options: viewer:=false, foxglove:=true, render_hz:=10
"""

import math
import os
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetLaunchConfiguration
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ENVS = '/hsim/assets/envs'


def profile(context):
    """config/robots/<robot>.yaml (robot:=wisco default). Sensor mounts there match the sim's sensor models."""
    name = LaunchConfiguration('robot').perform(context).strip() or 'wisco'
    path = os.path.join(get_package_share_directory('hsim_bringup'), 'config', 'robots', f'{name}.yaml')
    if not os.path.exists(path):
        raise RuntimeError(f'no robot profile {path} (robots: g1, wisco)')
    return yaml.safe_load(open(path))


def sensors_of(context, prof):
    """The sensor set: the argument, else the profile's default."""
    return LaunchConfiguration('sensors').perform(context).strip() or prof['default_sensors']


def arg(context, name):
    """A launch argument's value."""
    return LaunchConfiguration(name).perform(context).strip()


def spawn_pose(env, build_dir='build'):
    """The environment's spawn pose (x, y, yaw) from its built objects.yaml; the origin without an environment."""
    if not env:
        return 0.0, 0.0, 0.0
    objs = yaml.safe_load(open(os.path.join(ENVS, env, build_dir, 'objects.yaml')))
    sp = objs.get('spawn', {})
    return float(sp.get('x', 0.0)), float(sp.get('y', 0.0)), math.radians(float(sp.get('yaw_deg', 0.0)))


def ros2_control_nodes(world, update_rate, viewer, backend, prof):
    """A controller-manager backend for `world` (the profile's training scene, or an env's world_ros2_control/):
    the backend's files (hsim_sim/ros2_control_backend.py or ethercat_sim_backend.py), the controller manager with
    MuJoCo inside, the spawners, and the supervisor (description, standing start, /sim/reset).
    """
    # the controller manager finds the plugins through this environment; a shell opened before the workspace was
    # built cannot see them, and a plugin would then fail with only a logged error
    for pkg in ('hsim_mujoco_plugins', 'mujoco_ros2_control', 'wisco_ethercat_ros2_hw'):
        try:
            get_package_share_directory(pkg)
        except Exception:
            raise RuntimeError(f'package {pkg} not found in this shell: build the workspace, then '
                               'source /hsim/ros2_ws/install/setup.bash (or open a new tools/ros.sh shell)')
    out = tempfile.mkdtemp(prefix=f'hsim_{backend}_')
    viewer_rate = 50.0 if viewer else 0.0
    if backend == 'ethercat_sim':
        from hsim_sim.ethercat_sim_backend import build
        files = build(world, out, update_rate, ethercat=prof.get('ethercat_sim'), viewer_rate=viewer_rate)
        # the robot computer's controller manager; the hardware plugin publishes /clock
        manager = Node(package='controller_manager', executable='ros2_control_node', output='screen',
                       parameters=[{'use_sim_time': True}, files['controllers']],
                       remappings=[('/imu_sensor_broadcaster/imu', '/imu')])
    else:
        from hsim_sim.ros2_control_backend import build
        files = build(world, out, update_rate, viewer_rate=viewer_rate)
        manager = Node(package='mujoco_ros2_control', executable='ros2_control_node', output='screen',
                       parameters=[{'use_sim_time': True}, files['controllers'], files['plugins']],
                       remappings=[('/imu_sensor_broadcaster/imu', '/imu')])
    controllers = files['controllers']
    spawn = files['controllers_to_spawn']
    targets = [c for c in spawn if c.endswith('_targets')]
    # one spawner, in order: a chained controller (leg_targets on parallel_joints) needs the one below it active
    spawner = Node(package='controller_manager', executable='spawner', output='screen',
                   arguments=[*spawn, '--param-file', controllers, '--controller-manager-timeout', '60'])
    return [
        manager,
        spawner,
        Node(package='hsim_sim', executable='backend_supervisor', name='backend_supervisor', output='screen',
             parameters=[{'robot_description_file': files['urdf'], 'controllers': spawn,
                          'hold_controllers': targets, 'use_sim_time': True,
                          'plugin_node': files.get('plugin_node', '/mujoco_ros2_control_node')}]),
    ]


def robot_nodes(context):
    """Every robot-side node for the resolved profile: MuJoCo, sensors, state estimation, localization, the walking
    policy and the sensor mounts.
    """
    env = arg(context, 'env')
    prof = profile(context)
    bdir = prof['env_build_dir']
    sim_time = {'use_sim_time': True}
    viewer = arg(context, 'viewer') == 'true'
    stereo = sensors_of(context, prof) == 'stereo'
    body = prof['bodies']
    mounts = prof['mounts']
    sim_params = {'render_hz': float(arg(context, 'render_hz')),
                  'viewer_hz': 50.0 if viewer else 0.0,
                  'lidar_hz': 0.0 if stereo else 10.0,
                  'root_body': body['root'], 'torso_body': body['torso'],
                  'fall_height': float(prof['fall_height']),
                  'lidar_mount': [float(v) for v in mounts['lidar_on_torso']],
                  'use_sim_time': False}            # the sim is the clock source
    nodes = []
    backend = arg(context, 'backend')
    if backend not in ('sim', 'ros2_control', 'ethercat_sim'):
        raise RuntimeError(f'backend:={backend}: sim, ros2_control or ethercat_sim')
    ros2_control = backend != 'sim'                 # both controller-manager backends
    if ros2_control and not stereo:
        raise RuntimeError(f'backend:={backend} has no simulated lidar: use sensors:=stereo')

    if env:
        build = os.path.join(ENVS, env, bdir)
        world, map_yaml = os.path.join(build, 'world.mjb'), os.path.join(build, 'map.yaml')
        rc_world = os.path.join(build, 'world_ros2_control', 'world.xml')
        needed = [world, map_yaml] + ([rc_world] if ros2_control else [])
        for f in needed:
            if not os.path.exists(f):
                raise RuntimeError(f'{f} missing: run tools/build/env_build.py assets/envs/{env} --robot {prof["name"]} on the Mac first')
        if ros2_control:
            nodes += ros2_control_nodes(rc_world, float(arg(context, 'update_rate')), viewer, backend, prof)
        else:
            nodes.append(Node(package='hsim_sim', executable='mujoco_sim', name='sim', output='screen',
                              parameters=[{'scene': world, **sim_params}]))
        if arg(context, 'map_server') == 'true':
            nodes += [
                Node(package='nav2_map_server', executable='map_server', name='map_server', output='screen',
                     parameters=[{'yaml_filename': map_yaml, 'use_sim_time': True}]),
                Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
                     name='lifecycle_manager_map', output='screen',
                     parameters=[{'autostart': True, 'node_names': ['map_server'], 'use_sim_time': True}]),
            ]
    elif ros2_control:
        nodes += ros2_control_nodes(prof['world'], float(arg(context, 'update_rate')), viewer, backend, prof)
    else:
        nodes.append(Node(package='hsim_sim', executable='mujoco_sim', name='sim', output='screen',
                          parameters=[{'scene': prof['world'], **sim_params}]))

    # Head RGB-D camera, in both sensor modes (perception and visual SLAM need images). It renders the built world,
    # so it needs an env. ZED X (stereo mode) or the G1's RealSense D435i-like head camera (lidar mode).
    world = os.path.join(ENVS, env, bdir, 'world.mjb') if env else None
    if world is not None:
        cam = {'hfov_deg': 110.0, 'max_depth': 12.0} if stereo else \
              {'hfov_deg': 87.0, 'max_depth': 6.0, 'baseline': 0.05, 'width': 640, 'height': 360}
        nodes += [
            Node(package='hsim_sim', executable='stereo_camera', name='stereo_camera', output='screen',
                 parameters=[{'world': world, 'torso_body': body['torso'],
                              'camera_mount': [float(v) for v in mounts['camera'][:3]],
                              'camera_pitch': float(mounts['camera'][3]), **cam, 'use_sim_time': True}]),
            Node(package='tf2_ros', executable='static_transform_publisher', name='camera_optical_mount',
                 arguments=['--roll', '-1.5708', '--yaw', '-1.5708', '--frame-id', 'head_camera',
                            '--child-frame-id', 'head_camera_optical'], parameters=[sim_time]),
        ]
    elif stereo and not ros2_control:
        raise RuntimeError('sensors:=stereo needs an env (the camera renders a built world)')

    if stereo and world is not None:
        nodes += [
            Node(package='rtabmap_util', executable='point_cloud_xyz', name='depth_to_cloud', output='screen',
                 parameters=[{'decimation': 2, 'max_depth': 8.0, 'voxel_size': 0.05, 'approx_sync': False,
                              'use_sim_time': True}],
                 remappings=[('depth/image', '/camera/head/depth/image_raw'),
                             ('depth/camera_info', '/camera/head/depth/camera_info'),
                             ('cloud', '/camera/head/points')]),
            Node(package='pointcloud_to_laserscan', executable='pointcloud_to_laserscan_node', name='depth_to_scan',
                 output='screen',
                 # min_height 0.10 (was 0.05): stereo depth error on the floor (a few cm, more when the camera is
                 # pitched down and sees lots of floor) put phantom obstacles above 5 cm in the aisle
                 parameters=[{'target_frame': 'base_footprint', 'transform_tolerance': 0.2,
                              'min_height': 0.10, 'max_height': 1.5, 'angle_min': -0.96, 'angle_max': 0.96,
                              'angle_increment': 0.0087, 'scan_time': 0.2, 'range_min': 0.3, 'range_max': 8.0,
                              'use_inf': True, 'use_sim_time': True}],
                 remappings=[('cloud_in', '/camera/head/points'), ('scan', '/scan')]),
        ]

    # State estimation: realistic by default (leg odometry needs the robot's own kinematic model)
    legodo = {'robot_model': prof['world'], 'torso_body': body['torso'],
              'sole_offset': float(prof['sole_offset']), 'use_sim_time': True}
    if arg(context, 'estimator') == 'truth':
        nodes += [
            Node(package='hsim_state_estimation', executable='state_estimation_stub',
                 name='state_estimation', output='screen', parameters=[sim_time]),
            Node(package='hsim_state_estimation', executable='leg_odometry', name='leg_odometry',
                 output='screen', parameters=[legodo]),    # body height/tilt/waist transforms
        ]
    else:
        ekf = os.path.join(get_package_share_directory('hsim_state_estimation'), 'config', 'ekf.yaml')
        nodes += [
            Node(package='hsim_state_estimation', executable='leg_odometry', name='leg_odometry',
                 output='screen', parameters=[legodo]),
            Node(package='robot_localization', executable='ekf_node', name='state_estimation',
                 output='screen', parameters=[ekf, sim_time],
                 remappings=[('odometry/filtered', '/odometry/filtered')]),
        ]

    # map -> odom: the spawn pose, as an operator would give it (no drift correction)
    if arg(context, 'localization') == 'initial':
        x, y, yaw = spawn_pose(env, bdir)
        nodes.append(Node(package='tf2_ros', executable='static_transform_publisher', name='localization',
                          arguments=['--x', str(x), '--y', str(y), '--yaw', str(yaw),
                                     '--frame-id', 'map', '--child-frame-id', 'odom'],
                          parameters=[sim_time]))

    # Walking policy (robot profile's unless policy:= is given) and where the sensors sit on the body
    # (robot_state_publisher takes these over with the URDF later)
    policy = arg(context, 'policy') or prof['policy']
    lm, im, cm = mounts['lidar'], mounts['imu'], mounts['camera']
    nodes += [
        Node(package='hsim_locomotion', executable='policy_node', name='locomotion', output='screen',
             parameters=[{'policy_dir': policy}, sim_time]),
        Node(package='tf2_ros', executable='static_transform_publisher', name='lidar_mount',
             arguments=['--x', str(lm[0]), '--y', str(lm[1]), '--z', str(lm[2]),
                        '--frame-id', 'base_footprint', '--child-frame-id', 'lidar_link'],
             parameters=[sim_time]),
        Node(package='tf2_ros', executable='static_transform_publisher', name='camera_mount',
             arguments=['--x', str(cm[0]), '--y', str(cm[1]), '--z', str(cm[2]),
                        '--pitch', str(cm[3]), '--frame-id', 'torso_link', '--child-frame-id', 'head_camera'],
             parameters=[sim_time]),
        Node(package='tf2_ros', executable='static_transform_publisher', name='imu_mount',
             arguments=['--x', str(im[0]), '--y', str(im[1]), '--z', str(im[2]),
                        '--frame-id', 'base_link', '--child-frame-id', 'imu_link'],
             parameters=[sim_time]),
    ]
    return nodes


def generate_launch_description():
    """Declare the launch arguments; robot nodes come from `robot_nodes`, then twist_mux, safety, teleop and the
    viewer bridge.
    """
    cfg = os.path.join(get_package_share_directory('hsim_bringup'), 'config')
    sim_time = {'use_sim_time': True}
    return LaunchDescription([
        # Ctrl-C: give nodes 2 s to stop, then 2 s after SIGTERM, before force-killing (default 5 + 10 s)
        SetLaunchConfiguration('sigterm_timeout', '2'),
        SetLaunchConfiguration('sigkill_timeout', '2'),
        DeclareLaunchArgument('env', default_value='', description='environment, e.g. test/table_room'),
        DeclareLaunchArgument('robot', default_value='wisco', description='robot profile: wisco | g1 | g1_arms_free (config/robots/)'),
        DeclareLaunchArgument('policy', default_value='', description="policy folder (default: the robot profile's)"),
        DeclareLaunchArgument('estimator', default_value='ekf', description='ekf (realistic) | truth (debug)'),
        DeclareLaunchArgument('localization', default_value='initial', description='initial | none'),
        DeclareLaunchArgument('map_server', default_value='true'),
        DeclareLaunchArgument('sensors', default_value='',
                              description="lidar (G1-style) | stereo (ZED X-style); default: the robot profile's"),
        DeclareLaunchArgument('viewer', default_value='true',
                              description='stream state to the Mac viewer and accept its key commands'),
        DeclareLaunchArgument('render_hz', default_value='0.0', description='camera image for Foxglove, 0 = off'),
        DeclareLaunchArgument('foxglove', default_value='false'),
        DeclareLaunchArgument('backend', default_value='sim',
                              description='sim | ros2_control | ethercat_sim (see the docstring)'),
        DeclareLaunchArgument('update_rate', default_value='200',
                              description='backend:=ros2_control / ethercat_sim: controller_manager (bus) rate, Hz'),

        OpaqueFunction(function=robot_nodes),
        Node(package='twist_mux', executable='twist_mux', name='twist_mux', output='screen',
             parameters=[os.path.join(cfg, 'twist_mux.yaml'), sim_time],
             remappings=[('cmd_vel_out', '/cmd_vel')]),
        Node(package='hsim_safety', executable='safety_monitor', name='safety', output='screen',
             parameters=[sim_time]),
        Node(package='hsim_teleop', executable='viewer_teleop', name='teleop', output='screen',
             condition=IfCondition(LaunchConfiguration('viewer'))),
        Node(package='hsim_teleop', executable='viewer_bridge', name='viewer_bridge', output='screen',
             parameters=[sim_time], condition=IfCondition(LaunchConfiguration('viewer'))),
        Node(package='foxglove_bridge', executable='foxglove_bridge', name='foxglove_bridge',
             parameters=[{'port': 8765, 'address': '0.0.0.0', 'use_sim_time': True}],
             condition=IfCondition(LaunchConfiguration('foxglove'))),
    ])
