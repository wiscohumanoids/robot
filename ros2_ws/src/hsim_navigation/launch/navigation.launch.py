"""Localization + Nav2 for the walking humanoid (G1 or Wisco).

    build_dir:=build      the env's build folder (Wisco: build_wisco; autonomy.launch.py sets it from the robot profile)
    robot_radius:=0.30    footprint radius for both costmaps (robot profile)
    obstacle_range:=3.0   mark obstacles only this close (depth cameras: stereo depth error grows with the square of
                          the distance, ~5 cm at 3 m but ~30 cm at 8 m, and far marks land in the aisle and are never
                          cleared by a forward-only sensor); clearing still reaches 1 m further. Default: nav2.yaml's.

    mapping:=map   AMCL localizes on the env's known geometry map (served by map_server in
                   sim_stack); initial pose = env spawn pose, as an operator would give it
    mapping:=slam  slam_toolbox builds the map while working and localizes on it (publishes /map)
    mapping:=rtabmap  RTAB-Map visual SLAM from the stereo camera's color + depth (sensors:=stereo)

Nav2: NavFn global planner, Regulated Pure Pursuit controller, spin/backup/wait recoveries,
velocity smoother -> /cmd_vel_nav (twist_mux decides between it and teleop).
"""

import math
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ENVS = '/hsim/assets/envs'
NAV_NODES = ['controller_server', 'planner_server', 'behavior_server', 'bt_navigator', 'velocity_smoother']


def nodes(context):
    cfg = os.path.join(get_package_share_directory('hsim_navigation'), 'config')
    mapping = LaunchConfiguration('mapping').perform(context)
    env = LaunchConfiguration('env').perform(context).strip()
    sim_time = {'use_sim_time': True}
    nav2 = os.path.join(cfg, 'nav2.yaml')
    raw = [('cmd_vel', 'cmd_vel_nav_raw')]
    bdir = LaunchConfiguration('build_dir').perform(context).strip() or 'build'
    sensors = LaunchConfiguration('sensors').perform(context).strip() or 'lidar'
    radius = LaunchConfiguration('robot_radius').perform(context).strip()
    obst = LaunchConfiguration('obstacle_range').perform(context).strip()
    if radius or obst:
        params = yaml.safe_load(open(nav2))
        changed = False
        for cm in ('local_costmap', 'global_costmap'):
            p = params[cm][cm]['ros__parameters']
            if radius and abs(float(p.get('robot_radius', 0.0)) - float(radius)) > 1e-6:
                p['robot_radius'] = float(radius)
                changed = True
            if obst:
                scan = p['obstacle_layer']['scan']
                scan['obstacle_max_range'] = float(obst)
                scan['raytrace_max_range'] = float(obst) + 1.0
                changed = True
        if changed:
            nav2 = '/tmp/hsim_nav2_robot.yaml'
            with open(nav2, 'w') as f:
                yaml.safe_dump(params, f)
    if mapping in ('slam', 'rtabmap'):
        # The map is still growing, and a global costmap sized to it can't hold a goal beyond its edge
        # (e.g. a fridge the camera already sees). A 24 x 24 m window around the robot keeps the map
        # inside it and plans through not-yet-mapped space (allow_unknown) to anything within ~12 m.
        params = yaml.safe_load(open(nav2))
        gc = params['global_costmap']['global_costmap']['ros__parameters']
        gc.update({'rolling_window': True, 'width': 24, 'height': 24})
        # Never plan through space nobody has seen: an unseen wall isn't on the map, so planning through unknown
        # sent the robot out of the room (it then hit or discovered the wall and replanned). The behavior tree picks
        # goals on floor it has seen, and looks around first.
        params['planner_server']['ros__parameters']['GridBased']['allow_unknown'] = False
        # Keep the path while it stays valid: replanning every second against a young, changing SLAM map made the
        # path (and the robot) flip back and forth. Nav2's stock tree: replan only if the path becomes invalid.
        params['bt_navigator']['ros__parameters']['default_nav_to_pose_bt_xml'] = os.path.join(
            get_package_share_directory('nav2_bt_navigator'), 'behavior_trees',
            'navigate_w_recovery_and_replanning_only_if_path_becomes_invalid.xml')
        nav2 = '/tmp/hsim_nav2_slam.yaml'
        with open(nav2, 'w') as f:
            yaml.safe_dump(params, f)

    if mapping == 'slam':
        loc = Node(package='slam_toolbox', executable='async_slam_toolbox_node', name='localization',
                   output='screen', parameters=[os.path.join(cfg, 'slam.yaml'), sim_time,
                                                {'use_lifecycle_manager': True}])
    elif mapping == 'map':
        x = y = yaw = 0.0
        if env:
            sp = yaml.safe_load(open(os.path.join(ENVS, env, bdir, 'objects.yaml'))).get('spawn', {})
            x, y, yaw = float(sp.get('x', 0)), float(sp.get('y', 0)), math.radians(float(sp.get('yaw_deg', 0)))
        loc = Node(package='nav2_amcl', executable='amcl', name='localization', output='screen',
                   parameters=[os.path.join(cfg, 'amcl.yaml'), sim_time,
                               {'initial_pose': {'x': x, 'y': y, 'z': 0.0, 'yaw': yaw}}])
    elif mapping == 'rtabmap':
        # RTAB-Map SLAM from scratch every launch (no prior map), odometry = the EKF (leg odometry + IMU) via TF.
        # Both sensor modes give it the head camera's color + depth (visual features, loop closures).
        #  lidar:  + the 2D lidar scan: occupancy grid from the scan (accurate, 360 deg) and ICP refinement of links
        #  stereo: occupancy grid from the depth camera (ZED X), obstacles 0.10-1.5 m (floor noise below)
        lidar = sensors == 'lidar'
        rt = {
            'frame_id': 'base_footprint', 'odom_frame_id': 'odom', 'map_frame_id': 'map',
            'subscribe_rgb': True, 'subscribe_depth': True, 'subscribe_scan': lidar, 'approx_sync': True,
            'sync_queue_size': 20, 'topic_queue_size': 10,
            'publish_tf': True, 'wait_for_transform': 0.3, 'database_path': '/tmp/rtabmap.db',
            'Reg/Force3DoF': 'true', 'Grid/RangeMax': '6.0',
            'Grid/MaxObstacleHeight': '1.5', 'Grid/MaxGroundHeight': '0.10',
            'RGBD/LinearUpdate': '0.1', 'RGBD/AngularUpdate': '0.1', 'Rtabmap/DetectionRate': '2',
            'Mem/IncrementalMemory': 'true', 'RGBD/OptimizeFromGraphEnd': 'false',
            # ray tracing: the floor between the sensor and what it hits is FREE (otherwise seen floor stays unknown
            # and the planner, which no longer crosses unknown space, has nowhere to go); drop isolated noise cells
            'Grid/RayTracing': 'true', 'Grid/NoiseFilteringRadius': '0.1', 'Grid/NoiseFilteringMinNeighbors': '3',
            'use_sim_time': True}
        rt.update({'Grid/Sensor': '0', 'Reg/Strategy': '1', 'Icp/VoxelSize': '0.05', 'Icp/MaxCorrespondenceDistance': '0.1'}
                  if lidar else {'Grid/Sensor': '1', 'Reg/Strategy': '0'})
        loc = Node(package='rtabmap_slam', executable='rtabmap', name='localization', output='screen',
                   arguments=['-d'],                          # fresh map every launch
                   parameters=[rt],
                   remappings=[('rgb/image', '/camera/head/color/image_raw'),
                               ('depth/image', '/camera/head/depth/image_raw'),
                               ('rgb/camera_info', '/camera/head/color/camera_info'),
                               ('scan', '/scan')])
    else:
        raise RuntimeError(f'mapping:={mapping}: use map, slam, or rtabmap')

    lifecycle = [] if mapping == 'rtabmap' else [   # RTAB-Map is not a lifecycle node
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager', name='lifecycle_manager_localization',
             output='screen', parameters=[sim_time, {'autostart': True, 'node_names': ['localization']}])]
    return [
        loc, *lifecycle,
        Node(package='nav2_controller', executable='controller_server', name='controller_server',
             output='screen', parameters=[nav2, sim_time], remappings=raw),
        Node(package='nav2_planner', executable='planner_server', name='planner_server',
             output='screen', parameters=[nav2, sim_time]),
        Node(package='nav2_behaviors', executable='behavior_server', name='behavior_server',
             output='screen', parameters=[nav2, sim_time], remappings=raw),
        Node(package='nav2_bt_navigator', executable='bt_navigator', name='bt_navigator',
             output='screen', parameters=[nav2, sim_time]),
        Node(package='nav2_velocity_smoother', executable='velocity_smoother', name='velocity_smoother',
             output='screen', parameters=[nav2, sim_time],
             remappings=[('cmd_vel', 'cmd_vel_nav_raw'), ('cmd_vel_smoothed', '/cmd_vel_nav')]),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager', name='lifecycle_manager_navigation',
             output='screen', parameters=[sim_time, {'autostart': True, 'node_names': NAV_NODES}]),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('mapping', default_value='map', description='map (AMCL on known map) | slam | rtabmap'),
        DeclareLaunchArgument('env', default_value=''),
        DeclareLaunchArgument('build_dir', default_value='build', description="env build folder (Wisco: build_wisco)"),
        DeclareLaunchArgument('robot_radius', default_value='', description='costmap footprint radius, m (default: nav2.yaml)'),
        DeclareLaunchArgument('obstacle_range', default_value='', description='mark obstacles only this close, m (stereo: 3.0)'),
        DeclareLaunchArgument('sensors', default_value='lidar', description='lidar | stereo (what RTAB-Map uses)'),
        OpaqueFunction(function=nodes),
    ])
