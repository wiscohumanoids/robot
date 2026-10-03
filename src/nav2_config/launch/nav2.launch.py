"""nav2_config / nav2.launch.py

Starts Nav2's navigation servers in place of nav_stub, with the velocity output
routed through this repo's contract: Nav2 must never publish /cmd_vel itself,
because cmd_vel_mux is its only publisher (teleop > nav > zero, 0.5 s watchdog).

Nav2's own navigation_launch.py ends the chain at /cmd_vel, so this file builds
the same node set with different remaps:

    controller_server --cmd_vel_raw--> velocity_smoother --/cmd_vel_nav--> cmd_vel_mux
    behavior_server  --cmd_vel_raw-->  (recovery moves: spin, backup, wait)

Use with the stub stack minus nav_stub:

    ros2 launch bringup full_stack.launch.py sim:=false navigation:=external
    ros2 launch nav2_config nav2.launch.py

STATUS: runs against the stub stack (see README.md). Parameters default to
config/nav2_params.yaml: conservative guesses, not yet tuned on the real gait or robot.
/odom, which Nav2's controller and bt_navigator read, comes from odom_adapter_stub.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml

# Nodes brought up and activated by the lifecycle manager (same set as Nav2's
# navigation_launch.py, which is what the stock params file is written for).
LIFECYCLE_NODES = [
    'controller_server',
    'smoother_server',
    'planner_server',
    'behavior_server',
    'bt_navigator',
    'waypoint_follower',
    'velocity_smoother',
]

# Internal topic between the controller and the smoother. Not part of the contract.
RAW_CMD_VEL = 'cmd_vel_raw'
# The contract topic cmd_vel_mux listens to for navigation.
NAV_CMD_VEL = '/cmd_vel_nav'


def generate_launch_description():
    params_file = LaunchConfiguration('params_file')
    autostart = LaunchConfiguration('autostart')
    log_level = LaunchConfiguration('log_level')

    # This stack has no simulated clock on the stubs-only path, so force wall time
    # (also protects against a params file copied from Nav2's stock one, which sets it True).
    configured_params = RewrittenYaml(
        source_file=params_file,
        root_key='',
        param_rewrites={'use_sim_time': 'false', 'autostart': 'true'},
        convert_types=True)

    def server(package, executable, name, remappings=None):
        return Node(
            package=package, executable=executable, name=name, output='screen',
            parameters=[configured_params],
            arguments=['--ros-args', '--log-level', log_level],
            remappings=remappings or [])

    return LaunchDescription([
        SetEnvironmentVariable('RCUTILS_LOGGING_BUFFERED_STREAM', '1'),
        DeclareLaunchArgument(
            'params_file',
            default_value=os.path.join(
                get_package_share_directory('nav2_config'), 'config', 'nav2_params.yaml'),
            description="Nav2 parameters. Default is this package's config/nav2_params.yaml."),
        DeclareLaunchArgument(
            'autostart', default_value='true',
            description='Activate the Nav2 lifecycle nodes automatically.'),
        DeclareLaunchArgument('log_level', default_value='info', description='Nav2 log level.'),

        # Controller publishes to the internal topic, not to cmd_vel.
        server('nav2_controller', 'controller_server', 'controller_server',
               [('cmd_vel', RAW_CMD_VEL)]),
        server('nav2_smoother', 'smoother_server', 'smoother_server'),
        server('nav2_planner', 'planner_server', 'planner_server'),
        # Recovery behaviors (spin, backup, ...) also publish cmd_vel; they must take the
        # same route as the controller or they would bypass cmd_vel_mux.
        server('nav2_behaviors', 'behavior_server', 'behavior_server',
               [('cmd_vel', RAW_CMD_VEL)]),
        server('nav2_bt_navigator', 'bt_navigator', 'bt_navigator'),
        server('nav2_waypoint_follower', 'waypoint_follower', 'waypoint_follower'),
        # Smoother limits acceleration, then hands the command to cmd_vel_mux.
        server('nav2_velocity_smoother', 'velocity_smoother', 'velocity_smoother',
               [('cmd_vel', RAW_CMD_VEL), ('cmd_vel_smoothed', NAV_CMD_VEL)]),

        Node(
            package='nav2_lifecycle_manager', executable='lifecycle_manager',
            name='lifecycle_manager_navigation', output='screen',
            arguments=['--ros-args', '--log-level', log_level],
            parameters=[{'autostart': autostart, 'node_names': LIFECYCLE_NODES}]),
    ])
