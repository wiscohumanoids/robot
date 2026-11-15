"""Manipulation skill (/skills/manipulate). Needs the VLA server reachable at vla_url (SSH tunnel on the Mac)."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory('hsim_manipulation'), 'config', 'manipulation.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('vla_url', default_value='http://host.docker.internal:8200'),
        Node(package='hsim_manipulation', executable='manipulation_node', name='manipulation', output='screen',
             parameters=[cfg, {'vla_url': LaunchConfiguration('vla_url')}]),
    ])
