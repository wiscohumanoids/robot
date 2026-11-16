"""Operator e-stop (node name: `operator`):  ros2 run hsim_safety estop on|off

Publishes a latched /safety/estop_request; the safety monitor turns it into a twist_mux lock.
"""
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ('on', 'off'):
        print('usage: ros2 run hsim_safety estop on|off')
        return 1
    rclpy.init()
    n = Node('operator')
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
    pub = n.create_publisher(Bool, '/safety/estop_request', qos)
    pub.publish(Bool(data=sys.argv[1] == 'on'))
    end = time.monotonic() + 1.5          # stay up long enough for the monitor to receive it
    while time.monotonic() < end:
        rclpy.spin_once(n, timeout_sec=0.1)
    print(f'e-stop {"ENGAGED" if sys.argv[1] == "on" else "released"}')
    n.destroy_node()
    rclpy.try_shutdown()
    return 0
