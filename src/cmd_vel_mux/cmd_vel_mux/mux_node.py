"""cmd_vel_mux / mux_node

STATUS: real. This is a small, complete node.

FREQUENCY: 50 Hz (PUBLISH_RATE_HZ), always publishing, so downstream nodes
(locomotion_runner, state_estimation) get a steady /cmd_vel and a stale source
turns into an explicit zero command instead of silence.

INPUTS:
  - geometry_msgs/Twist on /cmd_vel_teleop (owner: teleop_input)
  - geometry_msgs/Twist on /cmd_vel_nav    (owner: nav_stub today; when Nav2
    replaces it, remap Nav2's velocity output to this topic)

OUTPUT:
  geometry_msgs/Twist on /cmd_vel. This node is its only publisher. Priority
  teleop > navigation > zero, each source expiring after `timeout_s`.
  See INTERFACE_CONTRACT.md.
"""
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

from cmd_vel_mux.mux_logic import select_command

PUBLISH_RATE_HZ = 50.0


class CmdVelMux(Node):

    def __init__(self):
        super().__init__('cmd_vel_mux')
        self.declare_parameter('timeout_s', 0.5)
        self._timeout_s = float(self.get_parameter('timeout_s').value)
        self._teleop = None  # (receive_time_s, (vx, vy, wz))
        self._nav = None
        self._last_source = None

        self.create_subscription(Twist, '/cmd_vel_teleop', self._on_teleop, 10)
        self.create_subscription(Twist, '/cmd_vel_nav', self._on_nav, 10)
        self._pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_timer(1.0 / PUBLISH_RATE_HZ, self._on_timer)
        self.get_logger().info(
            f'cmd_vel_mux up at {PUBLISH_RATE_HZ} Hz: '
            f'/cmd_vel_teleop > /cmd_vel_nav > zero, timeout {self._timeout_s}s')

    @staticmethod
    def _as_tuple(msg: Twist):
        return (msg.linear.x, msg.linear.y, msg.angular.z)

    def _on_teleop(self, msg: Twist):
        self._teleop = (time.monotonic(), self._as_tuple(msg))

    def _on_nav(self, msg: Twist):
        self._nav = (time.monotonic(), self._as_tuple(msg))

    def _on_timer(self):
        source, (vx, vy, wz) = select_command(
            time.monotonic(), self._teleop, self._nav, self._timeout_s)
        if source != self._last_source:
            self.get_logger().info(f'/cmd_vel source -> {source}')
            self._last_source = source
        msg = Twist()
        msg.linear.x = vx
        msg.linear.y = vy
        msg.angular.z = wz
        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelMux()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
