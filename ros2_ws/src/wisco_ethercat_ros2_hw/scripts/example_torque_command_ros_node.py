#!/usr/bin/env python3
"""Effort-mode example: pulse one joint to break stiction, then let it coast.

Sends a full command array (one value per controller joint, zeros for the
others), reads the pulsed joint's state by name, and clamps the pulse to the
controller's max_command for that joint.

  ros2 run wisco_ethercat_ros2_hw example_torque_command_ros_node.py --ros-args -p joint:=left_elbow
"""

import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from wisco_ethercat_ros2_hw.ros_tools import get_controller_limits


class StictionBreakerNode(Node):

    def __init__(self):
        super().__init__('example_torque_command_ros_node')
        self.declare_parameter('controller', 'effort_controller')
        self.declare_parameter('joint', '')  # empty: the controller's first joint
        self.declare_parameter('pulse_torque', 1.35)  # Nm, before clamping
        self.declare_parameter('pulse_duration', 0.25)  # s
        self.declare_parameter('coast_duration', 1.3)  # s

        p = lambda name: self.get_parameter(name).value  # noqa: E731
        controller = p('controller')
        limits = get_controller_limits(self, controller)
        if limits['interface_name'] != 'effort':
            raise RuntimeError('%s commands %s, not effort' % (controller, limits['interface_name']))
        self.joints = limits['joints']
        self.joint = p('joint') or self.joints[0]
        if self.joint not in self.joints:
            raise RuntimeError('%s does not control %s (it controls %s)'
                               % (controller, self.joint, ', '.join(self.joints)))
        self.index = self.joints.index(self.joint)
        limit = limits['max_command'][self.index]
        self.pulse_torque = max(-limit, min(limit, float(p('pulse_torque'))))
        if self.pulse_torque != float(p('pulse_torque')):
            self.get_logger().warn('pulse_torque clamped to %.2f Nm (max_command)' % self.pulse_torque)
        self.pulse_duration = float(p('pulse_duration'))
        self.coast_duration = float(p('coast_duration'))

        self.position = float('nan')
        self.velocity = float('nan')
        self.state = 'PULSE'
        self.state_start = time.monotonic()
        self.last_log = 0.0

        self.pub = self.create_publisher(Float64MultiArray, '/%s/commands' % controller, 10)
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.create_timer(0.01, self.control_loop)
        self.get_logger().info('pulsing %s with %.2f Nm' % (self.joint, self.pulse_torque))

    def joint_state_callback(self, msg):
        if self.joint in msg.name:
            i = msg.name.index(self.joint)
            if i < len(msg.position):
                self.position = msg.position[i]
            if i < len(msg.velocity):
                self.velocity = msg.velocity[i]

    def control_loop(self):
        now = time.monotonic()
        if now - self.state_start >= (self.pulse_duration if self.state == 'PULSE'
                                      else self.coast_duration):
            self.state = 'COAST' if self.state == 'PULSE' else 'PULSE'
            self.state_start = now
        torque = self.pulse_torque if self.state == 'PULSE' else 0.0
        cmd = [0.0] * len(self.joints)
        cmd[self.index] = torque
        self.pub.publish(Float64MultiArray(data=cmd))
        if now - self.last_log >= 0.1:
            self.last_log = now
            self.get_logger().info('[%s] torque %.2f Nm | pos %.3f | vel %.4f' % (
                self.state, torque, self.position, self.velocity))

    def stop(self):
        for _ in range(3):
            self.pub.publish(Float64MultiArray(data=[0.0] * len(self.joints)))


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = StictionBreakerNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            try:
                node.stop()
            except Exception:  # noqa: BLE001 - context may already be shut down
                pass
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
