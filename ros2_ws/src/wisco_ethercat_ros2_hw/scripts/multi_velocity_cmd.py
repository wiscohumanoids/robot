#!/usr/bin/env python3
"""Velocity-mode demo: swing every joint gently around where it started.

Each joint tracks  p_d(t) = p0 + A * sin(w t - i * phase_offset)  with
v = dp_d/dt + gain * (p_d - p), so the motion stays centred on the start
position instead of drifting the way an open-loop velocity wave does. The
amplitude is capped by the joint's limits as reported by the controller, and
commands are clamped to the controller's max_command.

  ros2 run wisco_ethercat_ros2_hw multi_velocity_cmd.py
  ros2 run wisco_ethercat_ros2_hw multi_velocity_cmd.py --ros-args -p max_amplitude_rad:=0.3
"""

import math

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from wisco_ethercat_ros2_hw.ros_tools import get_controller_limits, wait_for_joint_states


class MultiVelocityCommand(Node):

    def __init__(self):
        super().__init__('multi_velocity_cmd')
        self.declare_parameter('controller', 'velocity_controller')
        self.declare_parameter('max_amplitude_rad', 0.5)
        self.declare_parameter('limit_fraction', 0.8)  # use at most this much of the room to a limit
        self.declare_parameter('period_s', 4.0)
        self.declare_parameter('phase_offset', 0.4)
        self.declare_parameter('gain', 2.0)  # 1/s, position error to velocity
        self.declare_parameter('ramp_s', 2.0)
        self.declare_parameter('rate_hz', 100.0)
        # Accepted so the launch file can pass them; this script reads the controller instead.
        self.declare_parameter('joints_file', '')
        self.declare_parameter('models_file', '')

        p = lambda name: self.get_parameter(name).value  # noqa: E731
        controller = p('controller')
        limits = get_controller_limits(self, controller)
        if limits['interface_name'] != 'velocity':
            raise RuntimeError('%s commands %s, not velocity' % (controller, limits['interface_name']))
        self.joints = limits['joints']
        self.max_command = limits['max_command']

        start = wait_for_joint_states(self, self.joints)
        self.p0 = [start[j][0] for j in self.joints]
        self.amplitude = []
        for j, p0, lo, hi in zip(self.joints, self.p0, limits['position_min'], limits['position_max']):
            room = min(p0 - lo, hi - p0)
            a = min(float(p('max_amplitude_rad')), float(p('limit_fraction')) * room)
            if a <= 0.0:
                self.get_logger().warn('%s starts at or past a limit (%.3f not in [%.3f, %.3f]); '
                                       'holding it still' % (j, p0, lo, hi))
                a = 0.0
            self.amplitude.append(a)

        self.omega = 2.0 * math.pi / float(p('period_s'))
        self.phase = float(p('phase_offset'))
        self.gain = float(p('gain'))
        self.ramp_s = max(1e-3, float(p('ramp_s')))
        self.position = list(self.p0)

        self.pub = self.create_publisher(Float64MultiArray, '/%s/commands' % controller, 10)
        self.create_subscription(JointState, '/joint_states', self._on_joint_states, 10)
        self.t0 = self.get_clock().now()
        self.create_timer(1.0 / float(p('rate_hz')), self._step)
        self.get_logger().info('swinging %s around their start positions, amplitudes %s rad' % (
            ', '.join(self.joints), ', '.join('%.2f' % a for a in self.amplitude)))

    def _on_joint_states(self, msg):
        index = {name: i for i, name in enumerate(msg.name)}
        for k, joint in enumerate(self.joints):
            i = index.get(joint)
            if i is not None and i < len(msg.position):
                self.position[k] = msg.position[i]

    def _step(self):
        t = (self.get_clock().now() - self.t0).nanoseconds * 1e-9
        ramp = min(1.0, t / self.ramp_s)
        out = []
        for k in range(len(self.joints)):
            arg = self.omega * t - k * self.phase
            a = ramp * self.amplitude[k]
            p_d = self.p0[k] + a * math.sin(arg)
            v_d = a * self.omega * math.cos(arg)
            v = v_d + self.gain * (p_d - self.position[k])
            out.append(max(-self.max_command[k], min(self.max_command[k], v)))
        self.pub.publish(Float64MultiArray(data=out))

    def stop(self):
        # Belt and braces: the controller also stops the joints once commands
        # stop arriving (command_timeout).
        for _ in range(3):
            self.pub.publish(Float64MultiArray(data=[0.0] * len(self.joints)))


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = MultiVelocityCommand()
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
