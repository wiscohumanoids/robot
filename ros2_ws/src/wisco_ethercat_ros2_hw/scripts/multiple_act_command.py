#!/usr/bin/env python3
"""Effort-mode demo: a feedforward torque wave sized to break each joint's stiction.

Per-joint stiction comes from the joint table (k_stiction on a row) or the
actuator model's default, and the torque is capped at the controller's
max_command, so this script and the controller cannot disagree on limits.

  ros2 launch wisco_ethercat_ros2_hw test_bench.launch.py control_mode:=effort demo:=true
"""

import math
import os

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from wisco_ethercat_ros2_hw.hardware_config import load_joint_profiles
from wisco_ethercat_ros2_hw.ros_tools import get_controller_limits

PACKAGE = 'wisco_ethercat_ros2_hw'


class MultiActuatorTorqueController(Node):

    def __init__(self):
        super().__init__('multi_actuator_torque_controller')
        config_dir = os.path.join(get_package_share_directory(PACKAGE), 'config')
        self.declare_parameter('controller', 'effort_controller')
        self.declare_parameter('joints_file', os.path.join(config_dir, 'joints.yaml'))
        self.declare_parameter('models_file', os.path.join(config_dir, 'actuator_models.yaml'))
        self.declare_parameter('speed_factor', 1.5)
        self.declare_parameter('amplitude', 1.0)
        self.declare_parameter('phase_offset', 0.4)
        # Stiction feedforward never exceeds this fraction of the joint's max_command.
        self.declare_parameter('stiction_fraction', 0.4)

        p = lambda name: self.get_parameter(name).value  # noqa: E731
        controller = p('controller')
        limits = get_controller_limits(self, controller)
        if limits['interface_name'] != 'effort':
            raise RuntimeError('%s commands %s, not effort' % (controller, limits['interface_name']))
        self.joints = limits['joints']
        self.torque_limit = dict(zip(self.joints, limits['max_command']))
        profiles = load_joint_profiles(p('models_file'), p('joints_file'))
        missing = [j for j in self.joints if j not in profiles]
        if missing:
            raise RuntimeError('joints %s are not in %s' % (missing, p('joints_file')))
        self.profiles = {j: profiles[j] for j in self.joints}

        self.speed_factor = float(p('speed_factor'))
        self.amplitude = float(p('amplitude'))
        self.phase_offset = float(p('phase_offset'))
        self.stiction_fraction = float(p('stiction_fraction'))

        self.start_time = self.get_clock().now()
        self.last_print_time = self.start_time
        self.vel_alpha = 0.35
        self.velocity = {j: 0.0 for j in self.joints}
        self.current_cmd = {j: 0.0 for j in self.joints}
        self.target_v = {j: 0.0 for j in self.joints}

        self.pub = self.create_publisher(Float64MultiArray, '/%s/commands' % controller, 10)
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.get_logger().info('feedforward torque wave on %s' % ', '.join(
            '%s (%s)' % (j, self.profiles[j]['model']) for j in self.joints))

    def joint_state_callback(self, msg):
        now = self.get_clock().now()
        t = (now - self.start_time).nanoseconds / 1e9
        for i, name in enumerate(msg.name):
            if name in self.velocity and i < len(msg.velocity):
                self.velocity[name] = ((1.0 - self.vel_alpha) * self.velocity[name] +
                                       self.vel_alpha * msg.velocity[i])

        ramp = min(1.0, t / 2.0)
        cmd = []
        for idx, joint in enumerate(self.joints):
            target_v = ramp * self.amplitude * math.cos(self.speed_factor * t -
                                                        idx * self.phase_offset)
            limit = self.torque_limit[joint]
            stiction = min(self.profiles[joint]['k_stiction'], self.stiction_fraction * limit)
            torque = max(-limit, min(limit, stiction * math.tanh(2.0 * target_v)))
            self.target_v[joint] = target_v
            self.current_cmd[joint] = torque
            cmd.append(torque)
        self.pub.publish(Float64MultiArray(data=cmd))

        if (now - self.last_print_time).nanoseconds / 1e9 >= 1.0:
            self.last_print_time = now
            lines = ['[feedforward torque]']
            for joint in self.joints:
                lines.append('%-22s target vel %+6.3f  vel %+6.3f  torque %+6.3f / %.2f Nm' % (
                    joint, self.target_v[joint], self.velocity[joint], self.current_cmd[joint],
                    self.torque_limit[joint]))
            self.get_logger().info('\n'.join(lines))

    def stop(self):
        for _ in range(3):
            self.pub.publish(Float64MultiArray(data=[0.0] * len(self.joints)))


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = MultiActuatorTorqueController()
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
