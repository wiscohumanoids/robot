"""safety / safety_node

STATUS: real, with software-only inputs today. This is the ROS2 side of the
safety interface; the other side (a physical wireless E-stop's firmware
bridging into ROS2) is a separate project. See INTEGRATION_POINTS.md "Safety".

FREQUENCY: 100 Hz (PUBLISH_RATE_HZ).

INPUTS:
  - std_msgs/Bool on /manual_estop. Used for testing, and by whatever bridge
    node a real wireless E-stop firmware project connects here. true = estop
    asserted, latched until a false is received.
  - humanoid_interfaces/RobotState on /robot_state: checked for NaN/Inf in
    joint positions/velocities/efforts, which would otherwise silently
    propagate into lowlevel_control's torque computation.

OUTPUT:
  humanoid_interfaces/SafetyStatus on /safety_status, at 100 Hz.
  lowlevel_control's JointImpedanceController subscribes to this and zeroes
  its effort output whenever is_safe is false, refusing to re-enable until a
  subsequent message reports is_safe=true again.
"""
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool

from humanoid_interfaces.msg import SafetyStatus, RobotState

PUBLISH_RATE_HZ = 100.0


class SafetyNode(Node):

    def __init__(self):
        super().__init__('safety')
        self._estop_active = False
        self._fault_flags = []

        self.create_subscription(Bool, '/manual_estop', self._on_manual_estop, 10)
        # /robot_state arrives at 1 kHz and only the latest value matters, so keep a
        # depth-1 best-effort queue instead of buffering 10 messages the check never needs.
        latest_qos = QoSProfile(
            depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(RobotState, '/robot_state', self._on_robot_state, latest_qos)
        self._pub = self.create_publisher(SafetyStatus, '/safety_status', 10)

        self.create_timer(1.0 / PUBLISH_RATE_HZ, self._on_timer)
        self.get_logger().info(f'safety up at {PUBLISH_RATE_HZ} Hz')

    def _on_manual_estop(self, msg: Bool):
        if msg.data != self._estop_active:
            self.get_logger().warn(f'estop_active -> {msg.data} (via /manual_estop)')
        self._estop_active = msg.data

    def _on_robot_state(self, msg: RobotState):
        finite = (all(map(math.isfinite, msg.joint_positions))
                  and all(map(math.isfinite, msg.joint_velocities))
                  and all(map(math.isfinite, msg.joint_efforts)))
        self._fault_flags = [] if finite else ['nan_or_inf_in_robot_state']

    def _on_timer(self):
        msg = SafetyStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.estop_active = self._estop_active
        msg.fault_flags = self._fault_flags
        msg.is_safe = (not self._estop_active) and (len(self._fault_flags) == 0)
        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SafetyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
