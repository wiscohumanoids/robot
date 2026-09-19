"""wbc_stub / wbc_node

STATUS: STUB arbitration, standing in for a real whole-body controller. There
is no dynamics model, no QP and no contact reasoning: only a fixed priority
rule (legs and waist always from locomotion_runner, arms from
manipulation_runner only while its output is fresh).

FREQUENCY: 500 Hz (CONTROL_RATE_HZ), faster than either input
source (50 Hz / 10 Hz) and slower than lowlevel_control (1000 Hz); see
RESEARCH_NOTES.md "Multi-rate architecture" for why this layer sits where it
does (cross-process DDS topic, not the 1kHz in-process controller boundary).

INPUTS:
  - humanoid_interfaces/JointTargets on /locomotion/joint_targets (source="locomotion")
  - humanoid_interfaces/JointTargets on /manipulation/joint_targets (source="manipulation"),
    used only if received within MANIPULATION_STALENESS_S seconds.

OUTPUT:
  humanoid_interfaces/JointCommand on /joint_command, effort_feedforward
  always zero (a real WBC would fill this with e.g. gravity compensation).
"""
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy

from humanoid_interfaces.msg import JointTargets, JointCommand

NUM_JOINTS = 23  # CANONICAL_JOINT_ORDER, see humanoid_interfaces/config/canonical_joint_order.yaml

CONTROL_RATE_HZ = 500.0
MANIPULATION_STALENESS_S = 0.3  # ignore manipulation targets older than this

ARM_JOINTS = slice(13, 23)  # canonical indices 13-22, both arms
ZERO_EFFORT = [0.0] * NUM_JOINTS


class WbcStubNode(Node):

    def __init__(self):
        super().__init__('wbc_stub')

        self._latest_locomotion = None       # JointTargets
        self._latest_manipulation = None     # JointTargets
        self._manipulation_received_at = 0.0

        latest_qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT)

        self.create_subscription(
            JointTargets, '/locomotion/joint_targets', self._on_locomotion, latest_qos)
        self.create_subscription(
            JointTargets, '/manipulation/joint_targets', self._on_manipulation, latest_qos)
        self._cmd_pub = self.create_publisher(JointCommand, '/joint_command', 10)

        self.create_timer(1.0 / CONTROL_RATE_HZ, self._on_timer)
        self.get_logger().info(f'wbc_stub up at {CONTROL_RATE_HZ} Hz (pass-through arbitration)')

    def _on_locomotion(self, msg: JointTargets):
        self._latest_locomotion = msg

    def _on_manipulation(self, msg: JointTargets):
        self._latest_manipulation = msg
        self._manipulation_received_at = time.monotonic()

    def _arbitrate(self):
        """Returns (positions, velocities) for the merged command, or None if
        locomotion has not published yet.

        Legs and waist always come from locomotion. Arms come from manipulation
        only while its output is fresh. This is a slice overwrite, not a
        whole-body solve: it does not check that the merged pose is dynamically
        consistent (see wbc_stub/README.md).
        """
        if self._latest_locomotion is None:
            return None

        positions = list(self._latest_locomotion.positions)
        velocities = list(self._latest_locomotion.velocities)

        manipulation_fresh = (
            self._latest_manipulation is not None
            and (time.monotonic() - self._manipulation_received_at) < MANIPULATION_STALENESS_S)
        if manipulation_fresh:
            positions[ARM_JOINTS] = self._latest_manipulation.positions[ARM_JOINTS]
            velocities[ARM_JOINTS] = self._latest_manipulation.velocities[ARM_JOINTS]
        return positions, velocities

    def _on_timer(self):
        merged = self._arbitrate()
        if merged is None:
            return  # locomotion_runner has not published yet: do not command garbage

        cmd = JointCommand()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.position_target, cmd.velocity_target = merged
        cmd.effort_feedforward = ZERO_EFFORT  # a real WBC would fill this in
        self._cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = WbcStubNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
