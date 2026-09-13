"""wbc_stub / wbc_node

STATUS: STUB arbitration, standing in for a real whole-body controller. No
dynamics model, no QP, no contact reasoning -- a fixed priority rule (legs +
waist always from locomotion_runner, arms from manipulation_runner only
while its output is fresh). See ARCHITECTURE.md's honesty about what this is
not.

FREQUENCY: 500 Hz (CONTROL_RATE_HZ) -- deliberately faster than either input
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

from wbc_stub.joint_order import load_canonical_joint_order, NUM_JOINTS

CONTROL_RATE_HZ = 500.0
MANIPULATION_STALENESS_S = 0.3  # ignore manipulation targets older than this

ARM_JOINTS = slice(13, 23)  # canonical indices 13-22, both arms


class WbcStubNode(Node):

    def __init__(self):
        super().__init__('wbc_stub')
        self.joint_names = load_canonical_joint_order()

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

    def _arbitrate(self) -> JointTargets:
        """Legs+waist always from locomotion. Arms from manipulation only if
        fresh. See wbc_stub/README.md for why this is a stand-in, not a real
        whole-body controller (it does not check torque/dynamics consistency
        between the two sources at all -- it just overwrites array slices)."""
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

        merged = JointTargets()
        merged.positions = positions
        merged.velocities = velocities
        merged.source = 'manipulation' if manipulation_fresh else 'locomotion'
        return merged

    def _on_timer(self):
        merged = self._arbitrate()
        if merged is None:
            return  # nothing published yet by locomotion_runner -- don't command garbage

        cmd = JointCommand()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.position_target = merged.positions
        cmd.velocity_target = merged.velocities
        cmd.effort_feedforward = [0.0] * NUM_JOINTS  # a real WBC would fill this in
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
