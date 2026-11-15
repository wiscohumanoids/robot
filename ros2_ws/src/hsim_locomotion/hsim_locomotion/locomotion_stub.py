"""STUB locomotion node (node name: `locomotion`).

Stands in for the ONNX policy node. Reads /cmd_vel like the real policy will,
but always outputs the default standing pose as joint targets at 50 Hz.
Publishes /locomotion/status (1 Hz) with mode=stub so the stack monitor and
anyone reading logs can see it isn't the real policy.
"""

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState

POLICY_HZ = 50.0
CMD_TIMEOUT_S = 0.5


class LocomotionStub(Node):
    def __init__(self):
        super().__init__('locomotion')
        self.declare_parameter('joint_names', Parameter.Type.STRING_ARRAY)
        self.declare_parameter('default_joint_pos', Parameter.Type.DOUBLE_ARRAY)
        self.joint_names = list(self.get_parameter('joint_names').value)
        self.default_pos = list(self.get_parameter('default_joint_pos').value)

        self.cmd = None
        self.cmd_time = None
        self.pub_targets = self.create_publisher(JointState, '/locomotion/joint_targets', 10)
        self.pub_status = self.create_publisher(DiagnosticArray, '/locomotion/status', 10)
        self.create_subscription(TwistStamped, '/cmd_vel', self.on_cmd, 10)
        self.create_timer(1.0 / POLICY_HZ, self.tick)
        self.create_timer(1.0, self.publish_status)
        self.get_logger().info('STUB locomotion: holds default pose, ignores commands')

    def on_cmd(self, msg: TwistStamped):
        self.cmd = msg.twist
        self.cmd_time = self.get_clock().now()

    def tick(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.default_pos
        self.pub_targets.publish(msg)

    def publish_status(self):
        age = None
        if self.cmd_time is not None:
            age = (self.get_clock().now() - self.cmd_time).nanoseconds * 1e-9
        active = age is not None and age < CMD_TIMEOUT_S
        st = DiagnosticStatus(
            level=DiagnosticStatus.OK, name='locomotion', hardware_id='g1',
            message='stub: holding default pose')
        st.values = [
            KeyValue(key='mode', value='stub'),
            KeyValue(key='policy_id', value='none'),
            KeyValue(key='cmd_active', value=str(active)),
            KeyValue(key='cmd_vx', value=f'{self.cmd.linear.x:.3f}' if active else '0.000'),
            KeyValue(key='cmd_age_s', value=f'{age:.3f}' if age is not None else 'never'),
        ]
        arr = DiagnosticArray(status=[st])
        arr.header.stamp = self.get_clock().now().to_msg()
        self.pub_status.publish(arr)


def main():
    rclpy.init()
    node = LocomotionStub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
