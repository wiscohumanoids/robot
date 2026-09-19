"""state_estimation_stub / state_estimation_node

STATUS: STUB. Real interface, fake internals: it does not estimate anything, it
dead-reckons the commanded /cmd_vel (assuming the robot tracks it perfectly)
starting from a fixed initial pose. That is enough for navigation and the
task layer to develop against a /robot_pose that moves when they command it.
The real replacement is the IEKF (own repo) or robot_localization.

FREQUENCY: 100 Hz (PUBLISH_RATE_HZ).

INPUT:
  - geometry_msgs/Twist on /cmd_vel (from cmd_vel_mux)

OUTPUTS (this node is the ONLY publisher of both):
  - geometry_msgs/PoseStamped on /robot_pose, frame_id "map"
  - TF edge  odom -> base_link
    (map -> odom is owned by slam_stub and is identity, so the pose of
     base_link in `map` equals its pose in `odom` today.)
See INTERFACE_CONTRACT.md for the full TF ownership table.
"""
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from rclpy.node import Node
from tf2_ros import TransformBroadcaster

from state_estimation_stub.kinematics import integrate_body_velocity, yaw_to_quaternion

PUBLISH_RATE_HZ = 100.0


class StateEstimationStub(Node):

    def __init__(self):
        super().__init__('state_estimation')
        self.declare_parameter('initial_x', 0.0)
        self.declare_parameter('initial_y', 0.0)
        self.declare_parameter('initial_yaw', 0.0)
        self._x = float(self.get_parameter('initial_x').value)
        self._y = float(self.get_parameter('initial_y').value)
        self._yaw = float(self.get_parameter('initial_yaw').value)
        self._cmd = (0.0, 0.0, 0.0)

        self.create_subscription(Twist, '/cmd_vel', self._on_cmd, 10)
        self._pose_pub = self.create_publisher(PoseStamped, '/robot_pose', 10)
        self._tf = TransformBroadcaster(self)
        self.create_timer(1.0 / PUBLISH_RATE_HZ, self._on_timer)
        self.get_logger().info(
            f'state_estimation STUB up at {PUBLISH_RATE_HZ} Hz: dead-reckoning /cmd_vel '
            f'from ({self._x}, {self._y}, yaw={self._yaw}); publishing /robot_pose + odom->base_link')

    def _on_cmd(self, msg: Twist):
        self._cmd = (msg.linear.x, msg.linear.y, msg.angular.z)

    def _on_timer(self):
        dt = 1.0 / PUBLISH_RATE_HZ
        self._x, self._y, self._yaw = integrate_body_velocity(
            self._x, self._y, self._yaw, *self._cmd, dt)
        qx, qy, qz, qw = yaw_to_quaternion(self._yaw)
        stamp = self.get_clock().now().to_msg()

        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = 'map'
        pose.pose.position.x = self._x
        pose.pose.position.y = self._y
        pose.pose.orientation.x, pose.pose.orientation.y = qx, qy
        pose.pose.orientation.z, pose.pose.orientation.w = qz, qw
        self._pose_pub.publish(pose)

        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = 'odom'
        tf.child_frame_id = 'base_link'
        tf.transform.translation.x = self._x
        tf.transform.translation.y = self._y
        tf.transform.rotation.x, tf.transform.rotation.y = qx, qy
        tf.transform.rotation.z, tf.transform.rotation.w = qz, qw
        self._tf.sendTransform(tf)


def main(args=None):
    rclpy.init(args=args)
    node = StateEstimationStub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
