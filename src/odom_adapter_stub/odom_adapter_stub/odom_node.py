"""odom_adapter_stub / odom_node

STATUS: STUB. Nav2's controller and bt_navigator read the robot's current velocity
from a nav_msgs/Odometry topic, and nothing else in the stack publishes one. This
node builds it from the odom -> base_link TF that state_estimation owns: the pose
is the transform, the velocity is the pose change between two samples, expressed
in the body frame. The real replacement is state estimation publishing /odom
itself (the IEKF or robot_localization), at which point this node goes away.

FREQUENCY: 50 Hz (PUBLISH_RATE_HZ).
INPUT:
  - TF odom -> base_link (from state_estimation)
OUTPUT (this node is the only publisher):
  - nav_msgs/Odometry on /odom, frame_id "odom", child_frame_id "base_link".
    Covariances are left at zero: Nav2 does not use them.
"""
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

from odom_adapter_stub.odom_logic import body_velocity, yaw_from_quaternion

PUBLISH_RATE_HZ = 50.0


class OdomAdapterStub(Node):

    def __init__(self):
        super().__init__('odom_adapter')
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._pub = self.create_publisher(Odometry, '/odom', 10)
        self._prev = None   # (stamp_ns, x, y, yaw) of the last transform used
        self.create_timer(1.0 / PUBLISH_RATE_HZ, self._on_timer)
        self.get_logger().info(
            f'odom_adapter STUB up at {PUBLISH_RATE_HZ} Hz: /odom from TF odom->base_link')

    def _on_timer(self):
        try:
            tf = self._tf_buffer.lookup_transform('odom', 'base_link', Time())
        except TransformException:
            return   # TF not up yet
        t = tf.transform.translation
        q = tf.transform.rotation
        stamp_ns = Time.from_msg(tf.header.stamp).nanoseconds
        x, y, yaw = t.x, t.y, yaw_from_quaternion(q.x, q.y, q.z, q.w)

        if self._prev is None or stamp_ns <= self._prev[0]:
            # first sample, or no new transform since last tick: nothing to difference
            if self._prev is None:
                self._prev = (stamp_ns, x, y, yaw)
            return
        dt = (stamp_ns - self._prev[0]) * 1e-9
        vx, vy, wz = body_velocity(self._prev[1], self._prev[2], self._prev[3], x, y, yaw, dt)
        self._prev = (stamp_ns, x, y, yaw)

        msg = Odometry()
        msg.header.stamp = tf.header.stamp
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link'
        msg.pose.pose.position.x, msg.pose.pose.position.y = x, y
        msg.pose.pose.orientation = q
        msg.twist.twist.linear.x, msg.twist.twist.linear.y = vx, vy
        msg.twist.twist.angular.z = wz
        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = OdomAdapterStub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
