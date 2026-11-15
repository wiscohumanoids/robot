"""STUB state estimator (node name: `state_estimation`).

Stands in for the IEKF (Phase 3). Copies the sim's ground-truth pose to
/odometry/filtered and broadcasts the odom -> base_footprint TF (2D), which this node owns.
"""

import math

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


class StateEstimationStub(Node):
    def __init__(self):
        super().__init__('state_estimation')
        self.pub = self.create_publisher(Odometry, '/odometry/filtered', 10)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(Odometry, '/sim/ground_truth/odom', self.on_truth, 10)
        self.get_logger().info('STUB state estimation: passing through ground truth')

    def on_truth(self, msg: Odometry):
        self.n = getattr(self, 'n', 0) + 1
        if self.n % 4:                 # 50 Hz, like the real EKF
            return
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_footprint'
        self.pub.publish(msg)

        t = TransformStamped()
        t.header = msg.header
        t.child_frame_id = 'base_footprint'
        p = msg.pose.pose.position
        t.transform.translation.x, t.transform.translation.y = p.x, p.y
        o = msg.pose.pose.orientation
        yaw = math.atan2(2 * (o.w * o.z + o.x * o.y), 1 - 2 * (o.y * o.y + o.z * o.z))
        t.transform.rotation.z, t.transform.rotation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self.tf.sendTransform(t)


def main():
    rclpy.init()
    node = StateEstimationStub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
