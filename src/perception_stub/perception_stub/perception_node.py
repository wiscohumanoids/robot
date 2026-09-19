"""perception_stub / perception_node

STATUS: STUB. The interface is real; the content is fake. There is no camera and
no detector: it reports one cube at a fixed position, forever. The real
replacement is ORB-SLAM3 + FoundationPose (or similar) wired into ROS,
publishing the same ObjectPoseArray.

FREQUENCY: 30 Hz (PUBLISH_RATE_HZ).

OUTPUT (this node is the only publisher):
  humanoid_interfaces/ObjectPoseArray on /object_poses, header.frame_id "map"
  (perception is responsible for transforming into `map`; consumers never
  need TF for this).

PARAMETERS (read once at startup): cube_x, cube_y, cube_z in metres, in `map`.
The defaults put a cube on a table 1.5 m ahead and 0.3 m left of the start pose.
"""
import rclpy
from rclpy.node import Node

from humanoid_interfaces.msg import ObjectPose, ObjectPoseArray

PUBLISH_RATE_HZ = 30.0


class PerceptionStub(Node):

    def __init__(self):
        super().__init__('perception')
        self.declare_parameter('cube_x', 1.5)
        self.declare_parameter('cube_y', 0.3)
        self.declare_parameter('cube_z', 0.8)
        cube = ObjectPose()
        cube.object_id = 'cube_1'
        cube.label = 'cube'
        cube.pose.position.x = float(self.get_parameter('cube_x').value)
        cube.pose.position.y = float(self.get_parameter('cube_y').value)
        cube.pose.position.z = float(self.get_parameter('cube_z').value)
        cube.pose.orientation.w = 1.0
        cube.confidence = 1.0

        # The scene never changes, so build the message once and only re-stamp it.
        self._msg = ObjectPoseArray()
        self._msg.header.frame_id = 'map'
        self._msg.objects = [cube]

        self._pub = self.create_publisher(ObjectPoseArray, '/object_poses', 10)
        self.create_timer(1.0 / PUBLISH_RATE_HZ, self._on_timer)
        pos = cube.pose.position
        self.get_logger().info(
            f'perception stub up at {PUBLISH_RATE_HZ} Hz: cube_1 at ({pos.x}, {pos.y}, {pos.z}) in map')

    def _on_timer(self):
        self._msg.header.stamp = self.get_clock().now().to_msg()
        self._pub.publish(self._msg)


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionStub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
