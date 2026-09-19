"""slam_stub / slam_node

STATUS: STUB. The interface is real; the content is fake. It publishes a static,
empty (all free) map and an identity map -> odom transform: no drift, and the map
origin is where the robot started. The real replacement is ORB-SLAM3 (or
slam_toolbox) wired into ROS.

OUTPUTS (this node is the only publisher of both):
  - nav_msgs/OccupancyGrid on /map, QoS transient_local (late joiners such as
    Nav2 get the latest map), re-sent every MAP_REPUBLISH_S seconds
  - TF edge  map -> odom  at TF_RATE_HZ
See INTERFACE_CONTRACT.md for the full TF ownership table.
"""
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import TransformBroadcaster

TF_RATE_HZ = 20.0
MAP_REPUBLISH_S = 5.0
MAP_SIZE_CELLS = 200      # 200 x 200 cells
MAP_RESOLUTION = 0.05     # m/cell -> a 10 m x 10 m arena
MAP_ORIGIN = -5.0         # m; map spans [-5, 5] on both axes


class SlamStub(Node):

    def __init__(self):
        super().__init__('slam')
        map_qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._map_pub = self.create_publisher(OccupancyGrid, '/map', map_qos)
        self._grid = self._build_grid()   # built once; only the stamp changes on re-send
        self._tf = TransformBroadcaster(self)
        self.create_timer(1.0 / TF_RATE_HZ, self._on_tf_timer)
        self.create_timer(MAP_REPUBLISH_S, self._publish_map)
        self._publish_map()
        self.get_logger().info(
            'slam STUB up: empty 10m x 10m /map + identity map->odom '
            f'({TF_RATE_HZ} Hz TF)')

    @staticmethod
    def _build_grid():
        grid = OccupancyGrid()
        grid.header.frame_id = 'map'
        grid.info.resolution = MAP_RESOLUTION
        grid.info.width = MAP_SIZE_CELLS
        grid.info.height = MAP_SIZE_CELLS
        grid.info.origin.position.x = MAP_ORIGIN
        grid.info.origin.position.y = MAP_ORIGIN
        grid.info.origin.orientation.w = 1.0
        grid.data = [0] * (MAP_SIZE_CELLS * MAP_SIZE_CELLS)  # 0 = free
        return grid

    def _publish_map(self):
        self._grid.header.stamp = self.get_clock().now().to_msg()
        self._map_pub.publish(self._grid)

    def _on_tf_timer(self):
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = 'map'
        tf.child_frame_id = 'odom'
        tf.transform.rotation.w = 1.0
        self._tf.sendTransform(tf)


def main(args=None):
    rclpy.init(args=args)
    node = SlamStub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
