"""Perception bridge for the Mac viewer (node name: `viewer_bridge`). Display only.

Sends what the ROBOT knows, never simulator truth, to tools/viewer.sh's perception panel over UDP
(viewer_host:viewer_port, default host.docker.internal:9872). Everything is in the map frame.

  b"HPER" + zlib(JSON), 10 Hz:
      pose     [x, y, yaw]                  robot's own estimate (TF map -> base_footprint)
      scan     [[x, y], ...]                latest /scan returns
      cloud    [[x, y, z], ...]             /camera/head/points (stereo mode), thinned
      plan     [[x, y], ...]                Nav2's current path (/plan)
      objects  [{label, id, p, s, c}, ...]  object memory (/world/objects)
      status   {state, skill, reply, say}   planner / behavior progress
  b"HMAP" + zlib(JSON header + "\\n" + grid bytes), when /map changes (at most 1 Hz):
      header {w, h, res, ox, oy}; grid = one byte per cell: 0 free, 1 occupied, 2 unknown

The viewer works without this node (just no perception panel); this node works without the viewer.
"""

import json
import math
import socket
import time
import zlib

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan, PointCloud2
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from vision_msgs.msg import Detection3DArray

MAX_SCAN, MAX_CLOUD, MAX_PLAN = 450, 1800, 120
MAX_UDP = 9000     # Docker Desktop drops container -> Mac UDP datagrams above ~9.2 kB (measured: 9000 ok, 9300 lost)


def quat_to_mat(x, y, z, w):
    """Rotation matrix from a quaternion (x, y, z, w)."""
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class ViewerBridge(Node):
    """Node `viewer_bridge`: streams the robot's own view of the world (map, scan, plan, objects, status) to the Mac
    viewer.
    """
    def __init__(self):
        """Resolve the viewer address, set up TF and the subscriptions, start the live and map timers."""
        super().__init__('viewer_bridge')
        self.declare_parameter('viewer_host', 'host.docker.internal')
        self.declare_parameter('viewer_port', 9872)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')
        host = self.get_parameter('viewer_host').value
        port = self.get_parameter('viewer_port').value
        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        try:
            self.addr = (socket.gethostbyname(host), port)
        except OSError as e:
            self.get_logger().warn(f'perception panel off: cannot resolve {host} ({e})')
            self.addr = None
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.scan, self.cloud, self.plan, self.objects = [], [], [], []
        self.status = {'state': '', 'skill': '', 'reply': '', 'say': ''}
        self.map_msg, self.map_sent = None, 0.0

        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, '/map', self.on_map, latched)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, '/camera/head/points', self.on_cloud, qos_profile_sensor_data)
        self.create_subscription(Path, '/plan', self.on_plan, 5)
        self.create_subscription(Detection3DArray, '/world/objects', self.on_objects, 5)
        self.create_subscription(String, '/planner/status', self.on_status, 10)
        self.create_subscription(String, '/planner/reply', lambda m: self.status.update(reply=m.data), 10)
        self.create_subscription(String, '/robot/say', lambda m: self.status.update(say=m.data), 10)
        self.create_timer(0.1, self.send_live)
        self.create_timer(0.25, self.send_map)
        self.get_logger().info(f'perception panel stream -> udp {self.addr}')

    # ------------------------------------------------------------------ helpers
    def to_map(self, frame, stamp=None):
        """(R, t) taking points in `frame` to the map frame, or None."""
        for t in ([Time.from_msg(stamp)] if stamp else []) + [Time()]:
            try:
                tf = self.tf.lookup_transform(self.map_frame, frame, t)
                q, p = tf.transform.rotation, tf.transform.translation
                return quat_to_mat(q.x, q.y, q.z, q.w), np.array([p.x, p.y, p.z])
            except Exception:
                continue
        return None

    def send(self, magic, payload: bytes):
        """Compress and send one packet with its 4-byte tag; packets above the UDP limit are dropped with a warning.
        """
        if self.addr is None:
            return
        data = magic + zlib.compress(payload, 3)
        if len(data) > MAX_UDP:
            if not getattr(self, 'oversize_warned', False):
                self.get_logger().warn(f'{magic.decode()} packet {len(data)} B > {MAX_UDP} B: not sent')
                self.oversize_warned = True
            return
        try:
            self.sock.sendto(data, self.addr)
        except OSError:
            pass                                   # viewer not running: fine

    # ------------------------------------------------------------------ inputs
    def on_map(self, m):
        """Store the latest map."""
        self.map_msg = m

    def on_scan(self, m):
        """Send the scan as map-frame points."""
        T = self.to_map(m.header.frame_id, m.header.stamp)
        if T is None:
            return
        r = np.asarray(m.ranges, np.float32)
        a = m.angle_min + np.arange(len(r)) * m.angle_increment
        ok = np.isfinite(r) & (r >= m.range_min) & (r <= m.range_max)
        r, a = r[ok], a[ok]
        if len(r) > MAX_SCAN:
            k = np.linspace(0, len(r) - 1, MAX_SCAN).astype(int)
            r, a = r[k], a[k]
        pts = np.stack([r * np.cos(a), r * np.sin(a), np.zeros_like(r)], 1)
        R, t = T
        self.scan = np.round((pts @ R.T + t)[:, :2], 2).tolist()

    def on_cloud(self, m):
        """Send a decimated camera point cloud as map-frame points."""
        from sensor_msgs_py import point_cloud2
        T = self.to_map(m.header.frame_id, m.header.stamp)
        if T is None:
            return
        pts = point_cloud2.read_points_numpy(m, field_names=('x', 'y', 'z'), skip_nans=True)
        if len(pts) == 0:
            return
        if len(pts) > MAX_CLOUD:
            pts = pts[np.random.default_rng(0).choice(len(pts), MAX_CLOUD, replace=False)]
        R, t = T
        self.cloud = np.round(pts @ R.T + t, 2).tolist()

    def on_plan(self, m):
        """Keep the Nav2 plan, decimated to at most MAX_PLAN points."""
        p = [[round(s.pose.position.x, 2), round(s.pose.position.y, 2)] for s in m.poses]
        if len(p) > MAX_PLAN:
            p = [p[int(i)] for i in np.linspace(0, len(p) - 1, MAX_PLAN)]
        self.plan, self.plan_t = p, time.monotonic()

    def on_objects(self, m):
        """Keep the object memory in the compact form the viewer draws."""
        self.objects = [{
            'id': d.id, 'label': d.results[0].hypothesis.class_id if d.results else d.id,
            'c': round(float(d.results[0].hypothesis.score), 2) if d.results else 0.0,
            'p': [round(d.bbox.center.position.x, 2), round(d.bbox.center.position.y, 2), round(d.bbox.center.position.z, 2)],
            's': [round(d.bbox.size.x, 2), round(d.bbox.size.y, 2), round(d.bbox.size.z, 2)],
        } for d in m.detections]

    def on_status(self, m):
        """Keep the planner state and the current plan for the viewer overlay."""
        try:
            s = json.loads(m.data)
        except ValueError:
            return
        self.status['state'] = s.get('state', '')
        plan = s.get('plan') or []
        if plan:
            self.status['skill'] = ' -> '.join(f"{k.get('name')}({k.get('target') or k.get('text') or ''})" for k in plan)

    # ------------------------------------------------------------------ outputs
    def send_live(self):
        """Send the live packet: robot pose, plan, objects and status (stale plans are dropped)."""
        T = self.to_map(self.base_frame)
        pose = None
        if T is not None:
            R, t = T
            pose = [round(float(t[0]), 3), round(float(t[1]), 3), round(math.atan2(R[1, 0], R[0, 0]), 3)]
        if self.plan and time.monotonic() - getattr(self, 'plan_t', 0) > 3.0:
            self.plan = []                            # Nav2 stopped publishing a path: goal done
        # One UDP datagram: with a depth camera the point cloud (plus a kitchen's worth of objects) can push it past
        # MAX_UDP, and an oversize datagram is dropped whole, so the panel would never update. Thin the cloud, then
        # the scan, until it fits.
        cloud, scan = self.cloud or [], self.scan or []
        while True:
            msg = {'pose': pose, 'scan': scan, 'cloud': cloud, 'plan': self.plan,
                   'objects': self.objects, 'status': self.status}
            payload = json.dumps(msg, separators=(',', ':')).encode()
            if len(zlib.compress(payload, 3)) + 4 <= MAX_UDP or (not cloud and len(scan) < 50):
                break
            if cloud:
                cloud = cloud[::2] if len(cloud) > 100 else []
            else:
                scan = scan[::2]
        self.send(b'HPER', payload)

    def send_map(self):
        """Send the occupancy map when it changes (and resend it now and then for a viewer that started late)."""
        # a new map at most once a second; the last one again every 3 s for viewers opened later
        m, now = self.map_msg, time.monotonic()
        if m is None:
            if getattr(self, 'map_payload', None) and now - self.map_sent > 3.0:
                self.map_sent = now
                self.send(b'HMAP', self.map_payload)
            return
        if now - self.map_sent < 1.0:
            return
        self.map_msg, self.map_sent = None, now
        g = np.asarray(m.data, np.int16).reshape(m.info.height, m.info.width)
        cells = np.full(g.shape, 2, np.uint8)
        cells[(g >= 0) & (g < 50)] = 0
        cells[g >= 50] = 1
        res, step = m.info.resolution, 1
        while True:                                   # thin very large SLAM maps to fit one datagram
            c = cells[::step, ::step]
            head = {'w': int(c.shape[1]), 'h': int(c.shape[0]), 'res': res * step,
                    'ox': m.info.origin.position.x, 'oy': m.info.origin.position.y}
            payload = json.dumps(head).encode() + b'\n' + c.tobytes()
            if len(zlib.compress(payload, 3)) < MAX_UDP - 8 or step >= 8:
                break
            step *= 2
        self.map_payload = payload
        self.send(b'HMAP', payload)


def main():
    """Entry point: spin the node until Ctrl-C."""
    rclpy.init()
    node = ViewerBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
