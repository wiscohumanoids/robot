"""PERCEPTION CLIENT (node name: `perception`): real open-vocabulary detection on the robot's camera.

Sends the ZED X head camera's RGB-D frames to the perception server on the GPU VM (Grounding DINO
finds the named objects, SAM 2 cuts out their pixels, the depth under each mask gives the 3D position)
and publishes what comes back on /perception/detections, exactly like the line-of-sight stub did
(camera-frame centres, boxes axis-aligned with the map), so world_model, the planner and the behavior
tree are unchanged.

  camera  /camera/head/color/image_raw (rgb8), /camera/head/depth/image_raw (32FC1 m), .../color/camera_info
          (sensors:=stereo); frame head_camera_optical
  server  POST {server_url}/detect  (VM 127.0.0.1:8100, reached through `ssh -N -L 8100:localhost:8100 hsim-gpu`;
          from the Docker container the Mac is host.docker.internal)
  out     /perception/detections  vision_msgs/Detection3DArray (frame head_camera_optical, image stamp)
          /perception/status      std_msgs/String JSON (mode, latency, detections, failures), 1 Hz

What it asks for: DEFAULT_VOCAB + the object labels of the current environment (the kinds of things in
the room, not where they are; vocab_from_env) + every object the planner's current plan names
(/planner/status). A new plan target triggers an immediate request, so find_object(mug) doesn't wait. The
current plan's objects are also sent as `priority_prompts`; the server's default (priority-mode none) just
treats them as prompts, because extra single-name passes hallucinated objects in the bench.

If the server is unreachable (max_failures in a row) and fallback:=stub, it falls back to the
line-of-sight stub (privileged positions) and keeps probing /health every retry_s to switch back.
"""

import json
import threading
import time

import numpy as np
import rclpy
import yaml
from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

from . import server_link as sl
from .perception_stub import PerceptionStub


def quat_to_mat(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class PerceptionClient(PerceptionStub):
    def __init__(self):
        super().__init__()                    # stub: params world/objects/..., GT subscription, fallback tick
        for name, default in (('server_url', 'http://host.docker.internal:8100'), ('request_hz', 1.5),
                              ('timeout_s', 3.0), ('min_score', 0.3), ('fallback', 'stub'), ('max_failures', 3),
                              ('retry_s', 5.0), ('box_threshold', -1.0), ('text_threshold', -1.0),
                              ('jpeg_quality', 90), ('depth_downsample', 2), ('vocab_from_env', True),
                              ('extra_prompts', ['']), ('map_frame', 'map')):
            self.declare_parameter(name, default)
        g = lambda n: self.get_parameter(n).value
        self.url = g('server_url').rstrip('/')
        self.period = 1.0 / max(g('request_hz'), 0.1)
        self.timeout = g('timeout_s')
        self.min_score = g('min_score')
        self.fallback = g('fallback')
        self.max_failures = g('max_failures')
        self.retry_s = g('retry_s')
        bt, tt = g('box_threshold'), g('text_threshold')
        self.thresholds = (bt if bt > 0 else None, tt if tt > 0 else None)
        self.jpeg_q, self.ds = g('jpeg_quality'), g('depth_downsample')
        self.map_frame = g('map_frame')

        vocab = list(sl.DEFAULT_VOCAB)
        if g('vocab_from_env') and self.get_parameter('objects').value:
            vocab += sl.env_vocab(yaml.safe_load(open(self.get_parameter('objects').value)))
        vocab += [p for p in g('extra_prompts') if p]
        self.base_vocab = list(dict.fromkeys(vocab))
        self.plan_vocab = []
        self.priority = []

        self.lock = threading.Lock()
        self.rgb, self.depth, self.K = {}, {}, None       # stamp(ns) -> msg; latest camera intrinsics
        self.last_done = -1
        self.mode = 'server'                              # 'server' | 'fallback'
        self.failures, self.last_probe = 0, 0.0
        self.stats = {'latency_ms': None, 'server_ms': None, 'detections': 0, 'requests': 0, 'errors': 0, 'last_error': ''}
        self.wake = threading.Event()

        self.tf2 = Buffer()
        self.tf2_listener = TransformListener(self.tf2, self)
        self.pub_status = self.create_publisher(String, '/perception/status', 5)
        self.create_subscription(Image, '/camera/head/color/image_raw', lambda m: self.on_image(m, self.rgb), 2)
        self.create_subscription(Image, '/camera/head/depth/image_raw', lambda m: self.on_image(m, self.depth), 2)
        self.create_subscription(CameraInfo, '/camera/head/color/camera_info', self.on_info, 2)
        self.create_subscription(String, '/planner/status', self.on_planner, 10)
        self.create_timer(1.0, self.publish_status)
        self.worker = threading.Thread(target=self.loop, daemon=True)
        self.worker.start()
        self.get_logger().info(f'perception CLIENT: server {self.url}, {len(self.base_vocab)} prompts, '
                               f'{1 / self.period:.1f} Hz, fallback={self.fallback}')

    # ------------------------------------------------------------------ inputs
    def on_image(self, msg, store):
        key = Time.from_msg(msg.header.stamp).nanoseconds
        with self.lock:
            store[key] = msg
            for k in sorted(store)[:-4]:
                del store[k]

    def on_info(self, msg):
        k = msg.k
        self.K = {'fx': k[0], 'fy': k[4], 'cx': k[2], 'cy': k[5], 'width': msg.width, 'height': msg.height}

    def on_planner(self, msg):
        names = sl.planner_targets(msg.data)
        new = [n for n in names if n not in self.base_vocab and n not in self.plan_vocab]
        if names:
            self.plan_vocab = list(dict.fromkeys(self.plan_vocab + names))[-20:]
            self.priority = names[:4]               # the current plan's objects get their own detector pass
        if new:
            self.get_logger().info(f'planner asked about {new}: looking now')
            self.wake.set()

    # ------------------------------------------------------------------ fallback
    def tick(self):                                       # stub timer: only publishes in fallback mode
        if self.mode == 'fallback':
            super().tick()

    # ------------------------------------------------------------------ server loop
    def latest_pair(self):
        with self.lock:
            common = sorted(set(self.rgb) & set(self.depth))
            if not common or common[-1] == self.last_done:
                return None
            k = common[-1]
            return k, self.rgb[k], self.depth[k]

    def camera_pose(self, frame, stamp):
        try:
            tf = self.tf2.lookup_transform(self.map_frame, frame, Time.from_msg(stamp), Duration(seconds=0.2))
        except Exception:
            try:
                tf = self.tf2.lookup_transform(self.map_frame, frame, Time())
            except Exception:
                return None
        q, p = tf.transform.rotation, tf.transform.translation
        T = np.eye(4)
        T[:3, :3] = quat_to_mat(q.x, q.y, q.z, q.w)
        T[:3, 3] = [p.x, p.y, p.z]
        return T

    def loop(self):
        while rclpy.ok():
            self.wake.wait(self.period)
            self.wake.clear()
            if self.mode == 'fallback':
                if time.monotonic() - self.last_probe > self.retry_s:
                    self.last_probe = time.monotonic()
                    try:
                        if sl.get_json(self.url + '/health', 2.0).get('status') == 'ok':
                            self.mode, self.failures = 'server', 0
                            self.get_logger().info('perception server is back: using real detections')
                    except Exception:
                        pass
                continue
            pair = self.latest_pair()
            if pair is None or self.K is None:
                continue
            key, rgb_msg, dep_msg = pair
            try:
                self.process(key, rgb_msg, dep_msg)
                self.failures = 0
            except Exception as e:                        # timeouts, connection refused, bad replies
                self.failures += 1
                self.stats['errors'] += 1
                self.stats['last_error'] = f'{type(e).__name__}: {e}'[:200]
                if self.failures == 1 or self.failures == self.max_failures:
                    self.get_logger().warn(f'perception server request failed ({self.failures}): {self.stats["last_error"]}')
                if self.failures >= self.max_failures and self.fallback == 'stub':
                    self.mode = 'fallback'
                    self.last_probe = time.monotonic()
                    self.get_logger().warn('perception server unreachable: falling back to the line-of-sight stub')

    def process(self, key, rgb_msg, dep_msg):
        rgb = sl.image_to_numpy(rgb_msg)
        depth = sl.image_to_numpy(dep_msg)
        T = self.camera_pose(rgb_msg.header.frame_id, rgb_msg.header.stamp)
        prompts = list(dict.fromkeys(self.base_vocab + self.plan_vocab))
        stamp = Time.from_msg(rgb_msg.header.stamp).nanoseconds * 1e-9
        req = sl.build_request(rgb, depth, self.K, T, prompts, rgb_msg.header.frame_id, stamp, self.jpeg_q, self.ds,
                               *self.thresholds, priority=self.priority)
        t0 = time.monotonic()
        res = sl.post_json(self.url + '/detect', req, self.timeout)
        self.stats['latency_ms'] = round((time.monotonic() - t0) * 1000, 1)
        self.stats['server_ms'] = res.get('timing_ms', {}).get('total')
        self.stats['requests'] += 1
        self.last_done = key
        out = Detection3DArray()
        out.header = rgb_msg.header
        for label, score, c, q, size in sl.detections_to_boxes(res.get('detections', []), T, self.min_score):
            det = Detection3D()
            det.header = out.header
            det.bbox.center.position.x, det.bbox.center.position.y, det.bbox.center.position.z = c
            (det.bbox.center.orientation.x, det.bbox.center.orientation.y,
             det.bbox.center.orientation.z, det.bbox.center.orientation.w) = map(float, q)
            det.bbox.size.x, det.bbox.size.y, det.bbox.size.z = size
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = label
            hyp.hypothesis.score = score
            hyp.pose.pose = det.bbox.center
            det.results = [hyp]
            out.detections.append(det)
        self.stats['detections'] = len(out.detections)
        self.pub.publish(out)

    def publish_status(self):
        self.pub_status.publish(String(data=json.dumps({'mode': self.mode, 'server': self.url, 'failures': self.failures,
                                                        'prompts': len(self.base_vocab) + len(self.plan_vocab), **self.stats})))


def main():
    rclpy.init()
    node = PerceptionClient()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
