"""World model (node name: `world_model`): the robot's memory of objects it has seen.

Input:  /perception/detections (camera-frame detections: label, position, size, score; no IDs)
        TF map <- head_camera, i.e. the robot's OWN estimate of where it is (localization)
Output: /world/objects  vision_msgs/Detection3DArray in the map frame (id = track id), 2 Hz
        /world/summary  std_msgs/String, JSON for the planner: objects, relations, robot pose

What it does, like a real robot's semantic memory:
  - places each detection on the map using the robot's estimated pose (so localization
    errors show up as object position errors)
  - associates repeated sightings to the same object (same label, close enough) and
    averages position/size; confidence grows with sightings
  - remembers when each object was last seen; never invents objects it has not seen
  - derives relations from geometry: "red block is on table"
  - multi-view confirmation (confirm_views, default 1 = off): with real detectors (perception:=server, launch sets 2)
    an object is committed only after consistent sightings (same label, within the association gate) from
    `confirm_views` distinct views. A sighting is a new view if the camera moved >= min_view_move m or turned
    >= min_view_turn_deg since the track's last view (and >= min_view_dt later), or after static_view_dt s of
    persistence from the same spot (a standing robot staring at an object). Consecutive frames from one viewpoint
    repeat the same mistakes, so time alone is not a view. Tentative tracks are not published and expire after
    tentative_timeout_s without a sighting, so a one-frame hallucination never reaches the planner.
    /world/summary objects carry `views`; the summary also counts `tentative` tracks.
"""

import json
import math
import time

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose


def quat_to_mat(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Track:
    def __init__(self, tid, label, pos, size, score, t):
        self.id, self.label = tid, label          # id None while tentative (assigned at confirmation)
        self.pos, self.size = pos, size
        self.confidence = 0.5 * score
        self.n, self.first_seen, self.last_seen = 1, t, t
        self.views, self.last_view = 1, t         # distinct views (see the module docstring)
        self.view_cam = None                      # camera (position, forward axis) at the last counted view

    def update(self, pos, size, score, t, alpha=0.3, cam=None, view_rule=None):
        self.pos = (1 - alpha) * self.pos + alpha * pos
        self.size = (1 - alpha) * self.size + alpha * size
        self.confidence = min(0.99, 1 - (1 - self.confidence) * (1 - 0.5 * score))
        self.n += 1
        self.last_seen = t
        if view_rule is None or view_rule(self, t, cam):
            self.views += 1
            self.last_view, self.view_cam = t, cam


class WorldModel(Node):
    def __init__(self):
        super().__init__('world_model')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('assoc_distance', 0.4)
        self.declare_parameter('confirm_views', 1)          # 1 = publish on first sighting (stub); 2+ for real detectors
        self.declare_parameter('min_view_dt', 0.3)
        self.declare_parameter('min_view_move', 0.1)
        self.declare_parameter('min_view_turn_deg', 5.0)
        self.declare_parameter('static_view_dt', 2.0)
        self.declare_parameter('tentative_timeout_s', 15.0)
        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.assoc = self.get_parameter('assoc_distance').value
        self.confirm_views = int(self.get_parameter('confirm_views').value)
        self.min_view_dt = self.get_parameter('min_view_dt').value
        self.min_view_move = self.get_parameter('min_view_move').value
        self.min_view_cos = math.cos(math.radians(self.get_parameter('min_view_turn_deg').value))
        self.static_view_dt = self.get_parameter('static_view_dt').value
        self.tentative_timeout = self.get_parameter('tentative_timeout_s').value
        self.tracks: list[Track] = []
        self.counter = {}
        self.tf = Buffer()
        # TF arrives on its own executor thread (see main) while detection callbacks wait for it
        self.tf_listener = TransformListener(self.tf, self)
        self.work = MutuallyExclusiveCallbackGroup()
        self.pub_objects = self.create_publisher(Detection3DArray, '/world/objects', 5)
        self.pub_summary = self.create_publisher(String, '/world/summary', 5)
        self.create_subscription(Detection3DArray, '/perception/detections', self.on_detections, 10,
                                 callback_group=self.work)
        self.create_timer(0.5, self.publish, callback_group=self.work)
        self.get_logger().info('world model: tracking objects seen by perception, in the map frame')

    def to_map(self, frame, stamp):
        try:
            tf = self.tf.lookup_transform(self.map_frame, frame, Time.from_msg(stamp), Duration(seconds=0.5))
        except Exception:
            # Only accept the latest pose if it is from (almost) the same moment: a stale pose
            # during a turn would fling the object across the room.
            try:
                tf = self.tf.lookup_transform(self.map_frame, frame, Time())
            except Exception:
                return None
            t_tf = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
            self.last_gap = t_tf - (stamp.sec + stamp.nanosec * 1e-9)
            if abs(self.last_gap) > 0.1:
                self.dropped = getattr(self, 'dropped', 0) + 1
                return None
        t, q = tf.transform.translation, tf.transform.rotation
        return np.array([t.x, t.y, t.z]), quat_to_mat(q.x, q.y, q.z, q.w)

    def on_detections(self, msg: Detection3DArray):
        if not msg.detections:
            return
        T = self.to_map(msg.header.frame_id, msg.header.stamp)
        if T is None:
            return                                   # not localized yet: can't place anything
        p_mc, R_mc = T
        self.placed = getattr(self, 'placed', 0) + 1
        now = self.get_clock().now().nanoseconds * 1e-9
        for det in msg.detections:
            c = det.bbox.center.position
            pos = p_mc + R_mc @ np.array([c.x, c.y, c.z])
            q = det.bbox.center.orientation
            R_box = R_mc @ quat_to_mat(q.x, q.y, q.z, q.w)   # box axes in the map frame
            size = np.abs(R_box @ np.array([det.bbox.size.x, det.bbox.size.y, det.bbox.size.z]))
            label = det.results[0].hypothesis.class_id
            score = det.results[0].hypothesis.score
            gate = max(self.assoc, 0.5 * float(np.max(size[:2])))
            best, best_d = None, gate
            for tr in self.tracks:
                if tr.label != label:
                    continue
                dd = float(np.linalg.norm(tr.pos[:2] - pos[:2]))
                if dd < best_d:
                    best, best_d = tr, dd
            cam = (p_mc.copy(), R_mc[:, 2].copy())        # camera position and optical (forward) axis in the map
            if best:
                best.update(pos, size, score, now, cam=cam, view_rule=self.new_view)
            else:
                best = Track(None, label, pos, size, score, now)
                best.view_cam = cam
                self.tracks.append(best)
            if best.id is None and best.views >= self.confirm_views:
                self.confirm(best)

    def new_view(self, tr, t, cam):
        dt = t - tr.last_view
        if dt < self.min_view_dt:
            return False
        if dt >= self.static_view_dt or tr.view_cam is None or cam is None:
            return True
        moved = float(np.linalg.norm(cam[0] - tr.view_cam[0]))
        turned = float(np.dot(cam[1], tr.view_cam[1]))     # cos of the angle between forward axes
        return moved >= self.min_view_move or turned <= self.min_view_cos

    def confirm(self, tr):
        k = self.counter.get(tr.label, 0) + 1
        self.counter[tr.label] = k
        tr.id = f"{tr.label.replace(' ', '_')}_{k}"
        extra = f' after {tr.views} views' if self.confirm_views > 1 else ''
        self.get_logger().info(f'new object: {tr.id} at ({tr.pos[0]:.2f}, {tr.pos[1]:.2f}, {tr.pos[2]:.2f}){extra}')

    def expire_tentative(self, now):
        if self.confirm_views <= 1:
            return
        self.tracks = [tr for tr in self.tracks if tr.id is not None or now - tr.last_seen < self.tentative_timeout]

    def relations(self):
        rel = {}
        conf = [tr for tr in self.tracks if tr.id is not None]      # relations only between committed objects
        for a in conf:
            a_bottom = a.pos[2] - a.size[2] / 2
            for b in conf:
                if a is b:
                    continue
                b_top = b.pos[2] + b.size[2] / 2
                inside = all(abs(a.pos[i] - b.pos[i]) <= b.size[i] / 2 + 0.05 for i in (0, 1))
                if inside and abs(a_bottom - b_top) < 0.08 and a.size[2] < b.size[2] * 2:
                    rel.setdefault(a.id, []).append(b.id)
        return rel

    def robot_pose(self):
        try:
            tf = self.tf.lookup_transform(self.map_frame, self.base_frame, Time())
        except Exception:
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return {'x': round(t.x, 2), 'y': round(t.y, 2), 'yaw_deg': round(math.degrees(yaw))}

    def merge_duplicates(self):
        """Two tracks with the same label that converged onto the same spot are one object."""
        merged = True
        while merged:
            merged = False
            for i, a in enumerate(self.tracks):
                for b in self.tracks[i + 1:]:
                    gate = max(self.assoc, 0.5 * float(max(a.size[0], a.size[1], b.size[0], b.size[1])))
                    if a.label == b.label and np.linalg.norm(a.pos[:2] - b.pos[:2]) < gate:
                        # a confirmed track always absorbs a tentative one (keeps its id)
                        keep, drop = (a, b) if (a.id is not None, a.n) >= (b.id is not None, b.n) else (b, a)
                        w = drop.n / (keep.n + drop.n)
                        keep.pos = (1 - w) * keep.pos + w * drop.pos
                        keep.n += drop.n
                        keep.views += drop.views
                        keep.last_seen = max(keep.last_seen, drop.last_seen)
                        keep.confidence = max(keep.confidence, drop.confidence)
                        self.tracks.remove(drop)
                        if keep.id is None and keep.views >= self.confirm_views:
                            self.confirm(keep)
                        merged = True
                        break
                if merged:
                    break

    def publish(self):
        self.merge_duplicates()
        self.expire_tentative(self.get_clock().now().nanoseconds * 1e-9)
        confirmed = [tr for tr in self.tracks if tr.id is not None]
        self.get_logger().info(f"detections placed {getattr(self, 'placed', 0)}, dropped {getattr(self, 'dropped', 0)} "
                               f"(last pose/detection time gap {getattr(self, 'last_gap', 0.0):+.2f} s)",
                               throttle_duration_sec=5.0)
        now = self.get_clock().now()
        arr = Detection3DArray()
        arr.header.stamp = now.to_msg()
        arr.header.frame_id = self.map_frame
        for tr in confirmed:
            d = Detection3D()
            d.header = arr.header
            d.id = tr.id
            d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.center.position.z = map(float, tr.pos)
            d.bbox.center.orientation.w = 1.0
            d.bbox.size.x, d.bbox.size.y, d.bbox.size.z = map(float, tr.size)
            h = ObjectHypothesisWithPose()
            h.hypothesis.class_id, h.hypothesis.score = tr.label, float(tr.confidence)
            h.pose.pose = d.bbox.center
            d.results = [h]
            arr.detections.append(d)
        self.pub_objects.publish(arr)

        t = now.nanoseconds * 1e-9
        rel = self.relations()
        summary = {
            'frame': self.map_frame,
            'robot': self.robot_pose(),
            'objects': [{
                'id': tr.id, 'label': tr.label,
                'position': [round(float(v), 2) for v in tr.pos],
                'size': [round(float(v), 2) for v in tr.size],
                'confidence': round(tr.confidence, 2),
                'seen_ago_s': round(t - tr.last_seen, 1),
                'times_seen': tr.n,
                'views': tr.views,
                'on': rel.get(tr.id, []),
            } for tr in confirmed],
            'tentative': len(self.tracks) - len(confirmed),
        }
        self.pub_summary.publish(String(data=json.dumps(summary)))


def main():
    rclpy.init()
    node = WorldModel()
    ex = MultiThreadedExecutor(num_threads=3)     # TF listener + detection/publish work in parallel
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
