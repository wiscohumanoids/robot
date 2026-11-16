"""LINE-OF-SIGHT PERCEPTION STUB (node name: `perception`). Replaced by real detectors in Phase 4.

Stands in for "camera + open-vocabulary detector + pose estimator" (Grounded-SAM +
FoundationPose). It is honest about WHAT the robot can see, and privileged about WHERE:

  privileged input   /sim/ground_truth/state (true world state) + the world model file
  honest output      only objects that are inside the head camera's field of view, within
                     range, large enough to detect at that distance, and not hidden behind
                     anything (checked with rays). Positions are relative to the camera with
                     distance-dependent noise. No identities: "a red block here", not "object 3".

Publishes /perception/detections (vision_msgs/Detection3DArray, frame head_camera, ~5 Hz).
The bounding box is axis-aligned with the world (its orientation is given in the camera frame).
"""

import math

import mujoco
import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

from builtin_interfaces.msg import Time

PREFIX = 'robot/'
# Head camera on the torso, about where the real G1's head camera sits, tilted down
CAMERA_MOUNT = np.array([0.08, 0.0, 0.45])
CAMERA_PITCH = math.radians(25.0)


def to_time(t):
    sec, nsec = divmod(int(round(t * 1e9)), 1_000_000_000)
    return Time(sec=sec, nanosec=nsec)


def mat_to_quat(R):
    """Rotation matrix -> quaternion (x, y, z, w)."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.ravel())          # (w, x, y, z)
    return q[1], q[2], q[3], q[0]


class PerceptionStub(Node):
    def __init__(self):
        super().__init__('perception')
        self.declare_parameter('world', '')
        self.declare_parameter('objects', '')
        self.declare_parameter('rate_hz', 5.0)
        self.declare_parameter('fov_h_deg', 87.0)
        self.declare_parameter('fov_v_deg', 58.0)
        self.declare_parameter('max_range', 6.0)
        self.declare_parameter('min_angular_size_deg', 1.2)   # smaller than this: too few pixels
        self.declare_parameter('seed', 1)
        # robot profile (hsim_bringup/config/robots/<robot>.yaml): torso body, camera position in it, pitch (+ = down)
        self.declare_parameter('torso_body', 'torso_link')
        self.declare_parameter('camera_mount', CAMERA_MOUNT.tolist())
        self.declare_parameter('camera_pitch', CAMERA_PITCH)

        self.m = mujoco.MjModel.from_binary_path(self.get_parameter('world').value)
        self.d = mujoco.MjData(self.m)
        for g in range(self.m.ngeom):                           # the camera never sees the robot itself
            if self.m.body(self.m.geom_bodyid[g]).name.startswith(PREFIX):
                self.m.geom_group[g] = 4 if self.m.geom_group[g] <= 2 else 5
        self.groups = np.array([1, 1, 1, 1, 0, 0], np.uint8)
        self.torso = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY,
                                       PREFIX + self.get_parameter('torso_body').value)
        self.cam_mount = np.array(self.get_parameter('camera_mount').value, dtype=float)
        self.cam_pitch = float(self.get_parameter('camera_pitch').value)
        self.rng = np.random.default_rng(self.get_parameter('seed').value)
        self.fov_h = math.radians(self.get_parameter('fov_h_deg').value) / 2
        self.fov_v = math.radians(self.get_parameter('fov_v_deg').value) / 2
        self.max_range = self.get_parameter('max_range').value
        self.min_ang = math.radians(self.get_parameter('min_angular_size_deg').value)

        objs = yaml.safe_load(open(self.get_parameter('objects').value))['objects']
        self.objects = []
        for o in objs:
            if not o.get('detectable', True):
                continue
            bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, o['body'])
            if bid < 0:
                continue
            subtree = {bid}
            for b in range(bid + 1, self.m.nbody):
                if self.m.body_parentid[b] in subtree:
                    subtree.add(b)
            geoms = [g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] in subtree]
            self.objects.append({'label': o['label'], 'bodies': subtree, 'geoms': geoms})
        self.state = None
        self.gid = np.zeros(1, np.int32)

        self.pub = self.create_publisher(Detection3DArray, '/perception/detections', 5)
        self.create_subscription(Float64MultiArray, '/sim/ground_truth/state', self.on_state, 5)
        self.create_timer(1.0 / self.get_parameter('rate_hz').value, self.tick)
        self.get_logger().info(f'STUB perception (line-of-sight): {len(self.objects)} detectable objects, '
                               f'FOV {2 * math.degrees(self.fov_h):.0f}x{2 * math.degrees(self.fov_v):.0f} deg, '
                               f'range {self.max_range} m')

    def on_state(self, msg):
        self.state = msg.data

    def aabb(self, geoms):
        corners = []
        for g in geoms:
            c, half = self.m.geom_aabb[g, :3], self.m.geom_aabb[g, 3:]
            R, p = self.d.geom_xmat[g].reshape(3, 3), self.d.geom_xpos[g]
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        corners.append(p + R @ (c + half * np.array([sx, sy, sz])))
        corners = np.array(corners)
        return corners.min(0), corners.max(0)

    def visible(self, cam, point, obj):
        v = point - cam
        dist = np.linalg.norm(v)
        hit = mujoco.mj_ray(self.m, self.d, cam, v / dist, self.groups, 1, -1, self.gid)
        if hit < 0:
            return False
        return self.m.geom_bodyid[self.gid[0]] in obj['bodies'] or hit >= dist - 0.02

    def tick(self):
        if self.state is None:
            return
        t = self.state[0]
        self.d.qpos[:] = self.state[1:1 + self.m.nq]
        mujoco.mj_kinematics(self.m, self.d)

        # camera pose (world): torso pose * mount, pitched down; camera axes x fwd, y left, z up
        R_t = self.d.xmat[self.torso].reshape(3, 3)
        cam = self.d.xpos[self.torso] + R_t @ self.cam_mount
        cp, sp = math.cos(self.cam_pitch), math.sin(self.cam_pitch)
        R_pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        R_wc = R_t @ R_pitch                                 # camera -> world
        R_cw = R_wc.T

        out = Detection3DArray()
        out.header.stamp = to_time(t)
        out.header.frame_id = 'head_camera'
        box_q = mat_to_quat(R_cw)                            # world-aligned box, seen from the camera
        for obj in self.objects:
            lo, hi = self.aabb(obj['geoms'])
            center = (lo + hi) / 2
            size = hi - lo
            rel = R_cw @ (center - cam)
            dist = float(np.linalg.norm(rel))
            if dist > self.max_range or rel[0] <= 0.1:
                continue
            if abs(math.atan2(rel[1], rel[0])) > self.fov_h or abs(math.atan2(rel[2], rel[0])) > self.fov_v:
                continue
            if 2 * math.atan2(float(np.linalg.norm(size)) / 2, dist) < self.min_ang:
                continue
            top = hi[2] - 0.01
            samples = [center, np.array([center[0], center[1], top])]
            for dx, dy in ((0.3, 0.3), (-0.3, 0.3), (0.3, -0.3), (-0.3, -0.3)):
                samples.append(np.array([center[0] + dx * size[0], center[1] + dy * size[1], top]))
            seen = sum(self.visible(cam, s, obj) for s in samples)
            if seen == 0:
                continue
            noise = 0.005 + 0.01 * dist
            meas = rel + self.rng.normal(0.0, noise, 3)
            det = Detection3D()
            det.header = out.header
            det.bbox.center.position.x, det.bbox.center.position.y, det.bbox.center.position.z = map(float, meas)
            (det.bbox.center.orientation.x, det.bbox.center.orientation.y,
             det.bbox.center.orientation.z, det.bbox.center.orientation.w) = map(float, box_q)
            sz = size * (1.0 + self.rng.normal(0.0, 0.05, 3))
            det.bbox.size.x, det.bbox.size.y, det.bbox.size.z = map(float, np.abs(sz))
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = obj['label']
            hyp.hypothesis.score = float(np.clip(0.95 - 0.07 * dist, 0.3, 0.95) * (0.6 + 0.4 * seen / len(samples)))
            hyp.pose.pose = det.bbox.center
            det.results = [hyp]
            out.detections.append(det)
        self.pub.publish(out)


def main():
    rclpy.init()
    node = PerceptionStub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
