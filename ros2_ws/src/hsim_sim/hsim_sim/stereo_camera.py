"""Simulated ZED X-style stereo camera (node name: `stereo_camera`). Part of the SIMULATOR.

Like the sim node, this is the world side: it reads the simulator's state
(/sim/ground_truth/state) to know where the head is, renders what the camera sees, and
turns perfect rendered depth into stereo-like depth:

  - error grows with distance squared:  sigma_z = z^2 * sigma_disparity / (focal_px * baseline)
  - holes where the image has no texture to match (plain walls, glossy counters)
  - limited range, a few random dropouts
It runs in its own process (rendering takes ~110 ms per frame on CPU), at `rate_hz`.

Publishes (frame head_camera_optical: x right, y down, z forward):
  /camera/head/color/image_raw    sensor_msgs/Image rgb8
  /camera/head/depth/image_raw    sensor_msgs/Image 32FC1, meters, NaN = no depth
  /camera/head/color/camera_info, /camera/head/depth/camera_info
"""

import math

import mujoco
import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float64MultiArray

PREFIX = 'robot/'
CAMERA_MOUNT = np.array([0.08, 0.0, 0.45])     # on the torso, same as the head_camera frame
CAMERA_PITCH = math.radians(25.0)


def to_time(t):
    sec, nsec = divmod(int(round(t * 1e9)), 1_000_000_000)
    return Time(sec=sec, nanosec=nsec)


def box_std(gray, k=7):
    """Local standard deviation (k x k window) via integral images."""
    pad = k // 2
    g = np.pad(gray, pad, mode='edge').astype(np.float64)
    def box(a):
        c = np.cumsum(np.cumsum(a, 0), 1)
        c = np.pad(c, ((1, 0), (1, 0)))
        return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)
    m = box(g)
    return np.sqrt(np.maximum(box(g * g) - m * m, 0.0))


class StereoCamera(Node):
    def __init__(self):
        super().__init__('stereo_camera')
        for name, default in (('world', ''), ('rate_hz', 5.0), ('width', 640), ('height', 320),
                              ('hfov_deg', 110.0), ('baseline', 0.12), ('disparity_noise_px', 0.15),
                              ('min_depth', 0.3), ('max_depth', 12.0), ('texture_threshold', 2.5),
                              ('dropout', 0.01), ('seed', 2),
                              # robot profile: torso body, camera position in it (m) and pitch (rad, + = down)
                              ('torso_body', 'torso_link'), ('camera_mount', CAMERA_MOUNT.tolist()),
                              ('camera_pitch', CAMERA_PITCH)):
            self.declare_parameter(name, default)
        g = lambda n: self.get_parameter(n).value
        self.m = mujoco.MjModel.from_binary_path(g('world'))
        self.d = mujoco.MjData(self.m)
        self.W, self.H = g('width'), g('height')
        hfov = math.radians(g('hfov_deg'))
        self.fx = (self.W / 2) / math.tan(hfov / 2)
        vfov = 2 * math.atan((self.H / 2) / self.fx)                # square pixels
        self.m.vis.global_.fovy = math.degrees(vfov)
        self.baseline, self.sig_d = g('baseline'), g('disparity_noise_px')
        self.zmin, self.zmax, self.tex, self.drop = g('min_depth'), g('max_depth'), g('texture_threshold'), g('dropout')
        self.rng = np.random.default_rng(g('seed'))

        self.rgb_r = mujoco.Renderer(self.m, height=self.H, width=self.W)
        self.dep_r = mujoco.Renderer(self.m, height=self.H, width=self.W)
        self.dep_r.enable_depth_rendering()
        for gi in range(self.m.ngeom):         # the camera doesn't see the robot's own body
            if self.m.body(self.m.geom_bodyid[gi]).name.startswith(PREFIX):
                self.m.geom_group[gi] = 4 if self.m.geom_group[gi] <= 2 else 5
        self.opt = mujoco.MjvOption()          # default: groups 0-2 drawn, 4/5 (robot) not
        self.torso = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, PREFIX + g('torso_body'))
        self.cam_mount = np.array(g('camera_mount'), dtype=float)
        self.cam_pitch = float(g('camera_pitch'))
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.state = None

        self.info = CameraInfo()
        self.info.width, self.info.height = self.W, self.H
        self.info.distortion_model = 'plumb_bob'
        self.info.d = [0.0] * 5
        cx, cy = (self.W - 1) / 2, (self.H - 1) / 2
        self.info.k = [self.fx, 0.0, cx, 0.0, self.fx, cy, 0.0, 0.0, 1.0]
        self.info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.info.p = [self.fx, 0.0, cx, 0.0, 0.0, self.fx, cy, 0.0, 0.0, 0.0, 1.0, 0.0]

        self.pub_rgb = self.create_publisher(Image, '/camera/head/color/image_raw', 2)
        self.pub_dep = self.create_publisher(Image, '/camera/head/depth/image_raw', 2)
        self.pub_ci = self.create_publisher(CameraInfo, '/camera/head/color/camera_info', 2)
        self.pub_di = self.create_publisher(CameraInfo, '/camera/head/depth/camera_info', 2)
        self.create_subscription(Float64MultiArray, '/sim/ground_truth/state', lambda m: setattr(self, 'state', m.data), 2)
        self.create_timer(1.0 / g('rate_hz'), self.tick)
        self.get_logger().info(f'stereo camera (ZED X-like): {self.W}x{self.H}, {g("hfov_deg"):.0f}x{math.degrees(vfov):.0f} deg, '
                               f'{g("rate_hz")} Hz, depth {self.zmin}-{self.zmax} m, '
                               f'noise at 3 m ~{100 * 9 * self.sig_d / (self.fx * self.baseline):.0f} cm')

    def tick(self):
        if self.state is None:
            return
        t = self.state[0]
        self.d.qpos[:] = self.state[1:1 + self.m.nq]
        mujoco.mj_forward(self.m, self.d)

        R_t = self.d.xmat[self.torso].reshape(3, 3)
        pos = self.d.xpos[self.torso] + R_t @ self.cam_mount
        cp, sp = math.cos(self.cam_pitch), math.sin(self.cam_pitch)
        fwd = R_t @ np.array([cp, 0.0, -sp])                       # optical axis in the world
        self.cam.lookat[:] = pos + fwd
        self.cam.distance = 1.0
        self.cam.azimuth = math.degrees(math.atan2(fwd[1], fwd[0]))
        self.cam.elevation = math.degrees(math.asin(max(-1.0, min(1.0, fwd[2]))))

        self.rgb_r.update_scene(self.d, camera=self.cam, scene_option=self.opt)
        rgb = self.rgb_r.render()
        self.dep_r.update_scene(self.d, camera=self.cam, scene_option=self.opt)
        z = self.dep_r.render().astype(np.float32)

        # stereo-like depth
        gray = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
        textured = box_std(gray) > self.tex
        sigma = z * z * self.sig_d / (self.fx * self.baseline)
        z = z + self.rng.normal(0.0, 1.0, z.shape).astype(np.float32) * sigma
        valid = textured & (z > self.zmin) & (z < self.zmax) & (self.rng.random(z.shape) > self.drop)
        z = np.where(valid, z, np.nan).astype(np.float32)

        stamp = to_time(t)
        for pub, data, enc, step in ((self.pub_rgb, rgb, 'rgb8', self.W * 3), (self.pub_dep, z, '32FC1', self.W * 4)):
            img = Image()
            img.header.stamp, img.header.frame_id = stamp, 'head_camera_optical'
            img.height, img.width, img.encoding, img.step = self.H, self.W, enc, step
            img.data = np.ascontiguousarray(data).tobytes()
            pub.publish(img)
        for pub in (self.pub_ci, self.pub_di):
            self.info.header.stamp, self.info.header.frame_id = stamp, 'head_camera_optical'
            pub.publish(self.info)


def main():
    rclpy.init()
    node = StereoCamera()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
