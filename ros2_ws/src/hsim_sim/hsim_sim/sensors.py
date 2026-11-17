"""Sensor models for the MuJoCo sim: what the robot's real sensors would report.

The physics engine knows the true state; real sensors don't. These models turn the
true state into realistic measurements:

  ImuModel    gyro + accelerometer with noise and bias, and an onboard attitude filter
              (like a 6-axis IMU without magnetometer): tilt stays accurate because gravity
              corrects it, heading slowly drifts with the gyro bias.
  Lidar       a head-mounted multi-beam lidar (rays cast into the scene), flattened to a
              2D scan: closest hit per direction within the robot's height band. This is
              what pointcloud_to_laserscan does with a real Livox MID-360.
  Battery     state of charge drained by actual motor power plus electronics load.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

GRAVITY = 9.81


def quat_mul(a, b):
    """Hamilton product of two quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_rotate_inv(q, v):
    """Rotate world vector v into the frame of unit quaternion q = (w, x, y, z)."""
    w, x, y, z = q
    u = np.array([x, y, z])
    return v * (2 * w * w - 1) - np.cross(u, v) * 2 * w + u * 2 * np.dot(u, v)


def yaw_of(q):
    """Yaw of a (w, x, y, z) quaternion."""
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class ImuModel:
    """6-axis IMU + complementary attitude filter. Defaults are in the range of a
    consumer MEMS IMU (e.g. BMI088 class).

    tilt_gain: how hard the filter pulls roll/pitch toward the accelerometer each 5 ms step. 0.001 = ~5 s time
    constant, like a tuned onboard AHRS (tilt error < 1 deg while walking). At 0.02 (0.25 s)
    the filter followed the body's own sway accelerations, tilt was off by up to 7-10 deg, and the walking
    policy, correcting a tilt that wasn't there, rocked about 2.5x more while standing than with the true tilt."""

    def __init__(self, rng: np.random.Generator, dt: float,
                 gyro_noise=0.0035, gyro_bias_sigma=0.001, gyro_bias_walk=2e-5,
                 accel_noise=0.03, accel_bias_sigma=0.02, tilt_gain=0.001):
        """Noise densities and bias parameters per axis; draws the initial biases."""
        self.rng, self.dt = rng, dt
        self.gyro_noise, self.gyro_bias_walk = gyro_noise, gyro_bias_walk
        self.accel_noise, self.tilt_gain = accel_noise, tilt_gain
        self.gyro_bias = rng.normal(0.0, gyro_bias_sigma, 3)
        self.accel_bias = rng.normal(0.0, accel_bias_sigma, 3)
        self.q = None                                     # filter estimate (w, x, y, z)

    def reset(self, true_quat):
        """Power-on: tilt from gravity (accurate), heading arbitrary -> start at yaw 0."""
        yaw = yaw_of(true_quat)
        undo_yaw = np.array([math.cos(-yaw / 2), 0.0, 0.0, math.sin(-yaw / 2)])
        self.q = quat_mul(undo_yaw, np.asarray(true_quat, float))
        self.q /= np.linalg.norm(self.q)

    def measure(self, true_gyro, true_accel):
        """Returns (orientation wxyz, gyro, accel) as the IMU driver would publish them."""
        self.gyro_bias += self.rng.normal(0.0, self.gyro_bias_walk, 3)
        gyro = np.asarray(true_gyro) + self.gyro_bias + self.rng.normal(0.0, self.gyro_noise, 3)
        accel = np.asarray(true_accel) + self.accel_bias + self.rng.normal(0.0, self.accel_noise, 3)

        # attitude filter: integrate the gyro ...
        dq = np.concatenate([[1.0], 0.5 * gyro * self.dt])
        self.q = quat_mul(self.q, dq)
        self.q /= np.linalg.norm(self.q)
        # ... and pull roll/pitch toward the measured gravity direction when not accelerating hard
        if abs(np.linalg.norm(accel) - GRAVITY) < 1.5:
            up_est = quat_rotate_inv(self.q, np.array([0.0, 0.0, 1.0]))
            up_meas = accel / np.linalg.norm(accel)
            err = np.cross(up_meas, up_est)              # body-frame correction toward measured gravity
            corr = np.concatenate([[1.0], 0.5 * self.tilt_gain * err])
            self.q = quat_mul(self.q, corr)
            self.q /= np.linalg.norm(self.q)
        return self.q.copy(), gyro, accel


class Lidar:
    """Multi-beam lidar flattened to a 2D scan (closest obstacle per direction within a
    height band). Rays start at a mount point on the robot and are cast in a level frame
    that turns with the robot's heading, like a gimbal-stabilized scan."""

    def __init__(self, m: mujoco.MjModel, rng: np.random.Generator, n_azimuth=360,
                 elevations_deg=np.linspace(-65.0, 5.0, 16), range_min=0.1, range_max=10.0,
                 band=(0.05, 1.5), noise=0.01, dropout=0.01):
        """Precompute the ray directions for the scan pattern and the buffers for mj_multiRay."""
        self.m, self.rng = m, rng
        self.n_az = n_azimuth
        self.range_min, self.range_max = range_min, range_max
        self.band, self.noise, self.dropout = band, noise, dropout
        az = np.linspace(-math.pi, math.pi, n_azimuth, endpoint=False)
        el = np.radians(np.asarray(elevations_deg))
        self.az = az
        A, E = np.meshgrid(az, el, indexing="ij")         # [n_az, n_el]
        self.dirs_local = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1)
        self.cos_el = np.cos(E)
        self.n = self.dirs_local.shape[0] * self.dirs_local.shape[1]
        self.geomid = np.zeros(self.n, np.int32)
        self.dist = np.zeros(self.n)
        # groups 4/5 hold the robot's own geometry (see MujocoSim), so the lidar never sees it
        self.groups = np.array([1, 1, 1, 1, 0, 0], np.uint8)

    def scan(self, d: mujoco.MjData, origin: np.ndarray, yaw: float) -> np.ndarray:
        """Cast every ray from `origin` at heading `yaw` and return the 2D scan: nearest return per azimuth within
        the height band.
        """
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        vec = (self.dirs_local.reshape(-1, 3) @ R.T).ravel()
        mujoco.mj_multiRay(self.m, d, origin, vec, self.groups, 1, -1, self.geomid, self.dist,
                           None, self.n, self.range_max / 0.2)
        dist = self.dist.reshape(self.dirs_local.shape[:2])
        hit = dist >= 0
        z = origin[2] + dist * self.dirs_local[..., 2]            # height of each hit
        valid = hit & (z > self.band[0]) & (z < self.band[1])
        horiz = np.where(valid, dist * self.cos_el, np.inf)
        horiz = np.where(horiz > self.range_max, np.inf, horiz)
        ranges = horiz.min(axis=1)                         # closest obstacle per direction
        finite = np.isfinite(ranges)
        ranges[finite] += self.rng.normal(0.0, self.noise, finite.sum())
        ranges[self.rng.random(self.n_az) < self.dropout] = np.inf
        ranges[ranges < self.range_min] = np.inf
        return ranges


class Battery:
    """Electric model of the G1 pack (~9 Ah at ~54 V nominal, ~490 Wh)."""

    def __init__(self, capacity_wh=486.0, start_soc=1.0, base_load_w=60.0, efficiency=0.65,
                 drain_multiplier=1.0):
        """Pack capacity, starting charge, base load and drive efficiency."""
        self.capacity_wh, self.soc = capacity_wh, start_soc
        self.base_load_w, self.eff, self.mult = base_load_w, efficiency, drain_multiplier
        self.power_w = base_load_w

    def step(self, torque: np.ndarray, joint_vel: np.ndarray, dt: float):
        """Drain the pack by base load plus mechanical joint power over `dt`."""
        mech = float(np.abs(torque * joint_vel).sum())
        self.power_w = self.base_load_w + mech / self.eff
        self.soc = max(0.0, self.soc - self.mult * self.power_w * dt / 3600.0 / self.capacity_wh)

    @property
    def voltage(self) -> float:
        """Pack voltage from the state of charge."""
        return 44.0 + 10.0 * self.soc                      # ~54 V full, ~44 V empty
