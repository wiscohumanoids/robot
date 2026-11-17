"""Device axes to robot axes, and the heading convention. No ROS.

The board does not sit level and facing forward: on Wisco V1 it stands upright at the back of the pelvis
(robots/wisco_v1/model/robot_numbers.yaml, sensors.imu). `mount_wxyz` is the rotation of the device's output axes
expressed in the pelvis frame (q_pelvis_device), the same quaternion as the `imu` site in the robot's MJCF. The
driver publishes in `imu_link`, a frame at the IMU with the pelvis's axes (x forward, y left, z up), which is what the
walking policy was trained on (the `imu_policy` site):

    gyro_link  = R_pd * gyro_device          acc_link = R_pd * acc_device
    q_world_link = q_world_nav * q_nav_device * conj(q_pelvis_device)

The device's navigation frame has z up, so tilt (what the policy and the state estimator use) needs no change.
Heading: the simulated IMU starts at yaw 0 and drifts (no magnetometer); with `zero_yaw` the driver removes the first
sample's heading the same way, so the robot starts facing yaw 0 in its own world frame.
"""

import math

import numpy as np


def quat_mul(a, b):
    """Hamilton product of two (w, x, y, z) quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw])


def quat_conj(q):
    """Conjugate (inverse of a unit quaternion)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalize(q):
    """Unit quaternion, or None for a zero one."""
    q = np.asarray(q, float)
    n = np.linalg.norm(q)
    return q / n if n > 0 else None


def quat_to_mat(q):
    """Rotation matrix of a unit (w, x, y, z) quaternion."""
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def yaw_of(q):
    """Heading (rad) of a (w, x, y, z) quaternion: the angle of its x axis in the horizontal plane."""
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def yaw_quat(yaw):
    """Rotation about z by `yaw`."""
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


class MountConverter:
    """Turns device-axis samples into imu_link samples (pelvis axes), with optional heading zeroing."""

    def __init__(self, mount_wxyz, zero_yaw=True):
        """`mount_wxyz`: q_pelvis_device. `zero_yaw`: remove the first orientation's heading."""
        q = quat_normalize(mount_wxyz)
        if q is None:
            raise ValueError('mount quaternion is zero')
        self.q_pd = q
        self.R_pd = quat_to_mat(q)
        self.zero_yaw = zero_yaw
        self.q_world_nav = None     # set from the first orientation when zero_yaw

    def vector(self, v):
        """A device-axis vector (gyro, acceleration) in imu_link axes."""
        return self.R_pd @ np.asarray(v, float)

    def orientation(self, q_nav_device):
        """imu_link orientation in the robot's world frame, or None when the device sends a zero quaternion."""
        q = quat_normalize(q_nav_device)
        if q is None:
            return None
        q_nav_link = quat_mul(q, quat_conj(self.q_pd))
        if self.q_world_nav is None:
            self.q_world_nav = yaw_quat(-yaw_of(q_nav_link)) if self.zero_yaw else np.array([1.0, 0.0, 0.0, 0.0])
        out = quat_mul(self.q_world_nav, q_nav_link)
        return out if out[0] >= 0 else -out

    def reset_heading(self):
        """Zero the heading again on the next sample (e.g. after a reconnect)."""
        self.q_world_nav = None


def tilt_deg(q_world_link):
    """Angle between the link's z axis and world up, degrees."""
    return math.degrees(math.acos(max(-1.0, min(1.0, quat_to_mat(q_world_link)[2, 2]))))
