"""Pure planar helpers for the odom adapter (no rclpy, unit-testable anywhere)."""
import math


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def yaw_from_quaternion(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def body_velocity(x0, y0, yaw0, x1, y1, yaw1, dt):
    """Body-frame (vx, vy, wz) that moved a planar pose from (x0, y0, yaw0) to
    (x1, y1, yaw1) in `dt` seconds (REP-103: x forward, y left).

    The world-frame displacement is rotated into the body frame using the
    heading halfway between the two samples, which keeps turning motion accurate.
    Raises ValueError if dt <= 0.
    """
    if dt <= 0.0:
        raise ValueError('dt must be positive')
    dyaw = wrap_angle(yaw1 - yaw0)
    mid = yaw0 + dyaw / 2.0
    dx, dy = x1 - x0, y1 - y0
    c, s = math.cos(mid), math.sin(mid)
    return (dx * c + dy * s) / dt, (-dx * s + dy * c) / dt, dyaw / dt
