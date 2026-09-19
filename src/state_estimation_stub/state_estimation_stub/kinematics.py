"""Pure planar kinematics helpers (no rclpy, unit-testable anywhere)."""
import math


def yaw_to_quaternion(yaw):
    """Returns (x, y, z, w) for a rotation of `yaw` radians about +z."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def integrate_body_velocity(x, y, yaw, vx, vy, wz, dt):
    """Advance a planar pose by a BODY-frame velocity (REP-103: x fwd, y left).

    Returns the new (x, y, yaw). Assumes perfect tracking of the command --
    this is exactly why it's a stub: a real estimator fuses IMU, kinematics
    and vision instead of trusting the command.
    """
    c, s = math.cos(yaw), math.sin(yaw)
    x += (vx * c - vy * s) * dt
    y += (vx * s + vy * c) * dt
    yaw = math.atan2(math.sin(yaw + wz * dt), math.cos(yaw + wz * dt))  # wrap to (-pi, pi]
    return x, y, yaw
