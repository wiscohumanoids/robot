"""Pure go-to-goal logic for nav_stub (no rclpy, unit-testable anywhere)."""
import math


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def yaw_from_quaternion(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def distance_to_goal(x, y, gx, gy):
    return math.hypot(gx - x, gy - y)


def go_to_goal_command(x, y, yaw, gx, gy,
                       k_lin=0.8, k_ang=1.5, max_lin=0.3, max_ang=0.8):
    """Unicycle-style proportional controller. Returns a body-frame
    (vx, vy, wz) with vy always 0 (the stub never strafes).

    Turns toward the goal first and only drives forward as the heading error
    shrinks (forward speed is scaled by cos(error), floored at 0), so it never
    backs away from the goal.
    """
    dist = distance_to_goal(x, y, gx, gy)
    heading_err = wrap_angle(math.atan2(gy - y, gx - x) - yaw)
    wz = max(-max_ang, min(max_ang, k_ang * heading_err))
    vx = min(max_lin, k_lin * dist) * max(0.0, math.cos(heading_err))
    return vx, 0.0, wz


def reached(x, y, gx, gy, tolerance=0.10):
    return distance_to_goal(x, y, gx, gy) <= tolerance
