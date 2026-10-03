import math

import pytest

from odom_adapter_stub.odom_logic import body_velocity, wrap_angle, yaw_from_quaternion
from state_estimation_stub.kinematics import integrate_body_velocity, yaw_to_quaternion


def test_stationary_is_zero():
    assert body_velocity(1.0, 2.0, 0.3, 1.0, 2.0, 0.3, 0.1) == (0.0, 0.0, 0.0)


def test_forward_when_facing_x():
    vx, vy, wz = body_velocity(0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.5)
    assert (round(vx, 9), round(vy, 9), round(wz, 9)) == (0.2, 0.0, 0.0)


def test_forward_is_body_frame_when_facing_y():
    vx, vy, _ = body_velocity(0.0, 0.0, math.pi / 2, 0.0, 0.1, math.pi / 2, 1.0)
    assert abs(vx - 0.1) < 1e-9 and abs(vy) < 1e-9


def test_pure_rotation():
    vx, vy, wz = body_velocity(0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.5)
    assert vx == 0.0 and vy == 0.0 and abs(wz - 0.4) < 1e-12


def test_yaw_difference_wraps_across_pi():
    _, _, wz = body_velocity(0.0, 0.0, math.pi - 0.01, 0.0, 0.0, -math.pi + 0.01, 1.0)
    assert abs(wz - 0.02) < 1e-9


def test_rejects_non_positive_dt():
    with pytest.raises(ValueError):
        body_velocity(0, 0, 0, 1, 0, 0, 0.0)


def test_recovers_the_velocity_state_estimation_integrated():
    """Round trip against the stub estimator: differencing its poses gives back the command."""
    cmd = (0.3, 0.05, 0.4)
    dt = 0.01
    pose = (1.0, -2.0, 0.7)
    for _ in range(200):
        nxt = integrate_body_velocity(*pose, *cmd, dt)
        vx, vy, wz = body_velocity(*pose, *nxt, dt)
        assert abs(vx - cmd[0]) < 1e-3 and abs(vy - cmd[1]) < 1e-3 and abs(wz - cmd[2]) < 1e-6
        pose = nxt


def test_yaw_from_quaternion_inverts_yaw_to_quaternion():
    for yaw in (-3.0, -1.0, 0.0, 0.5, 2.5):
        assert abs(wrap_angle(yaw_from_quaternion(*yaw_to_quaternion(yaw)) - yaw)) < 1e-12
