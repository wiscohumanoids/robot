import math

from state_estimation_stub.kinematics import integrate_body_velocity, yaw_to_quaternion


def test_forward_along_x_when_yaw_zero():
    x, y, yaw = integrate_body_velocity(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5)
    assert (round(x, 6), round(y, 6), round(yaw, 6)) == (0.5, 0.0, 0.0)


def test_forward_is_body_frame_when_facing_y():
    x, y, _ = integrate_body_velocity(0.0, 0.0, math.pi / 2, 1.0, 0.0, 0.0, 1.0)
    assert abs(x) < 1e-9 and abs(y - 1.0) < 1e-9


def test_lateral_velocity_is_left_of_heading():
    x, y, _ = integrate_body_velocity(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0)
    assert abs(x) < 1e-9 and abs(y - 1.0) < 1e-9


def test_yaw_wraps():
    _, _, yaw = integrate_body_velocity(0.0, 0.0, math.pi - 0.01, 0.0, 0.0, 1.0, 0.1)
    assert -math.pi <= yaw <= math.pi and yaw < 0.0


def test_quaternion_is_unit_and_rotates_about_z():
    qx, qy, qz, qw = yaw_to_quaternion(math.pi / 2)
    assert abs(qx) < 1e-12 and abs(qy) < 1e-12
    assert abs(qx**2 + qy**2 + qz**2 + qw**2 - 1.0) < 1e-12
    assert abs(qz - math.sqrt(0.5)) < 1e-9
