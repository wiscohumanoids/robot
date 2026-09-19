import math

from nav_stub.nav_logic import (
    distance_to_goal, go_to_goal_command, reached, wrap_angle, yaw_from_quaternion)


def test_facing_goal_drives_straight():
    vx, vy, wz = go_to_goal_command(0.0, 0.0, 0.0, 2.0, 0.0)
    assert vx > 0.0 and vy == 0.0 and abs(wz) < 1e-9


def test_speed_is_capped():
    vx, _, _ = go_to_goal_command(0.0, 0.0, 0.0, 100.0, 0.0, max_lin=0.3)
    assert abs(vx - 0.3) < 1e-9


def test_goal_behind_turns_without_reversing():
    vx, _, wz = go_to_goal_command(0.0, 0.0, 0.0, -2.0, 0.1)
    assert vx == 0.0 and wz != 0.0


def test_turn_direction_matches_goal_side():
    _, _, wz_left = go_to_goal_command(0.0, 0.0, 0.0, 0.0, 2.0)
    _, _, wz_right = go_to_goal_command(0.0, 0.0, 0.0, 0.0, -2.0)
    assert wz_left > 0.0 > wz_right


def test_reached_uses_tolerance():
    assert reached(0.95, 0.0, 1.0, 0.0, tolerance=0.1)
    assert not reached(0.5, 0.0, 1.0, 0.0, tolerance=0.1)


def test_closed_loop_converges():
    # simulate the stub against dead-reckoning (what state_estimation_stub does)
    x = y = yaw = 0.0
    dt = 0.05
    for _ in range(4000):
        if reached(x, y, 1.5, -0.7):
            break
        vx, vy, wz = go_to_goal_command(x, y, yaw, 1.5, -0.7)
        x += vx * math.cos(yaw) * dt
        y += vx * math.sin(yaw) * dt
        yaw = wrap_angle(yaw + wz * dt)
    assert reached(x, y, 1.5, -0.7)


def test_yaw_from_quaternion_roundtrip():
    assert abs(yaw_from_quaternion(0, 0, math.sin(0.4), math.cos(0.4)) - 0.8) < 1e-9
    assert distance_to_goal(0, 0, 3, 4) == 5.0
