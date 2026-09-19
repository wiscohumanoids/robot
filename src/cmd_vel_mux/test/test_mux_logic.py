from cmd_vel_mux.mux_logic import select_command

T = 0.5


def test_never_received_is_zero():
    assert select_command(10.0, None, None, T) == ('none', (0.0, 0.0, 0.0))


def test_teleop_beats_nav_when_both_fresh():
    src, cmd = select_command(10.0, (9.9, (1.0, 0.0, 0.0)), (9.9, (0.3, 0.0, 0.1)), T)
    assert (src, cmd) == ('teleop', (1.0, 0.0, 0.0))


def test_silent_teleop_hands_over_to_nav():
    src, cmd = select_command(10.0, (9.0, (1.0, 0.0, 0.0)), (9.9, (0.3, 0.0, 0.1)), T)
    assert (src, cmd) == ('nav', (0.3, 0.0, 0.1))


def test_final_teleop_zero_is_honoured_until_timeout():
    # teleop's last message is an explicit zero; it must still win for `timeout`
    src, cmd = select_command(10.0, (9.8, (0.0, 0.0, 0.0)), (9.9, (0.3, 0.0, 0.0)), T)
    assert (src, cmd) == ('teleop', (0.0, 0.0, 0.0))


def test_everything_stale_is_zero_watchdog():
    assert select_command(10.0, (5.0, (1.0, 0.0, 0.0)), (5.0, (1.0, 0.0, 0.0)), T) == (
        'none', (0.0, 0.0, 0.0))
