"""Pure selection logic for cmd_vel_mux (no rclpy, unit-testable anywhere)."""

ZERO = (0.0, 0.0, 0.0)


def select_command(now, teleop, nav, timeout_s):
    """Pick which velocity command to forward.

    now:       current time, seconds (any monotonic clock, same one used for stamps)
    teleop/nav: None if never received, else (receive_time_s, (vx, vy, wz))
    timeout_s: a source older than this is considered silent

    Priority: teleop > nav > zero. Returns (source_name, (vx, vy, wz)).
    A silent teleop hands control back to nav; both silent -> zero (this is the
    watchdog that stops the robot if whatever was driving it dies).
    """
    if teleop is not None and (now - teleop[0]) <= timeout_s:
        return 'teleop', teleop[1]
    if nav is not None and (now - nav[0]) <= timeout_s:
        return 'nav', nav[1]
    return 'none', ZERO
