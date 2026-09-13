# teleop_input

**Status:** Real, not a stub.
**Frequency:** 10 Hz.
**Publishes:** `humanoid_interfaces/VelocityCommand` on `/cmd_vel`.
**Subscribes:** `sensor_msgs/Joy` on `/joy` (optional -- only used if a
joystick driver is running).

Keyboard control (raw terminal input, same technique as ROS's
`teleop_twist_keyboard`): `w`/`s` = forward/backward, `a`/`d` = strafe
left/right, `q`/`e` = yaw left/right, `space` = zero all axes, `x` = quit.
Commands latch (hold) until changed -- you don't need to hold a key down.

Run standalone:

```bash
ros2 run teleop_input teleop_node
```

This is the entry point for the whole vertical stack described in
`ARCHITECTURE.md` -- everything below this node exists to turn `/cmd_vel`
into (eventually, on real hardware) motor torque.
