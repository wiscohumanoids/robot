# teleop_input

**Layer:** 5 (navigation & velocity arbitration). **Status:** real.
**Frequency:** 10 Hz, **only while a command is nonzero** (plus one final zero).
**Publishes:** `geometry_msgs/Twist` on **`/cmd_vel_teleop`** (not `/cmd_vel`,
which belongs solely to `cmd_vel_mux`).
**Subscribes:** `sensor_msgs/Joy` on `/joy` (optional, only if a joystick driver is running).

Keyboard control (raw terminal input, same technique as ROS's
`teleop_twist_keyboard`): `w`/`s` = forward/backward, `a`/`d` = strafe
left/right, `q`/`e` = yaw left/right, `space` = zero all axes, `x` = quit.
Commands latch (hold) until changed; you don't need to hold a key down.

Only `linear.x`, `linear.y` and `angular.z` are used (REP-103 `base_link`).

## Run it: in its own terminal

```bash
ros2 run teleop_input teleop_node
```

**Keyboard input needs a real TTY, and `ros2 launch` does not give its nodes one**,
so `bringup`'s `full_stack.launch.py` does *not* start teleop by default
(`teleop:=true` starts it there, but then only a joystick works). Start the stack,
then run the command above in a second terminal.

## Why it goes quiet when idle

`cmd_vel_mux` gives teleop priority over navigation for as long as teleop keeps
publishing. If teleop published a stream of zeros forever, navigation could never
move the robot. So teleop publishes only while a command is active, and one last
zero when it returns to zero; a moment later `cmd_vel_mux` hands control back to
navigation. See `cmd_vel_mux/README.md`.
