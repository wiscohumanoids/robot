# safety

**Status:** Real (software-only inputs today).
**Frequency:** 100 Hz.
**Subscribes:** `/manual_estop` (`std_msgs/Bool`), `/robot_state` (`RobotState`,
checked for NaN/Inf).
**Publishes:** `/safety_status` (`SafetyStatus`).

## Testing the estop path end-to-end

```bash
ros2 topic pub /manual_estop std_msgs/msg/Bool "{data: true}" --once
```

Should cause `/safety_status.is_safe` to go false within one 100 Hz tick, and
`lowlevel_control`'s `JointImpedanceController` to zero its torque output and
log that it has disabled (see `lowlevel_control/README.md`).

## Interface to a real hardware E-stop

A wireless E-stop is a separate firmware project, out of scope here. Whatever
bridges that firmware to ROS2 (micro-ROS agent, serial bridge node, etc.)
needs to publish `humanoid_interfaces/SafetyStatus` with `estop_active=true`
directly on `/safety_status` — or, more simply, publish `std_msgs/Bool` on
`/manual_estop` the same way the test command above does, and let this node
do the rest. Either integration point works; `/manual_estop` is the lower-
effort one for a firmware bridge to target.
