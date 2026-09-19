# safety

**Layer:** 9 (safety). **Status:** REAL (software-only inputs today).
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
must publish `std_msgs/Bool` on **`/manual_estop`** the same way the test command
above does, and let this node do the rest.

**It must not publish on `/safety_status`.** Under the [interface contract](../../INTERFACE_CONTRACT.md) that topic has exactly
one owner: this node. This node
publishes `is_safe: true` at 100 Hz whenever nothing is wrong, so a second
publisher asserting `estop_active` would be interleaved with (and immediately
overwritten by) those messages, and `lowlevel_control` would flicker back to
"safe". (An earlier version of this document allowed either topic; that was a
flaw in the design, not a valid option.)
