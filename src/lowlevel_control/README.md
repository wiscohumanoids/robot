# lowlevel_control

**Status:** Real.
**Frequency:** whatever `controller_manager`'s `update_rate` says,
`bringup/config/controllers.yaml` sets it to **1000 Hz** in this repo.
**Type:** a real `controller_interface::ControllerInterface` plugin
(`lowlevel_control/JointImpedanceController`), loaded by `controller_manager`
and run inside its real-time `update()` loop, not a plain ROS2 node with its
own timer, which is what makes its rate an actual real-time guarantee (given
a correctly configured `PREEMPT_RT` system, see `RESEARCH_NOTES.md` §4)
rather than a best-effort `rclcpp::Timer`.

## What it does

Per joint, per `update()` call:

```
tau = tau_ff + Kp*(q_d - q) + Kd*(qd_d - qd)
```

- `q_d`, `qd_d`, `tau_ff` come from the latest `humanoid_interfaces/JointCommand`
  on `/joint_command` (published by `wbc_stub` at 500 Hz, handed to this
  controller via a `realtime_tools::RealtimeBuffer`, not read directly in the
  subscription callback, so a slow/blocked ROS graph never stalls the 1kHz
  loop).
- `q`, `qd` come from the `position`/`velocity` state interfaces of whichever
  hardware plugin is active.
- `tau` is written to the `effort` command interface.
- If no `JointCommand` has ever arrived, or `/safety_status.is_safe` is
  false, `tau` is forced to `0.0` for every joint (see "Safety" below),
  the controller starts disabled and only ever actuates in response to a
  real command.
- Every tick, it also reads back `position`/`velocity`/`effort` state plus
  the `imu_imu` sensor interfaces (see `g1_description`'s `<ros2_control>`
  block) and publishes `humanoid_interfaces/RobotState` on `/robot_state`,
  including a `gravity_vector` computed directly from the IMU orientation
  quaternion (no `tf2` dependency, to avoid any allocation in the real-time
  path: see the inline math comment in `joint_impedance_controller.cpp`).

## Gains

Default gains (used only if `bringup/config/controllers.yaml` doesn't
override them) mirror the per-joint-group values `bipedal_nav` originally
used for its `position_pid` interface: legs stiff (`Kp=1500, Kd=150`, except
hip-yaw/waist at `Kp=500, Kd=50`), arms soft (`Kp=100, Kd=10`), wrists
softest (`Kp=50, Kd=5`), see `kDefaultKp`/`kDefaultKd` in
`joint_impedance_controller.cpp`.

## Torque limits

`effort_limits` (per joint, `controllers.yaml`) clamps `tau` before it is written.
It must equal the URDF `<limit effort>` (a test enforces this). It has to live here:
the vendored MuJoCo bridge's URDF-limit clamp never activates (its
`has_effort_limits` flag is never set), so without this the sim applies unbounded
torque and the physics diverges. `max_effort` is an optional extra global clamp.

## Safety

Subscribes to `/safety_status` (`SafetyStatus`) and forces zero torque
whenever `is_safe` is false: see `safety/README.md` and
`INTEGRATION_POINTS.md` "Safety" for the full E-stop interface description.

## Sim/hardware symmetry

This controller does not know or care whether `left_hip_pitch_joint/effort`
is backed by `mujoco_ros2_control/MujocoSystem` or
`ethercat_bridge/EthercatHardwareInterface`: it only claims the
`effort`/`position`/`velocity` interfaces `g1_description`'s
`<ros2_control>` block exports, which are identical either way. See
`ARCHITECTURE.md` "Sim/hardware symmetry" for the verification this relies on.
