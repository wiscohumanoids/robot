# hsim_mujoco_plugins — MuJoCo plugins for the sim stack's ros2_control backends

The sim stack can run the robot three ways (`sim_stack.launch.py backend:=...`, see `hsim_sim/README.md`). The two
that put MuJoCo inside a ros2_control controller manager need these plugins:

| Plugin | Backend | What |
|---|---|---|
| `hsim_mujoco_plugins/GroundTruthPlugin` (a `mujoco_ros2_control` plugin) | `ros2_control` | `/sim/ground_truth/state` and the viewer stream from inside the upstream `mujoco_ros2_control` hardware plugin |
| `hsim_mujoco_plugins/VirtualRobotSystem` (a ros2_control hardware plugin) | `ethercat_sim` | the robot's EtherCAT hardware, simulated: a CiA402 drive per motor, the master's logic, MuJoCo as the robot |

Both publish the world's state in the MuJoCo node's formats (`include/hsim_mujoco_plugins/ground_truth_stream.hpp`):
`/sim/ground_truth/state` (`std_msgs/Float64MultiArray`, `[time, qpos...]`, 20 Hz), which the simulated head camera
renders the world from and graders read, and the UDP stream for `tools/viewer.sh` (`b"HSIM"`, nq, time, qpos).

## GroundTruthPlugin

Loaded by the upstream [`mujoco_ros2_control`](https://github.com/ros-controls/mujoco_ros2_control) hardware plugin
through its plugin interface (`MuJoCoROS2ControlPluginBase`, Apache-2.0). Configured by
`hsim_sim/ros2_control_backend.py` (`plugins.yaml`, under `mujoco_plugins.ground_truth`): `state_topic`, `state_rate`,
`viewer_host`, `viewer_port`, `viewer_rate` (0 = off). It runs in the control thread's read step.

## VirtualRobotSystem: the EtherCAT layer in simulation

It stands in for `ethercat_driver_ros2` on the robot computer, so the controllers above it are the robot's, configured
as on the robot, and see what they would see there:

```
policy ─► leg_targets / arm_targets ─► [parallel_joints] ─► command interfaces of each MOTOR
          (JointTargetController)      (ankles, waist:       position, velocity, effort, kp, kd, mode, ...
                                        joints -> 2 motors)            │
                                                                       ▼
                                   Cia402DriveMaster (EcCiA402Drive's processData: scaling, auto-enable,
                                   0x607A handling)  ── RxPDO ──► RmdxMockSlave (CiA402 state machine,
                                                     ◄─ TxPDO ──  mode 5: kp (q* - q) + kd (q̇* - q̇) + τff)
                                                                       │ torque at every physics step
                                                                       ▼
                                   MuJoCo: single joints directly; a two-motor mechanism's crank torques
                                   reach its two serial joints as Jᵀτ (rod geometry, parallel_linkage.hpp)
```

- **One hardware joint per motor**, with the interfaces `EcCiA402Drive` exports (state: position, velocity, effort,
  mode_of_operation, status_word, error_code; command: position, velocity, effort, kp, kd, mode_of_operation,
  reset_fault, max_torque), in SI units through the drive's counts and per-mille scaling.
- **Bus timing as on the robot**: a command written in an update reaches the drive one cycle later, and the drive's
  reading reaches the state interfaces one cycle later. The drives' servo law runs at every physics step.
- **Drives come up through the CiA402 state machine** (the master's automatic transitions), in mode 5 unless a
  controller writes `mode_of_operation`.
- **MuJoCo's position actuators on the driven joints are switched off**: the drives' torques are the only actuation.
- **Simulated time**: `write()` steps one bus cycle of physics, publishes `/clock` and paces to real time; the
  controller manager runs with `use_sim_time`. `~/set_pause` blocks the control loop (so `/clock` stops);
  `~/reset_world` resets to a keyframe without turning time back. The world is frozen at the keyframe until the
  first unpause and for a few cycles after a reset, while the drives enable and the new targets travel the bus.
- Also: `/sim/ground_truth/odom`, and the IMU as a sensor component's state interfaces (`imu_sensor_broadcaster`).

The drive and master models are `wisco_ethercat_ros2_hw`'s (`include/wisco_ethercat_ros2_hw/mock/`), the same ones
its controller tests run against. Where the RMD-X manual is silent they are guesses, marked in `RmdxParams` (mode 5's
gain encoding among them). Parameters: see the header, `include/hsim_mujoco_plugins/virtual_robot_system.hpp`;
`hsim_sim/ethercat_sim_backend.py` writes them.

`test/test_virtual_robot.cpp` runs the plugin on Wisco V1 in the test room (skipped until
`tools/env_build.py assets/envs/test/table_room --robot wisco` has built it): every drive reaches mode 5, the
two-motor mechanisms read what the rod geometry gives, and mode-5 holds over the bus keep the robot upright.
