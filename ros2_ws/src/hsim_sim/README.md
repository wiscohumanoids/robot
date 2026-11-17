# hsim_sim — the simulated world and the robot's sensors

| Executable | Node | What |
|---|---|---|
| `mujoco_sim` | `sim` | MuJoCo 3.11 world of the robot profile (`robot:=`, Wisco V1 by default) at 200 Hz (training physics). Sensors with realistic models (`sensors.py`): IMU (noise, bias, onboard attitude filter), head lidar → `/scan` (10 Hz, `sensors:=lidar`; the G1's default, Wisco has none), battery → `/battery`. Services `/sim/reset`, test-only `/sim/set_body_pose`. Privileged `/sim/ground_truth/*` for graders only. |
| `stereo_camera` | `stereo_camera` | ZED X-style color + depth (`sensors:=stereo`, Wisco's default), stereo depth-error model, own process, ~2.5–5 Hz on CPU |
| `sim_stub` | `sim` | kinematic stand-in (no physics) for testing upper layers |
| `joint_target_bridge` | `joint_target_bridge` | `backend:=ros2_control` only: `/locomotion/joint_targets` (by name) → `/joint_targets_controller/commands` (controller order); startup pause → stand → reset → unpause; `/sim/reset` |
| `ros2_control_backend` | (CLI) | writes the ros2_control backend's files for a trained scene (below) |

## ros2_control backend

`sim_stack.launch.py backend:=ros2_control` replaces the sim node with the robot's control path: the upstream
[`mujoco_ros2_control`](https://control.ros.org/jazzy/doc/mujoco_ros2_control/doc/index.html) plugin (Jazzy package
`ros-jazzy-mujoco-ros2-control`, in the container image) runs the profile's trained scene as ros2_control hardware.
At launch `ros2_control_backend.py` writes, for that scene:

- `scene.xml`: the same model and physics (checked equal: gains, masses, geometry, keyframe, timestep) with mjlab's
  `robot/` prefix removed and an IMU (`imu_quat`, `imu_gyro`, `imu_accel`) on the policy's IMU site;
- `robot.urdf`: only the `<ros2_control>` block (position command, position/velocity/effort state per joint, the IMU,
  ground-truth odometry of the floating base on `/sim/ground_truth/odom`);
- `controllers.yaml`: `joint_state_broadcaster` → `/joint_states`, `imu_sensor_broadcaster` → `/imu`, and
  `joint_targets_controller` (forward position) at `update_rate:=200` Hz.

Each joint keeps its MuJoCo `position` actuator, so the commanded position becomes the actuator's ctrl and MuJoCo
applies kp (target − q) − kd q̇ every 5 ms physics step: the trained law and rate, and the law the drives run in
MyActuator mode 5. Unlike the sim node there is no lockstep with the policy (as on the robot), no environment, cameras,
lidar or battery, no Mac viewer stream, and the IMU and joint states are noise-free.

```bash
ros2 launch hsim_bringup sim_stack.launch.py backend:=ros2_control viewer:=false            # Wisco
ros2 launch hsim_bringup sim_stack.launch.py robot:=g1 backend:=ros2_control viewer:=false
```
