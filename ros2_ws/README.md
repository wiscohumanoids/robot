# ros2_ws (ROS 2 Jazzy colcon workspace)

One package per layer of the stack. Each owns the topics and TF edges listed for its node in
[`docs/interface_contract.md`](../docs/interface_contract.md), and each has a README. Build inside the container:
`tools/ros.sh build` (or `colcon build --symlink-install` in `/hsim/ros2_ws`).

| Package | Layer |
|---|---|
| `hsim_sim` | simulated robot and sensors (MuJoCo) |
| `hsim_locomotion` | walking policy (ONNX) |
| `hsim_state_estimation` | leg odometry + EKF |
| `hsim_navigation` | localization / SLAM + Nav2 |
| `hsim_perception` | object detection client, line-of-sight stub, object memory |
| `hsim_behavior` | task behavior tree (C++) |
| `hsim_planner` | Claude task planner |
| `hsim_manipulation` | the `/skills/manipulate` skill (VLA) |
| `hsim_safety` | safety monitor and e-stop lock |
| `hsim_teleop` | bridges to the laptop viewer |
| `hsim_bringup` | launch files, robot profiles, interface contract, live checks |
| `hsim_interfaces` | custom messages, actions, services |
| `wisco_ethercat_ros2_hw` | ros2_control hardware interface for MyActuator drives on EtherCAT, `SafeCommandController`, `ArmController`, mock EtherCAT, arm calibration toolkit |
| `wisco_hipnuc_imu` | driver for the robot's HiPNUC IMU board on the RK3588: serial packets → `/imu` in pelvis axes, plus a bench and mount check |
| `arm01_description` | arm01's URDF, meshes, joint configs, calibration store (a link to `robots/arm01/`) |
