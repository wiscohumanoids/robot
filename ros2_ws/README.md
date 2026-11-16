# ros2_ws (ROS 2 Jazzy colcon workspace)

One package per layer; each owns the topics and TF edges listed for its node in `docs/interface_contract.md`, and
each has a README. Build in the container: `tools/ros.sh build` (or `colcon build --symlink-install` in `/hsim/ros2_ws`).

| Package | Layer |
|---|---|
| `hsim_sim` | simulated robot + sensors (MuJoCo) |
| `hsim_locomotion` | walking policy (ONNX) |
| `hsim_state_estimation` | leg odometry + EKF |
| `hsim_navigation` | localization / SLAM + Nav2 |
| `hsim_perception` | detection client, stub, object memory |
| `hsim_behavior` | task behavior tree (C++) |
| `hsim_planner` | Claude task planner |
| `hsim_manipulation` | `/skills/manipulate` (VLA) |
| `hsim_safety` | safety monitor, e-stop lock |
| `hsim_teleop` | Mac viewer bridges |
| `hsim_bringup` | launch files, robot profiles, interface contract, live checks |
| `hsim_interfaces` | custom messages, actions, services |
| `hsim_description`, `hsim_monitor` | planned (README only) |
| `wisco_ethercat_ros2_hw` | EtherCAT hardware configuration for the real robot |
| `lowlevel_control`, `mujoco_ros2_control` | ported ros2_control controller and MuJoCo plugin, not built yet (`COLCON_IGNORE`) |
