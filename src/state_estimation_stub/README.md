# state_estimation_stub

**Layer:** 4 (perception & state estimation). **Status:** STUB.
**Frequency:** 100 Hz.
**Subscribes:** `/cmd_vel` (`Twist`).
**Publishes:** `/robot_pose` (`PoseStamped`, frame `map`) and the TF edge
`odom -> base_link`, **the only publisher of both.**

## What is fake

Nothing is estimated. The node starts at a fixed pose (params `initial_x`,
`initial_y`, `initial_yaw`, default the origin) and dead-reckons the commanded
`/cmd_vel`, i.e. it assumes the robot tracks its command perfectly. That is
enough for navigation and the task layer to see a `/robot_pose` that moves when
they command motion. It knows nothing about what the simulated or real robot is
actually doing.

## What replaces it

The IEKF (own repo) or `robot_localization`, fusing IMU, leg kinematics and
vision. It must keep publishing `/robot_pose` in `map` and `odom -> base_link`
at >= 50 Hz. It should also provide the body-frame linear velocity that
`PolicyObservation.base_linear_velocity` needs on real hardware (zero today).

## Swap it out

```bash
ros2 launch bringup full_stack.launch.py state_estimation:=external
ros2 run <your_pkg> <your_node>
ros2 run bringup check_contract.py
```
The kinematics are in `kinematics.py` (no ROS imports, unit-tested).
