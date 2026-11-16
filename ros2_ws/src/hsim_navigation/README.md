# hsim_navigation — localization + Nav2 for a walking humanoid (any robot profile)

`launch/navigation.launch.py mapping:=map|slam|rtabmap` (started by `hsim_bringup/autonomy.launch.py`).

| Config | Notes |
|---|---|
| `config/nav2.yaml` | NavFn (plans through unknown space) + Regulated Pure Pursuit: never < 0.3 m/s while moving, turns in place at 0.9 rad/s, `max_angular_accel` 20 (decision 0004: avoids a deadband deadlock). Footprint r = 0.30 m, inflation 0.55 m. Rotation-aware progress checker. Velocity smoother → `/cmd_vel_nav`. |
| `config/amcl.yaml` | known map; omni motion model (humanoids sidestep); initial pose = env spawn |
| `config/slam.yaml` | slam_toolbox online async; map frame starts at the robot's power-on pose |
| RTAB-Map (`mapping:=rtabmap`, default in `autonomy.launch.py`; parameters in `launch/navigation.launch.py`) | visual SLAM from the head camera's color + depth (plus the scan on the G1); `obstacle_range:=3.0` for stereo depth, whose error grows with distance squared |

Tested (`tools/ros.sh autotest`, G1): 4–5 m goals in 10–12 s with ~0.25 m final error (AMCL and SLAM); a box dropped
in the path is avoided (14 s, never touched). Current results for every robot, including known failures: `STATUS.md`.
