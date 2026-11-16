# hsim_state_estimation — where the robot thinks it is (no ground truth)

| Executable | Node | What |
|---|---|---|
| `leg_odometry` | `leg_odometry` | body velocity from stance-foot kinematics + IMU (`/leg_odometry`, 50 Hz, <2% speed error); body TF `base_footprint → base_link` (pelvis height + tilt) and `base_link → torso_link` (waist joints) |
| robot_localization `ekf_node` | `state_estimation` | fuses leg odometry + IMU → `/odometry/filtered`, `odom → base_footprint` (config `config/ekf.yaml`) |
| `state_estimation_stub` | `state_estimation` | DEBUG ONLY (`estimator:=truth`): copies ground truth |

Measured: 8 cm position drift and 2° heading drift after 4.7 m of walking and turning (`tools/eval/odometry_drift.py`).
