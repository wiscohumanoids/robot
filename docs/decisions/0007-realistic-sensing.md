# 0007 — The robot stack only uses what real sensors give it

**Rule.** Nodes in the robot stack may not read simulator ground truth. Ground truth (`/sim/ground_truth/*`) is
**privileged**: only graders (`tools/eval/*`), tests, and explicitly labeled stubs may read it.

| Information | Source now | Realistic? |
|---|---|---|
| joint angles/speeds | encoders (`/joint_states`) | yes |
| orientation, gyro, accel | IMU model: noise, bias, onboard attitude filter (heading drifts ~3°/min) | yes |
| body velocity, odometry | leg odometry (stance-foot kinematics) + EKF; ~1.6% drift | yes |
| pelvis height, tilt, waist pose | leg_odometry TF from kinematics + IMU | yes |
| position on the map | AMCL on a known map, slam_toolbox, or RTAB-Map | yes |
| obstacles | head lidar (rays, 16 beams flattened) or stereo depth (ZED X-like noise model) | yes |
| battery | power drawn by the motors | yes |
| **which objects are where** | **line-of-sight perception stub**: honest visibility (FOV, range, size, occlusion), noisy relative positions, but reads true object positions | **stub until Phase 4** |

**Frames (REP-105/120 humanoid layout):** `map → odom` (localization) → `base_footprint` (EKF, 2D) → `base_link`
(pelvis: height + tilt) → `torso_link` (waist joints) → `head_camera` (→ `head_camera_optical`). Lidar: `base_footprint
→ lidar_link` (level scan). IMU: `base_link → imu_link`.

**Sensor setups (swappable):** `sensors:=lidar` (G1-style, default) or `sensors:=stereo` (ZED X-style). The stereo
camera renders in its own process at ~2.5–5 Hz on CPU in Docker (a real ZED X runs 15–60 Hz), so stereo localization
and navigation are noticeably less precise here (≈0.5 m vs ≈0.25 m final error).
