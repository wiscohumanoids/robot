# Unitree G1 models

## mjlab_1.6.0/ — the exact world the velocity policy trained in
| File | What |
|---|---|
| g1_velocity_flat.xml | **load this one.** Physics options from training (5 ms step, implicitfast, Newton, 10 iterations, pyramidal cones) + includes scene.xml |
| scene.xml | `export-scene Mjlab-Velocity-Flat-Unitree-G1` from mjlab 1.6.0, unmodified: G1, 29 position actuators with training PD gains, IMU + contact sensors, floor, `init_state` keyframe |
| assets/ | the 35 G1 meshes from mjlab's asset_zoo (Unitree, via MuJoCo Menagerie) |

Checked on 2026-11-08 against `models/locomotion/g1_velocity_flat_snapshot/manifest.yaml`: all 29 joints present, kp/kd match within rounding, base starts at 0.76 m.

Things a deploy loop must get right:
- Names carry a `robot/` prefix (`robot/left_hip_pitch_joint`, `robot/imu_ang_vel`).
- **Actuator order differs from the policy's joint order.** Always map by joint name, never by index.
- Policy runs every 4 physics steps (50 Hz).
