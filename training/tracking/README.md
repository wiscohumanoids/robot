# training/tracking (GPU VM)

Motion tracking (BeyondMimic-style, mjlab's tracking task): one policy reproduces one LAFAN1 human motion clip, whole
body. Runs in the `training/locomotion/` uv environment. Not used by the ROS stack yet: the walking policy is what
drives `/cmd_vel`, and a tracking policy plays one fixed motion rather than following commands.

| Path | What |
|---|---|
| `motion_to_npz.py` | retargeted CSV → motion npz (50 Hz + forward kinematics in the robot's training model), `--robot g1|wisco` |
| `train_g1_tracking.sh <clip>` | G1 (`Mjlab-Tracking-Flat-Unitree-G1`) on one clip of Unitree's G1-retargeted LAFAN1 |
| [`wisco_v1/`](wisco_v1/README.md) | Wisco V1: own LAFAN1 retargeter, the `Wisco-V1-Tracking-Flat` task, launchers |

Trained runs and motion files stay on the VM (`~/hsim_runs/`, `~/hsim_data/motions/`). Play one back with
`tools/play_mjlab.py`.
