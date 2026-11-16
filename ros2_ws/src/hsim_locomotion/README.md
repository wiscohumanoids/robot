# hsim_locomotion (Phase 1) — node `locomotion`

| Executable | Status | What |
|---|---|---|
| `policy_node` | **REAL** | ONNX policy from `models/locomotion/<id>/` (param `policy_dir`), 50 Hz on sim time, CPU |
| `locomotion_stub` | stub | holds the default pose |

Observation (same as training): body velocity from /odometry/filtered, gyro and gravity direction from /imu, joint positions/velocities of every manifest joint from /joint_states (matched by name), last action, and the /cmd_vel command (zeroed after 0.5 s without one). 99 values for the all-joints (29-joint) policy, 85 for a legs+waist (arms-free) policy.
Publishes /locomotion/joint_targets and /locomotion/status (`mode=policy`, policy id, inference time, tilted flag, driven joints, held-joints source).

## Manifest versions (models/locomotion/<id>/manifest.yaml, written by training/locomotion/scripts/export_manifest.py)
| Version | Policy drives | Other joints |
|---|---|---|
| 1 | every joint in `joints` | none |
| 2 | `action.joints` (e.g. 12 legs + 3 waist) | `held_joints.names` (arms): default pose until the first `sensor_msgs/JointState` on `held_joints.source_topic` (default `/arm/joint_targets`), then the latest commanded target per joint (no timeout; joints missing from a message keep their target; unknown names are ignored with a warning) |

/locomotion/joint_targets always carries every joint: the policy's joints first (action order), then the held joints.
The manifest/observation logic is in `policy_contract.py` (no ROS), shared with `tools/sim2sim_play.py`.
Tests: `python -m pytest ros2_ws/src/hsim_locomotion/test -q` (needs numpy, pyyaml, pytest).
