# humanoid_interfaces

**Purpose:** the message contract every other package in this repo is written
against, plus [`config/interface_contract.yaml`](config/interface_contract.yaml) --
the machine-readable list of every topic, action and TF edge, its owner, type and
rate (rendered in [INTERFACE_CONTRACT.md](../../INTERFACE_CONTRACT.md), enforced by
`tests/` and `ros2 run bringup check_contract.py`).

**Only invent a custom message where no standard one fits.** `/cmd_vel`,
`/joint_states`, `/robot_pose`, `/map` and `navigate_to_pose` use standard types on purpose.

**Real and consumed** -- unlike `bipedal_nav`'s `humanoid_interfaces`
package (which defined `SlamState`/`RobotState`/`ContactState`/etc. but had
zero real code importing them), every message here has a publisher and a subscriber
(real or stub) somewhere in `src/`. Grep for the message
name if you want to verify this yourself; it's a deliberate design goal, not
a claim to take on faith.

**Frequency:** N/A (interface definitions only, no nodes).

## Contents

| File | Published by | Consumed by |
|---|---|---|
| `msg/Skill.msg`, `msg/SkillSequence.msg` | `task_planner` (event, on `/skill_sequence`) | `behavior_tree` |
| `msg/ObjectPose.msg`, `msg/ObjectPoseArray.msg` | `perception` (30 Hz, on `/object_poses`, frame `map`) | `task_planner` |
| `msg/PolicyObservation.msg` | `locomotion_runner` (50 Hz, for logging/replay) | (loggable; policy consumes the equivalent in-process) |
| `msg/JointTargets.msg` | `locomotion_runner` (50 Hz), `manipulation_runner` (10 Hz) | `wbc_stub` |
| `msg/JointCommand.msg` | `wbc_stub` (500 Hz) | `lowlevel_control` |
| `msg/RobotState.msg` | `lowlevel_control` (1000 Hz, throttled where re-published) | `locomotion_runner`, `safety` |
| `msg/SafetyStatus.msg` | `safety` (100 Hz) | `lowlevel_control` |
| `action/ExecuteManipulation.action` | server: `manipulation_runner` | client: `behavior_tree` |
| `config/interface_contract.yaml` | N/A (data file) | `tools/gen_docs.py`, `tests/`, `bringup/check_contract.py` |
| `config/canonical_joint_order.yaml` | N/A (data file) | every node that touches a 23-element joint array |

`NavigateToPose` is intentionally **not** redefined here: navigation uses Nav2's
own `nav2_msgs/action/NavigateToPose` (served by `nav_stub` today, real Nav2 later).

## Why standard types where they exist

Earlier versions of this package defined a custom `VelocityCommand` for
`/cmd_vel`. It was removed: `/cmd_vel` is now a plain `geometry_msgs/Twist` so
Nav2, `teleop_twist_keyboard`, joysticks and rosbag tooling work unchanged (only
`linear.x`, `linear.y`, `angular.z` are meaningful for a ground-walking humanoid).
The joint-space messages stay custom because a `float64[23]` with a documented
order is safer than a `Float64MultiArray` that carries no length or order: a
reordered or truncated array is a typed error, not a silent runtime bug. Every
array field is index-aligned with `config/canonical_joint_order.yaml`.

## Canonical joint order

23 DOF, matching `g1_description`'s `g1_23dof_rev_1_0` variant exactly:

```
0  left_hip_pitch_joint        6  right_hip_pitch_joint     12 waist_yaw_joint
1  left_hip_roll_joint         7  right_hip_roll_joint      13 left_shoulder_pitch_joint
2  left_hip_yaw_joint          8  right_hip_yaw_joint       14 left_shoulder_roll_joint
3  left_knee_joint             9  right_knee_joint          15 left_shoulder_yaw_joint
4  left_ankle_pitch_joint     10  right_ankle_pitch_joint   16 left_elbow_joint
5  left_ankle_roll_joint      11  right_ankle_roll_joint    17 left_wrist_roll_joint
                                                             18 right_shoulder_pitch_joint
                                                             19 right_shoulder_roll_joint
                                                             20 right_shoulder_yaw_joint
                                                             21 right_elbow_joint
                                                             22 right_wrist_roll_joint
```

See `ARCHITECTURE.md` for the full reconciliation against `berkeley_humanoid`'s
29-DOF training environment.
