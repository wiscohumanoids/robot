# humanoid_interfaces

**Purpose:** the message contract every other package in this repo is written
against. **Real and consumed** -- unlike `bipedal_nav`'s `humanoid_interfaces`
package (which defined `SlamState`/`RobotState`/`ContactState`/etc. but had
zero real code importing them), every message here has at least one real
publisher and one real subscriber somewhere in `src/`. Grep for the message
name if you want to verify this yourself; it's a deliberate design goal, not
a claim to take on faith.

**Frequency:** N/A (interface definitions only, no nodes).

## Contents

| File | Published by | Consumed by |
|---|---|---|
| `msg/VelocityCommand.msg` | `teleop_input` (~10 Hz) | `locomotion_runner` |
| `msg/PolicyObservation.msg` | `locomotion_runner` (50 Hz, for logging/replay) | (loggable; policy consumes the equivalent in-process) |
| `msg/JointTargets.msg` | `locomotion_runner` (50 Hz), `manipulation_runner` (10 Hz) | `wbc_stub` |
| `msg/JointCommand.msg` | `wbc_stub` (500 Hz) | `lowlevel_control` |
| `msg/RobotState.msg` | `lowlevel_control` (1000 Hz, throttled where re-published) | `locomotion_runner`, `safety` |
| `msg/SafetyStatus.msg` | `safety` (100 Hz) | `lowlevel_control` |
| `action/ExecuteManipulation.action` | server: `manipulation_runner` | client: whatever task layer calls it (none yet -- see ARCHITECTURE.md gap list) |
| `config/canonical_joint_order.yaml` | N/A (data file) | every node that touches a 23-element joint array |

`NavigateToPose` is intentionally **not** redefined here -- this repo has no
navigation stack (bipedal_nav's was a non-functional stub, see the earlier
repo audit), so there is nothing yet to standardize an interface against.
When a real nav stack is added, use `nav2_msgs/action/NavigateToPose`
directly rather than inventing a parallel type.

## Why custom messages instead of `geometry_msgs/Twist` + `Float64MultiArray`

Two deliberate choices, both explained in-line in the `.msg` files themselves:

1. `VelocityCommand` instead of `Twist` on `/cmd_vel`: `Twist` carries 3 fields
   (`linear.z`, `angular.x`, `angular.y`) that are meaningless for a
   ground-locomoting humanoid and would be silently ignored -- a custom 3-field
   message makes the contract exact.
2. Named `float64[23]` arrays with documented order, instead of
   `std_msgs/Float64MultiArray`: a `MultiArray` carries no compile-time
   guarantee of length or order, so a bug that reorders or truncates the array
   is a silent runtime failure instead of a build-time one. Every array field
   in this package is commented with the exact 23-index order (also see
   `config/canonical_joint_order.yaml`).

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
