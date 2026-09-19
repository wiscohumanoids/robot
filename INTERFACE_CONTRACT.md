# Interface contract

This is the agreement that lets several teams build nodes in parallel without
waiting on each other. **Nodes never call each other. They publish and
subscribe (or call an action) through the interfaces below**, and neither side
knows or cares whether the other end is the real implementation, a stub, a
recording, or nothing.

> **Where does this fit?** [`STATUS.md`](STATUS.md) says what is real / stubbed /
> missing and what to do next. [`ARCHITECTURE.md`](ARCHITECTURE.md) explains the
> stack and its data flow. [`INTEGRATION_POINTS.md`](INTEGRATION_POINTS.md) says
> exactly where each team plugs in. This file is the contract they all obey.
> [`CONTRIBUTING.md`](CONTRIBUTING.md) is the workflow.

## The rules

1. **One publisher per topic. One publisher per TF edge.** Each entry below
   names its *owner*. If two nodes could publish the same thing, you have a bug,
   not a feature.
2. **A stub and the real node that replaces it honor the same entry.** Swapping
   one for the other changes nothing for anyone else and must not touch this
   contract.
3. **Changing the contract is a reviewed act.** A topic name, type, rate or
   owner, a TF edge, a skill name, or a message field is an *interface change*
   -- see [Changing the contract](#changing-the-contract).
4. **Standard message types wherever one exists** (`Twist`, `JointState`,
   `PoseStamped`, `OccupancyGrid`, `NavigateToPose`), so tooling and libraries
   (Nav2, rviz, rosbag, `robot_localization`) work unchanged. Custom messages only
   where nothing standard fits (`humanoid_interfaces`).

## How it is enforced

The machine-readable source of truth is
[`src/humanoid_interfaces/config/interface_contract.yaml`](src/humanoid_interfaces/config/interface_contract.yaml).
The tables below are **generated from it** (`python3 tools/gen_docs.py`) -- never
edit them by hand.

| Check | Where | Needs ROS? | Catches |
|---|---|---|---|
| Static contract tests | `pytest tests/` (runs in CI) | no | a node publishing a topic the contract doesn't list, or one owned by another package; a second TF parent; a stub missing its executable; a launch switch that doesn't exist; docs out of date; joint order / nominal pose drifting between files; a URDF that doesn't expand |
| Live contract check | `ros2 run bringup check_contract.py` against a running stack | yes | two publishers on one topic (by actual graph), wrong type, a topic below its minimum rate, a missing action server, a missing or doubly-parented TF edge |

## Frames

<!-- BEGIN GENERATED:frames -->
| Frame | Meaning |
|---|---|
| `map` | Global fixed frame. Owned by SLAM (slam_stub today). |
| `odom` | Smooth local frame. Drifts relative to map; never jumps. |
| `base_link` | Robot body frame (REP-103: x forward, y left, z up). Fixed to `pelvis` in the URDF. |
<!-- END GENERATED:frames -->

## Topics

<!-- BEGIN GENERATED:topics -->
| Topic | Type | Owner (sole publisher) | Consumers | Rate |
|---|---|---|---|---|
| `/user_intent` | `std_msgs/msg/String` | `speech_input` | `task_planner` | event |
| `/skill_sequence` | `humanoid_interfaces/msg/SkillSequence` | `task_planner` | `behavior_tree` | event |
| `/task_status` | `std_msgs/msg/String` | `behavior_tree` | - | 1 Hz |
| `/object_poses` | `humanoid_interfaces/msg/ObjectPoseArray` | `perception` | `task_planner` | 30 Hz |
| `/robot_pose` | `geometry_msgs/msg/PoseStamped` | `state_estimation` | `nav` | 100 Hz |
| `/map` | `nav_msgs/msg/OccupancyGrid` | `slam` | `nav` | latched |
| `/cmd_vel_teleop` | `geometry_msgs/msg/Twist` | `teleop` | `cmd_vel_mux` | event |
| `/cmd_vel_nav` | `geometry_msgs/msg/Twist` | `nav` | `cmd_vel_mux` | event |
| `/cmd_vel` | `geometry_msgs/msg/Twist` | `cmd_vel_mux` | `locomotion`, `state_estimation` | 50 Hz |
| `/locomotion/observation` | `humanoid_interfaces/msg/PolicyObservation` | `locomotion` | - | 50 Hz |
| `/locomotion/joint_targets` | `humanoid_interfaces/msg/JointTargets` | `locomotion` | `wbc` | 50 Hz |
| `/manipulation/joint_targets` | `humanoid_interfaces/msg/JointTargets` | `manipulation` | `wbc` | event |
| `/joint_command` | `humanoid_interfaces/msg/JointCommand` | `wbc` | `lowlevel_control` | 500 Hz |
| `/robot_state` | `humanoid_interfaces/msg/RobotState` | `lowlevel_control` | `locomotion`, `safety` | 1000 Hz (sim only) |
| `/joint_states` | `sensor_msgs/msg/JointState` | `joint_state_broadcaster` | `robot_state_publisher` | 1000 Hz (sim only) |
| `/manual_estop` | `std_msgs/msg/Bool` | `estop_bridge` | `safety` | event |
| `/safety_status` | `humanoid_interfaces/msg/SafetyStatus` | `safety` | `lowlevel_control` | 100 Hz |
<!-- END GENERATED:topics -->

`event` topics are published on demand (a zero-publisher graph is fine, two
publishers is not). `latched` topics use transient-local QoS. "sim only" topics
exist only when the MuJoCo/`ros2_control` path is running (`sim:=true`).

### Notes on specific interfaces

- **`/cmd_vel` is a standard `geometry_msgs/Twist`, and only `cmd_vel_mux`
  publishes it.** Teleop publishes `/cmd_vel_teleop`, navigation publishes
  `/cmd_vel_nav`; the mux forwards teleop over navigation and falls back to zero
  0.5 s after both go silent, which doubles as a watchdog if whatever was driving
  the robot dies. Only `linear.x`, `linear.y` and `angular.z` are meaningful for a
  ground-walking humanoid (REP-103 `base_link`). When real Nav2 arrives, remap its
  velocity output to `/cmd_vel_nav`; nothing else changes. *(This replaced an
  earlier custom `VelocityCommand` message that would have collided with Nav2.)*
- **`/manual_estop` is the only way an external E-stop enters the system.**
  The wireless E-stop's firmware bridge publishes `std_msgs/Bool` there. It must
  **not** publish on `/safety_status` -- that topic belongs to the `safety` node
  alone, and a second publisher would overwrite the E-stop with `is_safe: true`.
- **`/robot_pose` is in the `map` frame.** It equals `odom -> base_link`
  composed with `map -> odom`; consumers that need other frames use TF.
- **`/object_poses` is in the `map` frame** by contract: perception does the
  transform, consumers don't need TF for it.
- **Joint-space arrays** (`JointTargets`, `JointCommand`, `RobotState`,
  `PolicyObservation`) are 23 elements in the canonical order from
  [`canonical_joint_order.yaml`](src/humanoid_interfaces/config/canonical_joint_order.yaml).

## Actions

<!-- BEGIN GENERATED:actions -->
| Action | Type | Server | Clients |
|---|---|---|---|
| `navigate_to_pose` | `nav2_msgs/action/NavigateToPose` | `nav` | `behavior_tree` |
| `execute_manipulation` | `humanoid_interfaces/action/ExecuteManipulation` | `manipulation` | `behavior_tree` |
<!-- END GENERATED:actions -->

`navigate_to_pose` is deliberately Nav2's own action name and type, so the
behavior tree's client is unchanged when real Nav2 replaces `nav_stub`.

## TF ownership

TF is the one genuinely shared piece of global state. It works with many
writers because each **edge** has exactly one publisher and the framework chains
edges together -- SLAM owns `map -> odom`, state estimation owns
`odom -> base_link`, the URDF owns the rest.

<!-- BEGIN GENERATED:tf -->
| TF edge | Owner (sole publisher) | Rate | Note |
|---|---|---|---|
| `map` -> `odom` | `slam` | 20 Hz | SLAM's drift correction |
| `odom` -> `base_link` | `state_estimation` | 100 Hz | fast local motion |
| `base_link` -> `pelvis` | `robot_state_publisher` | static | from the URDF (fixed joint) |
<!-- END GENERATED:tf -->

Every other URDF link (joint frames, IMU, head, ...) hangs off `pelvis` and is
published by `robot_state_publisher` from `/joint_states` (sim/hardware path
only). `base_link` is a massless root added to the URDF for REP-105
compatibility; it is fixed to `pelvis` with an identity transform. A camera frame
will be added as a static URDF edge when perception mounts a camera.

## Skill vocabulary

`SkillSequence.skills[].name` is a shared vocabulary between the planner and the
behavior tree:

| Skill name | Behavior tree calls | Uses |
|---|---|---|
| `navigate_to` | `navigate_to_pose` (`nav2_msgs/NavigateToPose`) | `target_pose` (frame `map`) |
| `pick`, `place`, `handover`, ... (anything else) | `execute_manipulation` (`ExecuteManipulation`) with `task = name` | `object_id`, optional `target_pose` |

Adding a skill name is an interface change.

## Changing the contract

1. Edit `interface_contract.yaml` (and the `.msg`/`.action` file if a message changes).
2. Run `python3 tools/gen_docs.py` and `pytest tests/`.
3. Update the affected stub **and** every consumer in the same PR, or the live
   check and other teams' nodes break.
4. Get the PR reviewed by the interface owner (integration lead). Announce it to
   every team whose node is listed as a consumer.

Message field order/removal breaks every recorded rosbag and every node built
against the old definition; prefer adding fields at the end and adding new
messages over changing existing ones.

## Swapping a stub for your real node

```bash
ros2 launch bringup full_stack.launch.py locomotion:=external   # everything but the stub you replace
ros2 run my_pkg my_locomotion_node                              # yours: now the topic's only publisher
ros2 run bringup check_contract.py                              # confirm you honored the contract
```

Layer switches: `planner`, `behavior_tree`, `perception`, `state_estimation`,
`slam`, `navigation`, `locomotion`, `manipulation`, `wbc` (each `stub|external`),
plus `teleop`, `cmd_vel_mux`, `safety` (`true|false`), and `sim:=false` to run the
stubs without MuJoCo. Full details in
[`src/bringup/launch/full_stack.launch.py`](src/bringup/launch/full_stack.launch.py).

## What stubs can and cannot prove

Stubs prove **architectural correctness**: that data flows through the right
topics at the right rates with the right types, and they let everyone develop in
parallel. They cannot prove anything about tightly-coupled real-time behavior --
the 1 kHz loop's timing under load, EtherCAT cycle jitter, whether a policy
actually balances the real robot, sim-to-real gaps. Those only surface on
hardware; the stubbing has done its job when *only* those are left.
