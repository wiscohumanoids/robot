# Project status

*Last updated 2026-09-19. Read this first if you are new: it says where the
project stands, what is stubbed, which layer of the stack each piece belongs to,
and what each area should do next.*

## Where we are, in one paragraph

The repo contains a **complete, correctly-wired stack of stubs and real
low-level infrastructure**, from a spoken/typed intent at the top to (simulated)
motor torque at the bottom, with a frozen [interface contract](INTERFACE_CONTRACT.md)
between every layer. Every layer has either a real implementation or a stub that
honors its interface, so any team can replace its stub with the real thing
without touching anyone else. **No layer above low-level control does real work
yet**: the planner is canned, perception reports a hardcoded cube, state
estimation dead-reckons the command, navigation is a go-to-point controller, and
the locomotion "policy" is a sine-wave gait that cannot balance a robot. What is
real today is the plumbing, the message contract, the 1 kHz impedance
controller, the MuJoCo bridge, the safety gating, and the tooling that keeps all
of it honest.

## The stack at a glance

```
 user intent ─► task_planner ─► behavior_tree ─┬─► navigate_to_pose ─► nav ─► /cmd_vel_nav ─┐
   (speech)      (LLM)          (BT.CPP)       │                                            ├─► cmd_vel_mux ─► /cmd_vel
                                    ▲          └─► execute_manipulation ─► manipulation      │        ▲
 perception ─► /object_poses ───────┘                     │                 teleop ─► /cmd_vel_teleop ┘
 slam ─► /map, map→odom                                   │
 state_estimation ─► /robot_pose, odom→base_link          │      /cmd_vel ─► locomotion
                                                          ▼                     │
                                                   /manipulation/joint_targets  │ /locomotion/joint_targets
                                                          └────────► wbc ◄──────┘
                                                                      │ /joint_command   (500 Hz)
                                                                      ▼
                                              lowlevel_control (1 kHz) ─► MuJoCo  |  ethercat_bridge ─► real drives
                                                                      │
                                                     safety ◄── /manual_estop ◄── (E-stop firmware bridge)
```

## Status summary

<!-- BEGIN GENERATED:summary -->
| Status | Nodes | Meaning |
|---|---|---|
| **REAL** | 4 | Complete for its purpose today. |
| **STUB** | 9 | Correct interface, fake internals -- the internals are what a team builds. |
| **SCAFFOLD** | 1 | Correct structure/interface, hardware I/O stubbed. |
| **EXTERNAL** | 3 | Third-party ROS node used as-is (configure only). |
| **MISSING** | 2 | Nothing exists yet; the contract reserves the interface. |
<!-- END GENERATED:summary -->

## Every node, by layer

Status, what it does today, what will replace it, who owns that area, and the
launch flag that swaps it out (`stub|external` for stubs -- see
[the swap procedure](INTERFACE_CONTRACT.md#swapping-a-stub-for-your-real-node)).

<!-- BEGIN GENERATED:nodes -->
#### 1. Intent & speech

| Node | Status | Package | What it does today | Real thing that replaces it | Area | Launch switch |
|---|---|---|---|---|---|---|
| `speech_input` | **MISSING** | - | Turns speech/UI input into /user_intent. Until it exists, publish by hand with `ros2 topic pub`. | Whisper (or similar) speech-to-text wrapper | planning | - |

#### 2. Task planning

| Node | Status | Package | What it does today | Real thing that replaces it | Area | Launch switch |
|---|---|---|---|---|---|---|
| `task_planner` | **STUB** | `task_planner_stub` | Any intent -> canned [navigate_to, pick/place] plan for the nearest known object. | LLM + skill-library task planner | planning | `planner:=stub|external` |

#### 3. Task execution (behavior tree)

| Node | Status | Package | What it does today | Real thing that replaces it | Area | Launch switch |
|---|---|---|---|---|---|---|
| `behavior_tree` | **STUB** | `behavior_tree_stub` | Sequential executor: runs each skill via the navigate_to_pose / execute_manipulation actions, stops at first failure. | BehaviorTree.CPP tree with custom condition/action nodes | planning | `behavior_tree:=stub|external` |

#### 4. Perception & state estimation

| Node | Status | Package | What it does today | Real thing that replaces it | Area | Launch switch |
|---|---|---|---|---|---|---|
| `perception` | **STUB** | `perception_stub` | Reports one hardcoded cube in `map` at 30 Hz. | ORB-SLAM3 + FoundationPose wiring | perception | `perception:=stub|external` |
| `state_estimation` | **STUB** | `state_estimation_stub` | Dead-reckons /cmd_vel (assumes perfect tracking); publishes /robot_pose and odom->base_link. | Custom IEKF (own repo) or robot_localization | state-estimation | `state_estimation:=stub|external` |
| `slam` | **STUB** | `slam_stub` | Empty 10 m x 10 m /map and identity map->odom. | ORB-SLAM3 / slam_toolbox | perception | `slam:=stub|external` |

#### 5. Navigation & velocity arbitration

| Node | Status | Package | What it does today | Real thing that replaces it | Area | Launch switch |
|---|---|---|---|---|---|---|
| `nav` | **STUB** | `nav_stub` | navigate_to_pose action server; straight-line go-to-point, no obstacle avoidance. | Nav2 (configured, not built). Remap its velocity output to /cmd_vel_nav. | navigation | `navigation:=stub|external` |
| `cmd_vel_mux` | **REAL** | `cmd_vel_mux` | Sole publisher of /cmd_vel: teleop > navigation > zero, with a 0.5 s watchdog. | - | integration | `cmd_vel_mux:=true|false` |
| `teleop` | **REAL** | `teleop_input` | Keyboard/joystick -> /cmd_vel_teleop. Keyboard needs its own terminal (see README). | - | locomotion | `teleop:=true|false` |

#### 6. Skill runners (locomotion, manipulation)

| Node | Status | Package | What it does today | Real thing that replaces it | Area | Launch switch |
|---|---|---|---|---|---|---|
| `locomotion` | **STUB** | `locomotion_runner` | Sinusoidal hip/knee gait over a nominal crouch pose at 50 Hz. | Trained RL policy (berkeley_humanoid ONNX export) -- slot exists, adapter unverified | locomotion | `locomotion:=stub|external` |
| `manipulation` | **STUB** | `manipulation_runner` | execute_manipulation action server: scripted approach/grasp/retract, then success. | lerobot_alohamini inference wrapped as an action server | manipulation | `manipulation:=stub|external` |

#### 7. Whole-body arbitration

| Node | Status | Package | What it does today | Real thing that replaces it | Area | Launch switch |
|---|---|---|---|---|---|---|
| `wbc` | **STUB** | `wbc_stub` | Priority merge: legs+waist from locomotion, arms from manipulation when fresh (500 Hz). | Whole-body controller (QP / operational-space) | locomotion | `wbc:=stub|external` |

#### 8. Low-level control & hardware

| Node | Status | Package | What it does today | Real thing that replaces it | Area | Launch switch |
|---|---|---|---|---|---|---|
| `lowlevel_control` | **REAL** | `lowlevel_control` | ros2_control controller plugin: tau = tau_ff + Kp(q_d-q) + Kd(qd_d-qd) at 1 kHz; publishes /robot_state. | - | low-level | - |
| `ethercat_bridge` | **SCAFFOLD** | `ethercat_bridge` | SystemInterface plugin, correct state/command interfaces; bus I/O stubbed. Use with lowlevel_test.launch.py use_hardware:=true. | Real EtherCAT master + CiA402 PDO I/O (read()/write() TODOs) | low-level | - |
| `joint_state_broadcaster` | **EXTERNAL** | `joint_state_broadcaster` | Standard ros2_control broadcaster: /joint_states. | - | low-level | - |

#### 9. Safety

| Node | Status | Package | What it does today | Real thing that replaces it | Area | Launch switch |
|---|---|---|---|---|---|---|
| `safety` | **REAL** | `safety` | Aggregates /manual_estop + NaN check into /safety_status at 100 Hz; lowlevel_control zeroes torque when unsafe. | - | safety | `safety:=true|false` |
| `estop_bridge` | **MISSING** | - | Reserved. Today /manual_estop is driven by hand for tests. | Wireless E-stop firmware bridge (micro-ROS / serial). Must publish std_msgs/Bool on /manual_estop -- NOT on /safety_status. | safety | - |

#### 10. Robot model & simulation

| Node | Status | Package | What it does today | Real thing that replaces it | Area | Launch switch |
|---|---|---|---|---|---|---|
| `mujoco_sim` | **EXTERNAL** | `mujoco_ros2_control` | Vendored MuJoCo <-> ros2_control bridge (real, unmodified). sim:=false skips it and lowlevel_control. | - | low-level | `sim:=true|false` |
| `robot_state_publisher` | **EXTERNAL** | `robot_state_publisher` | Standard: URDF -> TF (base_link->pelvis static, joint frames from /joint_states). | - | integration | - |
<!-- END GENERATED:nodes -->

## Who owns what

"Area" is the responsibility, not a named person -- **assign a named owner per
area** (and a reviewer for the interface contract) and record it in
`CONTRIBUTING.md`.

<!-- BEGIN GENERATED:areas -->
| Area | Nodes it owns (status) |
|---|---|
| **integration** | `cmd_vel_mux` (real), `robot_state_publisher` (external) |
| **locomotion** | `teleop` (real), `locomotion` (stub), `wbc` (stub) |
| **low-level** | `lowlevel_control` (real), `ethercat_bridge` (scaffold), `joint_state_broadcaster` (external), `mujoco_sim` (external) |
| **manipulation** | `manipulation` (stub) |
| **navigation** | `nav` (stub) |
| **perception** | `perception` (stub), `slam` (stub) |
| **planning** | `speech_input` (missing), `task_planner` (stub), `behavior_tree` (stub) |
| **safety** | `safety` (real), `estop_bridge` (missing) |
| **state-estimation** | `state_estimation` (stub) |
<!-- END GENERATED:areas -->

## What has and has not been verified

Be skeptical of anything not on the first list.

**Verified (2026-09-19, on a macOS machine with no ROS installed):**
- All Python sources compile; unit tests pass for the pure logic (mux selection,
  dead-reckoning kinematics, go-to-goal controller incl. closed-loop
  convergence, canned planner, BT dispatch).
- The URDF expands for both hardware plugins (`mujoco_ros2_control/MujocoSystem`
  and `ethercat_bridge/EthercatHardwareInterface`), has a single `base_link`
  root, and lists the 23 joints in the canonical order. *(An earlier bug -- the
  URDF failed to expand, which would have crashed every launch file -- is fixed.)*
- The 23-joint order is identical in the URDF `ros2_control` block,
  `canonical_joint_order.yaml`, `controllers.yaml`, `ethercat_slaves.yaml`, the
  C++ default list and the MJCF; the two `NOMINAL_POSE` copies match the URDF
  initial values.
- The static contract tests (`pytest tests/`): one owner per topic/TF edge, every
  literal publisher in the code matches the contract, docs match the YAML.

**Not verified -- nobody has run this on a ROS machine yet:**
- `colcon build` of any package (message generation for the four new messages, the
  C++ controller and hardware plugin, the vendored MuJoCo bridge).
- Any `ros2 launch`, including `full_stack.launch.py` in both `sim:=true` and
  `sim:=false`, and every new stub node at runtime. The task-layer stubs were
  written without a ROS install to test against.
- `ros2 run bringup check_contract.py` (the live contract check).
- MuJoCo behavior: whether a teleop command visibly moves the sim, and what the
  free-floating robot does under the stub gait (expect it to fall; nothing here
  balances it).
- The Docker image build (arm64 support and OSMesa headless are new) and the
  GitHub Actions workflow.
- `nav2_msgs/NavigateToPose` field names against a real Humble install (written
  from memory of the Humble definition).

**First thing to do on a ROS machine:** `./docker/build.sh`, then inside the
container `colcon build`, `ros2 launch bringup full_stack.launch.py sim:=false`,
and `ros2 run bringup check_contract.py --no-sim`. Fix whatever breaks, then tick
this section.

## Next actions, by area

**Integration lead**
1. Do the first ROS build and launch above; get CI green.
2. Assign named owners per area; pick a `LICENSE` (package manifests say
   Apache-2.0, there is no license file) and add `CODEOWNERS` with real handles.
3. Decide the interface-change reviewer and put it in `CONTRIBUTING.md`.

**Planning** (`speech_input`, `task_planner`, `behavior_tree`)
- Speech front-end (Whisper) publishing `/user_intent`.
- LLM planner + skill library replacing `plan_for_intent`; grow the
  [skill vocabulary](INTERFACE_CONTRACT.md#skill-vocabulary).
- BehaviorTree.CPP tree with real condition/action nodes replacing the
  sequential executor; keep the same action clients and `/task_status`.

**Perception** (`perception`, `slam`)
- ORB-SLAM3 + FoundationPose wired in; publish `/object_poses` in `map`, `/map`,
  and `map -> odom`. Add the camera's static TF to the URDF.

**State estimation**
- IEKF (own repo) or `robot_localization` taking over `/robot_pose` and
  `odom -> base_link`. Also supplies `base_linear_velocity` that
  `PolicyObservation` needs on real hardware (zero today).

**Navigation**
- Nav2 configuration replacing `nav_stub`: keep the `navigate_to_pose` action,
  remap its velocity output to `/cmd_vel_nav`, feed it `/map` and TF.

**Locomotion** (`locomotion`, `wbc`, `teleop`)
- ONNX adapter in `locomotion_node.py` (see `INTEGRATION_POINTS.md`: the
  23<->29 DOF mapping and the nominal-pose/gait math that lives outside the
  ONNX graph are both unfinished and unverified).
- A real whole-body controller replacing `wbc_stub`'s priority merge.

**Manipulation**
- Wrap `lerobot_alohamini` inference as the `execute_manipulation` action server;
  resolve the transport (ZMQ) and joint-mapping gaps in `INTEGRATION_POINTS.md`.

**Low-level** (`lowlevel_control`, `ethercat_bridge`)
- Fill the `ethercat_bridge` TODOs (slave IDs, master init, CiA402 state
  machine, PDO read/write); confirm the unconfirmed items in `RESEARCH_NOTES.md`.
- PREEMPT_RT / CPU isolation on the real control computer.

**Safety**
- Firmware bridge publishing `std_msgs/Bool` on `/manual_estop`.
- Close the fail-open gaps below.

## Known gaps and risks

- **Safety fails open.** `lowlevel_control` treats "no `/safety_status` ever
  received" as safe, and never times out a stale `/joint_command`: if the
  `safety` node or the WBC dies, torque is not zeroed. Add heartbeat/staleness
  timeouts before any hardware run.
- **Nothing balances the robot.** `locomotion_runner`'s stub gait, on a
  free-floating MuJoCo base, is expected to fall. The sim currently proves
  plumbing, not standing.
- **The ONNX adapter is unfinished.** `_compute_action_onnx` forwards only joint
  positions into a padded 29-vector; see `INTEGRATION_POINTS.md`.
- **Stubs cannot prove real-time or dynamics behavior** (see
  [the contract](INTERFACE_CONTRACT.md#what-stubs-can-and-cannot-prove)).
- **Duplicated data.** The joint order and `NOMINAL_POSE` live in several files;
  the tests keep them consistent, but change them together.
- **Placeholder identity.** Maintainer emails are `wiscohumanoids@example.com`.
