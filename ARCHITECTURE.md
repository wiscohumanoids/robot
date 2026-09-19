# Architecture

> Companion documents: [`STATUS.md`](STATUS.md) (what is real / stubbed / missing,
> who owns what, next actions), [`INTERFACE_CONTRACT.md`](INTERFACE_CONTRACT.md)
> (every topic, action and TF edge with its single owner),
> [`INTEGRATION_POINTS.md`](INTEGRATION_POINTS.md) (where each team plugs in).

## What this repo is

`robot` is the canonical ROS2 Humble integration workspace for WiscoHumanoids.
It does not contain a working locomotion policy, manipulation policy, task
planner, perception system or EtherCAT master -- those are the jobs of the
teams (and of the `berkeley_humanoid` and `lerobot_alohamini` repos) working
against this scaffold. What this repo *does* provide, today, for real:

1. **One interface contract** (`humanoid_interfaces` + `interface_contract.yaml`):
   every topic, action and TF edge between layers, each with exactly one owner,
   enforced by tests and a live graph checker.
2. **A stub for every layer**, from spoken intent down to motor torque, each
   honoring its interface at the correct rate, so any team can develop its real
   node against everyone else's stubs and swap it in with one launch flag.
3. **One canonical robot model** (`g1_description`) every other piece agrees with.
4. **A working MuJoCo sim path** and **a real 1 kHz impedance controller**
   (`lowlevel_control`) behind a `ros2_control` interface identical for sim and
   hardware.
5. **A properly structured (not fake) `ros2_control` hardware-interface scaffold**
   for a real EtherCAT torque-mode drive (`ethercat_bridge`).

The design principle: **nodes talk through frozen interfaces, never through each
other**, so stubs and real implementations are interchangeable and integration is
progressive stub-replacement rather than a big-bang merge. The catch, stated
plainly: stubs prove architectural correctness, not real-time timing or
dynamics -- see [what stubs cannot prove](INTERFACE_CONTRACT.md#what-stubs-can-and-cannot-prove).

## The whole stack

Two halves. The **task stack** decides *what to do* (intent -> plan -> behavior ->
navigation/manipulation requests, using perception and state estimation). The
**control stack** turns a velocity command or arm target into motor torque. They
meet at `/cmd_vel` and the `execute_manipulation` action.

```
TASK STACK  (Python stubs today; each is replaced by the real thing behind the same interface)

 /user_intent       ┌──────────────┐ /skill_sequence ┌────────────────┐
 (speech, by hand) ►│ task_planner │────────────────►│ behavior_tree  │  STUB: canned plan / sequential executor
                    │  STUB (LLM)  │                 │ STUB (BT.CPP)  │  publishes /task_status
                    └──────▲───────┘                 └───┬────────┬───┘
                           │ /object_poses               │        │
                    ┌──────┴───────┐                     │        │ action: execute_manipulation
                    │  perception  │  STUB               │        │
                    │  30 Hz       │                     │        ▼
                    └──────────────┘   action:           │   (see control stack: manipulation_runner)
                                       navigate_to_pose  ▼
 ┌────────┐ /map, map→odom       ┌────────────┐  /cmd_vel_nav   ┌─────────────┐  /cmd_vel   50 Hz
 │  slam  │ STUB ───────────────►│    nav     │ ───────────────►│ cmd_vel_mux │────────────┐
 └────────┘                      │ STUB(Nav2) │                 │    REAL     │            │
 ┌──────────────────┐ /robot_pose│  20 Hz     │◄─ /robot_pose   └──────▲──────┘            │
 │ state_estimation │ odom→base  └────────────┘                        │ /cmd_vel_teleop    │
 │  STUB, 100 Hz    │◄──── /cmd_vel (dead-reckoned)              ┌─────┴──────┐             │
 └──────────────────┘                                            │   teleop   │ REAL, 10 Hz │
                                                                 └────────────┘             │
CONTROL STACK                                                                               │
                                                                                            ▼
 50 Hz   ┌────────────────────┐                        10 Hz  ┌───────────────────────┐
         │ locomotion_runner  │◄─── /cmd_vel                  │  manipulation_runner   │
         │ STUB gait / ONNX   │                               │  STUB action server    │
         │ slot (see below)   │                               │  (ExecuteManipulation) │
         └─────────┬──────────┘                               └───────────┬───────────┘
                   │ JointTargets (source="locomotion")                   │ JointTargets (source="manipulation")
                   └───────────────────────┬──────────────────────────────┘
                                           ▼
500 Hz                            ┌──────────────────┐
                                  │     wbc_stub      │  STUB: pass-through arbitration
                                  │  (arbitrate, no   │  (locomotion wins below the waist,
                                  │  real WBC yet)    │   manipulation wins at the arms —
                                  └─────────┬─────────┘   see wbc_stub/README.md)
                                            │ JointCommand
                                            ▼
1000 Hz                           ┌──────────────────┐
                                  │  lowlevel_control  │  REAL: JointImpedanceController
                                  │  (ros2_control     │  tau = tau_ff + Kp(q_d-q) + Kd(qd_d-qd)
                                  │   controller,      │  runs inside controller_manager's
                                  │   runs in-process  │  real-time update() loop
                                  │   with the hw      │
                                  │   plugin below)    │
                                  └─────────┬─────────┘
                                            │ effort command_interface
                           ┌────────────────┴───────────────┐
                           ▼                                 ▼
                ┌─────────────────────┐          ┌─────────────────────────────┐
                │ mujoco_ros2_control/  │          │  ethercat_bridge/             │
                │ MujocoSystem (sim)    │   OR     │  EthercatHardwareInterface    │
                │ REAL, vendored,       │  (arg)   │  REAL SCAFFOLD: read()/write()│
                │ working today         │          │  stubbed, clearly marked      │
                └─────────────────────┘          └─────────────────────────────┘
                           │                                 │
                           ▼                                 ▼
                  simulated G1 in MuJoCo            real G1 over EtherCAT (not yet
                  (runs today, no GPU)               wired to a real bus — TODO)

  (orthogonal, not in the vertical path)
100 Hz   ┌──────────────────┐   /manual_estop ◄── (E-stop firmware bridge, MISSING)
         │      safety       │  REAL: aggregates /manual_estop + NaN checks into
         │                   │  /safety_status; lowlevel_control zeroes effort and
         └──────────────────┘  refuses to re-enable while unsafe
```

Both hardware boxes export the **identical interface**: state
`position`/`velocity`/`effort`, command `effort`, per joint, in
`CANONICAL_JOINT_ORDER`. `lowlevel_control` and everything above it never
knows or cares which one is active — that's the whole point of standardizing
on a raw torque interface instead of `bipedal_nav`'s original `position_pid`
custom interface (which only `MujocoSystem`'s built-in PID understood, and
which a real torque-mode CiA402 drive has no equivalent for). This is
selected via the `hardware_plugin` xacro arg on `g1_description`'s URDF —
see `bringup/README.md` and `bringup/lowlevel_test.launch.py`.

### Coordinate frames and TF

`map -> odom -> base_link -> pelvis -> (URDF links)`. Each edge has exactly one
publisher: SLAM owns `map -> odom`, state estimation owns `odom -> base_link`,
`robot_state_publisher` owns everything from `base_link` down (the URDF, plus
joint frames from `/joint_states`). Full table:
[INTERFACE_CONTRACT.md](INTERFACE_CONTRACT.md#tf-ownership).

### Why `/cmd_vel` has a mux

Nav2 and every off-the-shelf teleop node publish `Twist` on `/cmd_vel`; two
publishers on one topic is what the contract forbids. So teleop publishes
`/cmd_vel_teleop`, navigation `/cmd_vel_nav`, and `cmd_vel_mux` (teleop >
navigation > zero, 0.5 s timeout) is the sole owner of `/cmd_vel`. The timeout
doubles as a watchdog: if whatever was driving dies, the robot is commanded to
stop rather than repeating its last command.

## Sim/hardware symmetry — why this works

Verified directly in `mujoco_ros2_control/src/mujoco_system.cpp`: the
`effort`-command branch of `write()` clamps the incoming torque to the
joint's `<limit effort="...">` from the URDF and writes it straight into
MuJoCo's `qfrc_applied` array — it does **not** require or interact with any
`<actuator>` element in the MJCF. Verified in `g1_description`'s
`g1_23dof_rev_1_0.xml`: there are **zero** `<actuator>` elements (a comment in
the file says so explicitly), and `<option timestep="0.001">` means physics
already steps at exactly 1kHz. Net effect: the sim path is not an
approximation of the 1kHz torque loop, it *is* the 1kHz torque loop, applied
to a simulated instead of a physical body. The only thing that changes
between sim and hardware is which `SystemInterface` plugin
`controller_manager` loads.

## Frequencies (enforced with ROS2 timers, not busy-loops)

Authoritative, machine-checked version: the `topics` table in
[INTERFACE_CONTRACT.md](INTERFACE_CONTRACT.md) (`check_contract.py` verifies
minimum rates on a live stack). Summary:

| Layer / node | Rate | Real or stub | Interface in | Interface out |
|---|---|---|---|---|
| `task_planner` | event | Stub | `/user_intent`, `/object_poses` | `/skill_sequence` |
| `behavior_tree` | event + 1 Hz status | Stub | `/skill_sequence` | actions `navigate_to_pose`, `execute_manipulation`; `/task_status` |
| `perception` | 30 Hz | Stub | (none) | `/object_poses` |
| `slam` | 20 Hz TF, latched map | Stub | (none) | `/map`, TF `map->odom` |
| `state_estimation` | 100 Hz | Stub | `/cmd_vel` | `/robot_pose`, TF `odom->base_link` |
| `nav` | 20 Hz while a goal is active | Stub | `navigate_to_pose` goal, `/robot_pose` | `/cmd_vel_nav` |
| `teleop_input` | 10 Hz while active | **Real** | keyboard / joystick | `/cmd_vel_teleop` |
| `cmd_vel_mux` | 50 Hz | **Real** | `/cmd_vel_teleop`, `/cmd_vel_nav` | `/cmd_vel` |
| `locomotion_runner` | 50 Hz | Stub (gait) + real ONNX slot | `/cmd_vel`, `RobotState` | `JointTargets` (`source=locomotion`) |
| `manipulation_runner` | 10 Hz | Stub (delayed-success action server) | `ExecuteManipulation` goal | `JointTargets` (`source=manipulation`), action result/feedback |
| `wbc_stub` | 500 Hz | Stub (pass-through arbitration) | both `JointTargets` streams | `JointCommand` |
| `lowlevel_control` | 1000 Hz | **Real** (impedance law, `ros2_control` controller) | `JointCommand`, hw state interfaces | hw `effort` command interface, `RobotState` |
| `ethercat_bridge` | 1000 Hz | **Real scaffold** (interface real, bus I/O stubbed) | `effort` command from controller | (would be) EtherCAT RxPDO/TxPDO |
| `mujoco_ros2_control` | 1000 Hz | **Real** (vendored, unmodified) | `effort` command from controller | simulated motion |
| `safety` | 100 Hz | **Real** | `/manual_estop`, `RobotState` | `SafetyStatus` |

## Canonical joint order (23 DOF)

Single source of truth: `humanoid_interfaces/config/canonical_joint_order.yaml`,
mirrored as an inline comment in `g1_description/g1_23dof.urdf.xacro`'s
`<ros2_control>` block.

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

This is the `g1_23dof_rev_1_0` variant from `bipedal_nav/src/g1_description`:
legs 6 DOF each, waist yaw only (roll+pitch locked), arms 5 DOF each
(shoulder pitch/roll/yaw, elbow, wrist roll only — no wrist pitch/yaw, rubber
hand end effector).

### Reconciliation with `berkeley_humanoid`

`berkeley_humanoid`'s `G1Env` (`environment/g1_env.py`) trains against the
**29-DOF** G1 variant: same 12 leg joints and same `waist_yaw_joint`, but adds
`waist_roll_joint` + `waist_pitch_joint` (2 extra) and, per arm,
`wrist_pitch_joint` + `wrist_yaw_joint` (4 extra) — 6 extra DOF total,
23 + 6 = 29. Confirmed by reading `g1_env.py`'s own qpos-layout docstring and
cross-checking joint names against `environment/g1_robot.xml`'s `<actuator>`
list.

This is **not** a blocking mismatch. Per the earlier audit of
`berkeley_humanoid`, `G1Env`'s `correction_scaling` vector is ~zero for every
joint except the leg hip/knee pitch joints used for the walking gait — i.e.
the policy already treats the 6 extra DOF as held-fixed, not as joints it
actively uses for balance. The 29<->23 conversion is therefore a safe
truncation/padding at the `locomotion_runner` boundary, not a requirement to
retrain the policy from scratch. See `INTEGRATION_POINTS.md` for the exact
index mapping the adapter code must implement, and verify the
`correction_scaling` assumption still holds before trusting it against a
newer checkpoint.

## What's real vs. stub

The per-node table (status, what it does today, what replaces it, owner,
launch switch) lives in [`STATUS.md`](STATUS.md) and is generated from the
contract, so it cannot drift from the code. The short version:

**Real, working today (as far as static checks can tell -- see STATUS.md for
what has not yet been run on a ROS machine):** `g1_description`,
`mujoco_ros2_control` (vendored), `humanoid_interfaces` and the contract
tooling, `teleop_input`, `cmd_vel_mux`, `lowlevel_control`'s
`JointImpedanceController`, `safety`'s gating logic.

**Real interface, stubbed internals (by design):** `task_planner`,
`behavior_tree`, `perception`, `slam`, `state_estimation`, `nav`,
`locomotion_runner`, `manipulation_runner`, `wbc_stub`, and `ethercat_bridge`'s
bus I/O.

**Not present at all:** the speech front-end (`speech_input`), the E-stop
firmware bridge (`estop_bridge`), a trained locomotion or manipulation policy
(they live in the other two repos; only the slots exist here), real EtherCAT
bus I/O and CiA402 state handling, and the `PREEMPT_RT`/CPU-isolation OS setup
(see `RESEARCH_NOTES.md`).

## Data flow, one task, top to bottom

1. A user intent (`"pick up the cube"`) reaches `/user_intent` -- typed by hand
   today; the speech front-end later.
2. `task_planner` turns it, using the latest `/object_poses` from `perception`,
   into a `SkillSequence` on `/skill_sequence`: `[navigate_to, pick]`.
3. `behavior_tree` runs the skills in order. For `navigate_to` it calls the
   `navigate_to_pose` action; for `pick` it calls `execute_manipulation`; it
   reports progress on `/task_status`.
4. **Navigation:** `nav` drives toward the goal using `/robot_pose`, publishing
   `/cmd_vel_nav`. `cmd_vel_mux` forwards it (unless someone is using teleop) as
   `/cmd_vel` at 50 Hz. `state_estimation` updates `/robot_pose` and
   `odom -> base_link`; `slam` provides `/map` and `map -> odom`.
5. `locomotion_runner`, on its own 50 Hz timer, reads the latest `/cmd_vel`
   (whatever was last published -- no blocking wait) plus the latest
   `RobotState`, assembles a `PolicyObservation`, runs its stub gait (or, once
   dropped in, the ONNX policy), and publishes `JointTargets` with
   `source="locomotion"`.
6. **Manipulation:** `manipulation_runner`, while it has an active goal, publishes
   `JointTargets` with `source="manipulation"` for the arm joints at 10 Hz.
7. `wbc_stub`, at 500 Hz, merges both sources (legs+waist from locomotion, arms
   from manipulation when fresh -- see `wbc_stub/README.md`) into one `JointCommand`.
8. `lowlevel_control`'s `JointImpedanceController`, inside `controller_manager`'s
   1000 Hz real-time `update()`, reads the latest cached `JointCommand` (via a
   realtime buffer, not in the ROS callback) and the hardware's `position`/
   `velocity` state, computes `tau` per joint, and writes the `effort` command.
9. Whichever `SystemInterface` is loaded applies that torque -- in sim to
   `qfrc_applied`; on hardware it would be packed into a CiA402 RxPDO
   `TargetTorque` frame (currently stubbed, see `ethercat_bridge/README.md`).
10. `lowlevel_control` re-publishes state as `RobotState` for `locomotion_runner`
    (next tick) and `safety`. `safety`, on its own 100 Hz timer, publishes
    `SafetyStatus` from `/manual_estop` and NaN checks; `lowlevel_control` zeroes
    its torque and refuses to re-enable while `is_safe` is false.

Every arrow above is an entry in the contract, and every box can be a stub or the
real thing.

## Explicit gaps (kept honest)

This repo proves the *pipeline*, not a walking or manipulating robot.

- **Nothing balances the robot.** `locomotion_runner`'s stub gait on the
  free-floating MuJoCo base is expected to fall over; the sim currently proves
  plumbing, not standing. `wbc_stub`'s leg-vs-arm priority split is a hand-picked
  convention, not a torque-consistent whole-body QP.
- **Safety fails open.** `lowlevel_control` treats "no `/safety_status` received
  yet" as safe and never times out a stale `/joint_command`, so a crashed
  `safety` node or WBC does not zero torque. Add heartbeat/staleness timeouts
  before any hardware run.
- **Estimation is assumed, not measured.** `state_estimation_stub` dead-reckons
  the command, and `PolicyObservation.base_linear_velocity` has no real-hardware
  source yet (zero).
- **The ONNX adapter is unfinished and unverified** (`INTEGRATION_POINTS.md`).
- **No inter-repo CI** connects this repo to `berkeley_humanoid` or
  `lerobot_alohamini`; the ONNX/action-server "slots" are structurally ready but
  the artifacts they load don't ship here.
- **Nothing here has been built or launched on a ROS machine yet** (only static
  checks and pure-logic unit tests have run) -- see STATUS.md.
