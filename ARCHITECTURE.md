# Architecture

## What this repo is

`robot` is the canonical ROS2 Humble integration workspace for WiscoHumanoids.
It does not contain a working locomotion policy, a working manipulation
policy, or a working EtherCAT master — those are the jobs of, respectively,
`berkeley_humanoid`, `lerobot_alohamini`, and the low-level team filling in
this repo's scaffold. What this repo *does* provide, today, for real:

1. One canonical robot model (`g1_description`) every other piece must agree
   with.
2. One message contract (`humanoid_interfaces`) every node actually uses.
3. A complete vertical stack, teleop input down to (simulated) motor motion,
   at the *correct frequency at every layer*, with every stub clearly labeled
   as a stub and every real component clearly labeled as real.
4. A working MuJoCo sim path: send a command, watch the simulated G1 respond,
   today, on a laptop, no GPU.
5. A properly structured (not fake) `ros2_control` hardware-interface scaffold
   for a real EtherCAT torque-mode drive, so the low-level team is filling in
   `read()`/`write()`/slave YAML values, not designing the interface from
   scratch.

## The vertical stack

```
 10 Hz   ┌──────────────────┐
         │   teleop_input    │  REAL: keyboard/joystick -> VelocityCommand
         └─────────┬─────────┘
                    │ /cmd_vel  (humanoid_interfaces/VelocityCommand)
                    ▼
 50 Hz   ┌──────────────────┐        10 Hz  ┌───────────────────────┐
         │ locomotion_runner │               │  manipulation_runner   │
         │ STUB gait / ONNX  │               │  STUB action server    │
         │ slot (see below)  │               │  (ExecuteManipulation) │
         └─────────┬─────────┘               └───────────┬───────────┘
                    │ JointTargets                        │ JointTargets
                    │ (source="locomotion")                │ (source="manipulation")
                    └───────────────┬───────────────────────┘
                                    ▼
500 Hz                    ┌──────────────────┐
                           │     wbc_stub      │  STUB: pass-through arbitration
                           │  (arbitrate, no   │  (locomotion wins below the waist,
                           │  real WBC yet)    │   manipulation wins at the arms —
                           └─────────┬─────────┘   see wbc_stub/README.md)
                                    │ JointCommand
                                    ▼
1000 Hz                   ┌──────────────────┐
                           │  lowlevel_control  │  REAL: JointImpedanceController
                           │  (ros2_control     │  tau = tau_ff + Kp(q_d-q) + Kd(qd_d-qd)
                           │   controller,      │  runs inside controller_manager's
                           │   runs in-process  │  real-time update() loop
                           │   with the hw      │
                           │   plugin below)    │
                           └─────────┬─────────┘
                                    │ effort command_interface
                    ┌───────────────┴───────────────┐
                    ▼                                ▼
         ┌─────────────────────┐          ┌─────────────────────────────┐
         │ mujoco_ros2_control/  │          │  ethercat_bridge/             │
         │ MujocoSystem (sim)    │   OR     │  EthercatHardwareInterface    │
         │ REAL, vendored,       │  (arg)   │  REAL SCAFFOLD: read()/write()│
         │ working today         │          │  stubbed, clearly marked      │
         └─────────────────────┘          └─────────────────────────────┘
                    │                                │
                    ▼                                ▼
           simulated G1 in MuJoCo          real G1 over EtherCAT (not yet
           (runs today, no GPU)             wired to a real bus — TODO)

  (orthogonal, not in the vertical path)
100 Hz   ┌──────────────────┐
         │      safety       │  REAL: watches /safety_status, zeroes effort
         │                   │  and disables lowlevel_control on estop/fault
         └──────────────────┘
```

Both boxes at the bottom export the **identical interface**: state
`position`/`velocity`/`effort`, command `effort`, per joint, in
`CANONICAL_JOINT_ORDER`. `lowlevel_control` and everything above it never
knows or cares which one is active — that's the whole point of standardizing
on a raw torque interface instead of `bipedal_nav`'s original `position_pid`
custom interface (which only `MujocoSystem`'s built-in PID understood, and
which a real torque-mode CiA402 drive has no equivalent for). This is
selected via the `hardware_plugin` xacro arg on `g1_description`'s URDF —
see `bringup/README.md` and `bringup/lowlevel_test.launch.py`.

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

| Layer | Rate | Real or stub | Interface in | Interface out |
|---|---|---|---|---|
| `teleop_input` | 10 Hz | **Real** | keyboard/joystick device | `VelocityCommand` on `/cmd_vel` |
| `locomotion_runner` | 50 Hz | Stub (gait) + real ONNX slot | `VelocityCommand`, `RobotState` | `JointTargets` (`source=locomotion`) |
| `manipulation_runner` | 10 Hz | Stub (delayed-success action server) | `ExecuteManipulation` goal | `JointTargets` (`source=manipulation`), action result/feedback |
| `wbc_stub` | 500 Hz | Stub (pass-through arbitration) | both `JointTargets` streams | `JointCommand` |
| `lowlevel_control` | 1000 Hz | **Real** (impedance law, `ros2_control` controller) | `JointCommand`, hw state interfaces | hw `effort` command interface, `RobotState` |
| `ethercat_bridge` | 1000 Hz | **Real scaffold** (interface real, bus I/O stubbed) | `effort` command from controller | (would be) EtherCAT RxPDO/TxPDO |
| `mujoco_ros2_control` | 1000 Hz | **Real** (vendored, unmodified) | `effort` command from controller | simulated motion |
| `safety` | 100 Hz | **Real** | `SafetyStatus` | gates `lowlevel_control` |

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

## What's real vs. stub, one more time, unambiguously

**Real, working today:**
- `g1_description` (ported robot model)
- `mujoco_ros2_control` (vendored sim bridge)
- `mujoco_sim` (G1-specific sim bringup)
- `humanoid_interfaces` (message contract, actually consumed)
- `teleop_input` (keyboard/joystick -> `/cmd_vel`)
- `lowlevel_control`'s `JointImpedanceController` (the PD/impedance law itself)
- `safety`'s estop-gating logic
- The full pipeline wiring: a teleop command today measurably moves the
  simulated G1 in MuJoCo, through every layer in the diagram above, at each
  layer's correct rate.

**Real interface, stubbed internals (by design, per the project brief):**
- `locomotion_runner`'s gait (sinusoidal/hold-pose instead of a trained
  policy — has a labeled slot for one)
- `manipulation_runner`'s task execution (returns success after a fixed
  delay instead of running inference — has a real action-server interface)
- `wbc_stub`'s arbitration (dumb priority pass-through instead of a real
  whole-body controller / QP solver)
- `ethercat_bridge`'s bus I/O (`read()`/`write()` return simulated/logged
  values instead of real PDO frames — the `SystemInterface` plugin structure
  itself is real and loadable by `controller_manager` today)

**Not present at all (explicit gaps, not silently assumed away):**
- Perception (no cameras, no object detection, no SLAM)
- Navigation (deferred to `nav2_msgs/action/NavigateToPose` whenever a real
  nav stack exists — none does yet)
- State estimation (no EKF; `base_linear_velocity` in `PolicyObservation` is
  sim ground truth today and has no real-hardware source yet)
- A trained locomotion or manipulation policy (both live in the *other* two
  repos and are not copied here — only the slots they plug into)
- Real EtherCAT bus I/O, CiA402 state-machine handling, and all
  `PREEMPT_RT`/CPU-isolation OS-level setup (see `RESEARCH_NOTES.md`)

## Data flow, top to bottom, one full tick

1. Operator presses a key. `teleop_input` publishes `VelocityCommand{vx, vy,
   vyaw}` on `/cmd_vel` at 10 Hz.
2. `locomotion_runner`, on its own 50 Hz timer, reads the latest
   `VelocityCommand` (whatever was last published — no blocking wait) plus
   the latest `RobotState`, assembles a `PolicyObservation`, runs its stub
   gait (or, once dropped in, the ONNX policy), and publishes `JointTargets`
   with `source="locomotion"`.
3. `manipulation_runner`, independently, on its own 10 Hz timer, if it has an
   active `ExecuteManipulation` goal, publishes `JointTargets` with
   `source="manipulation"` for the arm joints.
4. `wbc_stub`, on its own 500 Hz timer, reads the latest `JointTargets` from
   both sources, merges them (legs+waist from locomotion, arms from
   manipulation when active — see `wbc_stub/README.md`), and publishes a
   single `JointCommand`.
5. `lowlevel_control`'s `JointImpedanceController`, inside
   `controller_manager`'s 1000 Hz real-time `update()` call, reads the latest
   cached `JointCommand` (via a realtime buffer, not directly in the ROS
   callback) and the hardware's current `position`/`velocity` state
   interfaces, computes `tau` per joint, and writes it to the `effort`
   command interface.
6. Whichever `SystemInterface` is loaded (`mujoco_ros2_control/MujocoSystem`
   or `ethercat_bridge/EthercatHardwareInterface`) applies that torque —
   in sim, to `qfrc_applied`; on hardware, it would be packed into a CiA402
   RxPDO `TargetTorque` frame (currently stubbed, see
   `ethercat_bridge/README.md`).
7. `lowlevel_control` re-publishes the resulting state as `RobotState` for
   `locomotion_runner` (next tick) and `safety` to consume.
8. `safety`, on its own 100 Hz timer, checks fault conditions and publishes
   `SafetyStatus`; `lowlevel_control` zeroes its torque output and refuses to
   re-enable if `is_safe` is false.

## Explicit gaps for a whole-body/humanoid architecture (kept honest)

This repo proves the *pipeline*, not a walking or manipulating robot. Beyond
the "not present at all" list above, note specifically:
- `wbc_stub`'s leg-vs-arm priority split is a hand-picked convention (legs
  always win below the waist, arms always win at the shoulders/elbow/wrist),
  not a torque-consistent whole-body QP — a real WBC would need to reconcile
  both target sets against a full dynamics model and contact constraints,
  which nothing here does.
- No inter-repo automated CI/testing exists connecting this repo to
  `berkeley_humanoid` or `lerobot_alohamini` — the ONNX/action-server "slots"
  are structurally ready but the actual artifacts they'd load do not ship in
  this repo (see `INTEGRATION_POINTS.md`).
