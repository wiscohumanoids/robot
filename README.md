# WiscoHumanoids Hardware Interface

Demos:
https://drive.google.com/file/d/1t2eEQImacynxiRTeHf_oAAU96mAUvgAz/view
https://drive.google.com/file/d/1zwNxs48vFlNf9jD_jDFN_XyyGplYSpis/view

ros2_control bring-up for MyActuator RMD X-series drives (CiA402 over EtherCAT) using
[ICube ethercat_driver_ros2](https://github.com/ICube-Robotics/ethercat_driver_ros2).
This package is the base of the arm control stack: the arm's controllers are
ros2_control plugins that run on top of the hardware interface defined here.

Today it runs a test bench of four left-arm drives (2× X4-10, 2× X4-36) at 100 Hz on a
laptop without PREEMPT_RT. The target is the RK3588 as real-time EtherCAT master, with a
Jetson Orin (v1) sending task-level targets over DDS.

## Quick start

```bash
# ethercat_driver_ros2 must be in the same workspace (or installed)
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install && source install/setup.bash

# Without drives: everything on mock hardware
ros2 launch wisco_ethercat_ros2_hw test_bench.launch.py use_mock_hardware:=true

# With drives (EtherCAT master running, drives powered)
sudo systemctl start ethercat
ros2 launch wisco_ethercat_ros2_hw test_bench.launch.py
```

To run without sudo, give the EtherCAT device to your user:

```bash
echo 'KERNEL=="EtherCAT[0-9]*", MODE="0666"' | sudo tee /etc/udev/rules.d/99-ethercat.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

The launch file starts the hardware, `joint_state_broadcaster` and one command
controller. It does **not** move anything on its own: pass `demo:=true` to also start
the demo script for the selected mode.

| Argument | Default | Meaning |
|---|---|---|
| `joints_file` | `config/joints.yaml` | Joint table (see below) |
| `models_file` | `config/actuator_models.yaml` | Per-model numbers |
| `calibration_file` | `config/calibration.yaml` | Per-joint zero offsets |
| `control_mode` | from the joint table | `velocity` (CSV) or `effort` (CST) |
| `update_rate` | `100` | Controller manager rate **and** EtherCAT cycle, Hz |
| `joint_state_rate` | `100` | `/joint_states` rate, Hz (capped at `update_rate`) |
| `command_timeout` | `0.1` | Seconds without a command before the joints are stopped |
| `use_mock_hardware` | `false` | Run the drives on `mock_components/GenericSystem` (velocity commands move the joints; effort commands are only echoed to the effort state) |
| `demo` | `false` | Start `multi_velocity_cmd.py` (velocity) or `multiple_act_command.py` (effort) |

## One joint table

Everything that depends on the wiring is generated at launch from three files:

- **`config/joints.yaml`**: one row per drive: URDF joint name, `bus_position` (its
  physical place on the ring, 0 = first after the master), `model` and `direction`
  (±1, so a positive command moves the joint positively about its URDF axis). Optional
  per-joint overrides: `max_velocity_rad_s`, `max_effort_nm`, `max_torque_permille`,
  `damping_nm_s_per_rad`, `soft_limits_rad`, `k_stiction`. Also holds `control_mode`.
- **`config/actuator_models.yaml`**: encoder counts, torque scale, peak torque, speed,
  bring-up limits and the 0x6072 torque cap per model, plus vendor id and DC sync.
- **`config/calibration.yaml`**: zero offsets, so joint angle = direction × encoder −
  offset. The drives have no home-offset object, so the offset is folded into the
  PDO scale factors.

From these the launch file writes, into a temp directory that is removed on shutdown:

- one slave config per drive (from `config/drive_pdo_template.yaml`) with that drive's
  scale factors, direction, zero offset, mode of operation and torque cap,
- the hardware manifest the xacro reads to build the `ros2_control` block (ring order
  is the table's, in any mix of models), and
- the controller parameters: joint list, per-joint command limit, position limits from
  the URDF (or `soft_limits_rad`), damping.

The table is checked before anything starts: unknown joints or models, duplicate or
non-contiguous bus positions, bad directions and calibration entries for joints not in
the URDF all stop the launch with a message. Warnings (no zero offset, suspicious torque
scale) are printed. URDF joints not in the table run on mock hardware (state only), so
`/joint_states` always covers the whole robot.

**Adding or moving a drive** is one row in `config/joints.yaml`. Then run the bring-up
check (below).

**Switching CSV ↔ CST** is `control_mode` in the table or `control_mode:=effort` at
launch. The drive mode (SDO and 0x6060), the commanded PDO channel, the exported command
interface and the controller all follow. Commands go to
`/velocity_controller/commands` (rad/s) or `/effort_controller/commands` (Nm) as
`std_msgs/Float64MultiArray`, one value per joint in table order.

## Command controller

Both controllers are `wisco_ethercat_ros2_hw/SafeCommandController`, a forward
controller with the checks the stock ones lack:

- **Timeout**: no command for `command_timeout` → velocity 0, or in effort mode a
  damping torque (never a push). A crashed or stalled script stops the arm.
- **Magnitude**: each joint is clamped to its `max_command` (bring-up limits from the
  model table unless overridden).
- **Soft position limits**: within `soft_margin` (0.1 rad) of a limit, motion toward it
  is scaled down to zero at the limit; past a limit, outward commands are dropped and
  in effort mode outward motion is damped. Moving back in is always allowed. In effort
  mode this cannot stop a joint that gravity pushes through a limit; mechanical stops
  and the 0x6072 cap are still needed.
- **Bad input**: NaN commands are treated as stale; arrays of the wrong length are
  ignored with a warning; the joints are stopped on activate and deactivate.

Only the selected mode's command interface is exported, so nothing can command the
other one by accident.

## Bring-up procedure

1. `ros2 run wisco_ethercat_ros2_hw check_urdf.py` (offline). Reports left/right axis,
   limit, effort and velocity mismatches and URDF limits that disagree with the wired
   actuator model.
2. `ethercat slaves`: every drive is in PREOP/OP and the count matches the table.
3. Launch in velocity mode with the arm clear, then
   `ros2 run wisco_ethercat_ros2_hw bringup_check.py`. It jogs each drive in bus order
   and checks that only the named joint moved, by about the commanded amount, in the
   commanded direction, and asks you to confirm the right physical joint moved the
   right way. A failure means a wrong `bus_position` or `direction`.
4. Zero each joint at its URDF zero pose and record the offsets in
   `config/calibration.yaml` (bump `version`). Until then, position limits assume the
   encoder zero is the URDF zero, and the launch file says so.
5. Only then raise the bring-up limits in the model table or per joint.

## Real-time setup (for 1 kHz)

- PREEMPT_RT kernel; run `ros2_control_node` with RT priority (`limits.conf`: `rtprio`
  and `memlock` for your user).
- Isolate a core for the control loop (`isolcpus=`/`nohz_full=`) and pin the EtherCAT
  NIC's IRQs to it (`/proc/irq/*/smp_affinity`).
- Set `update_rate:=1000`: the launch file uses the same value for the EtherCAT
  `control_frequency`, so they cannot drift apart.
- Leave `joint_state_rate` at 100–250 Hz; the controllers read state interfaces
  directly, and a 1 kHz JointState stream only loads DDS.
- DC sync is on (`dc_assign_activate: 0x0300` in the model table). Check that value
  against the drive's ESI on the bench; `0` turns DC off.

## Architecture

```text
wisco_ws/src/
├── ethercat_driver_ros2/          # ICube, add separately
└── wisco_ethercat_ros2_hw/        # this package
    ├── config/joints.yaml              # joint -> bus position, model, direction
    ├── config/actuator_models.yaml     # per-model numbers
    ├── config/calibration.yaml         # zero offsets
    ├── config/drive_pdo_template.yaml  # PDO map shared by all RMD-X drives
    ├── launch/test_bench.launch.py     # generates configs, starts everything
    ├── urdf/test_bench.urdf.xacro      # robot + ros2_control block from the manifest
    ├── src/safe_command_controller.cpp # command controller (pluginlib)
    ├── wisco_ethercat_ros2_hw/         # Python: config generation, ROS helpers
    └── scripts/                        # demos, bring-up check, URDF check

ros2_control_node (one RT process)
├── EthercatDriver (SystemInterface): one CiA402 drive per row of joints.yaml
├── joint_state_broadcaster
└── SafeCommandController (velocity or effort), later the arm controllers
```

Controller plan: first a joint PD + gravity/friction feedforward controller plugin
(same process, reads the state interfaces, writes effort), validated on the bench; then
task-space control; a whole-body QP only once that works. The Jetson sends targets to
the controller's command topic at 50–100 Hz over DDS; it is not in the RT loop.

Hardware notes: all drives currently share one EtherCAT ring, so one bad connector
stops the whole arm. Route cables so each joint's connector can be reseated without
disassembly, and watch the working counter (`ethercat domains`).

## Arm stack (arm01, the 7-DOF test arm)

The calibration and control stack from the spec ("Arm Calibration and Control Stack
Spec"), on top of this package's hardware layer.

```bash
# the arm (mock hardware first)
ros2 launch wisco_ethercat_ros2_hw arm01.launch.py use_mock_hardware:=true
ros2 launch wisco_ethercat_ros2_hw arm01.launch.py

# one command per stage; --sim runs the same code on the sim backend
ros2 run wisco_ethercat_ros2_hw arm bringup --sim      # or: python3 -m arm_stack.cli bringup --sim
ros2 run wisco_ethercat_ros2_hw arm all --sim --quick --fresh   # every stage on one sim arm
ros2 run wisco_ethercat_ros2_hw arm all --sim --quick --fresh --no-mocap   # the chain without mocap
ros2 run wisco_ethercat_ros2_hw arm status
ros2 run wisco_ethercat_ros2_hw arm export pace ~/arm_logs/SESSION.mcap out
ros2 run wisco_ethercat_ros2_hw arm export lerobot out_dir ~/arm_logs/A.mcap ~/arm_logs/B.mcap
```

| Stage | Command | Writes |
|---|---|---|
| 1 Bring up | `arm bringup` | joint config `direction` (with `--write-config`) |
| 1 Indicator | `arm indicator` | nothing (report: 90° step vs protractor, torque sign, 0x6077 units, nulls) |
| 2 Zero | `arm zero` | `zero_offsets_rad` |
| 3 Limits | `arm limits` | `soft_limits_rad` |
| 4 Torque constant | `arm kt` | `kt_output_nm_per_a` |
| 5 Gravity comp | `arm gravity` | `link_inertials` |
| 6 Friction | `arm friction` | `friction` |
| 7 Latency | `arm latency` | `latency_ms` |
| 8 Kinematics | `arm pivot` | `validation.pivot_spread_mm` |
| Plate kinematics (no mocap) | `arm plate` | `zero_offsets_rad`, `kinematic_overlay`, `joint_stiffness_nm_per_rad` |
| Mocap kinematics | `arm mocap-calib` | `zero_offsets_rad`, `kinematic_overlay`, `joint_stiffness_nm_per_rad` |
| 9 PACE | `arm pace` | `pace` |
| Mocap repeatability | `arm mocap-repeat` | `validation.repeatability_rp_mm` (+ tuned settings to apply) |
| ISO 9283 repeatability | `arm repeat [--indicators]` | `validation.repeatability_rp_mm` |
| Release gate | `arm validate` | `validation.*`, report beside the calibration file |
| Float check | `arm gravity-check` | joint config `float_friction_gain` (with `--write-config`) |

### Float (teach / hand guiding) and gravity-check

```bash
arm float                                   # Ctrl-C ends it: damps to a stop, then a Mode P hold
arm float --lock shoulder_pitch,shoulder_roll --payload 0.8@0,0,0.05 --record   # MCAP every cycle
arm float --sim --duration 4                # sim: an operator pushes tool0 sideways, then lets go
arm gravity-check [--payload KG@x,y,z] [--write-config]
```

`arm float` (`arm_stack/control/float_mode.py`) runs Mode T with kp = 0: RT gravity comp
(plus the payload), Coulomb help `float_friction_gain x Fc x ramp(|qd|)` with a 0.02 rad/s
deadband and a 0.03 rad/s smoothstep ramp (zero at rest, so it cannot start a drift; gain < 1,
so a joint let go always stops), optional `--damping` (RT kd), and spring-damper walls from
10° inside the soft limits. Safety: any joint above 0.5 x `max_vel` latches strong damping
(no friction help) until all are below 0.25 x for 0.2 s; every command carries the bring-up
torque clamp; the 50 ms watchdog is unchanged. `--lock` holds joints with a stiff RT PD
(5 x / 2 x the joint gains) plus a slow integrator: the controller switches CSP/CST for the
whole arm, so a per-joint Mode P hold is not available. `--friction-gain` (one value or one
per joint) overrides the joint config's `float_friction_gain` (default 0.6).

`--payload KG@x,y,z` is a point mass in the tool0 frame. It rides on the command as an
optional trailing `[kg, x, y, z]` block (`ArmController` keeps it, adds it to the inertia of
tool0's parent link, and uses it in Mode P, Mode T and the watchdog fallback; kg 0 clears it,
commands without the block leave it; parameter `payload_frame`, default tool0). Float and
gravity-check always send one (none without `--payload`), so it is never left from an
earlier session.

`arm gravity-check` (`arm_stack/calib/gravity_check.py`), complementing validate's hold test:
1. at N poses (6, `--quick` 3) the arm floats with no friction help or damping for 3 s; per
   joint, the drift's acceleration through the mass matrix (with armature) plus the friction
   model gives the gravity residual; > 5% of peak gravity is flagged. Joints that do not move
   are reported as below the breakaway.
2. per joint, others locked: a torque kick each way, then float at gain g; bisection for the
   largest g that still stops within 2 s (above the breakaway the joint creeps), and 0.9 x that
   is the recommended `float_friction_gain`, written to the joint config by `--write-config`.

### Without mocap: divot plate and indicator cube

Kinematics and repeatability have a path that needs no mocap. `arm all` picks it when the
client reports no mocap (`--no-mocap` makes the sim do that): `plate` instead of
`mocap-calib`, `repeat` on the indicator cube instead of `mocap-repeat` (its tuning is skipped
with the reason), and `validate` measures repeatability on the cube.

- `arm plate`: a machined plate with 6 to 9 divots at known positions (`plate:` in the joint
  config; the numbers there are placeholders until the plate drawing exists,
  `needs_drawing: true`). Hand guide the probe tip into each divot in several configurations,
  then again with the 1 kg test mass hanging. One fit from tip positions only, with the plate's
  divot spacing as the length reference: plate pose, probe-tip offset, joint zero offsets, a
  rank-selected part of the kinematic overlay, stiffness and backlash (`calib/kinfit.py`, shared
  with mocap-calib). J1's zero folds into the plate pose and keeps its jig value. J7's zero and the
  probe's angle about J7 are one parameter: the tip offset is stored folded into J7's zero and
  origin (exact at tool0; its radial part is reported). The report gives each parameter's
  observability and 1 sigma and the condition number. The held-out divots (`plate.held_out`)
  are never fitted: their tip RMS and the pair distances to the other divots, measured against
  the drawing, validate the fit (with the parameters as stored). Pass: held-out RMS < 0.5 mm,
  condition number < 1e4.
- `arm repeat --indicators`: ISO 9283 (warm-up, 5 poses x 30 cycles, `--quick` 6) with three
  orthogonal dial indicators on a cube on the flange (`indicator_cube:` in the joint config,
  `arm_stack/hw/indicator_cube.py`). Translation only, one pose per fixture setup: set the stand
  and zero the indicators at a pose, run its cycles, move to the next. On hardware you type the
  three readings (mm) each cycle while the arm holds; digital indicators can go through
  `SerialIndicatorReader` (a stub until the interface is on the bench). `arm validate` without
  mocap asks before using the cube; the sim uses it automatically.

```bash
arm plate                      # hardware: prompts for each divot and configuration
arm repeat --indicators        # hardware: prompts per pose (fixture) and per cycle (readings)
arm plate --sim --quick        # sim operator: truth IK into each divot
arm repeat --sim --quick --indicators
arm all --sim --quick --fresh --no-mocap
```

### Single-joint bench (Phases 1 and 2, before the arm is assembled)

`config/arms/bench01.yaml` is one actuator on a stand with a lever on its output
(`urdf/bench01/bench01.urdf`: shaft horizontal, q = 0 with the lever level, +90° hanging
down, test masses on a hook 0.25 m out). The same stages run on it; `--arm` takes a name.
Every actuator, spares included, goes through it, so give its serial and model: results go
to `config/calib/bench01/<serial>/` and every version carries `actuator_serial`.

```bash
# Phase 1: bring-up and the indicator check (commands 0° then 90°, you type the protractor
# reading; also torque sign against gravity, 0x6077 units, and every null left in the joint row)
arm bringup   --arm bench01 --serial SN123 --model x4_36
arm indicator --arm bench01 --serial SN123 --model x4_36

# Phase 2: the lever rig, per actuator: zero (lever level), kt on the three hook masses
# (rig.test_masses_kg for the model), gravity (lever COM), friction, latency; then
# rig_<date>_v<N>.yaml beside the versions with the gate (Kt linear across 3 loads, friction R² > 0.9)
arm rig --arm bench01 --serial SN123 --model x4_36

# the same in sim
arm indicator --sim --arm bench01 --fresh
arm rig --sim --quick --arm bench01 --serial SIM-1 --fresh
arm all --sim --quick --arm bench01 --fresh    # every stage; pivot / mocap / validate SKIP on 1 joint
```

The rig's lever length, hook point and test masses are under `rig:` in `bench01.yaml`; kt uses
them instead of its default 0.15 m lever and computed loads. On hardware the stages prompt for
each mass and for the lever-level zero. Hardware runs still need a launch for bench01 (the
arm01 launch file is hard-wired to arm01's joint table).

Layout:

- `urdf/arm01/arm01.urdf`: URDF from the STEP (`tools/cad/extract_arm01.py` rebuilds the
  meshes and numbers; the ledger at the top of the URDF lists every frame convention and
  assumption). `arm01.ros2_control.xacro` adds the ros2_control block.
- `config/arms/arm01.yaml`: the arm's joint config (`control_mode: arm`).
  `config/calib/arm01/<date>_v<N>.yaml`: calibration versions, append-only, written by the stages.
- `src/arm_controller.cpp`, `include/.../arm_law.hpp`: the 1 kHz RT loop as a ros2_control
  controller. Mode P = CSP target (0x607A) + gravity/friction torque offset (0x60B2);
  Mode T = CST host PD + feedforward (0x6071). Pinocchio gravity from `robot_description` plus
  the calibration file, soft limits, torque/velocity clamps, bumpless switching, and a 50 ms
  watchdog that drops to gravity comp + damping (with the payload, if one was sent).
  `~/command` and `~/state` layouts are in `arm_controller.hpp`.
- `arm_stack/`: Python. `calib/` (stages), `control/` (the same law in Python, trajectories),
  `hw/` (ArmClient API; `ros_client.py` for the arm), `sim/` (sim backend with injectable
  errors), `model/` (Pinocchio model + calibration), `pace/` (CMA-ES, replay backend, GP-BO),
  `log/` (MCAP recorder, exports), `cli.py`.

Tests: `test/test_arm_law.cpp` (gtest) and `test/arm_stack/` (pytest, same law cases in
Python, optimizers, ISO 9283, recorder; `test_nomocap.py` runs the plate fit and the cube
repeatability in sim; `test_float.py`: float and gravity-check in sim, about 1 min). The sim's
operator hooks are `SimArm.apply_external_force` / `apply_external_torque`; its Coulomb +
stiction is a velocity-level impulse (sticks below the breakaway, no tanh chatter creeping
light joints). The Phase 0 gate (every procedure recovers its
injected errors in sim) is `ARM_PHASE0_GATE=1 python3 -m pytest test/arm_stack/test_phase0_gate.py`.

Mock EtherCAT (`test/mock/`): `RmdxMockSlave` is an RMD-X drive as a CiA402 slave (0x6040/0x6041
state machine, CSP/CST, 0x6072, a one-joint model, injectable faults, lost frames),
`Cia402DriveMaster` transcribes the upstream `EcCiA402Drive` per-cycle logic (the package is
not in the workspace, so it is not linked), and `MockEthercatSystem` is a ros2_control
`SystemInterface` over both. `test_mock_slave` tests the drive and master logic;
`test_arm_controller_mock` runs the real `ArmController` and `SafeCommandController` against it
(enable, fault and reset, CSP/CST switching, the 50 ms watchdog, torque clamps, rejections,
the payload block in gravity and in the watchdog fallback).
The mock's guesses about the RMD-X are the `RmdxParams` fields marked "guess".
`ArmController`'s `drive_feedback` (default off) needs `status_word` (0x6041) and
`mode_of_operation` (0x6061) state interfaces on each drive joint, which the ros2_control
block does not export yet.

Assumptions that need the bench (also in the URDF ledger):

- Zeros: J1, J3, J5, J6 and J7 read 0 at the CAD pose; J4 reads 0 with the elbow straight
  (the CAD has it flexed 24.7°, so the geometry past J4 was rotated back). J2's lower limit
  is 0 (the given ROM is +190 only). Joint signs follow the right-hand rule about the URDF
  axes; `arm bringup` checks them against gravity, and J3 is mounted inverted so its
  `direction` may need to be −1.
- The wrist pitch parallelogram (crank 41.75, rod 70.0, ground 69.9 mm) is modeled 1:1 with
  mimic joints; only `wrist_pitch` is driven.
- Link masses come from CAD volumes with placeholder densities until the weighed values
  land in `link_masses_kg`.
- The ArmController writes each command's torque clamp to 0x6072 (`max_torque` command
  interface, `write_max_torque: true` in arm mode), so the drive's own CSP loop is capped
  too, which matters when `arm limits` pushes into a hard stop. 0x6072 is in the default
  RxPDO 0x1600 (MyActuator ESI). **0x6072 = 0 is the drive's stall-current limit, not zero
  torque**, so the controller never writes less than 1% of the joint's `max_torque`.
- Drive facts from MyActuator's "EtherCAT control protocol V2.0" and ESI
  `MT-Device_260424.xml` (firmware 26042402):
  - 0x6071 / 0x6077 / 0x6072 are per mille of the **rated current** (0x6075, mA, read it
    over SDO), so `effort_nm_per_permille` = Kt x I_rated / 1000 in
    `config/actuator_models.yaml`. 0x6076 (rated torque) is not implemented.
  - 0x60B2 (torque offset) is not in the default 0x1600 = [6040, 607A, 60FF, 6071, 6072,
    6060, 2FFD]. Arm mode assigns the optional RxPDO 0x1603 = [60B0, 60B1, 60B2] next to
    it, which needs firmware 26042402 or newer. Older firmware: Mode P has no gravity
    feedforward.
  - The CSP/CSV loop gains are 0x2002-0x2008 (REAL32, SDO only). The manuals give neither
    units nor defaults and the loop is cascaded (position P into velocity PI), not a Nm/rad
    PD: read them on the bench (`ethercat upload -p <pos> -t float 0x2002 0`) before
  `mocap-repeat` tunes Kp/Kd. Saving to
    flash is Controlword 0 -> 0x1000 -> 0 within 3 s with the motor disabled.
  - Vendor 0x00202008 and product 0 (one generic device for every model), DC
    AssignActivate 0x0300, minimum cycle 250 us. Read 0x6502 from the drive; the ESI
    default is odd.
- The model table uses the V4 "-E" datasheets (X4-10, X4-36, X6-60). The V4.1 "-L"
  successors of the X4s have different torque and current: check the labels. The X4-10
  ratio (12.5 in the table, 12.6 in the user manual, 1:13 on the label), the X4s'
  counts_per_rev (131072 assumed for the 18-bit output encoder) and the X4s' Kt (the
  datasheet Kt disagrees with rated torque / rated current) are marked "verify".
- The RT clamps (`max_torque_nm`, `max_vel_rad_s`) are the rated values; 0x6072 caps each
  drive at 1.2x the bring-up effort limit.
- J7's axis sits about 62° from the forearm at zero because J6's zero is the CAD pose.
- The PACE stage fits with a per-joint replay backend by default. The Isaac Lab backend
  (`arm_stack/pace/isaac_backend.py`, `fit(..., backend='isaac')`) replays the same logs or
  `arm export pace` npz files on the full arm, one env per candidate, but is untested
  against a real Isaac Lab install (only against a fake `isaaclab` in the tests): build the
  USD with `tools/isaac/import_urdf.py` and check it with `tools/isaac/check_model.py`
  first. The USD fixes the parallelogram's mimic joints and merges them into link5, as
  Pinocchio does, so only `wrist_pitch` is driven. Each chirp stops below its
  joint's servo resonance (about 5 Hz on the shoulders) because the per-joint replay
  cannot follow the coupled arm above it, and PACE uses the drives' CSP gains from
  `drive_gains` in the joint config (read them over SDO) rather than fitting them.
  With the gains known the replay also simulates the held joints under their CSP loops
  (coupled replay), and it applies the calibrated `joint_stiffness_nm_per_rad`: the
  encoder reads the motor side, link angle + tau / k. That is why `arm all` runs
  plate / mocap-calib before PACE. A joint with no stiffness is replayed rigid (flagged),
  and its armature and viscous then soak up the compliance.
  The replay also clips each drive's total torque at `max_torque_nm` (0x6072), which a
  shoulder chirp near its servo resonance reaches. Joints whose chirp stays well below the
  bus rate get the stage 7 latency as their delay instead of fitting it, because delay and
  armature trade off there.
- `latency_ms` is the transport delay on the torque channel (0x6071 sent vs 0x6077 read,
  Mode T), not the spec's position cross-correlation, which includes the drive's servo lag
  (both are logged).
- The gravity fit's pass threshold has a floor of one 0x6077 count, so low-gravity wrist
  joints are not failed on quantization.
- The ROS client re-derives raw counts and per-mille from converted values, and reports the
  status word as enabled, until the drive manifest exposes 0x6064/0x6077/0x6041 as state
  interfaces.

## Scripts

| Script | Mode | What it does |
|---|---|---|
| `multi_velocity_cmd.py` | velocity | Swings each joint around its start position (position tracking, so no drift), amplitude capped by the limits |
| `multiple_act_command.py` | effort | Feedforward torque wave sized from each joint's `k_stiction` |
| `example_torque_command_ros_node` | effort | Pulses one joint (`-p joint:=...`) to break stiction |
| `bringup_check.py` | velocity | Wiring and direction check, see above |
| `check_urdf.py` | offline | URDF symmetry and actuator-limit check |
| `example_ros_node` (C++) | n/a | Minimal node that listens on `/effort_controller/commands` |

The scripts read the joint list and limits from the running controller, so they always
agree with what it will enforce. They publish zeros on exit; the controller's timeout
stops the joints anyway if a script dies.

## Known open items (need the bench, CAD or datasheet)

- Torque scale (`effort_nm_per_permille`) for X4-10 and X4-36 implies more than the
  rated torque at 1000 ‰; measure Kt and fix. Effort readings and CST commands are
  suspect until then.
- URDF: the right arm was exported in a different pose (elbow, forearm and wrist axes
  do not mirror the left), left/right wrist pitch effort and velocity disagree, and the
  shoulder effort/velocity limits do not match the X4-10s wired there. See
  `check_urdf.py`.
- Whether 0x6064 reports the output-side or motor-side encoder (the scale factors
  assume output side, 2^17 counts per output revolution).

## Debugging

- `ros2 control list_controllers` / `ros2 control list_hardware_interfaces`
- `ros2 topic echo /joint_states`
- `ethercat slaves`, `ethercat domains` (working counter should be N/N)
- `ros2 topic hz /joint_states`
