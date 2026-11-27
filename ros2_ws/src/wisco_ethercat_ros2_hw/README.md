# wisco_ethercat_ros2_hw — hardware interface and joint controllers

The `ros2_control` side of the robot: how the MyActuator RMD-X drives are configured on the EtherCAT bus, the
real-time controllers that run on top of them, a mock EtherCAT drive to test all of it without hardware, and the
Python toolkit that commissions, calibrates and identifies an arm. It runs on the **RK3588 (Rockchip)**, the robot's
real-time computer; the central computer (**Jetson Thor or Jetson Orin NX**) sends it targets over ROS 2 at 50–100 Hz
and is never inside the real-time loop. The drives' bus driver is ICube's
[`ethercat_driver_ros2`](https://github.com/ICube-Robotics/ethercat_driver_ros2) on the IgH EtherCAT master; both are
installed on the robot computer, not in the development container.

```
ros2_control_node (one real-time process on the RK3588)
├── EthercatDriver (hardware interface): one CiA 402 drive per row of the joint table
├── joint_state_broadcaster
├── SafeCommandController   velocity or torque commands with timeouts, clamps and soft limits
└── ArmController           arm control at the cycle rate: gravity compensation (Pinocchio) + PD, two drive modes
```

The arm this is tested on is described in [`robots/arm01/`](../../../robots/arm01/README.md) (the `arm01_description`
package): its model, joint config, calibration procedures and results.

## What is here

| Path | What |
|---|---|
| `config/actuator_models.yaml` | per-motor-model numbers: encoder counts, torque scale, peak torque and speed, bring-up limits, torque cap, vendor id, distributed-clock setting |
| `config/drive_pdo_template.yaml` | the PDO map shared by every RMD-X drive |
| `config/joints.yaml`, `config/calibration.yaml` | the test-bench joint table (left-arm drives) and its zero offsets |
| `wisco_ethercat_ros2_hw/hardware_config.py` | generates every drive's slave config, the hardware manifest and the controller parameters from a joint table, and checks the table first |
| `wisco_ethercat_ros2_hw/calibration_files.py` | finds an arm's calibration versions (`<date>_v<N>.yaml`, newest last), ignoring reports that share the folder |
| `urdf/ros2_control_macros.xacro`, `urdf/test_bench.urdf.xacro` | the `<ros2_control>` block built from the manifest; the test-bench robot |
| `launch/test_bench.launch.py`, `launch/arm01.launch.py` | bring up the bench or arm01, on drives or on mock hardware |
| `src/safe_command_controller.cpp`, `include/.../safety_logic.hpp` | `SafeCommandController` and its unit-tested safety functions |
| `src/arm_controller.cpp`, `src/gravity_model.cpp`, `include/.../arm_law.hpp` | `ArmController`: the arm's real-time control law |
| `src/joint_target_controller.cpp`, `config/wisco_joint_targets.yaml` | `JointTargetController`: a policy's joint targets (topic) to the drives, with timeout, ramp-in and limits |
| `src/parallel_linkage_controller.cpp`, `include/.../parallel_linkage.hpp`, `config/wisco_parallel_joints.yaml` | `ParallelLinkageController`: "PR mode" for the two-motor ankles and waist (joints ↔ motors through the rods) |
| `src/pp_test.cpp`, `src/pp_motion.hpp`, `config/pp_test.yaml` | `pp_test`: profile-position check of one drive or the whole bus, straight on the IgH master (see below); built only where the EtherCAT libraries are installed |
| `src/current_sensor_node.cpp`, `src/ads7128.hpp` | power-rail current readout (ADS7128 I2C ADC + ACS758 sensors) on `/power/currents` |
| `arm_stack/` | Python toolkit: commissioning and calibration stages, the same control law, a simulated arm, PACE system identification, IK, MCAP logging and export; `scripts/arm` is its CLI |
| `scripts/` | `bringup_check.py` (wiring and direction check), `check_urdf.py` (URDF vs actuator limits), `mocap_check.py` (mocap stream rate, gaps and noise), two demo scripts |
| `launch/mocap.launch.py` | motion capture into ROS 2 (upstream `vrpn_mocap`): each rigid body on `/vrpn_mocap/<body>/pose` |
| `tools/` | model checks, IK benchmark, workspace map, Isaac Lab USD import, parity golden-file writers (`make_parity_golden.py`, `make_linkage_golden.py`) (reports go to `build/analysis/`, not kept in the repo) |
| `test/` | C++ tests (safety logic, control law, mock drives, controllers against the mock bus, C++/Python gravity parity) and Python tests |

## One joint table

Everything that depends on the wiring is generated at launch from three files, then checked before anything starts:

- the **joint table** (`config/joints.yaml`, or an arm's `robots/<robot>/config/<arm>.yaml`): one row per drive with the
  URDF joint name, `bus_position` (place on the ring, 0 = first after the master), `model` and `direction` (±1, so a
  positive command moves the joint positively about its URDF axis), plus optional per-joint limits, damping and soft
  limits, and the table's `control_mode`;
- the **actuator model table** (`config/actuator_models.yaml`);
- the **calibration file**: zero offsets, so joint angle = direction × encoder − offset (the drives have no home-offset
  object, so the offset is folded into the PDO scale factors).

From these the launch writes, into a temporary directory removed on shutdown, one slave config per drive, the
hardware manifest the xacro reads, and the controller parameters. Unknown joints or models, duplicate or
non-contiguous bus positions, bad directions and calibration entries for joints not in the URDF stop the launch with a
message. URDF joints not in the table run on mock hardware (state only), so `/joint_states` always covers the robot.
Adding or moving a drive is one row in the joint table, followed by the bring-up check.

**Control modes** (`control_mode` in the table, or `control_mode:=` at launch):

| Mode | Drive mode of operation | Command | Used by |
|---|---|---|---|
| `velocity` | 9, CSV | rad/s on `/velocity_controller/commands` | bench bring-up |
| `effort` | 10, CST | N·m on `/effort_controller/commands` | torque tests |
| `arm` | starts in 8 (CSP); `ArmController` switches to 10 (CST) at runtime | position + torque offset, or torque | `ArmController` |

The drive mode, the commanded PDO channel, the exported command interface and the controller all follow the setting,
and only the selected mode's command interface is exported. The walking joints (legs, waist, head) are planned to run
MyActuator mode 5 (position + kp + kd every cycle, PD in the drive); see `robots/wisco_v1/README.md`.

## Controllers

**`SafeCommandController`**, a forward controller with the checks the stock ones lack:
- **timeout:** no command for `command_timeout` → velocity 0, or in torque mode a damping torque (never a push), so a
  crashed or stalled script stops the joint;
- **magnitude:** each joint clamped to its `max_command`;
- **soft position limits:** within the margin of a limit, motion toward it is scaled down to zero at the limit; past a
  limit, outward commands are dropped and, in torque mode, outward motion is damped; moving back in is always allowed;
- **bad input:** NaN commands count as stale, wrong-length arrays are ignored with a warning, the joints are stopped on
  activate and deactivate.

**`ArmController`**, the arm's control law, run every cycle:
- **Mode P:** CSP position target (0x607A) plus a gravity and friction torque offset (0x60B2), so the drive's own
  position loop tracks while the host cancels gravity.
- **Mode T:** CST torque (0x6071) = host PD + gravity + friction feedforward.
- Pinocchio gravity from `robot_description` and the calibration file, optional payload at the tool, soft limits,
  torque and velocity clamps, bumpless switching between modes, and a 50 ms watchdog that falls back to gravity
  compensation + damping. Options: hold position on startup or on command timeout, gravity compensation on/off, and a
  per-joint switch for drives whose firmware lacks the torque-offset object (those joints use the plain position path).
- It writes each command's torque clamp to the drive's torque limit (0x6072), so the drive's own position loop is capped
  too. 0x6072 = 0 means the drive's stall-current limit, not zero torque, so it never writes less than 1% of the limit.
- With a **relative** calibration (a resting-pose reference, see `robots/arm01/README.md`) it only holds the position it
  starts at (`hold_only`), or, after a passing local calibration, accepts position commands inside the measured local
  envelope (at most 20° wide per joint) with the fitted local gravity model; CAD gravity and torque mode stay off until the arm has a
  registered (URDF) calibration.
- Command and state layouts are documented in `include/wisco_ethercat_ros2_hw/arm_controller.hpp`.

## Policy targets on the robot

On the robot the walking policy runs on the central computer and publishes `/locomotion/joint_targets` (joint names
and positions, 50 Hz). `JointTargetController` takes them into the RK3588's real-time loop:

```
policy (Jetson) --/locomotion/joint_targets--> leg_targets (JointTargetController)
                                                 |-- hips, knees, waist yaw, head --> drives: position + kp + kd
                                                 '-- ankles, waist roll/pitch --> parallel_joints --> 2 motors each
```

| Phase | What it does |
|---|---|
| hold | on activation, hold the measured pose until targets arrive |
| ramp | blend from the measured pose to the policy's (moving) targets over `ramp_time` (2 s; Unitree's deployment does the same before its policy runs) |
| track | pass targets through, clamped to the URDF limits; kp and kd per joint from the policy's `manifest.yaml`, i.e. the PD it was trained with |
| safe | no message for `timeout` (50 ms): `damp` (kp 0, kd 8, Unitree's exit command) or `hold`; the two-motor joints hold. When messages return, ramp again |

It is a safe pass-through, not a whole-body controller: the drives run the PD (mode 5). One instance per policy
output: the walking policy's here; a manipulation policy's arm targets would get their own instance or the
ArmController. Joints a message leaves out keep their last target, and joints not in the instance are ignored.
Tested against fake drives (`test/test_joint_target_controller.cpp`) and chained into `parallel_joints` in a real
controller manager on mock motors (hold, ramp at the expected rate, exact final pose, timeout to damping). Config for
Wisco's 17 non-arm joints: `config/wisco_joint_targets.yaml`. The two computers' network and clock:
[`infra/robot_network/`](../../../infra/robot_network/README.md).

`/joint_states` on the robot: the hardware's joints are the motors, so the upstream joint_state_broadcaster would
publish `left_ankle_motor_a` where the policy, leg odometry and the rest expect `left_ankle_pitch`.
`ChainedJointStateBroadcaster` publishes the URDF's joints in one message, reading the two-motor joints from
`parallel_joints`' exported states (`joints`, `chained_joints`, `chain_prefix`, `topic`; load it after
`parallel_joints`). The whole chain runs in simulation on simulated drives: `sim_stack.launch.py
backend:=ethercat_sim` (`hsim_sim/README.md`).

## Parallel joints: ankles and waist ("PR mode")

Each ankle (pitch, roll) and the waist (roll, pitch) is driven by **two motors through push rods**, but the URDF,
the simulator and the walking policy treat them as two ordinary joints. `ParallelLinkageController` translates
between the two, as Unitree's G1 does in its "PR mode" (`unitree_sdk2`, G1 ankle example: `Mode::PR`, "Series Control
for Pitch/Roll Joints", versus `Mode::AB`, "Parallel Control for A/B Joints"), so the policy only ever sees pitch and
roll:

```
leg controller ──joint references──► ParallelLinkageController ──motor commands──► 2 drives per mechanism
 (pitch, roll)   ◄──joint states────   (rod geometry from CAD)   ◄──motor states──
```

- **Joints → motors** (`include/.../parallel_linkage.hpp`): closed form. Each rod has a fixed length, so each crank
  angle solves `P cos θ + Q sin θ = K`; of the two roots, the one the robot is assembled in (the CAD's) is kept.
- **Motors → joints**: Newton's method on that, warm-started from the last cycle (one or two iterations), with each
  step limited so a bad guess cannot overshoot. Velocities go through the Jacobian J, torques through Jᵀ.
- **Gains**: a joint-space PD seen at the motors is `J⁻ᵀ K J⁻¹`. In `pd_in_drive` mode (MyActuator mode 5) each drive
  gets its diagonal, and the dropped coupling term is reported; `torque` mode computes the exact joint PD here and
  sends motor torques. **At the ankles the diagonal is not enough:** the coupling is about 70 % of the diagonal, so
  with the trained 35 / 35 Nm/rad the ankle becomes about 124 Nm/rad in pitch and 20 in roll (the rod geometry fixes
  the ratio; no per-motor gains fix it). On simulated drives (`backend:=ethercat_sim`) the robot walks at 0.38 m/s
  instead of 0.55 for 0.5 commanded in `pd_in_drive`, and falls about 2 s after it stops; in `torque` mode it walks
  at 0.54 and stops cleanly (so the simulated robot runs `torque`: `ethercat_sim.command_mode` in hsim_bringup's
  `config/robots/wisco.yaml`; this file keeps `pd_in_drive` until the choice is made). The waist's coupling is about
  7 % and harmless. Options for the ankles: `torque` mode (needs a steady bus at a high enough rate), or mode 5 with
  the coupling sent as torque feedforward each cycle (not implemented).
- **Safety**: references are clamped to the joint limits. A motor reading the rods cannot produce, or one that only
  fits a physically impossible configuration (another rod assembly, a pose past the limits), holds the motors where
  they are and is counted. The controller will not activate on such readings: that is the check that the motors are
  calibrated and their directions right.
- **Geometry** is `robots/wisco_v1/linkage.json`, measured from the CAD; `robots/wisco_v1/linkage.py` is the Python
  reference, also used in training (per-motor limits through the rods, `training/locomotion/wisco_v1/wisco_mjlab/
  ankle_actuator.py`).

Setup (`config/wisco_parallel_joints.yaml`): one joint-table row per motor; calibrate each motor to read 0 at the
joint's zero pose (foot flat, torso upright), with the direction that turns its crank the way linkage.json does; the
upstream controller then commands `parallel_joints/<joint>/position` (and velocity, effort). Tested against the Python
reference (`test/test_parallel_linkage.cpp`, `test/test_linkage_golden.py`), against fake motors
(`test/test_parallel_linkage_controller.cpp`), and chained under a position controller in a real controller manager
on mock motors. Not yet on the drives: the motors' calibration and directions, and mode 5 itself.

## Run it

```bash
# on mock hardware (no drives), from a built and sourced workspace
ros2 launch wisco_ethercat_ros2_hw test_bench.launch.py use_mock_hardware:=true
ros2 launch wisco_ethercat_ros2_hw arm01.launch.py use_mock_hardware:=true

# on the drives (EtherCAT master running, drives powered)
sudo systemctl start ethercat
ros2 launch wisco_ethercat_ros2_hw test_bench.launch.py            # demo:=true also starts a demo script
ros2 launch wisco_ethercat_ros2_hw arm01.launch.py enable_motion:=true
```

| Launch argument | Default | Meaning |
|---|---|---|
| `description` | `urdf/test_bench.urdf.xacro` | robot description xacro (relative to this package's share or absolute) |
| `joints_file` | `config/joints.yaml` | joint table |
| `calibration_file` | `config/calibration.yaml` | zero offsets (arm01: its newest calibration version) |
| `control_mode` | from the table | `velocity`, `effort` or `arm` |
| `update_rate` | `100` | controller manager rate **and** EtherCAT cycle, Hz; raise toward 1000 only after checking loop timing on the target |
| `joint_state_rate` | `100` (arm01: `50`) | `/joint_states` rate (capped at `update_rate`) |
| `command_timeout` | `0.1` | seconds without a command before the joints are stopped |
| `use_mock_hardware` | `false` | `mock_components/GenericSystem` instead of drives |
| `enable_motion` | `false` | let the drives go to Operation Enabled (arm mode on real drives). Off by default: the arm launch starts with the motors disabled; set it after a reference or calibration exists |
| `demo` | `false` | start the demo script for the selected mode |

To run without sudo, give the EtherCAT device to your user:
`echo 'KERNEL=="EtherCAT[0-9]*", MODE="0666"' | sudo tee /etc/udev/rules.d/99-ethercat.rules` and reload udev.

## Bring-up procedure

1. `ros2 run wisco_ethercat_ros2_hw check_urdf.py` (offline): left/right axis, limit, effort and velocity mismatches,
   and URDF limits that disagree with the wired actuator models.
2. `ethercat slaves`: every drive is in PREOP/OP and the count matches the table.
3. Passive commissioning, with ROS stopped and the drives disabled: `ros2 run wisco_ethercat_ros2_hw arm commission`.
   You move each joint by hand to measured angles; it reads the encoders over SDO and derives direction, counts per
   output revolution, zero offset and software limits, and writes a calibration version.
4. For an arm, record a reference or calibrate before enabling motion (`arm reference`, `arm autocal`; see
   `robots/arm01/README.md`). For a bench joint, launch in velocity mode with the joints clear, then
   `ros2 run wisco_ethercat_ros2_hw bringup_check.py`: it jogs each drive in bus order and checks that only the named joint moved, by about the commanded amount, in the commanded
   direction. A failure means a wrong `bus_position` or `direction`.
5. Only then raise the bring-up limits in the model table or per joint. The full calibration chain for an arm is in
   `robots/arm01/README.md`.

## Motion capture

The arm calibrations that use mocap (`arm mocap-calib`, `mocap-repeat`, `validate`) read a
`geometry_msgs/PoseStamped` topic, set as `mocap.topic` in the arm's joint config. The upstream
[`vrpn_mocap`](https://index.ros.org/p/vrpn_mocap/) client provides it from any system that streams VRPN (OptiTrack
Motive, Vicon, Qualisys):

```bash
# MOCAP_HOST (the mocap computer) in the repo .env; VRPN streaming enabled in the mocap software
ros2 launch wisco_ethercat_ros2_hw mocap.launch.py
ros2 run wisco_ethercat_ros2_hw mocap_check.py /vrpn_mocap/arm01_flange/pose --rate 240   # rigid body held still
```

Then set `mocap.topic: /vrpn_mocap/arm01_flange/pose` and `length_bar_error_m` (from a length-bar test in the mocap
software) in `robots/arm01/config/arm01.yaml`. The same stream gives the walking robot's true base pose for
sim-to-real checks (`training/sim2real/README.md`).

## Real-time setup (toward 1 kHz)

[`infra/rk3588/`](../../../infra/rk3588/README.md) sets up the RK3588 and checks it (`setup_rt.sh`): PREEMPT_RT
kernel, cores 6-7 isolated, `rtprio`/`memlock` limits, NIC interrupts on the control core, a cyclictest report, and
the `update_rate` ramp (100 → 250 → 500 → 1000 Hz, each step checked for EtherCAT lost frames, working-counter changes
and controller overruns). The launch uses the same value for the EtherCAT cycle, so the two cannot drift apart.

| Measured on the RK3588 | Result |
|---|---|
| cyclictest, 1 ms period, cores 6-7, under load (`setup_rt.sh latency`) | not measured yet |
| highest `update_rate` passing `setup_rt.sh rate-check` | not measured yet (runs at the default 100 Hz) |
- Keep `joint_state_rate` at 50–250 Hz: the controllers read state interfaces directly, and a 1 kHz JointState stream
  only loads DDS.
- Distributed clocks are on (`dc_assign_activate: 0x0300` in the model table); check it against the drive's ESI.

## MyActuator drive facts

From MyActuator's EtherCAT control protocol and ESI file (`MT-Device`, firmware 26042402):
- **0x6071 / 0x6077 / 0x6072 are per mille of the rated *current*** (0x6075, mA, read over SDO), so
  `effort_nm_per_permille` = Kt × I_rated / 1000 in `config/actuator_models.yaml`. 0x6076 (rated torque) is not implemented.
- The torque offset 0x60B2 is not in the default RxPDO 0x1600 = [6040, 607A, 60FF, 6071, 6072, 6060, 2FFD]; arm mode
  adds the optional RxPDO 0x1603 = [60B0, 60B1, 60B2], which needs firmware 26042402 or newer.
- The drives' own CSP/CSV loop gains are 0x2002–0x2008 (REAL32, SDO only), a cascaded position-P / velocity-PI loop
  with no documented units: read them on the bench (`ethercat upload -p <pos> -t float 0x2002 0`) before tuning.
  Saving to flash is Controlword 0 → 0x1000 → 0 within 3 s with the motor disabled.
- One generic device for every model: vendor 0x00202008, product 0, revision 0x00010000; minimum cycle 250 µs.
- **Firmware:** all seven arm01 drives run the 26042402 PDO profile (the older 5.12 profile, with base PDOs only and no
  torque offset, is no longer used). Read a drive's version from 0x100A; `pp_test` prints it.
- The 26042402 ESI (`MT-Device_260424.xml`) maps kp and kd (0x2000 / 0x2001, int32) into RxPDO 0x1601, next to the
  controlword, targets and mode: the objects the walking joints' mode 5 needs. Profile-position parameters are in
  RxPDO 0x1602 (0x6081 velocity, 0x6083 / 0x6084 acceleration / deceleration, 0x6085 quick stop), and 0x603F (error
  code) is in TxPDO 0x1A00 (`config/pp_test.yaml`).
- **Torque clamps:** a joint row with `use_peak_torque_limit: true` may set `max_torque_nm` up to the model's peak
  torque; otherwise the ceiling is the rated torque. Loading refuses anything above.
- The model table uses the "-E" EtherCAT datasheets. Marked "verify" there: the X4-10 gear ratio (12.5 in the table,
  12.6 in the user manual, 1:13 on the label), the X4s' encoder counts per output revolution (131072 assumed), and the
  X4s' torque constant (the datasheet Kt disagrees with rated torque / rated current).

## Profile-position drive check (`pp_test`)

A direct test of the drives that bypasses `ros2_control`: it checks a drive's PDO mapping against the
profile-position configuration and, if asked, moves it one small step. Stop any launch that owns IgH master 0 first,
and identify drives by their bus position (`ethercat slaves`).

```bash
ros2 run wisco_ethercat_ros2_hw pp_test --position 0          # mapping only: controlword stays 0, nothing moves
ros2 run wisco_ethercat_ros2_hw pp_test --position 0 --move   # same checks, then +5° at 2°/s (type MOVE to confirm)
ros2 run wisco_ethercat_ros2_hw pp_test --all --move          # every drive in bus order, one confirmation
```
It compares the live CoE descriptors with `config/pp_test.yaml`, brings the drive to OP, reads back the SM2
(0x1601 + 0x1602) and SM3 (0x1A00) mappings, verifies the profile values over SDO, and for a move checks the step
reached its target within 0.15°. Keep the load supported and clear in the positive direction; a failure or Ctrl-C
stops the sequence.

## Mock EtherCAT

The simulated drive and master (`include/wisco_ethercat_ros2_hw/mock/`, library `wisco_ethercat_ros2_hw_mock_drives`):
`RmdxMockSlave` is an RMD-X drive as a CiA 402 slave (0x6040/0x6041 state machine, CSP, CSV, CST and MyActuator mode
5 with kp / kd from 0x2000 / 0x2001, the 0x6072 torque cap, injectable faults and lost frames), with either its own
one-joint model or a simulator's joint (`external_physics`: `measure()` before each bus cycle, `servo()` at every
physics step); `Cia402DriveMaster` reproduces the `EcCiA402Drive` per-cycle logic of `ethercat_driver_ros2`. The
mock's assumptions about the drive are the `RmdxParams` fields marked "guess".

- Tests (`test/mock/`): `MockEthercatSystem` is a `ros2_control` hardware interface over both, on the toy joint.
  `test_arm_controller_mock` runs the real `ArmController` and `SafeCommandController` against it: enable, fault and
  reset, CSP/CST switching, the watchdog, torque clamps, rejected commands, the payload. `test_mock_slave` covers the
  drive on its own.
- Simulation: `hsim_mujoco_plugins`' `VirtualRobotSystem` puts a drive on every motor of the MuJoCo robot
  (`backend:=ethercat_sim`).

**Mode 5 and upstream `EcCiA402Drive`**: upstream overwrites 0x607A with the actual position whenever 0x6061 is not
CSP, so a mode-5 drive behind it would only ever hold where it is (`UpstreamPositionOverrideFreezesModeFive` shows
it). The mock master lets 0x607A through in mode 5 too (`mit_position_passthrough`); the robot's `EcCiA402Drive` needs
the same exception before the legs can run mode 5.

## Safety

Software safety here is the command layer (`SafeCommandController`, `ArmController`'s clamps, soft limits and
watchdog). The hardware e-stop, which cuts motor power independent of all software, is in `hardware/estop/`.

## Current sensing

`ros2 run wisco_ethercat_ros2_hw current_sensor_node` reads seven ACS758 hall-effect current sensors through an ADS7128
ADC on I2C (`/dev/i2c-1`, address 0x10): main bus, the 24 V buck converter, all actuators, and four actuator rails. It
publishes `/power/currents` (`diagnostic_msgs/DiagnosticArray`, one status per rail with `current_a` and `raw_counts`)
every `poll_period_ms` (100). Parameters: `i2c_device`, `i2c_address`, `poll_period_ms`, `print` (also print a line
per sweep) and `print_raw_counts`. Zero offsets are set in `kSensors`: measure each channel unloaded with
`print:=true print_raw_counts:=true`. The rail names in `kSensors` are placeholders until they are matched to the power
distribution board; compare the readings with the simulated current draw in `robots/wisco_v1/reports/`.

## Open items (need the bench)

- The torque scale for the X4-10 and X4-36 implies more than their rated torque at 1000 ‰: measure Kt and correct it;
  effort readings and torque commands are suspect until then.
- Whether 0x6064 reports the output-side or motor-side encoder (the scale factors assume output side, 2^17 counts per
  output revolution).
- The test-bench URDF: the right arm's axes don't mirror the left, wrist pitch limits differ left/right, and the
  shoulder limits don't match the X4-10s wired there (`check_urdf.py`).

## Debugging

- `ros2 control list_controllers`, `ros2 control list_hardware_interfaces`
- `ros2 topic echo /joint_states`, `ros2 topic hz /joint_states`
- `ethercat slaves`; `ethercat domains` (the working counter should be N/N)
- `ros2 run wisco_ethercat_ros2_hw arm status`: the newest calibration version and its contents

Tests: `colcon test --packages-select wisco_ethercat_ros2_hw` (C++ and Python), or the Python part alone with
`python -m pytest test/arm_stack test/test_hardware_config.py` from this folder.
