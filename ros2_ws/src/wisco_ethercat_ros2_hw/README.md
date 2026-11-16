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
└── ArmController           1 kHz arm control: gravity compensation (Pinocchio) + PD, two drive modes
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
| `urdf/ros2_control_macros.xacro`, `urdf/test_bench.urdf.xacro` | the `<ros2_control>` block built from the manifest; the test-bench robot |
| `launch/test_bench.launch.py`, `launch/arm01.launch.py` | bring up the bench or arm01, on drives or on mock hardware |
| `src/safe_command_controller.cpp`, `include/.../safety_logic.hpp` | `SafeCommandController` and its unit-tested safety functions |
| `src/arm_controller.cpp`, `src/gravity_model.cpp`, `include/.../arm_law.hpp` | `ArmController`: the arm's real-time control law |
| `src/current_sensor_node.cpp`, `src/ads7128.hpp` | power-rail current readout (ADS7128 I2C ADC + ACS758 sensors); not yet built or published |
| `arm_stack/` | Python toolkit: commissioning and calibration stages, the same control law, a simulated arm, PACE system identification, IK, MCAP logging and export; `scripts/arm` is its CLI |
| `scripts/` | `bringup_check.py` (wiring and direction check), `check_urdf.py` (URDF vs actuator limits), two demo scripts |
| `tools/` | model checks, IK benchmark, workspace map, Isaac Lab USD import, parity golden-file writer (outputs go to `robots/arm01/docs/`) |
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

**`ArmController`**, the arm's 1 kHz control law:
- **Mode P:** CSP position target (0x607A) plus a gravity and friction torque offset (0x60B2), so the drive's own
  position loop tracks while the host cancels gravity.
- **Mode T:** CST torque (0x6071) = host PD + gravity + friction feedforward.
- Pinocchio gravity from `robot_description` and the calibration file, optional payload at the tool, soft limits,
  torque and velocity clamps, bumpless switching between modes, and a 50 ms watchdog that falls back to gravity
  compensation + damping. Options: hold position on startup or on command timeout, gravity compensation on/off, and a
  per-joint switch for drives whose firmware lacks the torque-offset object (those joints use the plain position path).
- It writes each command's torque clamp to the drive's torque limit (0x6072), so the drive's own position loop is capped
  too. 0x6072 = 0 means the drive's stall-current limit, not zero torque, so it never writes less than 1% of the limit.
- Command and state layouts are documented in `include/wisco_ethercat_ros2_hw/arm_controller.hpp`.

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
| `update_rate` | `100` | controller manager rate **and** EtherCAT cycle, Hz (1000 with PREEMPT_RT) |
| `joint_state_rate` | `100` | `/joint_states` rate (capped at `update_rate`) |
| `command_timeout` | `0.1` | seconds without a command before the joints are stopped |
| `use_mock_hardware` | `false` | `mock_components/GenericSystem` instead of drives |
| `enable_motion` | `false` | let the drives go to Operation Enabled (arm mode on real drives) |
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
4. Launch in velocity mode with the joints clear, then `ros2 run wisco_ethercat_ros2_hw bringup_check.py`: it jogs
   each drive in bus order and checks that only the named joint moved, by about the commanded amount, in the commanded
   direction. A failure means a wrong `bus_position` or `direction`.
5. Only then raise the bring-up limits in the model table or per joint. The full calibration chain for an arm is in
   `robots/arm01/README.md`.

## Real-time setup (for 1 kHz)

- PREEMPT_RT kernel; run `ros2_control_node` with real-time priority (`rtprio` and `memlock` limits for the user).
- Isolate a core for the control loop (`isolcpus=`, `nohz_full=`) and pin the EtherCAT NIC's interrupts to it.
- `update_rate:=1000`: the launch uses the same value for the EtherCAT cycle, so the two cannot drift apart.
- Keep `joint_state_rate` at 100–250 Hz: the controllers read state interfaces directly, and a 1 kHz JointState stream
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
- One generic device for every model: vendor 0x00202008, product 0; minimum cycle 250 µs.
- The model table uses the "-E" EtherCAT datasheets. Marked "verify" there: the X4-10 gear ratio (12.5 in the table,
  12.6 in the user manual, 1:13 on the label), the X4s' encoder counts per output revolution (131072 assumed), and the
  X4s' torque constant (the datasheet Kt disagrees with rated torque / rated current).

## Mock EtherCAT

`test/mock/`: `RmdxMockSlave` is an RMD-X drive as a CiA 402 slave (0x6040/0x6041 state machine, CSP and CST, the
0x6072 torque cap, a one-joint dynamics model, injectable faults and lost frames); `Cia402DriveMaster` reproduces the
`EcCiA402Drive` per-cycle logic of `ethercat_driver_ros2`; `MockEthercatSystem` is a `ros2_control` hardware interface
over both. `test_arm_controller_mock` runs the real `ArmController` and `SafeCommandController` against it: enable,
fault and reset, CSP/CST switching, the watchdog, torque clamps, rejected commands, the payload. The mock's assumptions
about the drive are the `RmdxParams` fields marked "guess".

## Safety

Software safety here is the command layer (`SafeCommandController`, `ArmController`'s clamps, soft limits and
watchdog). The hardware e-stop, which cuts motor power independent of all software, is in `hardware/estop/`.

## Current sensing

`src/current_sensor_node.cpp` reads seven ACS758 hall-effect current sensors through an ADS7128 ADC on I2C
(`/dev/i2c-1`, address 0x10): main bus, the 24 V buck converter, all actuators, and four actuator rails. Parameters:
`i2c_device`, `i2c_address`, `poll_period_ms` (100), `print_raw_counts`. It prints amperes per rail; zero offsets are
set in `kSensors` (measure each channel unloaded with `print_raw_counts:=true`). Next steps: add it to the build,
publish the readings on a ROS topic, and compare them with the simulated current draw in `robots/wisco_v1/reports/`.

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
