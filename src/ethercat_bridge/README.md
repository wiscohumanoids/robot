# ethercat_bridge

**Status:** Real scaffold. The `hardware_interface::SystemInterface` plugin
structure, lifecycle, exported interfaces, and slave-config parsing/
validation are real and functional. Actual EtherCAT bus I/O is stubbed,
loudly, everywhere it matters. See `RESEARCH_NOTES.md` for the research this
is built on and `INTEGRATION_POINTS.md` "Low-level / EtherCAT" for the
checklist to make it real.

**Frequency:** whatever `controller_manager`'s `update_rate` is,
`bringup/config/controllers.yaml` sets 1000 Hz.

## What's real today

- `EthercatHardwareInterface` is a real, loadable `pluginlib` plugin.
  Pass `hardware_plugin:=ethercat_bridge/EthercatHardwareInterface` to
  `g1_description`'s xacro and `controller_manager` will load it.
- It exports exactly the same interface contract as
  `mujoco_ros2_control/MujocoSystem`: `position`/`velocity`/`effort` state
  and `effort` command per joint, plus the same `imu_imu` sensor interfaces.
  This is what lets `lowlevel_control` run unmodified against either one.
- `on_configure()` really parses and validates
  `config/ethercat_slaves.yaml`: it checks the slave count matches the
  joint count, checks slave order matches `CANONICAL_JOINT_ORDER` exactly,
  and warns (but does not fail) if any `vendor_id`/`product_id` is still the
  `0x00000000` placeholder.

## What's stubbed: read this before assuming anything works

| Method | What it logs it would do | What it actually does |
|---|---|---|
| `on_configure()` | Open an EtherCAT master on the configured NIC, scan the bus | Nothing, no NIC is opened, no bus scan happens |
| `on_activate()` | CiA402 Controlword sequence per slave: Shutdown (`0x0006`) -> Switch On (`0x0007`) -> Enable Operation (`0x000F`) | Nothing, no frame is sent |
| `on_deactivate()` | Disable Voltage (`0x0000`) per slave | Nothing |
| `read()` | Decode each slave's TxPDO (ActualPosition `0x6064`, ActualVelocity `0x606C`, ActualTorque `0x6077`) | Nothing, state stays at whatever it was last set to (URDF `initial_value` at startup, static forever after) |
| `write()` | Encode `effort_command_` into RxPDO TargetTorque (`0x6071`) and run the master's send/receive cycle | Logs the torque it would send (throttled to 1/5s), sends nothing |
| IMU (`imu_imu` sensor interfaces) | Read a real pelvis IMU | Hardcoded identity orientation, zero gyro/accel. A real pelvis IMU is often not an EtherCAT device (typically a separate serial or CAN link), so this is tracked as its own open item, separate from the PDO TODOs above |

**This class will not move a real motor today.** It exists so the low-level
team is filling in `read()`/`write()`/`on_activate()`/`on_deactivate()` and
`config/ethercat_slaves.yaml`'s placeholder IDs, not designing a
`SystemInterface` plugin from scratch or guessing at the CiA402 handshake.

## Filling this in: the checklist

See `INTEGRATION_POINTS.md` "Low-level / EtherCAT" for the full list. In short:

1. Replace every `0x00000000` in `config/ethercat_slaves.yaml` with the real
   `vendor_id`/`product_id` from each drive's datasheet/ESI file.
2. Implement `on_configure()`'s master init + bus scan (SOEM or delegate to
   `ethercat_driver_ros2`; both are architecturally open, see
   `RESEARCH_NOTES.md` "Why we didn't vendor ethercat_driver_ros2 directly").
3. Implement the CiA402 Controlword sequence in `on_activate()`/
   `on_deactivate()`, **verify the Statusword decode table against the real
   drive datasheet first**, it's flagged unconfirmed in `RESEARCH_NOTES.md`.
4. Implement real TxPDO consumption in `read()` and RxPDO production in
   `write()`, confirming each object's engineering-unit scaling against the
   datasheet (CiA402 defines the object, not the vendor's units for it).
5. Set up `PREEMPT_RT`, CPU isolation, and `SCHED_FIFO`/`mlockall` on the
   actual control computer (`RESEARCH_NOTES.md` section 4), not part of
   this repo, but required before any of the above is meaningfully real-time.
6. Swap `bringup/lowlevel_test.launch.py`'s `hardware_plugin` argument from
   `mujoco_ros2_control/MujocoSystem` to
   `ethercat_bridge/EthercatHardwareInterface` and confirm a published
   `JointCommand` produces real motion; that's the milestone this whole
   repo is built around.
