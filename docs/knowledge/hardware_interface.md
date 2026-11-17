# The hardware interface: ros2_control, EtherCAT, CiA 402 and real-time Linux

How a joint target computed on the central computer becomes current in a motor winding, and what has to be true for
that to happen 1,000 times a second without ever missing. Read [controls.md](controls.md) first for why the targets
look the way they do; this document is about the plumbing underneath.

## The layers

```
central computer (Jetson Thor or Jetson Orin NX)          real-time computer (RK3588, Rockchip)
  policies, planner, Nav2  ── targets, 50–100 Hz ──►        ros2_control_node (one process, one real-time thread)
                              ROS 2 / DDS                     controllers: update()  ← read state, write commands
                                                              hardware interface: read() / write()
                                                              EtherCAT master (IgH): one frame per cycle, 1 kHz
                                                                    │
                                                              drives (CiA 402): state machine, own current loop (FOC)
                                                                    │
                                                              motors
```

Two rules hold the design together:

- **The real-time loop never crosses DDS.** Everything that must happen every millisecond (controllers ↔ hardware
  interface ↔ EtherCAT master) lives in one process on one computer. ROS topics carry targets *into* that process at a
  slower rate; if they stop arriving, the controllers keep the joints safe on their own.
- **Each layer reads whatever the layer above last sent.** Inputs are "latest value wins" (depth-1 queues, real-time
  buffers), never a blocking wait, so a slow publisher can delay a target but never stall the loop.

## ros2_control

`ros2_control` splits the robot into **hardware interfaces** (talk to devices) and **controllers** (compute commands),
both loaded as plugins into `controller_manager`, which runs one loop:

```
every cycle:   hardware.read()  →  every active controller.update()  →  hardware.write()
```

A hardware interface (`hardware_interface::SystemInterface`) goes through a lifecycle:

| Callback | What it does for a drive bus |
|---|---|
| `on_init(info)` | parse the `<ros2_control>` block of the URDF: joints, interfaces, parameters |
| `on_configure()` | start communication: EtherCAT master, scan the slaves, configure PDOs; state: comms up, not commanding |
| `on_activate()` | walk each drive through the CiA 402 state machine to Operation Enabled |
| `export_state_interfaces()` / `export_command_interfaces()` | expose named doubles (`joint/position`, `joint/effort`, …) that controllers read and write |
| `read()` / `write()` | copy the last received frame into state interfaces; copy command interfaces into the next frame. **Must be real-time safe**: no allocation, no blocking, no logging in the hot path |
| `on_deactivate()` / `on_cleanup()` | disable the drives; shut the master down |

Controllers claim command interfaces exclusively, so two controllers can never write the same joint. A controller's
`update()` has the same real-time rules; its ROS subscription callbacks hand data in through a
`realtime_tools::RealtimeBuffer` (lock-free, latest value) instead of touching the loop's data directly.

`mock_components/GenericSystem` is a stock hardware interface that echoes commands back as state: enough to run every
controller with no drives at all.

## EtherCAT

EtherCAT is Ethernet with a twist: one frame leaves the master, passes through every slave in a ring, and each slave
reads its outputs from and writes its inputs into the frame **as it passes**, in hardware. One frame per cycle carries
every drive's data, so a 1 kHz cycle over 30 drives is routine.

- **PDOs (process data objects)** are the fields exchanged every cycle, mapped from the drive's object dictionary:
  RxPDO (master → drive: controlword, target position / velocity / torque, mode of operation) and TxPDO (drive →
  master: statusword, actual position / velocity / torque). The mapping is configured once at startup.
- **SDOs (service data objects)** are slow, acknowledged reads and writes of any object (gains, rated current, limits),
  used for configuration and diagnostics, never in the loop.
- **Distributed clocks (DC)** synchronize every drive's clock to the first one, so all drives latch their positions and
  apply their commands at the same instant. Enabled per drive by the sync activation register (`0x0300` for these
  drives).
- **Working counter:** each slave that processes the frame increments it; `N/N` means every drive took part in the
  last cycle. A drop means a cable, a drive or a timing problem.
- **States:** INIT → PREOP (SDOs work) → SAFEOP (inputs flow) → OP (outputs flow). A drive stuck in PREOP+ERROR
  usually has a PDO mapping it rejects.

The master here is the IgH EtherCAT master (a kernel module plus the `ethercat` command-line tool); ROS reaches it
through ICube's `ethercat_driver_ros2`, whose `EcCiA402Drive` plugin handles one CiA 402 drive per configured slave.

## CiA 402: the drive profile

Every drive here speaks CiA 402 (IEC 61800-7-201), the standard profile for motor drives: the same object numbers and
the same state machine on every vendor's drive.

**State machine.** The master moves a drive between states by writing the **controlword** (0x6040) and reads where it
is from the **statusword** (0x6041):

| Controlword | Value | Moves to |
|---|---|---|
| Shutdown | 0x0006 | Ready to Switch On |
| Switch On | 0x0007 | Switched On |
| Enable Operation | 0x000F | **Operation Enabled** (the motor follows commands) |
| Disable Voltage | 0x0000 | Switch On Disabled |
| Quick Stop | 0x0002 | Quick Stop Active |
| Fault Reset | 0x0080, rising edge | Fault → Switch On Disabled |

| Statusword & mask | Value | State |
|---|---|---|
| & 0x4F | 0x00 | Not Ready to Switch On |
| & 0x4F | 0x40 | Switch On Disabled |
| & 0x6F | 0x21 | Ready to Switch On |
| & 0x6F | 0x23 | Switched On |
| & 0x6F | 0x27 | Operation Enabled |
| & 0x6F | 0x07 | Quick Stop Active |
| & 0x4F | 0x0F | Fault Reaction Active |
| & 0x4F | 0x08 | Fault |

Startup is Shutdown → Switch On → Enable Operation, one step per cycle, waiting for the statusword to confirm each.
Vendors may add their own bits above bit 10, so decode with the masks, never by comparing the whole word.

**Modes of operation** (0x6060, written each cycle; 0x6061 reports the active one):

| Mode | Name | The drive receives | The drive runs |
|---|---|---|---|
| 8 | CSP, cyclic synchronous position | target position (0x607A) every cycle, optional velocity and torque offsets (0x60B1, 0x60B2) | its own position → velocity → current loops |
| 9 | CSV, cyclic synchronous velocity | target velocity (0x60FF) | velocity → current loops |
| 10 | CST, cyclic synchronous torque | target torque (0x6071) | current loop only |

The more of the loop the drive runs, the less the host has to do in real time, but the less the host controls the
response. Torque-limit object 0x6072 caps whatever the drive's own loops produce.

## MyActuator RMD-X specifics

- **Mode 5 "PVT/MIT"** (vendor mode, EtherCAT protocol V2.0): every cycle the host sends position, velocity and
  feed-forward torque targets **plus kp and kd** (0x2000 / 0x2001, integer × 1000), and the drive computes
  τ = kp (q_des − q) + kd (q̇_des − q̇) + τ_ff at its own loop rate. This is exactly the law the walking policy is trained
  against, so the legs, waist and head use it.
- **Torque units:** 0x6071 / 0x6077 / 0x6072 are **per mille of rated current** (0x6075, mA), not of rated torque
  (0x6076 is not implemented). Converting needs the torque constant: N·m per ‰ = Kt × I_rated / 1000.
- **0x6072 = 0 means the stall-current limit, not zero torque**: a controller must never write 0 to mean "no torque".
- The torque offset 0x60B2 is not in the default RxPDO; arm mode maps an extra RxPDO (0x1603: 0x60B0, 0x60B1, 0x60B2),
  which needs recent drive firmware.
- The drive's own CSP/CSV gains (0x2002–0x2008, REAL32, SDO only) have no documented units: read them on the bench
  before relying on the drive's position loop.
- One generic EtherCAT device for every model (vendor 0x00202008), minimum cycle 250 µs.

All of this, with the open questions still to measure, is in `ros2_ws/src/wisco_ethercat_ros2_hw/README.md`.

## Real-time Linux

A 1 kHz loop has a 1 ms budget; the usual target is worst-case jitter under about 5 % of the period (50 µs). A stock
kernel cannot promise that. What it takes:

- a **PREEMPT_RT** kernel, so a high-priority thread preempts almost anything;
- the control process at **real-time priority** (`SCHED_FIFO`, via `rtprio` limits), with memory locked (`mlockall`,
  `memlock` limit) so it never page-faults;
- a **core isolated** for the loop (`isolcpus=`, `nohz_full=`) with the EtherCAT NIC's interrupts pinned to it;
- nothing in the loop that allocates, blocks, or logs.

Installing PREEMPT_RT alone does none of the rest. `controller_manager` runs its loop on such a thread when the
limits allow it; publishing `/joint_states` at 100–250 Hz rather than 1 kHz keeps DDS load off the machine.

## Safety layers

| Layer | Acts on | Example |
|---|---|---|
| Controllers | every cycle | timeouts (stale command → stop or damp), clamps, soft position limits, watchdog → gravity compensation + damping |
| Drives | every cycle | 0x6072 torque cap, their own fault detection (over-current, over-temperature) → Fault state |
| ROS safety node | 10–50 Hz | e-stop lock, fall detection, stale sensors, battery → zero commands |
| Hardware e-stop | always | handheld transmitter; motor power stays on only while heartbeats arrive; cuts the contactor independent of all software |

Each layer assumes the one above it can fail.

## In our stack

| Concept | This repo |
|---|---|
| Hardware interface + controllers | `ros2_ws/src/wisco_ethercat_ros2_hw/`: `ethercat_driver_ros2` + IgH master, `SafeCommandController`, `ArmController` |
| Drive firmware | MyActuator 26042402 PDO profile on every arm01 drive; `pp_test` checks a drive's mapping and moves it one step without ros2_control |
| Drive configuration | generated at launch from one joint table, the actuator model table and the calibration file (`hardware_config.py`) |
| Legs, waist, head | MyActuator mode 5 with the trained kp / kd (`robots/wisco_v1/README.md`, "Drive modes and gains"; not wired yet). The two-motor ankles and waist go through `ParallelLinkageController` ("PR mode": joints ↔ motors through the rods) |
| Arms | `ArmController` at the cycle rate (100 Hz by default, 1 kHz once timing is verified on the RK3588): CSP + gravity torque offset (0x60B2), or CST with host PD; tested on the arm01 test arm |
| No drives | `use_mock_hardware:=true` (GenericSystem), or the mock EtherCAT bus in `test/mock/` (simulated RMD-X drives, `include/wisco_ethercat_ros2_hw/mock/`: the CiA 402 state machine, CSP/CSV/CST and mode 5, run under the real controllers in tests) |
| Simulation | MuJoCo applies the same PD at the physics step: in the sim node, or under ros2_control with the upstream `mujoco_ros2_control` plugin (`sim_stack.launch.py backend:=ros2_control`, files from `hsim_sim/ros2_control_backend.py`); or the whole chain down to the PDOs on simulated drives with MuJoCo as the robot (`backend:=ethercat_sim`, `hsim_mujoco_plugins`' `VirtualRobotSystem`), two-motor joints included |
| Hardware e-stop | `hardware/estop/` |

Sources: ros2_control documentation (writing a hardware component; `SystemInterface`), ICube `ethercat_driver_ros2`
documentation (configuration, CiA 402 drive), CiA 402 / IEC 61800-7-201, the ROS 2 real-time programming guide,
MyActuator's EtherCAT control protocol and ESI file.
