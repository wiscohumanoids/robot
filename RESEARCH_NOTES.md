# Research notes: EtherCAT / ros2_control / real-time / multi-rate architecture

This document records what was actually found (with sources) before
`ethercat_bridge` and `lowlevel_control` were scaffolded, per the project
requirement to cite research rather than invent plausible-sounding hardware
integration code. Where a source didn't give a complete answer, that gap is
stated explicitly — this repo's EtherCAT code is a scaffold, and the gaps
below are exactly the TODOs left in the code for the low-level team to fill
with real datasheet/hardware values.

---

## 1. `ethercat_driver_ros2` (ICube-Robotics)

This is the reference implementation the design here follows conceptually.
**We do not take a hard dependency on it** (see "Why we didn't vendor
`ethercat_driver_ros2` directly" below), but `ethercat_bridge`'s
`EthercatHardwareInterface` mirrors its architecture so that swapping in the
real package later — or wiring `ethercat_bridge` to actually call it — is a
small diff, not a redesign.

**`<ros2_control>` URDF structure** (from the master hardware plugin's
perspective):

```xml
<ros2_control name="mySystem" type="system">
  <hardware>
    <plugin>ethercat_driver/EthercatDriver</plugin>
    <param name="master_id">0</param>
    <param name="control_frequency">100</param>
  </hardware>
</ros2_control>
```

Each EtherCAT slave is then declared as a module resource with an `alias`/
`position` pair identifying its address on the bus topology, e.g.:

```xml
<ec_module name="ECModule">
  <plugin>ethercat_plugins/ECModule</plugin>
  <param name="alias">0</param>
  <param name="position">1</param>
</ec_module>
```

**CiA402 drive slave config** (`slave_config.yaml`-style file consumed by the
`EcCiA402Drive` plugin) — confirmed field names and example values:

```yaml
vendor_id: 0x000000fb
product_id: 0x64400000
assign_activate: 0x0300      # distributed-clock sync-manager assignment/activation register
auto_fault_reset: false
```

RxPDO (master -> slave) entries seen in the reference config, each with
`index`, `sub_index`, `type`, and an optional `command_interface` binding:

- Controlword — `0x6040`, sub-index 0, `uint16`
- Target position — `0x607a`, sub-index 0, `int32`, `command_interface: position`
- Target velocity — `0x60ff`, sub-index 0, `int32`
- Target torque — `0x6071`, sub-index 0, `int16`
- Mode of operation — `0x6060`, sub-index 0, `int8`

TxPDO (slave -> master) entries, each with an optional `state_interface`
binding:

- Statusword — `0x6041`, sub-index 0, `uint16`
- Position actual value — `0x6064`, `state_interface: position`
- Velocity actual value — `0x606c`, `state_interface: velocity`
- Torque actual value — `0x6077`, `state_interface: effort`

`mode_of_operation` is also settable as a plain URDF `<param>` on the joint to
fix the drive's default CiA402 mode (position/velocity/torque profile) without
a runtime PDO write.

**Gap:** the exact `sync_cycle_time`/distributed-clock cycle-time field name
for pinning a 1ms (1kHz) cycle was not confirmed from the fetched pages — the
docs describe DC sync as a capability gated by `assign_activate` but didn't
spell out the millisecond-cycle parameter name in the excerpt retrieved.
**TODO for the low-level team:** confirm this against
`ICube-Robotics/ethercat_driver_ros2_examples` before wiring a real bus; our
`config/ethercat_slaves.yaml` scaffold assumes a field name
(`dc_sync.cycle_time_us: 1000`) that is our own placeholder, clearly marked,
not a confirmed upstream field.

Sources:
- https://icube-robotics.github.io/ethercat_driver_ros2/quickstart/configuration.html
- https://icube-robotics.github.io/ethercat_driver_ros2/user_guide/config_cia402_drive.html
- https://github.com/ICube-Robotics/ethercat_driver_ros2_examples
- https://github.com/ICube-Robotics/ethercat_driver_ros2/issues/47 (real-world PDO-mapping/ros2_control wiring discussion)
- https://github.com/ICube-Robotics/ethercat_driver_ros2/issues/214, #171 (real-world CiA402 slave bring-up failure modes — worth reading before hardware bring-up, e.g. drives getting stuck in PREOP+ERROR)

### Why we didn't vendor `ethercat_driver_ros2` directly

Two reasons, both practical: (1) it requires a real EtherCAT master stack
(SOEM or IgH) and a NIC with RT-capable driver support to even build/link
against meaningfully, which breaks this repo's "colcon build succeeds on a
laptop with no special hardware" requirement; (2) the project brief calls for
a *scaffold this team owns and fills in incrementally*, not an opaque
dependency. `ethercat_bridge` is written so that its `on_configure()`/`read()`/
`write()` TODOs are exactly the places a real integration would either (a)
call into `ethercat_driver_ros2`'s generic module plugins, or (b) call SOEM
directly — both paths are left open, and this is documented at the top of
`ethercat_bridge/src/ethercat_hardware_interface.cpp`.

---

## 2. `ros2_control` `hardware_interface::SystemInterface` plugin architecture

Confirmed lifecycle (this is what `EthercatHardwareInterface` and, for
reference, the vendored `MujocoSystem` both implement):

- **`on_init(const HardwareInfo& info)`** → `CallbackReturn`. Must call
  `SystemInterface::on_init(info)` first (parent fills `info_` from the URDF's
  `<ros2_control>` block: joint names, command/state interfaces, `<param>`
  values). Return `SUCCESS` iff all required parameters are present/valid,
  else `ERROR`.
- **`on_configure()` / `on_cleanup()`** → set up / tear down communication to
  hardware (for us: EtherCAT master init / slave scan / shutdown). Hardware
  state after `on_configure`: `INACTIVE` (comms up, not yet commanding).
- **`on_activate()` / `on_deactivate()`** → enable/disable hardware "power"
  (for us: CiA402 state machine transition into `Operation Enabled` /
  back to `Switched On`).
- **`export_state_interfaces()`** / **`export_command_interfaces()`** → build
  the list of `StateInterface`/`CommandInterface` objects from `info_`,
  pointing at internal member storage.
- **`read(const rclcpp::Time&, const rclcpp::Duration& period)`** → pull
  latest hardware state into internal storage. **Must be real-time safe**
  (no dynamic allocation, no blocking I/O) once in the `ACTIVE` state, since
  `controller_manager` calls this from its RT update thread.
- **`write(const rclcpp::Time&, const rclcpp::Duration& period)`** → push
  internal command storage out to hardware. Same real-time-safety
  requirement.

This maps directly onto the CiA402 PDO exchange: a real `read()` would consume
the last-received TxPDO frame (Statusword, ActualPosition, ActualVelocity,
ActualTorque) that the EtherCAT master's background thread already latched;
`write()` would populate the RxPDO frame (Controlword, TargetTorque) for the
master to send on the next bus cycle. Our scaffold's `read()`/`write()` are
structured with exactly this split (see code comments), but the "master's
background thread" and actual frame exchange are the stubbed part.

Sources:
- https://control.ros.org/rolling/doc/ros2_control/hardware_interface/doc/writing_new_hardware_component.html
- https://docs.ros.org/en/ros2_packages/humble/api/hardware_interface/generated/classhardware__interface_1_1SystemInterface.html
- https://github.com/ros-controls/ros2_control/blob/master/hardware_interface/include/hardware_interface/system_interface.hpp
- https://control.ros.org/rolling/doc/ros2_control_demos/example_7/doc/userdoc.html (full worked 6-DOF example, same pattern we followed for `export_state_interfaces`/`export_command_interfaces` shape)

---

## 3. CiA 402 drive state machine

Confirmed states and the startup transition sequence (the "typical happy
path" every real drive bring-up follows):

| Transition (Controlword written) | Hex | Resulting state |
|---|---|---|
| Shutdown | `0x0006` | Ready to Switch On |
| Switch On | `0x0007` | Switched On |
| Enable Operation | `0x000F` | Operation Enabled (motor can move) |
| Disable Voltage | `0x0000` | Switch On Disabled |
| Quick Stop | `0x0002` | Quick Stop Active |
| Fault Reset | `0x0080` (rising edge) | clears Fault -> Switch On Disabled |

Full state set per the standard (CiA 402 / IEC 61800-7-201): **Not Ready to
Switch On -> Switch On Disabled -> Ready to Switch On -> Switched On ->
Operation Enabled**, with **Quick Stop Active** and **Fault Reaction Active ->
Fault** as side branches reachable from most states. The drive's current
state is decoded by masking Statusword (`0x6041`) and matching a fixed
bit-pattern table; Controlword bit semantics: bit0=Switch On, bit1=Enable
Voltage, bit2=Quick Stop, bit3=Enable Operation, bit7=Fault Reset (bits 4-6
and 8-15 are mode-of-operation-specific, e.g. New Setpoint for profile
position mode).

**Gap:** the exact Statusword bitmask table (which bits of `0x6041` to mask
and compare, e.g. the commonly-cited `0x6F` mask) was **not** confirmed
verbatim from a fetched primary source in this session — the fetched
Synapticon page confirmed the state names and the three startup Controlword
values (`0x0006`/`0x0007`/`0x000F`) but the tool's summary explicitly noted the
full bitmask table was not present in what it returned. The `0x0002` (Quick
Stop) and `0x0080` (Fault Reset) values above are standard and consistent
across CiA402 secondary literature, but **the low-level team must verify the
Statusword decode table against the actual drive's datasheet** (every vendor
implements the same CiA402 object dictionary, but some add vendor-specific
bits above bit 10) before trusting `ethercat_bridge`'s (currently unimplemented,
TODO-marked) state-machine decode logic.

Sources:
- https://www.synapticon.com/en/motion-control-academy/cia-402-antriebsprofil-state-machine
- https://www.can-cia.org/can-knowledge/cia-402-series-canopen-device-profile-for-drives-and-motion-control
- https://doc.synapticon.com/actilink_s/system_integration/coe_cia_402.html
- https://lichirobotics.com/blog/control-systems/cia402-drive-profile-explained

---

## 4. PREEMPT-RT for a 1kHz control loop

Findings, with the specific numeric target this repo's `lowlevel_control` and
`ethercat_bridge` are designed against:

- A vanilla (non-RT) Linux kernel cannot guarantee arbitrary user-thread
  preemption; a 1kHz (1ms period) hard real-time loop requires the
  `PREEMPT_RT` kernel patch, not just `nice`/`chrt` on a stock kernel.
- Rule of thumb cited: keep worst-case jitter under **~5% of the update
  period** — for 1kHz that's a ≤50µs budget; one cited reference
  configuration achieved <3% jitter (~30µs) using `PREEMPT_RT` +
  `SCHED_RR` at priority 98.
- `ros2_control`'s own real-time thread model: the `controller_manager`'s
  main update loop (which calls every active controller's `update()`, and
  every hardware component's `read()`/`write()`) is intended to run as a
  single real-time thread; controller `update()` implementations must
  themselves be allocation-free/lock-free to preserve that guarantee (this is
  exactly why `JointCommand` is delivered to `lowlevel_control` via a
  `realtime_tools::RealtimeBuffer`-style non-blocking handoff from the ROS
  subscription callback rather than processed directly in the subscription
  callback).
- Caveat directly relevant to this repo: **installing `PREEMPT_RT` does not
  by itself make a program real-time** — the process must also be pinned to
  an isolated CPU core (`isolcpus`/`taskset`), run at `SCHED_FIFO`/`SCHED_RR`
  elevated priority, and avoid page faults (lock memory with `mlockall`) and
  any allocation/logging/blocking syscalls in the hot path. None of this is
  configured in this repo's stub yet — see `ethercat_bridge/README.md`'s
  "what's not done" list.

Sources:
- https://github.com/ros2/ros2_documentation/blob/humble/source/Tutorials/Demos/Real-Time-Programming.rst
- https://answers.ros.org/question/382893/best-practices-for-real-time-capabilites-in-ros2/
- https://link.springer.com/article/10.1186/s10033-023-00976-5
- https://roscon.ros.org/2015/presentations/RealtimeROS2.pdf

---

## 5. Multi-rate ROS2 architecture (10 / 50 / 500 / 1000 Hz coexisting cleanly)

Findings applied directly to this repo's layer frequencies (`teleop_input`
10Hz -> `cmd_vel_mux` 50Hz -> `locomotion_runner`/`manipulation_runner` 50/10Hz ->
`wbc_stub` 500Hz -> `lowlevel_control`/`ethercat_bridge` 1000Hz; the task-stack
stubs added later run at 1-100 Hz or on events, well inside the "plain DDS is
fine" range this section describes -- see the contract for each rate):

- A single node can cleanly run several independent rates using multiple ROS
  `Timer` objects (e.g. rclpy `create_timer` / rclcpp `create_wall_timer`)
  rather than needing one node per rate — used in this repo only where a
  layer genuinely has one job at one rate (which is every layer here; each
  package in the stack has exactly one timer).
- For frequencies **above** roughly 50Hz shared over the standard
  DDS/multi-process transport, latency degrades with node/topic count — the
  literature explicitly calls out that intra-process **composition** (loading
  nodes as components in one process, using intra-process communication) is
  the standard mitigation once you're pushing >1kHz data volumes across node
  boundaries. This repo's 1kHz boundary (`lowlevel_control` <-> whichever
  hardware plugin is active) deliberately stays **inside a single process**
  (`controller_manager` loads both the controller and the hardware component
  as plugins in one process) specifically to sidestep this — it is not a
  design accident that `JointCommand` (500Hz, cross-process, DDS topic) sits
  one layer *above* the 1kHz boundary rather than crossing it.
- Recommended layering pattern (perception -> planning -> control, decoupled
  per-layer polling instead of one global synchronized clock) matches this
  repo's structure: each layer reads whatever the layer below last published
  (via a cached "latest message wins" QoS, not a blocking wait), so a slow
  publisher never stalls a fast consumer — this is why `wbc_stub` and
  `lowlevel_control` both use `KEEP_LAST` depth-1 QoS on their inputs rather
  than a queued/reliable large-depth queue.

Sources:
- https://arxiv.org/pdf/2305.09933 ("Impact of ROS 2 Node Composition in Robotic Systems")
- https://arxiv.org/pdf/2101.02074 ("Latency Analysis of ROS2 Multi-Node Systems")
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12845773/ ("ROS 2-Based Architecture for Autonomous Driving Systems")
- https://roboticsbackend.com/ros-rate-roscpy-roscpp/

---

## 6. Task-stack and interface-design decisions (added with the task-layer scaffold)

**Read this differently from sections 1-5.** Those record research with cited
sources. This section records *design decisions* and, more importantly, the
**assumptions that were made without checking a primary source** (the task-stack
stubs were written on a machine with no ROS installation, and no web research was
done for them). Nothing below is cited because nothing below was looked up; each
item is a thing to verify on a ROS machine. Verified items get a tick in
`STATUS.md`'s "not verified" list.

**Decisions and their reasons**

- *Standard message types where they exist.* `/cmd_vel` is `geometry_msgs/Twist`
  (an earlier custom `VelocityCommand` was removed), navigation is Nav2's own
  `NavigateToPose` action, `/joint_states` is `sensor_msgs/JointState`. Reason:
  Nav2, rosbag, rviz, joystick and teleop tooling work unchanged, and a
  custom `/cmd_vel` type would have collided with Nav2's `Twist` publisher.
- *One publisher per topic and per TF edge*, enforced by
  `interface_contract.yaml` + `tests/` + `check_contract.py`. A topic with two
  publishers interleaves their messages (last-writer-wins per message), which
  for `/safety_status` would let a healthy `is_safe: true` overwrite an E-stop.
  A TF frame can have only one parent; two publishers of the same edge produce
  jitter, two parents produce a broken tree.
- *A small custom `cmd_vel_mux` instead of the `twist_mux` package.* Chosen so the
  arbitration (teleop > nav > zero with a 0.5 s timeout, doubling as a watchdog)
  is fully unit-testable without ROS. `twist_mux` is the standard alternative and
  is a reasonable replacement; its parameter names and Humble message type were
  **not** checked, and it would additionally need teleop to stop publishing zeros
  while idle (which `teleop_input` now does anyway).
- *`base_link` added to the URDF* as a massless root fixed to `pelvis`, so the
  REP-105 frame chain (`map -> odom -> base_link`) and Nav2's default
  `robot_base_frame` hold. The Unitree model's own root is `pelvis`.
- *Stubs live in separate packages, logic in ROS-free modules,* so each team owns a
  package and the interesting behavior (planner rules, go-to-goal controller,
  dead-reckoning, mux selection) is unit-tested with plain `pytest`.

**Assumptions to verify on a ROS Humble machine**

1. `nav2_msgs/action/NavigateToPose` in Humble: goal `pose` (`PoseStamped`) +
   `behavior_tree`; result `std_msgs/Empty`; feedback includes `current_pose`
   and `distance_remaining`. `nav_stub` and `behavior_tree_stub` were written
   from memory of this definition.
2. The rosdep key for the Python `tf2_ros` module in Humble is `tf2_ros_py`
   (used in the new packages' `package.xml`).
3. `ros2 launch` does not attach a TTY to launched nodes, so keyboard teleop only
   works from `ros2 run` in its own terminal. (Asserted from how `teleop_node`
   detects a TTY; not tested.)
4. How Nav2 is remapped so its velocity output goes to `/cmd_vel_nav` (which
   node/param) -- the intent is fixed by the contract, the wiring is unwritten.
5. `mock_components/GenericSystem` was considered as a no-MuJoCo stand-in for
   hardware (so CI could test `lowlevel_control`) and **not used**: whether it
   exports the `imu_imu` sensor interfaces `lowlevel_control` requires, and with
   what initial values (NaN would trip `safety`'s NaN check), is unknown. CI
   therefore runs the stubs-only stack (`sim:=false`) and skips the sim-only
   topics.
6. The MuJoCo `linux-aarch64` release tarball exists under the same URL pattern as
   `linux-x86_64` for `MUJOCO_VERSION=3.2.7` (used by the Dockerfile for arm64
   hosts), and OSMesa headless rendering works in that image with
   `MUJOCO_HEADLESS_OSMESA=ON`.
7. `rclpy` action clients/servers behave as `behavior_tree_stub` and `nav_stub`
   assume: goal/result futures completing from a worker thread under a
   `MultiThreadedExecutor`, and blocking `execute_callback`s with
   `time.sleep` (the same pattern `manipulation_runner` already used).

## Summary of open TODOs this research left for the hardware team

These are repeated as inline `# TODO` / `// TODO` comments at the exact call
sites in `ethercat_bridge/`, listed here as one checklist:

1. Confirm the real distributed-clock cycle-time YAML field name against
   `ethercat_driver_ros2_examples` (our `dc_sync.cycle_time_us` key is a
   placeholder).
2. Confirm the Statusword bitmask/decode table against the actual drive
   datasheet, not just the generic CiA402 spec, before trusting any
   state-machine decode logic.
3. Decide whether `ethercat_bridge` will (a) shell out to
   `ethercat_driver_ros2`'s generic slave plugins, or (b) link SOEM directly
   — both are architecturally possible with the current scaffold, neither is
   implemented.
4. Configure `PREEMPT_RT` + CPU isolation + `SCHED_FIFO`/`mlockall` on the
   actual target compute — none of this OS-level setup is part of this repo
   (it's a deployment/provisioning concern, tracked here so it isn't
   forgotten).

Task-stack items to verify are listed in section 6 above and in `STATUS.md`.
