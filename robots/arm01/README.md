# arm01 — 7-DOF test arm

A standalone prototype of a humanoid arm on a test stand, used to develop and validate the arm's hardware interface,
controller and calibration before they go on a full robot. It is a separate hardware item from Wisco V1
(`robots/wisco_v1/`, the full-body model). Seven joints: shoulder pitch, roll and yaw (2× MyActuator RMD-X6-60,
1× X4-36), elbow (X4-36), forearm yaw, wrist pitch and wrist yaw (3× X4-10); the wrist pitch is a near-parallelogram
linkage. Moving mass about 3.9 kg. All seven drives run the 26042402 firmware PDO profile.

This folder is the `arm01_description` ROS package (linked into the workspace as `ros2_ws/src/arm01_description`). The
controllers, drive configuration and the calibration toolkit that act on it live in
[`ros2_ws/src/wisco_ethercat_ros2_hw/`](../../ros2_ws/src/wisco_ethercat_ros2_hw/README.md).

## Files

| Path | What |
|---|---|
| `urdf/arm01.urdf` | the arm model, converted from the CAD export by `cad/import_full_assembly.py` (frame conventions and assumptions are listed at the top of the file); regenerate, don't hand-edit |
| `urdf/arm01.ros2_control.xacro` | the URDF plus the `<ros2_control>` block for the drives |
| `urdf/bench01.urdf` | the single-actuator bench rig: one actuator with a lever on its output |
| `meshes/` | visual and collision meshes |
| `config/arm01.yaml`, `config/bench01.yaml` | joint configs: bus order, motor models, directions, gains, home pose, limits, calibration fixtures |
| `calibration/<arm>/<date>_v<N>.yaml` | calibration versions, append-only, written by the calibration stages (bench results per actuator serial under `calibration/bench01/<serial>/`). `arm01/2026-11-15_v1.yaml` is the first resting-pose reference recorded on the physical arm |
| `cad/` | CAD import tool, the export's capture pose and contact angles (checked against the URDF by the tests, used by stop zeroing) and the frame-convention ledger |

The IK benchmark and workspace map (`wisco_ethercat_ros2_hw/tools/ik_benchmark.py`, `workspace_map.py`) write their
reports to `build/analysis/`; rerun them when the model or joint limits change.

## Bring it up

```bash
ros2 launch wisco_ethercat_ros2_hw arm01.launch.py use_mock_hardware:=true     # no drives
ros2 launch wisco_ethercat_ros2_hw arm01.launch.py                             # on the drives, motors disabled
ros2 launch wisco_ethercat_ros2_hw arm01.launch.py enable_motion:=true         # powered holding (needs a reference)
```
The launch loads the newest calibration version and runs `ArmController` at 100 Hz (`update_rate`). Motors stay
disabled unless `enable_motion:=true`; then the controller holds the position it reads at startup. It never moves to
the zero or home pose on its own. On real drives the launch refuses to enable motion without a usable calibration,
or if the joint table changed since it was written (its `joint_config_hash`); record a new reference then.

## First-time calibration

Run `arm status` to see whether the active coordinates are **relative** (a resting-pose reference) or **URDF**
(registered to the model). Every command below runs with the arm launch stopped unless noted; autocal starts its own.

1. **Watch the encoders** (optional, drives disabled): `arm telemetry --cyclic` streams joint positions and drive
   status; `arm telemetry --log-angles poses.csv` records labelled poses.
2. **Resting reference** (no fixture needed): let the arm settle, supported, with the drives powered but disabled, then
   `arm reference`. Every joint's resting position becomes 0 relative degrees; nothing is enabled or moved. This allows
   powered position holding only: CAD poses, gravity compensation and torque mode stay blocked. Record it once and reuse
   it (check encoder continuity after a power cycle); re-running it replaces the coordinates.
3. **Local calibration**: `arm autocal`. With a relative reference it captures the start pose, prints the motion
   envelope (±10° by default, `--travel-deg` down to 0.5°) and asks `yes` before enabling the motors. After you remove
   the support and press Enter, each joint in turn sweeps through the envelope (up to 12°/s) while the motors hold the
   others; holding torque is measured at 24 settled poses approached from both sides, and a local gravity model is
   fitted and checked on held-out poses. A passing run saves the measured envelope (typically ±7°) and the model; any
   failure (tracking, too little encoder motion, friction asymmetry, faults, saturation, a failed fit) keeps the
   previous calibration. Every joint needs clearance both ways: autocal never searches for stops.
4. **Registered (URDF) calibration**, when a rest or fixture holds the arm at a known model pose: without a relative
   reference, `arm autocal` prints the reference pose (`home_pose_rad`); you set the arm there with the motors
   disabled and confirm, it derives the zero offsets from that pose, then runs the same checks in a ±3° envelope.
5. **Measured coordinates** beyond the local envelope: `arm commission --write-config` (two known angles and the usable
   endpoints per joint, by hand), `arm kt` with test masses, `arm limits` (powered stop search on an assessed setup).
   With the powered launch running, `arm bringup --soak-s 10` jogs each joint by at most 3° with confirmation and
   checks holding.

**Stop zeroing**: `arm autocal --stop-zero --sim --quick --fresh` rehearses zeroing each joint against its mechanical
stop (the CAD contact angles in `cad/arm01_capture.json`; J1 uses its configured −90° / +180° range).
`arm autocal --verify-stop shoulder_pitch` checks one stop on the hardware: with the drives disabled, seat it by hand
twice and it reports the angle against the configured endpoint. `arm autocal --check-encoder <joint>` compares every
encoder during a small hand movement. Physical stop angles and contact loads still need measuring on the assembled arm.

Without hardware: `arm autocal --sim --quick --fresh` runs the whole automatic sequence on the simulated arm.

## Calibration and identification stages

One command per stage, from `ros2 run wisco_ethercat_ros2_hw arm <stage>` (or `python3 -m arm_stack.cli <stage>` in the
hardware package). `--sim` runs the identical procedure on a simulated arm with injected errors, so every stage can be
rehearsed before hardware; `arm all --sim --quick --fresh` runs the whole chain.

| Stage | Command | Writes |
|---|---|---|
| Passive commissioning (ROS stopped, drives disabled) | `arm commission` | directions, counts per revolution, zero offsets, software limits |
| Bring-up | `arm bringup` | joint `direction` (with `--write-config`) |
| Indicator check | `arm indicator` | report: 90° step vs a protractor, torque sign, torque units, unset values |
| Zero | `arm zero` | `zero_offsets_rad` |
| Limits | `arm limits` | `soft_limits_rad` (3° inside each found stop; the URDF range for joints with `hard_stops: false`) |
| Torque constant | `arm kt` | `kt_output_nm_per_a` |
| Gravity | `arm gravity` | `link_inertials` |
| Friction | `arm friction` | `friction` |
| Latency | `arm latency` | `latency_ms` (transport delay on the torque channel) |
| Kinematics | `arm pivot`, `arm plate` (divot plate, no mocap), `arm mocap-calib` | zero offsets, kinematic overlay, joint stiffness |
| Motor model (PACE) | `arm pace` | `pace`: per-joint armature, friction, damping and delay fitted to logged chirps |
| Repeatability | `arm repeat [--indicators]` (ISO 9283), `arm mocap-repeat` | `validation.repeatability_rp_mm` |
| Release gate | `arm validate` | `validation.*` and a report beside the calibration file |
| Hand-guiding check | `arm gravity-check` | `float_friction_gain` (with `--write-config`) |

Logs and data: every procedure can record to MCAP; `arm export pace <log> <out>` and
`arm export lerobot <out> <logs...>` turn recordings into PACE inputs or a LeRobot dataset. `arm status` prints the
newest calibration version.

**PACE** fits each joint's motor model by replaying the logged command through a simulated joint and matching the
measured response (CMA-ES, Levenberg-Marquardt or Gaussian-process Bayesian optimization). With the drives' own loop
gains known (read over SDO into `drive_gains`), the replay also simulates the other joints under their position loops,
and applies the calibrated joint stiffness, which is why kinematics calibration runs before PACE. Chirps stay below
each joint's servo resonance (about 5 Hz on the shoulders). An Isaac Lab replay backend exists for the full arm
(build the USD with `tools/isaac/import_urdf.py`); it is tested against a stand-in for Isaac Lab only.

**Hand-guiding** (`arm float`): torque mode with kp = 0, real-time gravity compensation (plus an optional payload),
friction help that fades in with speed so a released joint always stops, optional damping, and spring-damper walls
10° inside the soft limits. Any joint above half its speed limit latches strong damping. `--lock` holds chosen joints;
`--record` logs every cycle.

**Without mocap**: kinematics from a machined divot plate (`arm plate`: hand-guide the probe tip into each divot in
several configurations, with and without a test mass; held-out divots validate the fit) and repeatability from three
dial indicators on a cube on the flange (`arm repeat --indicators`). Plate and cube dimensions are in `config/arm01.yaml`
(placeholders until the plate drawing exists, `needs_drawing: true`).

**Single-actuator bench** (`config/bench01.yaml`): every actuator, spares included, goes through it before assembly.
```bash
arm bringup   --arm bench01 --serial SN123 --model x4_36
arm indicator --arm bench01 --serial SN123 --model x4_36
arm rig       --arm bench01 --serial SN123 --model x4_36   # zero, torque constant on three hook masses, gravity, friction, latency
```
The rig's lever length, hook point and test masses are under `rig:` in `bench01.yaml`.

## Assumptions to confirm on the bench

- Zeros and signs come from the CAD export's operating coordinates, not encoder readings: commissioning or `arm zero`
  sets the encoder zeros, `arm bringup` checks the signs against gravity (J3 is mounted inverted, so its `direction`
  may need to be −1).
- **q = 0 is the arm stretched out forward**, its worst gravity load, so every stage starts from `home_pose_rad` in
  `config/arm01.yaml` (shoulder pitch +90°, elbow +45°), which is also the default jig pose.
- Ranges are the export's operating range (J1 −90..+180°, J2 0..190°, J3 ±90°, J4 0..130°, J5 ±90°, J6 ±35°, J7 ±90°),
  not certified stops. J1 has no hard stop in the CAD, so `hard_stops: false` keeps its range as the soft limit and
  `arm limits` never drives it past.
- The wrist pitch is a near-parallelogram (output/crank 0.996–1.004): the J6 encoder drives `wrist_pitch` 1:1 within
  0.25° over ±35°; the crank and coupler are mimic joints.
- Link masses are the export's audited model (provisional actuator masses, solid PLA peg, harness estimate) until the
  weighed values go in `link_masses_kg`.
- Drive facts and torque-scale caveats for the MyActuator drives are in the hardware package README.

Tests for this arm (forward kinematics against the CAD capture, IK, joint-table checks, every stage in simulation,
C++/Python gravity parity) are in `ros2_ws/src/wisco_ethercat_ros2_hw/test/`.
