# Wisco V2 robot model: changes from the CAD export

The original CAD export (Legs V2 Full Assembly.step + FullAssmb.step) is kept unchanged in `original_from_team/`.
`robot.urdf`, `robot.xml` and `model_report.json` in this folder are **generated** from it plus the
numbers below by `../training/model/sync_robot_folder.py`. To change a number, edit its source
(table below) and rerun that script; don't hand-edit the generated files.

**Unchanged from CAD:** every link, mesh, mass, inertia, centre of mass, joint axis, joint position,
collision shape, and every joint range except ankle pitch. Checked: all body positions identical to
the CAD export (0.000 mm difference), total mass 40.551 kg in both.

## What changed (2026-11-10)

| Item | CAD export | Now | Source |
|---|---|---|---|
| Joint effort (peak torque) | 120 / 60 / 34 / 10 N*m, linkage joints approximate | Same motor peaks; ankle pitch 86.6, ankle roll 35.2, waist roll 78.0, waist pitch 83.7 N*m (two motors through the linkage, zero pose) | MyActuator datasheets |
| Joint velocity (no-load) | from old URDF | X8-150 16.55, X6-60 18.43, X4-36 11.62, X4-10 33.20 rad/s; linkage joints converted | MyActuator datasheets |
| Joint friction | "carried over from old URDF" | Datasheet backdrive torque: X8-150 0.34, X6-60 1.6, X4-36 1.14, X4-10 0.8 N*m (linkage joints summed) | MyActuator datasheets |
| Joint damping | 0.2-1.6 (guesses) | X8-150 0.13, X6-60 0.02, X4-36 0.031, X4-10 0.0 N*m*s/rad | Upper bound from datasheet no-load power, halved |
| Armature (MJCF only) | X8 0.05 (guess), others datasheet | X8-150 0.046 (rotor 1.2 kg*cm2); others unchanged (datasheet rotor inertia x gear ratio^2) | Datasheets; X8-150 sheet value is a likely typo, see note |
| Ankle pitch lower limit | -0.611 rad (-35 deg) | **-0.546 rad (-31.3 deg)** both ankles | Rod-linkage reach analysis: full roll range only reachable down to -0.566 (L) / -0.576 (R); 0.02 rad margin |
| IMU frame | at the board position, axes = pelvis axes (assumed) | Board orientation from the Legs V2 CAD: board x -> robot right, y -> down, z -> forward (URDF rpy -pi/2 0 -pi/2; MJCF quat 0.5 -0.5 0.5 -0.5) | CAD (EVAL_HI05 board on the IMU Mount) |
| Camera | `head_camera` frame | + `head_camera_optical` (ROS optical convention: z forward, x right, y down). Camera is a ZED X; tilt assumed level | Team confirmation |
| URDF transmissions | none | One per joint: gear ratio, motor model, planned control mode (comment) | Datasheets, control decision below |
| MJCF motor ctrlrange | per old URDF | = joint peak torque above | Datasheets |
| MJCF keyframes | zero, stand | + `home`: knees-bent pose the walking policies start from (hip pitch 0.3, knee 0.6, ankle pitch 0.3, shoulder roll 0.2, elbow 0.5; pelvis 0.688 m) | Training setup |
| model_report.json | joint values from old URDF | Same updates, plus `control_mode` per joint | |

The MyActuator parameter sheets used are saved in `../training/model/motor_datasheets/`; every
value with its source is in `../training/model/robot_numbers.yaml`.

## Planned motor control (decided 2026-11-10)
| Joints | Driver mode (MyActuator EtherCAT protocol V2.0) | Who computes torque |
|---|---|---|
| Legs, waist, head | Mode 5 "PVT / MIT": target position + kp + kd sent every cycle (objects 0x607A, 0x2000, 0x2001); driver computes tau = kp (p_des - p) + kd (v_des - v) + tau_ff | Driver, with **our** kp/kd (the same values the walking policy is trained with) |
| Arms (shoulder, elbow, forearm, wrist) | Mode 10 "CST": target torque (0x6071, per mille of rated torque) | ROS controls stack (gravity compensation + PD, Pinocchio) |
Not used for policy joints: mode 8 (CSP), which runs the driver's own position/velocity loops instead of our gains.
Gains (decided 2026-11-10, in `../training/model/robot_numbers.yaml` -> control.gains, also per joint in `model_report.json`).
Legs, waist: sent to the drivers in MIT mode, identical to what the walking policy is trained with. Arms: the values the
simulation uses to stand in for the ROS controls stack (on the robot the controls stack chooses its own). kd = 2 x 0.5 x
sqrt(kp x I), I = joint inertia at the home pose (damping ratio 0.5, close to RP1's kd; critical damping would give the hips
kd 21-24). Head (welded in training): kp 20, kd 0.5.

| Joint | kp (N*m/rad) | kd (N*m*s/rad) | RP1 kd (for comparison) |
|---|---|---|---|
| Hip pitch | 120 | 11.34 | 7.5 |
| Hip roll | 120 | 10.6 | 7.5 |
| Hip yaw | 50 | 1.95 | 2.5 |
| Knee | 120 | 4.97 | 7.5 |
| Ankle pitch (joint space, 2x X4-36 linkage) | 35 | 2.14 | 1.5 |
| Ankle roll (joint space) | 35 | 0.88 | 1.5 |
| Waist yaw | 100 | 7.52 | 7.5 |
| Waist roll (2x X6-60 linkage) | 100 | 11.89 | 7.5 |
| Waist pitch | 100 | 10.06 | 7.5 |
| Shoulder pitch (sim stand-in) | 40 | 3.7 | 2.5 |
| Shoulder roll (sim stand-in) | 40 | 3.42 | 0.75 |
| Shoulder yaw (sim stand-in) | 40 | 1.53 | 0.75 |
| Elbow (sim stand-in) | 30 | 1.73 | 0.75 |
| Forearm yaw (sim stand-in) | 20 | 0.9 | 0.75 |
| Wrist pitch (sim stand-in) | 20 | 0.58 | 0.75 |
| Wrist roll (sim stand-in) | 20 | 0.39 | 0.75 |


## Still open (details in ../training/HARDWARE_ISSUES.md)
- **X8-150 rotor inertia (H6):** the official sheet prints 0.001023 kg*m2 (10.23 kg*cm2), physically impossible for this
  actuator's rotor; likely a 10x typo. Model uses 1.2 kg*cm2. Please confirm with MyActuator.
- **IMU (H7):** check that the axes printed on the EVAL_HI05 board match the CAD part axes.
- **Masses (H1):** non-motor parts are CAD volume x assumed density; no wiring or battery. CAD mass properties with
  real materials, then weighing, go into `../training/model/robot_numbers.yaml` (masses.overrides).
- **Foot sole (H12), supply voltage per motor group (H13), ankle/waist zero calibration (H4), camera tilt (H8).**
