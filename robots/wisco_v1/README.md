# Wisco V1

The WiscoHumanoids humanoid: 31 revolute joints (legs 12, waist 3, arms 14, head 2), MyActuator RMD-X motors
(X8-150, X6-60, X4-36, X4-10), rod-linkage ankles and waist, ZED X head camera, EVAL_HI05 IMU in the pelvis, no lidar.
Total mass 40.6 kg as modelled (see "What is estimated").

Everything about the physical robot lives here: the model the simulator and training use, every number with its
source, pre-training checks, and the hardware findings. Training code: `training/locomotion/wisco_v1/` (walking) and
`training/tracking/wisco_v1/` (motion tracking). Hardware interface and e-stop: `hardware/README.md`.

## Files

| Path | What |
|---|---|
| `robot.urdf`, `robot.xml`, `model_report.json` | **generated** robot model (URDF; MuJoCo scene with IMU site, motor ranges, keyframes `zero`, `stand`, `home`; per-joint report). Don't hand-edit: see `CHANGES.md` |
| `original_from_team/` | the CAD export as received ("Legs V2 Full Assembly.step" + "FullAssmb.step", built 2026-11-09), never modified |
| `meshes/` | `<link>.stl` visual, `<link>_collision.stl` convex hull (feet use a box) |
| `linkage.json`, `linkage.py` | ankle and waist rod geometry; joint ↔ motor torque/velocity conversion |
| `stand_test.py` | 5 s standing check |
| `CHANGES.md` | what the generated model changes from the CAD export, with sources; planned drive modes and gains |
| `HARDWARE_ISSUES.md` | findings and open questions for the mechanical and electrical team (H1–H13) |
| `model/robot_numbers.yaml` | **the numbers file**: motors (datasheet values + sources), delay and bus, mass overrides, gains and control modes, default pose, locomotion action joints, IMU and camera poses |
| `model/motor_datasheets/` | the official MyActuator parameter sheets (images) |
| `model/robot_params.py` | derives every joint value (gains, limits, friction, torque-speed) from the files above |
| `model/sync_robot_folder.py` | regenerates `robot.urdf`, `robot.xml`, `model_report.json` from `original_from_team/` + the numbers |
| `model/linkage_tools.py` | ankle linkage analysis → `model/ankle_linkage.json` (reachable range, Jacobian grids) |
| `model/build_training_model.py` | `robot.xml` → `model/wisco_v1_train.xml` + `model/train_model_info.json` (what mjlab trains on) |
| `model/ankle_motor.py` | ankle joint torque → per-motor torque |
| `checks/` | pre-training checks in plain MuJoCo with mjlab's actuator model (`run_checks.py`, checks A1–A8; `g1_baseline.py` runs the same on the G1); results in `checks/results/` |
| `tools/current_estimate.py` → `reports/current_draw/` | per-motor and bus current from the walking policy (datasheet motor model) |

The sim world the ROS stack loads, `assets/robots/wisco_v1/mjlab_1.6.0/`, is built from the training task by
`tools/build_robot_world.py wisco`.

## Rebuild (Mac, repo `.venv`)

```
cd robots/wisco_v1/model
../../../.venv/bin/python linkage_tools.py          # ~6 min, only if linkage.json changes
../../../.venv/bin/python sync_robot_folder.py      # robot.urdf, robot.xml, model_report.json
../../../.venv/bin/python build_training_model.py   # model/wisco_v1_train.xml + train_model_info.json
cd ../checks && ../../../.venv/bin/python run_checks.py --actuator dc --delay 2
cd ../../.. && .venv-mjlab/bin/python tools/build_robot_world.py wisco   # the ROS sim world
```

## Changing numbers when real data arrives

| New data | Edit | Then |
|---|---|---|
| Measured / CAD masses | `model/robot_numbers.yaml` → `masses.overrides` | rebuild (above), rerun the checks, retrain |
| Motor bench tests (inertia, friction, damping, torque-speed) | `model/robot_numbers.yaml` → `motors` (value + source) | same |
| Measured command delay | `model/robot_numbers.yaml` → `delay` | same |
| Real driver PD gains | `model/robot_numbers.yaml` → `control.gains` (kp/kd) | same |
| IMU / camera pose | `model/robot_numbers.yaml` → `sensors` | same |

## Conventions

x forward, y left, z up, metres. Base link `pelvis`, origin midway between the hip pitch axes. Zero pose = standing
straight, arms down, feet flat; pelvis origin 0.7015 m above the floor. Every link frame sits at its joint pivot with
axes parallel to the base at the zero pose (all rpy = 0). The walking policies start from the knees-bent `home`
keyframe (pelvis 0.688 m).

## What comes from the CAD

Joint axes and pivots (from the motor and bearing cylinders), link geometry, part volumes, centre-of-mass locations,
inertia shapes, ankle/waist rod lengths and crank radii, IMU board pose. The legs come from Legs V2 (not the older
copy inside FullAssmb). Duplicate motors/bearings and the "wave" hands were left out; the YUBI gripper is the hand.

## What is estimated (change these when real numbers exist)

- **Mass:** motors use datasheet mass (X8-150 1.33, X6-60 0.82, X4-36 0.36, X4-10 0.33 kg; 53 % of the total).
  Everything else is volume × density: aluminium 2.70, fasteners/bearings steel 7.85, covers/fairings/hardstops
  plastic 1.25, sole 1.2, boards 1.9, gripper 1.4 g/cm³. No wiring or battery. A motor's whole mass sits on the link
  that holds its body (HARDWARE_ISSUES H1).
- Ankle rods and waist links are lumped into the shin and torso.
- Joint friction = datasheet backdrive torque; viscous damping = half the bound from the datasheet's no-load power;
  armature = rotor inertia × gear ratio² (X8-150's sheet value is a likely typo, H6). Bench tests should confirm (H10).
- Camera tilt (assumed level in the model; the sim profile tests 25° down, H8), foot sole material (H12), command
  delay and driver gains (H11).

## Ankles and waist

Modelled as two plain joints at the real pivots. Ankle: pitch then roll, roll axis 17.5 mm below the pitch axis,
pitch axis 61.5 mm above the floor. Waist: yaw, then roll and pitch through one universal joint 174 mm above the
pelvis origin. Joint effort/velocity limits for these four joints are the two-motor equivalents at the zero pose; use
`linkage.py` to turn a joint torque into each motor's torque. The ankle pitch lower limit is −0.546 rad (the full
roll range is unreachable below about −0.57 rad, H2).
