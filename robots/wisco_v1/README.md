# Wisco V1

The WiscoHumanoids humanoid: 31 revolute joints (legs 12, waist 3, arms 14, head 2), MyActuator RMD-X motors
(X8-150, X6-60, X4-36, X4-10), rod-linkage ankles and waist, a ZED X stereo camera on the head, an EVAL_HI05 IMU in
the pelvis, no lidar. Modelled mass 40.6 kg (see "What is estimated").

Everything about the physical robot is here: the model the simulator and training use, every number with its source,
pre-training checks, simulated current draw, and the hardware observations from simulation. Training code:
`training/locomotion/wisco_v1/` (walking) and `training/tracking/wisco_v1/` (motion tracking). The ROS stack uses it as
`robot:=wisco`.

## Files

| Path | What |
|---|---|
| `robot.urdf`, `robot.xml`, `model_report.json` | the robot model: URDF; MuJoCo scene with IMU site, motor ranges and keyframes `zero`, `stand`, `home`; per-joint report. Geometry from the CAD export; every number below written by `model/sync_robot_folder.py` |
| `meshes/` | `<link>.stl` visual, `<link>_collision.stl` convex hull (feet use a box) |
| `linkage.json`, `linkage.py` | ankle and waist rod geometry; joint ↔ motor torque and velocity conversion |
| `hardware_observations.md` | what simulation has found about the hardware, with what is needed back; updated by simulation runs |
| `model/robot_numbers.yaml` | **the numbers file**: motors (datasheet values + sources), command delay and bus, mass overrides, gains and control modes, default pose, locomotion action joints, IMU and camera poses |
| `model/motor_datasheets/` | the MyActuator parameter sheets the motor numbers come from |
| `model/robot_params.py` | derives every joint value (gains, limits, friction, torque-speed) from the numbers file |
| `model/sync_robot_folder.py` | writes those values into `robot.urdf`, `robot.xml`, `model_report.json` (idempotent) |
| `model/linkage_tools.py` | ankle linkage analysis → `model/ankle_linkage.json` (reachable range, Jacobian grids) |
| `model/build_training_model.py` | `robot.xml` → `model/wisco_v1_train.xml` + `model/train_model_info.json`, the model mjlab trains on |
| `model/ankle_motor.py` | ankle joint torque → per-motor torque |
| `checks/` | pre-training checks in plain MuJoCo with mjlab's actuator model (`run_checks.py`; `g1_baseline.py` runs the same on the G1); results go to `checks/results/` (not kept in the repo) |
| `tools/current_estimate.py` → `reports/current_draw/` | per-motor and bus current while walking, from the policy and a datasheet motor model |

The sim world the ROS stack loads, `assets/robots/wisco_v1/mjlab_1.6.0/`, is built from the training task with
`tools/build/build_robot_world.py wisco`.

## How the numbers flow

```
model/motor_datasheets/ ─┐
measurements, decisions ─┴─► model/robot_numbers.yaml ─► robot_params.py ─┬─► sync_robot_folder.py ─► robot.urdf / robot.xml / model_report.json
                                                                          ├─► build_training_model.py ─► model/wisco_v1_train.xml (mjlab)
                                                                          └─► training (gains, limits, randomization ranges)
```

| What the numbers file sets | Value | Source |
|---|---|---|
| Joint peak torque | motor peaks; ankle pitch 86.6, ankle roll 35.2, waist roll 78.0, waist pitch 83.7 N·m (two motors through the linkage, zero pose) | datasheets |
| Joint no-load speed | X8-150 16.55, X6-60 18.43, X4-36 11.62, X4-10 33.20 rad/s; linkage joints converted | datasheets |
| Joint friction | datasheet backdrive torque: X8-150 0.34, X6-60 1.6, X4-36 1.14, X4-10 0.8 N·m (linkage joints summed) | datasheets |
| Joint damping | X8-150 0.13, X6-60 0.02, X4-36 0.031, X4-10 0.0 N·m·s/rad | half the bound implied by the datasheet no-load power |
| Armature (MuJoCo) | rotor inertia × gear ratio²; X8-150 0.046 (its sheet value is not physical; see the observations) | datasheets |
| Ankle pitch lower limit | −0.546 rad (the full roll range is reachable only down to about −0.57 rad) | linkage reach analysis, 0.02 rad margin |
| IMU frame | board axes: x → robot right, y → down, z → forward (URDF rpy −π/2 0 −π/2) | CAD |
| Camera | ZED X; `head_camera_optical` frame in the ROS optical convention; tilt from `sensors` (assumed level) | CAD |
| Home keyframe | hip pitch 0.3, knee 0.6, ankle pitch 0.3, shoulder roll 0.2, elbow 0.5; pelvis 0.688 m | training setup |

Everything else (links, meshes, inertias, centres of mass, joint axes and positions, collision shapes, every other joint
range) is the CAD export's.

## Drive modes and gains

| Joints | MyActuator EtherCAT mode | Who computes the torque |
|---|---|---|
| Legs, waist, head | mode 5 "PVT/MIT": target position + kp + kd every cycle; the drive computes τ = kp (q_des − q) + kd (q̇_des − q̇) + τ_ff | the drive, with **the same kp/kd the walking policy is trained with** |
| Arms | position mode plus a gravity torque offset, or torque mode with the PD on the host | the arm controller (`ArmController`, gravity compensation with Pinocchio) |

Gains (`model/robot_numbers.yaml` → `control.gains`); kd = 2 × 0.5 × √(kp × I), with I the joint inertia at the home pose
(damping ratio 0.5). The arm values are what the simulation uses in place of the arm controller. Head (welded in
training): kp 20, kd 0.5. A reference humanoid's kd is listed for comparison.

| Joint | kp (N·m/rad) | kd (N·m·s/rad) | Reference kd |
|---|---|---|---|
| Hip pitch / roll | 120 / 120 | 11.34 / 10.6 | 7.5 |
| Hip yaw | 50 | 1.95 | 2.5 |
| Knee | 120 | 4.97 | 7.5 |
| Ankle pitch / roll (joint space, 2× X4-36 linkage) | 35 / 35 | 2.14 / 0.88 | 1.5 |
| Waist yaw / roll / pitch | 100 | 7.52 / 11.89 / 10.06 | 7.5 |
| Shoulder pitch / roll / yaw | 40 | 3.7 / 3.42 / 1.53 | 2.5 / 0.75 / 0.75 |
| Elbow | 30 | 1.73 | 0.75 |
| Forearm yaw, wrist pitch, wrist roll | 20 | 0.9 / 0.58 / 0.39 | 0.75 |

## Rebuild (laptop, repo `.venv`)

```
cd robots/wisco_v1/model
../../../.venv/bin/python linkage_tools.py          # ~6 min, only if linkage.json changes
../../../.venv/bin/python sync_robot_folder.py      # robot.urdf, robot.xml, model_report.json
../../../.venv/bin/python build_training_model.py   # model/wisco_v1_train.xml + train_model_info.json
cd ../checks && ../../../.venv/bin/python run_checks.py --actuator dc --delay 2
cd ../../.. && .venv-mjlab/bin/python tools/build/build_robot_world.py wisco   # the ROS sim world
```

## Changing numbers when real data arrives

| New data | Edit | Then |
|---|---|---|
| Measured or CAD masses | `model/robot_numbers.yaml` → `masses.overrides` | rebuild (above), rerun the checks, retrain |
| Motor bench tests (inertia, friction, damping, torque-speed) | `model/robot_numbers.yaml` → `motors` (value + source) | same |
| Gantry actuator fit (armature, friction, damping, delay per joint; `training/sim2real/pace.py`) | `model/robot_numbers.yaml` → `motors` and `delay` (source: the fit file) | same; narrow the dynamics randomization around the fitted values |
| Measured command delay | `model/robot_numbers.yaml` → `delay` | same |
| Real driver PD gains | `model/robot_numbers.yaml` → `control.gains` | same |
| IMU or camera pose | `model/robot_numbers.yaml` → `sensors` | same |

## Conventions

x forward, y left, z up, metres. Base link `pelvis`, origin midway between the hip pitch axes. Zero pose = standing
straight, arms down, feet flat; pelvis origin 0.7015 m above the floor. Every link frame sits at its joint pivot with
axes parallel to the base at the zero pose. The walking policies start from the `home` keyframe.

## What is estimated

- **Mass:** motors use datasheet mass (X8-150 1.33, X6-60 0.82, X4-36 0.36, X4-10 0.33 kg; 53 % of the total). Everything
  else is volume × density (aluminium 2.70, steel 7.85, plastic 1.25, sole 1.2, boards 1.9, gripper 1.4 g/cm³), with no
  wiring or battery; training randomizes every link's mass by ×0.85–1.20.
- Ankle rods and waist links are lumped into the shin and torso.
- Friction, damping and armature come from datasheets; bench tests should confirm them.
- Camera tilt, foot sole material, command delay and driver gains are not measured yet.

Details and what is needed back: `hardware_observations.md`.

## Ankles and waist

Modelled as two plain joints at the real pivots. Ankle: pitch then roll, the roll axis 17.5 mm below the pitch axis,
the pitch axis 61.5 mm above the floor. Waist: yaw, then roll and pitch through one universal joint 174 mm above the
pelvis origin. Joint torque and speed limits for these four joints are the two-motor equivalents at the zero pose;
`linkage.py` turns a joint torque into each motor's torque (and back: `joint_angles` solves the joints from the motor
angles). In training, every physics step applies each ankle motor's own limits through the rods (`ankle_actuator.py`).
On the robot, `ParallelLinkageController` (`ros2_ws/src/wisco_ethercat_ros2_hw`, "Parallel joints") does the same
translation in real time, so the policy keeps commanding plain pitch and roll joints.
