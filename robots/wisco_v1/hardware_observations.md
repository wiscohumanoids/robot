# Wisco V1 hardware observations

What simulation of `robots/wisco_v1/` has found about the physical robot. Each section says what was found and how,
what simulation does about it for now, and what is needed back from the hardware side. Simulation runs add to and
update these sections; when a value is measured, it goes into `model/robot_numbers.yaml` with its source and the
section is updated or removed.

**Nothing here blocks training in simulation.** Each item either has a simulation workaround or is covered by
randomizing the uncertain value during training. They matter before a policy runs on the real robot: the sooner the
measurements come back, the less retraining later.

| Section | Summary | State | Needed before | Who can close it |
|---|---|---|---|---|
| [Masses](#masses) | 53 % of the mass (31 motors) is datasheet mass; the rest is CAD volume × density, no wiring or battery | open | real robot (high priority: easy to measure, big effect) | mechanical: CAD mass properties now, weigh the robot later |
| [Ankle reach](#ankle-reach) | part of the ankle range is unreachable through the linkage | workaround | design review | mechanical: confirm the real limit |
| [Ankle torque](#ankle-torque) | pitch and roll share two X4-36 motors; walking needs more than their rated torque much of the time | open | real robot (motor sizing) | mechanical: ankle motor sizing and cooling once the real mass is known |
| [Ankle and waist zeros](#ankle-and-waist-zeros) | crank zero positions differ between left and right in the CAD | open | real robot (calibration) | mechanical: calibration procedure |
| [Grippers and hips](#grippers-and-hips) | with the arms straight down the grippers overlap the hips | workaround | real robot (rest pose) | mechanical: confirm |
| [Rotor inertia](#rotor-inertia) | from datasheets, except the X8-150, whose printed value is not physical | X8-150 open | real robot (sets the gains) | MyActuator: X8-150 rotor inertia; mechanical: installed variants |
| [IMU orientation](#imu-orientation) | from the CAD; only a board-axis check remains | nearly closed | real robot | mechanical: printed board axes = CAD axes |
| [Head camera](#head-camera) | ZED X confirmed; mount tilt unknown | open (minor) | perception on the robot | mechanical: mount tilt |
| [Shoulder roll inward limit](#shoulder-roll-inward-limit) | the arm hits the torso before the inward limit | open (minor) | manipulation | mechanical: intended range |
| [Friction and damping](#friction-and-damping) | friction = datasheet backdrive torque; damping bounded from the datasheet | bench test to confirm | real robot | bench test: breakaway and constant-speed sweeps |
| [Drive mode, gains and delay](#drive-mode-gains-and-delay) | mode and gains decided; delay and the drive's PD implementation not measured | bench test | real robot (high priority) | controls and electrical: bench checks |
| [Foot soles](#foot-soles) | flat box, assumed friction 0.6; real sole unknown | open | real robot | mechanical: sole material and drawing |
| [Supply voltage and heating](#supply-voltage-and-heating) | supply voltage per motor group unknown; heating modelled only on the ankles | open | real robot | electrical: bus voltages |

---

## Masses
- **Found:** each motor has its datasheet mass (X8-150 1.33, X6-60 0.82, X4-36 0.36, X4-10 0.33 kg): 31 motors,
  21.45 kg, 53 % of the 40.55 kg total. Every other part is CAD volume × an assumed density (aluminium 2.70, steel 7.85,
  plastic 1.25 g/cm³, …). No wiring, connectors or battery.
- **Why it matters:** mass and where it sits decide how much torque every joint needs. If the real robot is heavier, a
  policy trained on 40.6 kg may not hold it up.
- **Simulation meanwhile:** training scales every link's mass (and inertia with it) by ×0.85–1.20.
- **Possible now, without hardware:** export mass properties from the CAD with real materials assigned, per part or
  subassembly, plus spec-sheet masses for purchased parts (bearings, boards, battery, harness). Each number goes into
  `model/robot_numbers.yaml` (`masses.overrides`) with its source; then rebuild and retrain.
- **Needed back:** the CAD mass export; whether a battery is on board and where. Later: weigh the robot, ideally per
  subassembly (each leg below the hip, torso + arms, head).

## Ankle reach
- **Found:** solving the rod-and-crank geometry in `linkage.json` over the joint range (`model/linkage_tools.py`, full
  scan of roll at each pitch): the full roll range (±15°) is reachable only down to ankle pitch −0.566 rad (−32.4°) on
  the left and −0.576 rad (−33.0°) on the right, not the −0.611 rad (−35°) of the range-of-motion sheet. Beyond that a rod
  would have to stretch about 1 mm. Inside the reachable range the linkage is well behaved (worst Jacobian condition
  number 4.8).
- **Simulation meanwhile:** ankle pitch lower limit −0.546 rad (−31.3°): the reachable limit plus a 0.02 rad margin.
- **Needed back:** is the published range (pitch −35° to +45°, roll ±15°) meant to hold at every combination? If yes,
  check rod lengths and ball-joint positions; if no, what is the real combined limit (hard stops)?

## Ankle torque
- **Found:** the two X4-36 motors per ankle drive pitch and roll together through the rods (pitch leverage about 1.27×,
  roll about 0.52× at the zero pose). The 86.6 N·m pitch peak is available only with zero roll torque, and the reverse;
  rated (continuous) pitch torque with both motors is about 26.7 N·m.
- **Measured in simulation** (walking policy, 256 robots × 20 s, full curriculum difficulty, random commands): with only
  joint-level torque caps, at least one ankle motor was above its 10.5 N·m rated torque 65 % of the time and above its
  34 N·m peak 2.7 % of the time (p95 22, p99.9 42, max 77 N·m). With each motor clamped to its peak and torque-speed line
  through the linkage (as the drives would), some ankle motor is still above rated in 67 % of steps (22.7 % of
  motor-steps, RMS 10.1 N·m; 2 s moving RMS above rated 34 % of motor-time).
- **Simulation meanwhile:** training clamps each ankle motor through the linkage, penalizes any step above peak and
  penalizes sustained torque above rated (`training/locomotion/wisco_v1/wisco_mjlab/robot/ankle_actuator.py`,
  `linkage_penalty.py`).
- **Two X4-36 per ankle look undersized for this gait and mass.** Whether the real ankle overheats depends on the real
  mass (likely above 40.6 kg) and cooling.
- **Needed back:** once the real mass is known, check ankle motor sizing (X6-class motors or a higher linkage ratio) and
  cooling; the datasheet stall figure (X4-36: 34.5 N·m for 3 s raises it 50 °C) is the relevant limit.

## Ankle and waist zeros
- **Found:** at joint zero (foot flat), the crank angles relative to their CAD position are about 0.04 / 0.02 rad on the
  left ankle but 0.39 / −0.40 rad on the right: the cranks were drawn in a different pose on each side of the assembly.
- **Why it matters:** only for calibrating the real robot: each motor's encoder zero must map to the foot-flat pose.
- **Needed back:** a calibration procedure or jig for the ankle and waist motor zeros, and confirmation that the CAD crank
  positions match the real assembly.

## Grippers and hips
- **Found:** MuJoCo collision check at the all-zero pose: each gripper overlaps the hip pitch and hip roll links by
  4–5 cm, and the forearm and wrist touch the hip pitch link.
- **Simulation meanwhile:** the `home` pose rolls the shoulders out slightly and bends the elbows (shoulder roll 0.2,
  elbow 0.5 rad) so the hands clear the hips.
- **Needed back:** confirm this is only geometry at the zero pose (the arms cannot actually hang straight). If the real
  arms rest at zero, there is a real interference.

## Rotor inertia
- **Found:** from the MyActuator parameter sheets (`model/motor_datasheets/`): X4-10 0.25, X4-36 0.3, X6-60
  0.66 kg·cm²; armature = rotor inertia × gear ratio².
- **X8-150:** its sheet prints 0.001023 kg·m² (10.23 kg·cm²). That is not physical: a solid cylinder of the whole
  1.33 kg actuator at its 96 mm diameter would be 15.3 kg·cm², and the rotor is a fraction of that mass; most likely a
  10× typo for 1.023. Other X8-family listings: X8 0.805, X8 Pro 1.45, X8-120 1.5 kg·cm².
- **Simulation meanwhile:** X8-150 1.2 kg·cm² (armature 0.046 kg·m²), randomized 0.8–2.0 in training.
- **Why it matters:** armature sets how fast a joint can accelerate, and the gains are derived from joint inertia.
- **Needed back:** from MyActuator, the X8-150 rotor inertia and its reference shaft. From mechanical, which variants
  are installed: the X4-10 and X4-36 non-"L" versions are listed as discontinued, and the X4-10-L / X4-36-L parameters may
  differ.

## IMU orientation
- **Found (CAD):** EVAL_HI05 board, 34 × 24 × 3 mm, on the IMU mount at the back of the pelvis, upright, chip facing
  backward, at (−0.0518, −0.0003, 0.0345) m. Board x → robot right (−y), y → down (−z), z → forward (+x): URDF rpy
  (−π/2, 0, −π/2), quaternion wxyz (0.5, −0.5, 0.5, −0.5), in `robot.urdf` (`imu_link`) and `robot.xml` (`imu` site).
  Training reads the IMU in pelvis axes (an `imu_policy` site at the same point); on the robot the IMU driver
  (`ros2_ws/src/wisco_hipnuc_imu`, `mount_wxyz` in its config) applies this fixed board-to-pelvis rotation.
- **Needed back:** confirm the axes printed on the board match the CAD part axes:
  `ros2 run wisco_hipnuc_imu imu_check --mounted` with the robot hung level or standing; it passes when gravity lands on
  +z after the rotation, and otherwise prints which board axis points up.

## Head camera
- **Found:** no lidar. The camera is a ZED X (the stack's `sensors:=stereo` setup) at the CAD `head_camera` frame,
  8.5 cm forward of the head pivot.
- **Simulation meanwhile:** the camera looks straight ahead, level with the head (`ros2_ws/src/hsim_bringup/config/robots/wisco.yaml`).
- **Needed back:** the mount tilt.

## Shoulder roll inward limit
- **Found (range sweep):** driving each shoulder roll toward its inward limit (−0.436 rad, −25°) with the arm hanging and
  the elbow slightly bent, the upper arm hits the torso at about −0.17 rad (−10°) on the left and −0.23 rad (−13°) on the
  right; the left/right difference comes from the torso collision shape.
- **Why it matters:** little for walking; it matters for reaching across the body.
- **Needed back:** is −25° inward roll meant only with the arm raised forward? If the CAD says it clears the torso
  hanging down, the collision shapes or the torso geometry need a look.

## Friction and damping
- **Found:** joint friction is the datasheet backdrive torque (X8-150 0.34, X6-60 1.6, X4-36 1.14, X4-10 0.8 N·m;
  linkage joints use the linkage-weighted sum). The datasheet no-load power bounds viscous damping at X8-150 ≤ 0.26,
  X6-60 ≤ 0.04, X4-36 ≤ 0.06, X4-10 ≈ 0 N·m·s/rad.
- **Simulation meanwhile:** damping at half of each bound, randomized from 0 to the bound; friction randomized ±30 %.
- **Needed back:** a bench test per motor type (no robot needed): breakaway torque, and constant-speed runs in both
  directions to fit Coulomb + viscous friction (`ros2 run wisco_ethercat_ros2_hw arm friction` runs this on the arm).

## Drive mode, gains and delay
- **Why it matters:** the walking policy is trained assuming joint torque = kp (target − angle) − kd · speed with our
  kp and kd. Published sim-to-real work shows the delay matters (7.5 ms measured in ETH's PACE, 0–9 ms by bus read order
  on Menlo's Asimov) and that mismatched drive gains make a policy vibrate on hardware.
- **Decided:** MyActuator's EtherCAT protocol V2.0 mode 5 "PVT/MIT" computes τ = Kp (p_des − p) + Kd (v_des − v) + τ_ff
  in the drive, with Kp and Kd sent every cycle (0x2000 / 0x2001, integer × 1000). Legs, waist and head use mode 5 with
  the trained kp and kd; arms use position mode with a gravity torque offset, or torque mode, from the arm controller.
  A stock position mode (CSP with the drive's own cascaded loops) would not match the simulation.
- **Gains:** in `model/robot_numbers.yaml` (`control.gains`; table in `README.md`): hips and knee kp 120, hip yaw 50,
  ankles 35 (joint space), waist 100; kd at damping ratio 0.5 on the home-pose joint inertia.
- **Simulation meanwhile:** kp and kd randomized ±10 %, command delay 0–10 ms per episode. EtherCAT cycles are typically
  1 ms and deterministic, so the real delay is probably at the low end.
- **Consequences seen in simulation:** with these gains the robot cannot stand with zero policy action (passive standing
  needs about 150 N·m/rad per ankle), so the policy balances, as on the G1. Ankle friction through the linkage (2.9 N·m)
  against ankle kp 35 leaves a dead zone of about 5°; a lower measured friction or a higher ankle kp would shrink it.
- **Needed back (bench):** the drive's Kp and Kd units (output-side N·m/rad?) and the rate it runs the PD; the EtherCAT
  cycle rate; the measured command-to-motion delay (`arm latency` on the arm); ankle friction on the assembled ankle.

## Foot soles
- **Found:** each foot touches the ground through one flat box (210 × 85 mm) with friction 0.6, randomized 0.3–1.2 in
  training. The model gives the sole density 1.2 g/cm³ (rubber or plastic) and nothing about its shape.
- **Why it matters:** walking pushes through the feet; published work reports policies losing balance recovery when
  the foot contact geometry was slightly off.
- **Needed back:** sole material, thickness and drawing (flat, or with treads or edges?). Later: a slip test on the floor
  the robot will walk on.

## Supply voltage and heating
- **Found:** the datasheets rate the X4-10 and X4-36 at 24 V and the X6-60 and X8-150 at 48 V, and the simulation's
  speed limits use those voltages. MyActuator notes that above the rated voltage the speed must not exceed the rated
  speed. Real motors heat up: rated torque is continuous, peak torque lasts seconds.
- **Simulation meanwhile:** only the ankle motors are heat-limited (sustained-torque penalty, see
  [Ankle torque](#ankle-torque)); the ankle X4-36s are the motors most likely to overheat. Other joints are not measured
  yet (`tools/current_estimate.py` estimates current draw per motor).
- **Needed back (electrical):** the supply voltage to each motor group (one 48 V bus, or 24 V for the X4s?).
