# Hardware issues and open questions (for the mechanical team)

Model under review: `wiscohumanoids/robot/` (V2, built 2026-11-09 from "Legs V2 Full Assembly.step" + "FullAssmb.step").
Each item says what we found, how we found it, what we did in simulation meanwhile, and what we need back.
Status: OPEN (needs mech input), WORKAROUND (sim handles it for now), CLOSED.

**Does any of this block training in simulation? No.** Each item either has a sim workaround or is covered
by randomizing the uncertain value during training (the standard approach; the G1 model has similar
approximations). They matter before a policy runs on the **real robot**, so the sooner the measurements
come back, the less retraining later. "Needed before" says when each item must be resolved.

| ID | Area | Summary | Status | Needed before | Who can close it |
|---|---|---|---|---|---|
| H1 | Mass | 53 % of the mass (31 motors) is datasheet-exact; the rest is CAD volume x density; no wiring/battery | OPEN | Real robot (high priority: easy to measure, big effect) | Mech: CAD mass properties now; weigh the robot later |
| H2 | Ankle | Part of the ankle range of motion is physically unreachable | WORKAROUND | Design review only | Mech: confirm the real limit |
| H3 | Ankle | Shared ankle torque (modelled per motor now); measured walking demand: 65 % of time above X4-36 rated, 2.7 % above peak: motors look undersized | OPEN | Real robot (motor sizing) | Mech: ankle motor sizing/cooling once real mass (H1) is known |
| H4 | Ankle | Ankle motor zero positions differ a lot between left and right | OPEN | Real robot (calibration) | Mech: calibration procedure |
| H5 | Arms | With arms straight down, the grippers overlap the hips | WORKAROUND | Real robot (rest pose) | Mech: confirm |
| H6 | Motors | Rotor inertia now from MyActuator datasheets, except the X8-150 (its printed value is physically impossible) | MOSTLY CLOSED (X8-150 open) | Real robot (sets the gains) | MyActuator: X8-150 rotor inertia; mech: installed variants |
| H7 | Sensors | IMU orientation now from the CAD (EVAL_HI05 board); only a board-axis check remains | MOSTLY CLOSED | Real robot | Mech: confirm printed board axes = CAD part axes |
| H8 | Sensors | Head camera is a ZED X (confirmed); mount tilt to confirm | OPEN (minor) | Phase 4 perception | Mech: mount tilt |
| H9 | Arms | Shoulder roll can't reach its inward limit: the arm hits the torso | OPEN (minor) | Manipulation phase | Mech: confirm intended ROM |
| H10 | Motors | Friction = datasheet backdrive torque; viscous damping bounded from datasheet (old values were 4-25x too high) | MOSTLY CLOSED (bench test to confirm) | Real robot | Bench test: breakaway torque + constant-speed sweeps |
| H11 | Motors | Driver control mode, PD gains and command delay not decided / measured | OPEN | Real robot (high priority: decide before bench tests) | Controls/electrical: EtherCAT mode, gains, cycle rate |
| H12 | Feet | Sole contact is a flat box with assumed friction 0.6; real sole material/shape unknown | OPEN | Real robot | Mech: sole material and drawing |
| H13 | Motors | Supply voltage per motor and thermal (continuous torque) limits not modelled | OPEN | Real robot | Electrical: bus voltage(s); sim: add thermal penalty later |

---

## H1. Masses are estimates
- **Found:** `robot/README.txt` says every non-motor part's mass is volume x assumed density (aluminium 2.70, steel 7.85, plastic 1.25 g/cm3, ...). No wiring, connectors or battery are listed. Total 40.6 kg.
- **Why it matters:** mass and where it sits decides how much torque every joint needs. If the real robot is heavier, a policy trained on 40.6 kg may not hold it up.
- **Sim workaround:** training randomizes mass (see WORK_LOG).
- **Update 2026-11-10:** per `robot/README.txt`, the V2 model gives each motor its datasheet mass (X8-150 1.33, X6-60 0.82, X4-36 0.36, X4-10 0.33 kg); those numbers match the official sheets. 31 motors = 21.45 kg, 53 % of the 40.55 kg total. The other 19.1 kg is CAD volume x assumed density. Training randomizes every link's mass x0.85-1.20.
- **Now (no hardware needed):** export mass properties from the CAD with the real materials assigned (SolidWorks/Onshape "mass properties" per part or subassembly), plus spec-sheet masses for purchased parts (bearings, boards, battery, harness). Each number goes into `training/model/robot_numbers.yaml` (masses.overrides) with its source; rebuild and retrain.
- **Need back:** the CAD mass export above; is a battery on board, and where? Later: weigh the robot, ideally per subassembly (each leg below the hip, torso + arms, head).

## H2. One ankle pose is unreachable
- **Found:** solving the rod-and-crank geometry in `robot/linkage.json` across the joint range. At full toe-down (ankle pitch -0.61 rad) combined with full roll -0.26 rad, one rod would have to stretch about 1 mm (0.9 mm right, 1.1 mm left). `linkage.py` returns nonsense there (one crank angle came out as 10^12 rad).
- **Measured 2026-11-10** (`training/model/linkage_tools.py`, full scan of roll at each pitch): the full roll range (+/-15 deg) is only reachable down to ankle pitch **-0.566 rad (-32.4 deg) on the left and -0.576 rad (-33.0 deg) on the right**, not the -0.611 rad (-35 deg) in the range-of-motion sheet. Below that, some roll angles can't be reached. Inside the reachable range the linkage is well behaved (worst Jacobian condition number 4.8, no near-singular poses).
- **Sim workaround:** training uses ankle pitch lower limit **-0.546 rad (-31.3 deg)** (reachable limit + 0.02 rad margin).
- **Need back:** is the published ankle range of motion (pitch -35 to +45 deg, roll +/-15 deg) meant to hold at every combination? If yes, check rod length / ball joint positions; if no, what is the real combined limit (hard stop positions)?

## H3. Shared ankle torque
- **Found:** the two X4-36 motors per ankle drive pitch and roll together through the rods (pitch leverage about 1.27x, roll about 0.52x at zero pose). Max ankle pitch torque is about 86.6 N*m only when the roll torque is zero, and the reverse.
- **Why it matters:** a plain joint model lets the policy use full pitch and full roll at once, which the real ankle can't do. Rated (continuous) ankle pitch torque with both motors is about 26.7 N*m.
- **Sim workaround:** training penalizes any step where either ankle motor would need more than its peak torque (computed through the linkage).
- **Measured in simulation (2026-11-10, J3d walking policy, 256 robots x 20 s, full curriculum difficulty, random walking commands):**
  with only joint-level torque caps, at least one ankle motor was above its **10.5 N*m rated torque in 65 % of walking time**
  and above its **34 N*m peak in 2.7 %** (p95 22 N*m, p99.9 42 N*m, max 77 N*m). The simulation now clamps each motor to its
  peak and torque-speed line through the linkage (as the drivers would) and penalizes sustained torque above rated; with the
  clamp, the same policy still has some ankle motor above rated in 67 % of steps (22.7 % of motor-steps, RMS 10.1 N*m,
  sustained 2 s RMS above rated 34 % of motor-time).
- **Two X4-36 per ankle look undersized for this gait and mass.** Rated (continuous) ankle pitch torque with both motors is
  ~26.7 N*m at the joint; walking needs more for a large part of the time. Whether the real ankle overheats depends on the
  real mass (H1, likely heavier than 40.6 kg) and cooling.
- **Need back:** once the real mass is known, check ankle motor sizing (e.g. X6-class motors or a higher linkage ratio) and
  cooling/thermal limits; the datasheet stall table (X4-36: 34.5 N*m for 3 s raises it 50 C) is the relevant limit.

## H4. Ankle motor zero offsets differ left vs right
- **Found:** at joint zero (foot flat), the crank angles relative to their CAD position are about 0.04 / 0.02 rad on the left ankle but 0.39 / -0.40 rad on the right. The cranks were drawn in a different pose on each side of the CAD assembly.
- **Why it matters:** only for calibrating the real robot: each motor's encoder zero must be mapped to the foot-flat pose.
- **Need back:** a calibration procedure or jig for setting ankle (and waist) motor zeros, and confirmation that the CAD crank positions match the real assembly.

## H5. Grippers overlap the hips with arms straight down
- **Found:** MuJoCo collision check at the all-zero pose: each gripper overlaps the hip pitch and hip roll links by 4-5 cm, and the forearm/wrist touch the hip pitch link.
- **Sim workaround:** the default training pose keeps the shoulders rolled out slightly and elbows bent (exact values in WORK_LOG), so the hands clear the hips.
- **Need back:** confirm this is just geometry at the zero pose (arms can't actually hang straight). If the real robot's arms rest at zero, there's a real interference.

## H6. Rotor inertia (armature)
- **Update 2026-11-10:** read the official MyActuator parameter sheets (saved in `training/model/motor_datasheets/`, values in `training/model/robot_numbers.yaml`). X4-10 0.25, X4-36 0.3, X6-60 0.66 kg*cm2 ("Moment of Inertia"); armature = that x gear ratio squared, which matches what the model used. **CLOSED for these three** (pending bench confirmation).
- **X8-150 still open:** its sheet prints "Moment of Inertia 0.001023 kg*m2" (= 10.23 kg*cm2, a different unit from the other sheets). That is physically impossible for its rotor: a solid cylinder made of the entire 1.33 kg actuator at its full 96 mm diameter would only be 15.3 kg*cm2, and the rotor is a fraction of that mass. Most likely a 10x typo for 1.023. Other X8-family listings: X8 0.805, X8 Pro 1.45, X8-120 1.5 kg*cm2. Sim uses **1.2 kg*cm2** (armature 0.046 kg*m2) and randomizes 0.8-2.0 in training.
- **Why it matters:** armature sets how fast a joint can accelerate and, in the training recipe we copied, it also sets the joint gains directly.
- **Need back:** from MyActuator: the X8-150 rotor inertia and whether it is referenced to the motor shaft (likely a unit/typo error on the sheet). From mech: which variants are installed. The X4-10 and X4-36 non-"L" versions are listed as discontinued; the current ones are X4-10-L and X4-36-L, whose parameters may differ.

## H7. IMU orientation
- **Found:** the IMU frame is at the CAD part location with axes assumed aligned to the pelvis.
- **Update 2026-11-10 (from the Legs V2 CAD):** EVAL_HI05 board, 34 x 24 x 3 mm, on the "IMU Mount" at the back of the pelvis, upright, chip facing backward, at (-0.0518, -0.0003, 0.0345) m. Board x -> robot right (-y), y -> down (-z), z -> forward (+x); URDF rpy (-pi/2, 0, -pi/2), quaternion wxyz (0.5, -0.5, 0.5, -0.5). Now in robot/robot.urdf (imu_link) and robot/robot.xml (imu site); both checked to agree. Training reads the IMU in pelvis axes (a separate "imu_policy" site at the same point); on the robot the IMU driver applies this fixed board-to-pelvis rotation.
- **Need back:** confirm the axes printed on the board match the CAD part axes.

## H8. No lidar; head camera mount
- **Found:** the robot has no lidar. The sim stack will use its ZED X stereo setup (`sensors:=stereo`) mounted at the `head_camera` frame (on the head, 8.5 cm forward of the head pivot).
- **Update 2026-11-10:** camera model confirmed as **ZED X** (matches the stack's `sensors:=stereo` setup). Mount position comes from the CAD `head_camera` frame.
- **Need back:** only the mount tilt (the model assumes the camera looks straight ahead, level with the head).

## H9. Shoulder roll inward limit unreachable
- **Found (Phase A range sweep, 2026-11-10):** driving each shoulder roll toward its inward limit (-0.436 rad, -25 deg per the range-of-motion sheet) with the arm hanging and elbow slightly bent, the elbow / upper arm hits the torso at about -0.17 rad (-10 deg) on the left and -0.23 rad (-13 deg) on the right. The left/right difference comes from the torso collision shape.
- **Why it matters:** low for walking. Matters for manipulation later (reaching across the body).
- **Need back:** is -25 deg inward roll only meant with the arm raised forward? If the CAD says it should clear the torso hanging down, the collision shapes or the torso geometry need a look.

## H10. Joint friction and damping
- **Found:** the V2 model's joint friction (0.34-2.9 N*m) was "carried over from the old URDF". On the wrists (X4-10, 0.8 N*m friction vs 4 N*m rated) it creates a visible dead zone in simulation: a 0.2 rad wrist command stops 7-18 % short.
- **Update 2026-11-10:** those friction values turn out to be exactly the datasheet **backdrive torques** (X8-150 0.34, X6-60 1.6, X4-36 1.14, X4-10 0.8 N*m; ankle/waist = linkage-weighted sums), so they are datasheet-based, not guesses. The **viscous damping** values (0.2-1.6) were guesses and too high: the datasheet's no-load power bounds viscous loss at X8-150 <= 0.26, X6-60 <= 0.04, X4-36 <= 0.06, X4-10 ~0 N*m*s/rad (4-25x lower). Sim now uses half of each bound and randomizes 0 to the bound; friction is randomized +/-30 %.
- **Need back:** a bench test per motor type to confirm (no robot needed): breakaway torque, and constant-speed runs in both directions to fit Coulomb + viscous friction.

## H11. Command delay and driver gains
- **Found:** the sim had no delay between a policy command and the motor acting. Published measurements: 7.5 ms (PACE, ETH), 0-9 ms by bus read order (Menlo's Asimov). The sim now randomizes 0-10 ms per episode (estimate).
- **Why it matters:** delay changes how a policy must react; gains matter even more: Menlo's Asimov policy vibrated on hardware because the sim's PD gains didn't match the real drivers.
- **Update 2026-11-10:** the stack is EtherCAT + ROS 2. EtherCAT cycles are typically 1 ms and deterministic, so the real delay is probably at the low end of our 0-10 ms range; measure it on the bench.
- **The key decision is the driver control mode**, because the policy is trained assuming joint torque = kp x (target - angle) - kd x speed with our kp/kd:
  - an impedance/"MIT-style" mode (driver takes target position + kp + kd) or a torque mode with that PD on the computer at >= 1 kHz: matches the sim. Program the same kp/kd as `training/model/robot_params.py`.
  - a stock position mode (CSP) with MyActuator's own cascaded position/velocity/current loops: does NOT match the sim; the sim would then need a model of those loops.
- **Decided 2026-11-10:** MyActuator's EtherCAT protocol V2.0 has mode 5 "PVT/MIT": tau = Kp (p_des - p) + Kd (v_des - v) + t_ff, with Kp/Kd sent every cycle (0x2000/0x2001, integer x1000). **Legs, waist and head use mode 5 with our kp/kd; arms use mode 10 (CST, torque from the ROS controls stack).** Leg gains: RP1-style moderate values (hips/knee ~120, hip yaw ~40-60, ankles ~30-40, waist ~100), replacing the stiffer formula gains. Recorded in robot/CHANGES.md.
- **Update 2026-11-10 (gains set):** values now in `training/model/robot_numbers.yaml` (control.gains), table in robot/CHANGES.md: hips/knee kp 120, hip yaw 50, ankles 35 (joint space), waist 100; kd at damping ratio 0.5 on the home-pose joint inertia. Training randomizes kp/kd +/-10 % until the driver's implementation is bench-checked.
- **Consequences seen in sim (Phase A):** with these moderate gains the robot can no longer stand with zero policy action (passive standing needs ~150 N*m/rad per ankle); the policy balances, as on the G1. Ankle friction (2.9 N*m through the linkage) vs ankle kp 35 leaves a ~5 deg dead zone; a measured, lower ankle friction or a higher ankle kp would shrink it.
- **Need back (bench):** confirm the driver's Kp/Kd units (output-side N*m/rad?) and the rate it runs the PD; EtherCAT cycle rate; measured command-to-motion delay; ankle linkage friction on the assembled ankle.

## H12. Foot sole contact
- **Found:** each foot touches the ground through one flat box (210 x 85 mm) with friction 0.6 (copied from the G1 recipe; randomized 0.3-1.2 in training). The source model says sole density 1.2 g/cm3 (a rubber/plastic sole), nothing about its shape or material.
- **Why it matters:** walking pushes through the feet; Menlo reported their policy stopped learning balance recovery when the foot/toe contact geometry was slightly off.
- **Need back:** sole material, thickness and drawing (flat, or with treads/edges?). Later: a slip test on the floor the robot will walk on.

## H13. Supply voltage and motor heating
- **Found:** the datasheets rate the X4-10 and X4-36 at 24 V and the X6-60/X8-150 at 48 V. The sim's speed limits use those rated voltages. MyActuator notes: above the rated voltage the speed "must not exceed the motor's rated speed". The sim also lets motors run at peak torque indefinitely; real motors heat up (rated = continuous, peak = seconds; e.g. X4-36 34.5 N*m for 3 s raises it 50 C).
- **Need back (electrical):** the supply voltage to each motor group (one 48 V bus for all, or 24 V for the X4s?).
- **Update 2026-11-10:** sim now penalizes sustained ankle motor torque above rated (2 s moving RMS per motor) and clamps each
  ankle motor to its peak + torque-speed line. Measured ankle demand: see H3 (65 % of walking time above rated, 2.7 % above
  peak before the clamp). The ankle X4-36s are the motors most likely to overheat; other joints not yet measured.
