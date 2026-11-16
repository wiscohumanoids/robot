# Work and observation log: Wisco V1 walking policy

Newest entries at the bottom. Each entry: what was done, decisions (and why), issues hit.
Hardware problems found along the way also go in `HARDWARE_ISSUES.md` (for the mech team).
Training runs (Phase C) will get one line each in the run table at the end.

---

## 2026-11-09: starting point
- Source model: `robots/wisco_v1/` (V2, from "Legs V2 Full Assembly.step" + "FullAssmb.step"). Treated as read-only;
  everything generated lives under `wiscohumanoids/training/`.
- Earlier review of V1 (before this log): loads cleanly, exact left/right symmetry, valid inertias, ankles at the real
  pivots with a rod linkage model, stands under simple PD at 16 % of rated knee torque. Open items carried into
  HARDWARE_ISSUES H1-H8.
- Recipe: copy mjlab 1.6.0's Unitree G1 velocity task ("Mjlab-Velocity-Flat-Unitree-G1", the one we trained on the
  GPU VM), swap in our robot. mjlab source read from the 1.6.0 package (downloaded to a scratch folder; mjlab's G1
  files: `asset_zoo/robots/unitree_g1/g1_constants.py`, `tasks/velocity/config/g1/env_cfgs.py`, `rl_cfg.py`).

### Decisions
| Decision | Why |
|---|---|
| Policy drives 29 joints: legs 12, waist 3, arms 14 (incl. wrists); head 2 welded | Same as the G1 setup (it drove all 29 of its joints; it has no head) |
| Sensing in the ROS stack: ZED X stereo setup (`sensors:=stereo`) at the `head_camera` frame | Robot has no lidar ("vedx" = ZED X) |
| Gains from mjlab's G1 formula: kp = armature*(2pi*10)^2, kd = 4*armature*2pi*10, action scale = 0.25*effort/kp | Proven recipe; armature from the V1 model_report.json |
| Default pose knees bent: hip pitch 0.30, knee 0.60, ankle pitch 0.30, shoulder roll 0.20, elbow 0.50 | Same posture as G1's KNEES_BENT pose (hip -0.312 / knee 0.669 / ankle -0.363 about +y = ours about -y). Straight knees sit at a singularity next to the knee limit. Arm values keep the grippers off the hips (H5) |
| Keep the V1 model's joint friction and passive damping | G1 has none, but ours are the model's best estimate of real motors. Revisit in Phase C (see A4 wrists) |
| Linkage limits: ankle pitch lower -0.546 rad in training (source -0.611) | Full roll range unreachable below -0.566/-0.576 (H2) |
| Ankle linkage penalty: per-motor torque via precomputed Jacobian grid | Real ankle can't do full pitch + roll torque together (H3). Running linkage.py's solver per step for thousands of robots would be far too slow on the GPU |

### Issues hit
- mjlab can't be installed on this Mac (needs Python < 3.14; only 3.14 here) and the GPU VM is off. So Phase A uses a
  plain-MuJoCo copy of mjlab's actuator/collision/solver settings (`robots/wisco_v1/checks/sim_replica.py`), and Phase B's first VM
  run is the real mjlab integration test.
- `robots/wisco_v1/linkage.py` diverges near the edge of the ankle range (crank angle 10^12 rad). Wrote `model/linkage_tools.py`
  (scan + bisection). One-time run takes 6 min.
- scipy isn't in the project venv: used plain bisection instead of installing anything.
- MjSpec resolves `meshdir` against the source file while compiling; fixed by rewriting the saved XML's meshdir to
  `../meshes/` so the training model works from its own folder (and on the VM).

## 2026-11-10: Phase A (pre-training checks): no blockers
Full report: `robots/wisco_v1/checks/results/REPORT.md`. Summary: A1, A2, A3, A5, A6, A7, A8 pass; A4 fails on 5 non-critical joints.

### Observations
- **Stiffer than the G1.** The formula gives ankle pitch kp 499 (G1 28.5) and X4-36 arm joints kp 154 with action scale
  0.055 rad (G1 arms: kp 14, scale 0.44). Cause: high gear ratios, so large rotor inertia at the joint. All stable.
  First suspect if Phase C shows arms or ankles that barely move.
- **Stands with zero policy action** for 10 s (tilt 6.4 deg, knees sag 0.087 rad). The G1 run through the same test
  (`g1_baseline.py`, mjlab's own G1 gains) falls over. So "stand unaided" is stricter than needed.
- **Torso is heavy for the waist gains:** hung in the air, a 0.2 rad torso roll command ends at 0.35 rad.
- **Wrists have a friction dead zone** (0.8 N*m vs kp 15.8 -> 0.05 rad). Not important for walking.
- **Falls:** the G1's tilt > 70 deg rule caught every fall (limp, pushes, tilted start) in 0.6-1.4 s. No extra rule needed.
- **Sizing:** max 21 contacts / 113 constraint rows, well inside mjlab's G1 settings (njmax 300).

### Decisions
- Changed A3's pass rule from "tilt < 5 deg" to "doesn't fall (tilt < 20 deg) and settles": the 5 deg number was my own
  guess, and the G1 fails the test outright, so it isn't a fair bar.
- No gain changes before training: start from the unmodified recipe so Phase C changes are measured against it.

### Issues hit
- A 2 mm pelvis / hip-pitch-link contact showed up in every hung-in-the-air test. Cause: welding the pelvis to the
  world turns off MuJoCo's parent-child contact filter. Free-standing robot: no self-contacts. Test artifact only.

### New hardware items
- H2 updated with the measured reachable ankle limits; H9 (shoulder roll inward blocked by torso); H10 (joint friction
  unverified, wrist dead zone).

## 2026-11-10: Phase B (mjlab task written; VM steps pending)
Task `Wisco-V1-Velocity-Flat` in `training/locomotion/wisco_v1/wisco_mjlab/`, a copy of the G1 flat velocity task with these
differences (all marked `# WISCO:` in `env_cfg.py`):
- robot: `robot_cfg.py` (one position actuator per joint, gains from `robot_params.py`; knees-bent home pose,
  pelvis 0.688 m; G1-style collisions: feet condim 3 / friction 0.6, self-collision on)
- names: foot sites `left_foot`/`right_foot`, foot geoms `*_foot_collision`, torso body `torso`, pose-reward
  tolerances re-keyed to our joint names (forearm_yaw treated like a wrist joint: 0.3)
- **added:** per-part mass + inertia randomization at startup, x0.85 to x1.20 (`dr.pseudo_inertia`; skewed up
  because CAD masses leave out wiring, H1)
- **added:** ankle motor overload penalty, weight -1.0: sum over the 4 ankle motors of
  relu(|motor torque| / 34 N*m - 1)^2 (H3), plus a metric `ankle_motor_peak_nm` (max per episode) to watch
- PPO settings identical to the G1's; experiment name `wisco_v1_velocity`
- launchers `train.py` / `play.py` register the task, then run mjlab's own unchanged train / play

### Checked on the Mac (mjlab can't run here)
- All files compile. Pose-reward patterns match each of the 29 joints exactly once (walking and running).
- Ankle penalty maths (bilinear grid + closed-form 2x2 solve, written out in numpy) equals `ankle_motor.py`
  to 1e-14 N*m, and is within 1.5 % of `robots/wisco_v1/linkage.py`'s exact solver at 80 random poses.
- mjlab API names used by `vm/check_env.py` confirmed against the mjlab 1.6.0 source.

### Decisions
- Motor strength randomization: not in the first run (G1 recipe doesn't use it); add in Phase D.
- Mass randomization included from the first run: masses are known to be uncertain (H1), unlike the G1's.

### Waiting on
- GPU VM is off. Next on the VM: sync, GPU status (leave pi0.5 alone), `check_env.py`, `smoke_test.sh`.

## 2026-11-10: hardware issues triaged
- Camera confirmed as ZED X (H8 reduced to mount tilt only).
- Added "needed before" and "who can close it" to HARDWARE_ISSUES. None block sim training. High priority for
  the real robot: H1 (weigh it), H6 (motor rotor inertia), H7 (IMU orientation).

## 2026-11-10: motor values from MyActuator datasheets; DC motor model; delay; updated URDF
Goal: make the sim as accurate as possible without hardware; every number traceable to a source.

### What was done
- Read the official MyActuator parameter sheets for X4-10, X4-36, X6-60, X8-150 (myactuator.com detail pages;
  tables are images, saved in `model/motor_datasheets/`). All values with sources: `model/motor_specs.json`.
- `model/robot_params.py` now derives every joint value from `motor_specs.json` (linkage joints through their
  Jacobians at the zero pose). `model/mass_overrides.json` added (empty) for measured masses later.
- Finding: the model's peak torques, speeds, motor masses, 3 of 4 rotor inertias and all joint friction
  values were already datasheet values (friction = "backdrive torque"). Earlier "estimate" labels were too harsh.
- Real changes to the model:
  | Item | Before | Now | Source |
  |---|---|---|---|
  | Viscous damping | 0.2-1.6 (guess) | X8 0.13, X6 0.02, X4-36 0.031, X4-10 0 (half of bound) | bound from datasheet no-load power |
  | X8-150 armature | 0.050 (guess) | 0.058 (1.5 kg*cm2 rotor) | family scaling + X8-120 sibling; printed value impossible (H6) |
  | Torque-speed limit | none | max torque = min(peak, slope*(w_noload - w)), slope = Kt*Ke*N/R | datasheet motor constants; X8 slope 42.5 vs X8-120 dyno 38.1 |
  | Command delay | 0 | 0-10 ms random per episode | PACE 7.5 ms, Menlo 0-9 ms (estimate, H11) |
- mjlab task (`robot_cfg.py`, `env_cfg.py`): actuators are now `DcMotorActuatorCfg` (torque-speed curve +
  delay); startup randomization of armature (X8 x0.67-1.67, others +/-10 %), friction +/-30 %, damping 0-bound.
- `model/export_urdf.py` -> `model/wisco_v1_updated.urdf`: source URDF + datasheet effort/velocity/friction/damping,
  training ankle limit, mass overrides, and `<transmission>` blocks (gear ratios). For the ROS side; training uses MJCF.

### Phase A rerun (`robots/wisco_v1/checks/results/README.md`)
- Position actuator, DC motor, DC motor + 10 ms delay: all pass the same checks as before; nothing unstable.
- Stand tilt 6.4 -> 4.1 deg; knee sag 0.087 -> 0.062 rad.
- Torso roll now settles in 0.74 s with 9 % overshoot (was 0.41 s): the old guessed waist damping (0.85) hid how
  lightly damped the waist is. Watch in Phase C.

### Decisions
- Use mjlab's `DcMotorActuatorCfg` (PD torque + linear torque-speed clip), not MuJoCo's native `<dcmotor>`:
  the native one's PID works in volts, so the G1 gain recipe would not carry over. Phase A showed the explicit
  PD is stable with our gains at 5 ms.
- X8-150 rotor inertia 1.5 kg*cm2 with a wide random range until MyActuator answers.
- robots/wisco_v1/robot.urdf stays untouched (mech team's file); updated copy is generated.

### Issues hit
- MyActuator spec tables are images; read them as images. The X8-150 page's characteristic curve is labelled
  X8-P20-120 (a different motor), so it was used only as a cross-check.
- MuJoCo 3.11 MjSpec joint `damping` is a 3-vector; MuJoCo's URDF importer needs `<mujoco><compiler meshdir>`
  (source URDF already had one; export rewrites it).
- zsh doesn't word-split `$var` in loops (`${=var}` needed); first rerun silently ran nothing.

### Hardware items
- H1, H6, H10 updated (mostly closed from datasheets), H11 added (delay, driver gains).

## 2026-11-10: estimates inventory
- Listed everything in the sim that is estimated or chosen (answer to "what else are we estimating"). New hardware items:
  H12 (foot sole contact), H13 (supply voltage per motor group, motor heating); H11 expanded: the driver control
  mode (impedance/torque vs stock position mode) decides whether the sim's PD model matches the robot.

## 2026-11-10: robots/wisco_v1/ folder synced for the team; IMU from CAD; control modes decided
- robots/wisco_v1/robot.urdf, robot.xml, model_report.json are now generated by `robots/wisco_v1/model/sync_robot_folder.py` from the
  untouched CAD export (copied to robots/wisco_v1/original_from_team/) + motor_specs.json. Changes listed in robots/wisco_v1/CHANGES.md. Checks:
  identical geometry to CAD (0.000 mm), mass 40.551 kg, ROS `check_urdf` parses (root pelvis), URDF and MJCF IMU frames
  agree to 1e-12, team's stand_test.py still stands. `model/export_urdf.py` and `wisco_v1_updated.urdf` removed (replaced).
- IMU orientation from the Legs V2 CAD (H7 mostly closed). Training uses a pelvis-aligned "imu_policy" site at the same point.
- X8-150 rotor inertia 1.5 -> 1.2 kg*cm2 (range 0.8-2.0) after a wider search (H6).
- Control: legs/waist/head in MyActuator EtherCAT mode 5 (MIT, our kp/kd on the driver), arms in mode 10 (CST torque from the
  ROS controls stack). Leg gains will move to RP1-style values (decision; implemented in the next step).
- Issue: URDF angles written to 6 significant digits (-1.5708) made the URDF/MJCF IMU frames differ by 4e-6; now 10 decimals.
- Phase A rerun (DC model, 10 ms delay): same passes. With X8 armature 0.046 the formula gains drop (kp 228 -> 182): knee sag
  0.11 rad, tilt 8.6 deg, hip pitch step now overshoots > 25 % (A4). Expected; these formula gains are replaced by RP1-style gains next.

## 2026-11-10: helper A, J3 prep: one numbers file, RP1-style gains, arms-free velocity recipe
- **One numbers file:** `model/robot_numbers.yaml` replaces motor_specs.json + mass_overrides.json (deleted): motors
  (value + source), joint_to_motor, delay, bus (EtherCAT; cycle and supply voltage unknown), masses.overrides, control
  (modes, gains, head gains, locomotion action joints, default pose), sensors (IMU pose from CAD, ZED X pose/tilt).
  robot_params.py / sync_robot_folder.py / build_training_model.py read it; robots/wisco_v1/ files regenerated (model_report.json
  now carries kp/kd per joint). pyyaml was already in both venvs and the VM env.
- **Gains:** RP1-style kp (hips/knee 120, hip yaw 50, ankles 35, waist 100, sim arms 20-40). kd = 2*0.5*sqrt(kp*I) with I the
  home-pose joint inertia (fixed base, incl. armature): hips 10.6-11.3, knee 5.0, ankle pitch 2.1, ankle roll 0.9, waist 7.5-11.9.
  Deviation from "about critical damping": zeta = 1 on swing-limb inertia gives hips kd 21-24 (3x RP1's 7.5) and would spend
  most of the motor torque at walking speed; zeta 0.5 lands near RP1's kd. Action scale = 0.25*peak/kp (hips/knee 0.25, ankle
  pitch 0.62, ankle roll 0.25, waist 0.20-0.30).
- **arms-free recipe** (`training/locomotion/wisco_v1/wisco_mjlab/env_cfg.py`, custom terms `arms_free_terms.py`): 15 actions (legs + waist, order of
  robot_numbers.yaml locomotion_action_joints); arms position-held (DC motor model, sim arm gains) with targets moving toward
  goals resampled every 1-3 s (30 % default pose, 70 % uniform within 60 % of the soft range, speed 0.3-2.5 rad/s); 0-1 kg added
  to each hand; tracking std 0.25 m/s / 0.35 rad/s; command subclass with 18 % pure-turn samples (16 % after standing envs;
  measured 15.6-16.2 %), yaw +-1.5 at every curriculum stage; command delay 0-10 ms; pushes 0.8 m/s; DR: armature (X8 x0.67-1.67,
  others +/-10 %), kp/kd +/-10 %, friction +/-30 %, damping 0-bound, mass x0.85-1.20; ankle penalty kept. Posture and
  joint-limit rewards act on the 15 policy joints only. Task ID unchanged: `Wisco-V1-Velocity-Flat`.
- **Checks:** Mac CPU (.venv-mjlab): env builds/steps, obs actor 85 / critic 97, no NaN; 3 PPO iterations run, checkpoints + ONNX
  export written. `vm/check_env.py` ALL PASS on the Mac (32 envs) and on the VM GPU (1024 envs, ~16.9k env steps/s while sharing
  the GPU). Phase A rerun (DC, 10 ms): see `robots/wisco_v1/checks/results/README.md`.
- **Issues:** local mjlab venv has wandb >= 0.29, which rsl-rl's W&B writer rejects -> TensorBoard logger locally; the VM env has
  wandb 0.28.2; `train_velocity.sh` uses TensorBoard by default (LOGGER=wandb to switch). check_env.py hard-coded cuda -> now falls
  back to CPU.
- **Flags:** (1) RP1-style gains: no passive standing with zero action (needs ~150 N*m/rad per ankle); the policy must balance
  (the G1 is the same). (2) Ankle pitch dead zone ~5 deg (friction 2.9 N*m vs kp 35). First knobs if the policy shows ankle
  chatter: ankle kp 60-80 or measured ankle friction.
- **Launch J3 (coordinator, VM):** `~/humanoid-sim-stack/training/locomotion/wisco_v1/vm/train_velocity.sh --log-root ~/hsim_runs/J3_wisco_v1_velocity_arms_free`
  (defaults 4096 envs, 20000 iterations, TensorBoard). Smoke: `training/locomotion/wisco_v1/vm/smoke_test.sh`.

## 2026-11-10: helper C, J4 prep: Wisco tracking (LAFAN1 dance1 retargeted to Wisco)
Files: `tracking/retarget_lafan1.py`, `tracking/wisco_tracking/` (task `Wisco-V1-Tracking-Flat`), `tracking/train.py`,
`tracking/play.py`, `tracking/train_wisco_tracking.sh`, `tracking/compare_strip.py`,
`tracking/motions/dance1_subject2.csv` (+ `.report.json`, `.debug.npz`, `preview/g1_vs_wisco_t*.png`).

### Retargeting
- Source: original Ubisoft LAFAN1 BVH `dance1_subject2` (CC BY-NC-ND 4.0), the same take as the G1 dance clip, so the
  two robots can be compared frame by frame against Unitree's own G1 retarget.
- GMR could not be used: installing/running its GitHub code was blocked by the permission system. Wrote an own
  retargeter (numpy + MuJoCo only): BVH FK -> "bone-direction" targets (human bone directions, Wisco segment lengths,
  starting from Wisco's own hip/shoulder pivots) -> per-frame damped-least-squares IK on the training model, warm-started.
- Issues hit and fixes:
  | Problem | Cause | Fix |
  |---|---|---|
  | Robot hunched forward | pelvis/chest frames from joint positions follow the human spine curve (pelvis 20.5 deg, chest 8.9 deg off upright) | calibrate both offsets at frame 0 (LAFAN1's standing T-pose) |
  | 10 s of garbage at 32-41 s (IK residual up to 2.3) | backward heel kick: IK took a "hip yaw spun ~180 deg" solution and stuck at the limits | retarget-only bounds (hip yaw +/-0.9, shoulder/forearm yaw +/-1.6 rad), stronger twist regularization, re-solve legs from home if a frame's error > 0.25 |
  | Shoulder joint jumps up to 97 rad/s | redundant arm solutions flipping near straight arms | continuity term (pull to previous frame), legs-only re-solve, 50 ms Gaussian smoothing |
- Final quality (whole 131 s clip), compared with Unitree's G1 retarget of the same take using the same measure:
  | Measure | Wisco (ours) | G1 (Unitree) |
  |---|---|---|
  | planted-foot speed p50 / p95 (ankle body, m/s) | 0.055 / 0.30 | 0.058 / 0.24 |
  | max joint speed p99 / max (rad/s) | 11.7 / 15.3 | 20.1 / 32.7 |
  | joint-limit violations | 0 (ankle pitch min -0.518 >= -0.546) | n/a |
  | self-collision frames > 1 cm | 0.7 % (max 66 mm) | n/a |
  | foot penetration max | 35 mm | n/a |
  | IK residual mean / p95 | 0.13 / 0.51 (worst in the 33 s heel kick: foot orientation at the ankle-roll limit) | n/a |
  Visual check: `motions/preview/g1_vs_wisco_t*.png` (G1 top, Wisco bottom, same moments): poses match through the clip.
  Known differences: Wisco arms a bit more bent than the G1's in relaxed stretches; foot roll is clamped at +/-15 deg.

### Tracking task
- Copy of mjlab's G1 tracking config: anchor `torso`, 14 key bodies (G1 list with `left/right_hand` for the wrist links),
  end effectors ankle_roll links + hands, all 29 joints are actions (ACTION_SCALE from robot_cfg).
- Added the same Wisco realism as the velocity task: mass/inertia x0.85-1.20, armature/friction/damping randomization,
  ankle-motor overload penalty (DC motors + 0-10 ms delay come with robot_cfg).
- Local check (Mac, CPU, 4 envs, 2 PPO iterations): builds, trains, exports ONNX (inputs obs[160] + time_step).
  All 8 startup randomization events active.

## 2026-11-10: J3 fix: difficulty curriculum (coordinator: run 1 learned to stand still)
- **Diagnosis (coordinator, model_750):** episode length ~1000 but per-step velocity error ~0.63 m/s, yaw error ~1 rad/s; reward
  mostly upright + pose. Causes: tracking std 0.25 / 0.35 from iteration 0 (0.5 m/s off -> ~2 % reward instead of 37 %), and every
  disturbance at full strength from iteration 0.
- **Fix (same design as the G1 arms-free task, helper B):** `arms_free_terms.py` difficulty curriculum. Level L in [0, 1], gated on finished
  episodes (evaluated every >= 240 env steps with >= 64 finished episodes): +0.1 if mean episode length > 0.8 x max AND linear
  tracking reward rate > 50 % of its weight; -0.1 if mean length < 0.4 x max. L scales: arm target range 0 -> 60 % of soft range,
  hand payload max 0 -> 1 kg (resampled for all envs every 10-20 s), push 0.5 -> 0.8 m/s. After L = 1, level L2 (same gate)
  tightens tracking std from the G1 defaults 0.5 / 0.707 to 0.25 / 0.35 (lowering takes L2 down first). Mass + motor DR stay on
  from the start. Logged: Curriculum/level, Curriculum/level_info/{level2, lin_std, ang_std, ep_len_frac, lin_track_frac}.
  Play mode holds L = L2 = 1 (fixed_level term).
- **Tests (Mac CPU):** gating unit test (3 good evals -> L 0.3; good length + low tracking -> hold; short episodes -> L 0.2;
  forced L = 1 -> push 0.8, arm goals up to 1.71 rad, hand mass up to 1.39 kg; 10 more good evals -> L2 = 1, stds 0.25 / 0.35),
  play env at L = L2 = 1, 3 PPO iterations with the curriculum tags in the log, check_env.py ALL PASS (check 7 now forces L = 1).
- **Pose reward vs the G1 recipe:** same weight (1.0), same std_standing (0.05) and the same walking/running stds per joint type;
  only difference: averaged over our 15 policy joints instead of the G1's 29 (arms aren't policy joints). Not more attractive.
- **Other things in the Wisco task that favour standing over the G1 task (not changed, watch them):**
  (1) ankle_motor_overload (weight -1, zero below 34 N*m per motor): stepping raises ankle torque, standing never triggers it;
  watch Episode_Reward/ankle_motor_overload. (2) self_collisions (-1): at high L the randomly moving arms can hit the thighs when
  the legs swing; the G1's policy-controlled arms avoid that. (3) ~5 deg ankle pitch dead zone (friction 2.9 N*m vs kp 35) makes
  precise foot placement harder. (4) No passive standing with RP1-style gains (the G1 is the same).
- **Launch (fresh run):** `~/humanoid-sim-stack/training/locomotion/wisco_v1/vm/train_velocity.sh --log-root ~/hsim_runs/J3b_wisco_v1_vel`

## 2026-11-10: J3 fix 2 (coordinator): non-arm self-collision, arm targets off the body, stricter gates, angmom weight, J3c resume
- **Self-collision (G1 run hit -0.70/s at full difficulty from scripted arm contacts):** mjlab contact sensors take one
  secondary element (`secondary_policy="first"` silently used only the pelvis), so the penalty now sums mjlab's
  self_collision_cost over 5 sensors with no arm links: left leg vs right-leg subtree, both legs vs pelvis, torso_yaw_link,
  torso_roll_link, torso. Test (Mac CPU, real env, forced poses, 3 steps): arms driven into torso: old sensor 5, new 0;
  hands driven into thighs: old 4, new 0; left leg crossing right leg: old 7, new 7.
- **Arm targets off the body:** per-joint (toward-lower, toward-upper) fractions of the soft range around default, x level
  (`robot_cfg.ARM_TARGET_FRACS`): shoulder pitch 0.25/0.6, shoulder roll 0/0.6 (outward only, H9), shoulder yaw 0.25/0.1,
  elbow 0.3/0.6, forearm yaw and wrists 0.3/0.3. MuJoCo sweep, 3000 random arm poses with legs at home and walking-like:
  old rule (60 % every joint) 5.3 % touch the body -> new 0.0 %. Shoulder yaw toward its upper limit alone touched 23 %.
- **Gates:** raise L (then L2) by 0.05 only if episode length > 0.8 x max AND linear tracking > 50 % AND angular tracking
  > 40 % of their weights, at most once per 50 iterations; lower by 0.1 if length < 0.4 x max. Unit-tested (incl. J3b-like
  lin 0.46 -> no raise). New log tag Curriculum/level_info/ang_track_frac.
- **Why forward tracking stalls at ~46 % (J3b iteration 957 vs G1 J1b iteration 923, per-second episode rewards):**
  angular_momentum -0.336 vs -0.080 (4.2x, while Wisco moves less); pose 0.80 vs 0.54 (standing near default pays);
  action_rate -0.71 vs -1.21; foot_clearance/swing -0.07/-0.04 vs -0.19/-0.06; ankle_motor_overload -0.001 (negligible);
  self_collisions -0.008 vs -0.65. The dominant stepping cost unique to Wisco is angular momentum: whole-body (incl. the
  scripted arms) and larger for a heavier, taller robot. **Applied:** angular_momentum weight -0.02 -> -0.005 (about the
  G1's cost). Not changed: pose (same as G1 recipe), ankle overload (negligible).
- **Eval note:** play mode held L = 1 (1 kg payload, 60 % arm range) while J3b trained at L = 0.2, so evals were harsher than
  training. Play now reads WISCO_PLAY_LEVEL / WISCO_PLAY_LEVEL2 (default 1.0); evaluate at the training level for progress and
  at 1.0 for the final policy.
- **J3c resume:** J3b model_1000.pt copied to ~/hsim_runs/J3c_wisco_v1_vel/wisco_v1_velocity/J3b_model_1000/. Verified on the
  Mac (CPU copy of the checkpoint): loads, continues at iterations 1000-1001, writes checkpoints + ONNX. Curriculum restarts at L = 0.
  check_env.py ALL PASS on the Mac and the VM GPU (1024 envs); yaw-share check now uses >= 2000 samples.
- **Launch J3c:** `~/humanoid-sim-stack/training/locomotion/wisco_v1/vm/train_velocity.sh --log-root ~/hsim_runs/J3c_wisco_v1_vel --agent.resume True --agent.load-run J3b_model_1000 --agent.load-checkpoint model_1000.pt`

## 2026-11-10: J3 fix 3 (coordinator): policy-body angular momentum, start level, J3d resume
- **Problem (coordinator, J3c):** as Curriculum/level rose 0.07 -> 0.35, Episode_Reward/angular_momentum went -0.02 -> -1.10/s
  even at weight -0.005; tracking and mean reward collapsed (72 -> 13).
- **Root cause found (GPU only):** the whole-body subtreeangmom sensor (`robot/root_angmom`) jumps ~10x right after the hand-payload
  mass change and stays there, while the robot's motion is unchanged. J3c model_1300 rollout at L = 0.25, 1024 envs on the VM GPU,
  payload event forced at t = 100: whole-body |L|^2 2.5-2.7 before -> 22-30 after; policy-body |L|^2 1.8-1.9 throughout. Same
  rollout on the Mac CPU: no jump (whole body 2.9-4.2). So on the mujoco_warp GPU path the whole-body value is computed about a
  wrong centre of mass after a runtime mass change (dr.body_mass + set_const). The scripted arms alone add little (~40 % at L = 0.25).
  **Also affects the G1 arms-free task if it uses the same payload event + mjlab's angular_momentum term** (told the coordinator).
- **Fix:** `arms_free_terms.policy_body_angular_momentum_sq`: |L|^2 of the 18 policy-controlled bodies (pelvis, legs, torso links, welded
  head; not arms/hands) about their own COM, from body COM poses/velocities and model mass/inertia (no subtree data).
  Weight -0.005 kept. Validation with all bodies selected and mass DR off: equals MuJoCo's subtree_angmom exactly (0.2305,
  0.8277, 8.6036, 0.0517). Arms-only motion -> exactly 0; legs 2 rad/s -> 0.69; torso 2 rad/s -> 4.76. Old whole-body value
  kept as metric `angmom_wholebody_sq`; `Metrics/policy_angmom_mean` logged.
- Trained policy (J3c model_1300) at L = 0.25: deterministic CPU: policy-body 1.88 vs whole-body 2.66 (cost/s -0.009 vs -0.013);
  stochastic: 2.19 vs 3.24; GPU with a payload change: 1.80 vs 19.5 (-0.009 vs -0.098).
- **Start level:** WISCO_START_LEVEL / WISCO_START_LEVEL2 (default 0) set the initial curriculum levels; the first raise from there
  waits the 50-iteration dwell.
- **J3d:** J3c model_1300 copied to ~/hsim_runs/J3d_wisco_v1_vel/wisco_v1_velocity/J3c_model_1300/. Verified on the Mac (CPU copy):
  loads, continues 1300-1301, start level 0.25. check_env.py ALL PASS on the VM GPU (1024 envs, start level 0.25).
- **Launch J3d:** `WISCO_START_LEVEL=0.25 ~/humanoid-sim-stack/training/locomotion/wisco_v1/vm/train_velocity.sh --log-root ~/hsim_runs/J3d_wisco_v1_vel --agent.resume True --agent.load-run J3c_model_1300 --agent.load-checkpoint model_1300.pt`

## 2026-11-10: J3 fix 4 (coordinator): per-motor ankle limits, thermal penalty, J3e resume
- **Problem (coordinator, J3d model_2150, 256 envs x 1000 steps, level 1):** per-motor X4-36 torque above the 34 N*m peak in
  2.67 % of robot-steps (p99.9 42.3, max 76.8), above rated 10.5 in 64.8 % (p95 22). Joint caps (86.6 / 35.2) don't stop
  per-motor overload; the overload penalty (-0.002/s) was too weak.
- **Fix 1, `ankle_actuator.py`:** one actuator for the 4 ankle joints (subclass of mjlab's DcMotorActuator, compute overridden):
  joint PD with per-joint kp/kd (gain DR still applies) -> motor torques tau_m = solve(J^T, tau_j) -> each motor clamped with
  mjlab's dc_motor_clip (peak 34, saturation 116.8 = X4-36 slope x no-load speed, velocity limit 11.62 rad/s at the motor,
  motor speed = J qdot) -> tau_j = J^T tau_m. Other joints keep the per-joint DC model; ankle armature stays per joint (XML).
  Unit test (200k random commands per ankle, 93 % over the limit): max motor torque after clamp 34.0000, 0 above the
  torque-speed line; commands inside the envelope change < 6e-6 N*m. In the env (random actions x2, level 1, 32 envs):
  actuator's motor torques max 34.0000. Note: mapping the applied joint torques back after the step (post-step J) reads up to
  ~36 N*m because the ankle moves within the step; the penalties/metrics/check script now read the actuator's exact values.
- **Fix 2, thermal penalty `linkage_penalty.ankle_motor_thermal` (class term):** per motor EMA of tau^2 (time constant 2 s),
  penalty sum over 4 motors relu(sqrt(EMA)/10.5 - 1)^2, weight -1.0: all 4 motors at 50 % over rated -> -1.0/s (half the max
  linear-tracking reward of 2.0/s); 2 motors -> -0.5/s. Metrics: ankle_motor_rms_nm, ankle_motor_frac_above_rated (+ EMA RMS).
- **Fix 3, `vm/ankle_torque_check.py`:** coordinator's script, reads the exact clamped motor torques, adds % motor-steps above
  rated, RMS, 2 s moving RMS (thermal proxy), CPU fallback. J3d model_2200 with the new actuator (8 envs, CPU): 0.00 % above
  peak (max 34.0), any motor above rated in 66.6 % of steps, 22.7 % of motor-steps, RMS 10.1, sustained RMS > rated 34.3 %.
- **Fix 4, J3e:** `vm/resume_latest.sh <src> <dst>` copies the source run's newest checkpoint (numeric sort) to
  <dst>/wisco_v1_velocity/resume_src/ and resumes. Verified on the Mac with J3d model_2200 (CPU copy): loads, continues
  2200-2201, level 1. check_env.py ALL PASS on the VM GPU (1024 envs, level 1); check 4 now compares the grid mapping.
- HARDWARE_ISSUES H3 / H13 updated: two X4-36 per ankle look undersized for this gait and mass.
- **Launch J3e:** `WISCO_START_LEVEL=1.0 ~/humanoid-sim-stack/training/locomotion/wisco_v1/vm/resume_latest.sh ~/hsim_runs/J3d_wisco_v1_vel ~/hsim_runs/J3e_wisco_v1_vel`

---

## Training run table (Phase C)

| Run | Date | Change from previous | Iterations | Result | Next |
|---|---|---|---|---|---|

## 2026-11-10 ~14:00 UTC: J3e -> J3f (curriculum gate fix), G1 J1e -> J1f
- Problem: the L2 stage tightens the tracking-reward stds, and the same rewards decided whether L2 may advance.
  Tightening lowered the gate score by itself, so L2 stalled (Wisco at 0.65, G1 at 0.15) while behaviour was fine
  (Wisco 9/10 behaviour tests, 0 falls).
- Fix: two extra reward terms `gate_track_lin` / `gate_track_ang` (weight 1e-3, fixed starting stds 0.5 / 0.707) and
  the curriculum reads those. Thresholds unchanged for Wisco (0.5 / 0.4); G1 keeps its L2 turn gate at 0.35.
- J3f resumes J3e model_3000 at L=1, L2=0.65 (ankle clamp + thermal as before). J1f resumes J1e model_2650 at L=1, L2=0.15.
- G1 only, ~14:35 UTC, J1f -> J1g: an in-place turn test showed the G1 ignores turns below ~1 rad/s from standstill
  (0.3/0.5/0.8 -> 0.02/0.03/0.2; Wisco 0.05/0.3/0.6). Left and right are symmetric. The L2 turn gate deadlocked it: the
  turn-std tightening that fixes this was waiting on turning being good already. J1g: no turn gate on L2 (length and speed
  gates kept), 30 % in-place turn commands at 0.2-1.5 rad/s (was 20 % at 0.3-1.5). Resumes J1f model_3000 at L2 = 0.40.
- ~14:55 UTC, both robots (J1g -> J1h, J3f -> J3g): Wisco finished the curriculum (L2 = 1) and slow 0.2 m/s now passes, but
  in-place turns got worse (0.5 rad/s -> 0.06-0.2). Cause: mjlab's track_angular_velocity counts roll/pitch rate as error; at
  std 0.35 the rocking from stepping costs more than a slow turn earns. Fix: yaw-rate-only turn reward (track_yaw_rate, also
  used by the gate copy); roll/pitch still penalised by body_ang_vel. J1h resumes J1g model_3200 at L2 0.60, J3g resumes J3f
  model_3650 at L2 1.0.
- ~16:10 UTC, G1 only (J1h -> J1i): yaw-only reward fixed Wisco (10/10 behaviour tests, in-place turns 0.5-1.2 rad/s
  followed) and improved the G1's larger turns, but the G1 still ignores slow 0.2 m/s and 0.5 rad/s right turns after
  ~450 iterations at L2 = 1. J1i: 20 % of non-turn resamples get one small command (vx 0.1-0.35, vy 0.1-0.3 or wz
  0.15-0.5) and the forward command range is capped at 1.5 m/s (stock stages widen to 3 m/s).
- ~18:05 UTC, G1 only (J1i -> J1j): after ~1000 iterations of small-command practice the G1 still ignores 0.2 m/s and
  0.5 rad/s right (model_5150: 8/10, 0 falls; slow 0.003-0.02 m/s). Fixed-command video: it stands flat-footed. Head-to-head
  with the kitchen-demo all-joints policy (model_9999, arms still): all-joints 7/10 (sidestep 0.4 -> 0.13, right 0.8 turn -> 0.14), arms-free 8/10 with
  arm motion + payload. J1j: feet air-time reward 0.25 (0.05-0.5 s swings) whenever command > 0.1. Fallback: J1i model_5150.
- ~18:35 UTC: the air-time reward worked on the G1 within ~100 iterations (J1j model_5350: 9/10, slow 0.2 m/s -> 0.11,
  in-place turns now symmetric 0.5 -> 0.20/0.19). Same change for Wisco (J3g -> J3h): air_time 0.25 for commands > 0.1,
  resumed from J3g model_4200 (best: 10/10, but slow 0.2 m/s flickers 0.055-0.13 across checkpoints).
  resume_latest.sh now takes WISCO_CKPT_N to pick a specific checkpoint.
- ~20:10 UTC: J3h model_4650 = Wisco best so far (10/10 arms still AND arms random, 0 falls; in-place 0.5-1.2 rad/s both
  ways; ankle peak <= 34 N*m, 26.5 % above rated). At iteration 5000 the stock command curriculum widened Wisco's forward
  command to 2 m/s (G1 was already capped at 1.5): reward 66 -> 56. J3i: same 1.5 m/s cap for Wisco, resumed from J3h model_5050.
- ~20:40 UTC, manipulation: first closed-loop test of GR00T N1.6 checkpoint-5000 (25 % of the fine-tune), 20 held-out
  layouts: reported 3/20 but all 3 were "successes" at step 1-2 (cup spawned already counting as on the plate). Real result
  0/17. eval_closed_loop.py fixed: skips layouts with reset_ok False or success() already true at reset (as datagen does).
  Video (results/videos/groot_ck5000/): the arm moves purposefully toward the cup/plate area but doesn't complete a grasp.
- ~21:05 UTC: user approved pausing the motion-tracking runs (levelled off). J2a/J2b/J2c (G1 dance/jumps/fall-and-get-up,
  model_9500) and J4 (Wisco dance, model_9000) stopped; ONNX + checkpoint + motion copied to results/tracking_models/
  (README has the Mac playback commands, tested). GR00T fine-tune went from ~2.3 s/step to ~1.25 s/step.
- ~22:25 UTC, manipulation: GR00T N1.6 checkpoint-10000 (50 %), 30 held-out layouts (fixed eval, invalid / pre-solved
  layouts skipped): 8/30 = 27 % real successes (163-479 steps each), up from 0/17 at checkpoint-5000. Fine-tune continues.
- Wisco in the ROS stack (robot:=wisco): robot profiles config/robots/{g1,wisco}.yaml; Wisco deploy world with PD in the
  XML (tools/build_robot_world.py; sim2sim passes); policy models/locomotion/wisco_v1_velocity_arms_free; all envs built to
  build_wisco/. Smoke test passes. Autonomy suite: Wisco 4/11, G1 with stereo 1/11, G1 with lidar 11/11 -> the stereo-only
  autonomy path is the gap, not the robot swap (Wisco has no lidar, so this is needed for hardware too).
- ~23:00 UTC, Mac viewer with Wisco: (1) panel stuck on "waiting for viewer bridge": Docker Desktop drops container->Mac
  UDP datagrams above ~9.2 kB (measured: 9000 ok, 9300 lost); with the stereo point cloud the bridge's packets were bigger.
  viewer_bridge MAX_UDP 60000 -> 9000 and it now thins cloud/scan to fit (logs once if a packet still can't).
  (2) objects only appeared after the first command: kitchen_l1_s1's spawn faced a bare wall (0 of 27 objects in view),
  same for the G1. Spawn yaw set to face the room in l1 (60), l2 (180), l3 (-15), l5 (-90), l9 (180), l10 (-30); rebuilt
  for both robots. Objects now show ~8 s after launch with no command. G1 kitchen tests still pass.
  Also: container /dev/shm was 100 % full (828 stale DDS segments) -> shm_size 1gb in compose, ros.sh stop cleans it.
- ~23:40 UTC, "go to the sink -> couldn't, way blocked" (user report): reproduced 3/3 after a first navigate_to. Causes and
  fixes (hsim_behavior behavior_node.cpp unless noted):
  1. mid-walk re-planning sometimes found no valid side (perceived box changing while merged) and fell back to "0.55 m
     from the centre toward me", which in a kitchen lands inside the counter run -> Nav2 fails. Now keeps the chosen side.
  2. side picker treated cells off the map as free -> picked the counter's back side against the wall, staging spot
     outside the room. Off-map now counts as blocked; the chosen side gets a strong preference (no flipping sides).
  3. staging spot fixed at 0.65 m: in a narrow aisle Nav2 can't plan to it. Now pulled in to >= 0.45 m until it has
     0.32 m free around it.
  4. stereo robots: Nav2 marked obstacles out to 5.5 m; stereo depth error ~ z^2 (~30 cm at 8 m) left phantom marks in
     the aisle. navigation.launch.py obstacle_range:=3.0 for sensors:=stereo (clearing to 4 m).
  Result: fridge->sink / stove->sink 6/7 (was 0/3), arrivals 0.29-0.35 m, 0-6 deg off square.
  Arrival tightened: BT squares to 6 deg, accepts <= 10 deg, else reports it couldn't square up (was: any angle after
  5 s). Grader reach_check.py: furniture 0.15-0.40 m, <= 10 deg, in front of it (+-0.10 m); small objects <= 0.60 m,
  facing <= 20 deg (was <= 0.45 m / 20 deg and 0.65 m / 45 deg).
- GR00T checkpoint-15000: 15/30 = 50 % (ck-10000 27 %, ck-5000 0 %).
- ~01:30 UTC 2026-11-11: GR00T N1.6 fine-tune finished (20000 steps, 8.1 h, final loss 0.005). Final closed-loop test,
  checkpoint-20000, 50 held-out layouts: 26/50 = 52 % (ck-15000 50 % on 30, ck-10000 27 %, ck-5000 0 %); successful
  episodes 154-536 steps (median ~205 = ~10 s); server latency ~96 ms. Weakest cups: obj18 0/5, obj06 1/4.
  finetune.sh exited 1 after "Training completed!" (EXIT trap's kill of the finished monitor) -> job runner restarted it;
  stopped before it touched anything; trap now `|| true`; job marked done.
- navigate_to final approach (behavior_node.cpp): reach target 0.30 -> 0.27 m; look_around falls back to turning in
  place itself when Nav2's spin refuses (stops only if something is < 0.30 m); final approach tracks heading and the last
  few cm in the ODOM frame (AMCL heading with a forward-only camera is 10-20 deg off and jumps); square-up turns in
  pulses and judges the angle only from scans captured after stopping (5 Hz depth scan lag = 8-16 deg at 0.7 rad/s);
  sensor range only trusted when shorter than the map's (a level head camera loses a 0.75 m table inside ~0.8 m).
  Strict grader (0.15-0.40 m, <= 10 deg): G1 (lidar) 3/4 (table, fridge, sink; block 0.69 m), Wisco 0/4 but near misses
  (table 0.40 m @ 14 deg, fridge 0.43 m @ 2 deg, block 0.50 m @ 41 deg, sink once nav-failed).
- ~03:30 UTC 2026-11-11, "no cheating" autonomy (user request): no prebuilt map, no privileged perception, by default.
  * autonomy.launch.py defaults: mapping:=rtabmap (RTAB-Map SLAM from scratch every launch; map:=AMCL on the prebuilt
    map kept for tests/debug only), perception:=server (Grounding DINO + SAM 2 on the VM via the SSH tunnel), no stub
    fallback (perception_fallback:=none; VM down = the robot sees nothing).
  * Every robot runs a rendered head RGB-D camera: ZED X (110 deg, 12 m) in stereo mode; the G1's D435i-like head
    camera (87 deg, 6 m) in lidar mode. RTAB-Map uses color + depth in both, plus the lidar scan in lidar mode
    (grid from the scan, ICP links); grid from depth in stereo mode.
  * New robot profile g1_arms_free = G1 + arms-free walking policy (J1j model_6550; models/locomotion/g1_velocity_arms_free, sim2sim PASS).
    G1 training (J1j) stopped (plateaued since ~5k iterations). Wisco camera pitched 25 deg down (H8 test).
  * BT: with SLAM the map frame is the robot's start pose, not wall-aligned -> furniture sides now follow the room's
    dominant wall direction estimated from the SLAM map (Manhattan world); off-map = unknown under SLAM.
  * End-to-end ("look_around, navigate_to:sink", kitchen_l1_s1): 5/5 combos succeed, 0 falls, 0-10 deg off square;
    strict grade 2/5 (Wisco 0.25 m; G1 lidar 0.33 m), others 0.43 m / beside the sink.
- ~04:30 UTC 2026-11-11, user report (Wisco, SLAM + real perception): leaves the room / constant replanning; approach
  distance and alignment off. Fixes:
  * SLAM modes (navigation.launch.py): planner allow_unknown false (unseen walls aren't on the map, so planning through
    unknown went out of the room); bt_navigator uses Nav2's navigate_w_recovery_and_replanning_only_if_path_becomes_invalid
    (no 1 Hz replanning against a changing map); RTAB-Map Grid/RayTracing (seen floor = free) + noise filtering.
  * BT: approach spot moves only for > 0.6 m changes seen on 2 checks in a row; prefers staging on seen floor (SLAM);
    per-kind standoff (fridge 0.45, dishwasher/oven 0.42, cabinet/drawer 0.35, else 0.27 m) and appliances approached
    at the middle of their front.
  * Grader: per-kind windows (fridge 0.30-0.60, dishwasher/oven 0.28-0.55, cabinet 0.20-0.50, else 0.15-0.40) and
    "centered on its front" for appliances.
  Results (Wisco, l1 + l3, sink + fridge, 5 runs): never left the room or entered walls/furniture (true-path tracking),
  0 path replans, 0 staging moves, all plans succeeded, 0 falls. Fridge l1 PASS (0.30 m, 3 deg); sink varies run to
  run: 5 deg/0.18 m PASS vs 14 deg/off-center FAIL (real-detector sink estimate + camera ranging past the counter edge).
- ~06:40-07:45 UTC 2026-11-11, G1 arms-free smoothness fine-tune (user: wobbles / shuffles while standing, some stumbles).
  New eval `eval_velocity.py --script kitchen` (stands between short walks, nudges, turns; per-robot sway speed, foot
  lift-offs/s, action jitter). J1j model_6550 baseline: sway 0.064 m/s, jitter 0.011 (keeps swaying for seconds after moving).
  J1k (from 6550): stand-still penalty (drift + feet in air when command ~0), 25 % standing that turns don't replace,
  commands every 2-6 s, action rate -0.2, joint acc -2.5e-7, air time 0.2 -> steadier (sway 0.031-0.043) but lazy:
  0.8 rad/s turns halved, 0.2 m/s ignored. J1l (stricter) stopped; J1m (air time 0.25, action rate -0.15) equally lazy.
  J1n (from J1k model_7900): + precise tracking terms (std 0.1 m/s / 0.2 rad/s, weight 1 each). J1n model_8250: sway
  0.020 m/s, jitter 0.003, 0 foot lifts, 0.8 rad/s turns 0.55 both ways (as 6550), forward 1.0 -> 1.13, 0 falls; still weak
  below what the kitchen stack sends (0.2 m/s walk, 0.5 rad/s turn; Nav2 >= 0.3 m/s, 0.9 rad/s; squaring 0.7 rad/s).
  Deployed as models/locomotion/g1_velocity_arms_free (sim2sim PASS); previous model kept in models/locomotion/g1_velocity_arms_free_J1j.
- ~08:10 UTC: user still sees swaying / shuffling / stumbling with J1n. New probe tools/eval/steadiness.py (plain MuJoCo,
  kitchen-like stop/start script: torso rocking rate, tilt, steps, foot slide, drift). Standing / moving rocking:
  all-joints 3.5 / 9.1 deg/s, J1j 8.0 / 17.1, J1n 9.7 / 22.2 (foot slide moving 6.5 / 12.1 / 15.6 cm/s): the whole arms-free line rocks
  2-3x more than the all-joints policy, and J1n's precise tracking made it worse. g1_arms_free reverted to J1j (J1n kept in g1_velocity_arms_free_J1n).
  eval_velocity.py prints ROCKING (torso roll/pitch rate RMS, max pelvis tilt); J1j 6550 on VM: 16.3 deg/s, 8.2 deg.
  J1o (from J1j 6550): J1k stand-still/smoothness + torso roll/pitch rate -0.4 (was -0.05), pelvis tilt -10, foot slip
  -0.5 also while standing, no precise tracking. J1p: same, strong (-1.0, -20, -1.0). Both 2000 iterations in parallel.
- ~08:10 UTC: first anti-rocking checkpoints (~350 its): rocking 16.3 -> 5.4 (J1o) / 6.1 deg/s (J1p), tilt 8.2 -> 5.7 / 6.8 deg,
  but sluggish (kitchen 6/11 and 8/11; J1o 0.35 m/s -> 0.17). User: don't rush, give it real training time. Queued J1o2/J1p2
  (+3000 iterations each, start automatically from each run's final checkpoint; ~5000 total, ~2.5 h at 1.66 s/it with two
  runs sharing the GPU). VM tmux steady-monitor evaluates the newest checkpoint of each run every 30 min
  (kitchen/turns/standard + ROCKING) -> ~/hsim_runs/_steady_evals.log.
- ~10:10 UTC (overnight loop): J1o2 9700 / J1p2 9800 / J1q 8950: rocking kitchen 8.8 / 7.1 / 8.4 deg/s (base 15.8), standard
  16.9 / 13.7 / 16.6 (base 28.6), tilt ~8-9 deg (base 17), 0 falls; but left 0.8 rad/s in-place turn only 0.26-0.29 (base 0.61).
  New knob G1_YAW_W (turn tracking weight). J1o2 stopped (plateaued); J1r from J1p2 model_9800: strong anti-rocking + yaw
  weight 3.0 + 40 % turns.
- ~10:40 UTC: candidate J1p2 model_10450 (packaged as models/locomotion/g1_arms_free_candidate, not deployed): VM rocking kitchen 8.0 /
  standard 12.3 deg/s (base 15.8 / 28.6), tilt 5.7 (base 17), 0 falls, turn 0.8 0.48/0.46 (base 0.61/0.51); misses slow 0.2
  (0.06, base 0.17) and nudges. Mac plain MuJoCo (steadiness.py): rocking standing 5.7 vs 8.0, moving 10.3 vs 17.1 deg/s;
  drift/slide about the same; sim2sim PASS. Holding for J1p2 final / J1r (turn weight 3.0) before deploying.
- ~10:50 UTC: DEPLOYED J1r model_10150 as models/locomotion/g1_velocity_arms_free (robot:=g1_arms_free); J1j backup in g1_velocity_arms_free_J1j.
  VM: rocking kitchen 6.8 / standard 12.2 deg/s (base 15.8 / 28.6), tilt 7.6 (base 17), turn 0.8 in place 0.64/0.67 (base
  0.61/0.51), 0.5 rad/s turns now followed (0.29, base ignored), forward 1.0 -> 1.02, standard 9/10, 0 falls. Mac plain
  MuJoCo: rocking moving 8.8 vs 17.1 deg/s (all-joints 9.1), standing 5.3 vs 8.0, tilt 2.1-2.4 vs 3.2-3.3, drift 5.7 vs 6.0 cm/s;
  sim2sim PASS. Slow 0.2 m/s from standstill: 0.01 m/s (J1j 0.02, so no regression). J1r keeps training.
- ~10:55 UTC: J1q stopped (behind J1r); J1s from J1r 10150: + precise tracking 0.5, stand-still -2.0 (slow starts, drift).
- ~11:15 UTC: J1r model_10800: standard 10/10 (slow 0.2 -> 0.16), turns 0.69/0.59, but rocking 7.2/13.9 (deployed 10150: 6.8/12.2), tilt 9.0 (7.6): candidate, not deployed. J1p2 finished (not better). J1t from J1r 10150: + action rate -0.3, joint acc -5e-7.
- ~11:55 UTC: DEPLOYED J1r model_11500 as g1_velocity_arms_free (previous J1r 10150 in g1_velocity_arms_free_J1r10150). VM: kitchen rock 6.9
  (10150: 6.8), kitchen 10/11 (9/11), standing 0.038 m/s (0.046), standard 13.2 deg/s (12.2), tilt 9.5 (7.6), 9/10, turn 0.8
  0.75/0.65 (0.64/0.67), 0 falls. Mac steadiness: moving rock 7.9 vs 8.8, standing 4.7 vs 5.3 deg/s, slide moving 10.2 vs
  13.6 cm/s, fewer steps; sim2sim PASS.
- ~12:00 UTC: J1s stopped (precise 0.5 + stand -2 didn't help: rock 8.3/15.4, 8/11). J1t 10850: 14.0 deg/s, 10/10, turns 0.68/0.73 (watch). J1u from J1r 11500: tilt -40.
- ~12:35 UTC: J1t 11550 (VM: tilt 6.3, 10/10, rock 7.7/14.2) loses on Mac steadiness (moving rock 10.7 vs 7.9, standing 6.0 vs 4.7) -> not deployed. J1r peaked at 11500 (12150 rock 8.8).
- ~12:43 UTC: J1r done (peak 11500, deployed). J1u 11950: kitchen 11/11, rock 7.1/13.8, tilt 9.0, turns 0.60/0.66 (watch). J1v from J1r 11500: torso rate -2.0.
- ~13:05 UTC: J1t 12250 (VM better: rock 6.8/13.1, tilt 6.0, turns 0.71/0.76) loses on Mac again (standing rock 6.4 vs 4.7, moving 8.9 vs 7.9, drift 7.5 vs 5.8 cm/s, sim2sim tracking poor on a segment) -> not deployed. VM eval and Mac plain-MuJoCo disagree for J1t (higher action-rate/joint-acc weights).
- ~13:15 UTC: DEPLOYED J1v model_11900 as g1_velocity_arms_free (previous J1r 11500 in g1_velocity_arms_free_J1r11500). J1v = J1r settings with
  torso roll/pitch rate -2.0. VM: kitchen rock 6.6 (6.9), 10/11, standing 0.033 m/s (0.038), standard 11.1 deg/s (13.2), tilt
  5.7 (9.5), 10/10, turn 0.8 0.67/0.69, 0 falls. Mac: moving rock 7.2 vs 7.9, standing 4.3 vs 4.7 deg/s, slide moving 7.0 vs
  10.2 cm/s, steps 1.14 vs 1.49/s, drift 5.5 vs 5.8 cm/s; sim2sim PASS.
- ~13:35 UTC: J1t finishing (12900: drift 0.068, not better). J1w from J1v 11900: torso rate -3.0.
- ~13:55 UTC: J1v 12600 (VM rock 5.7/10.3) loses on Mac: moving rock 8.2 vs 7.2, steps 1.83 vs 1.14/s, slide 11.5 vs 7.0 cm/s, drift 6.5 vs 5.5 -> deployed J1v 11900 stays.
- ~14:20 UTC: J1u finishing (14000: rock 7.1/12.9, not better). J1x from J1v 11900: foot slip -2.0.
- ~14:35 UTC: J1w 12750 (torso -3.0): lowest VM rocking (5.3/8.9) but Mac sim2sim forward 1.0 -> 0.45 m/s (too slow), more steps/slide -> rejected. -3.0 trades speed for steadiness; watch later J1w checkpoints for speed recovery.
- ~14:55 UTC: J1v 13900 (VM 6.1/10.9) loses on Mac: moving rock 8.3 vs 7.2, steps 2.0 vs 1.14/s, slide 15.2 vs 7.0 -> J1v 11900 stays. Later J1v checkpoints drift toward more stepping.
- ~15:05 UTC: J1w 13350 rock 6.4/10.7, J1x 12500 6.1/10.9 (8/11) - neither clearly better. J1y queued after J1v: J1v settings + air time 0.1.
- ~15:35 UTC: J1w 14050 (VM 5.5/9.9, speed recovered: sim2sim forward 1.0 -> 1.10) vs deployed on Mac: rocking equal (7.3 vs 7.2 moving, 3.9 vs 4.3 standing) but steps 1.88 vs 1.14/s, slide 12.2 vs 7.0, drift 7.2 vs 5.5 -> not deployed (more shuffling).
- ~15:45 UTC: J1x 13150 (VM 5.9/10.4, 11/11) loses on Mac moving: rock 8.3 vs 7.2, steps 1.76 vs 1.14, slide 11.3 vs 7.0, forward 1.0 -> 0.88 -> not deployed. Standing slightly better (4.0, drift 5.0).
- ~15:55 UTC: Mac check of J1v neighbours: 11800 moving rock 6.3 / steps 1.22 / slide 8.1 / drift 6.5; 11950 8.7 / 1.49 / 7.5; 12000 7.3 / 1.08 / 6.3 / drift 7.2 (deployed 11900: 7.2 / 1.14 / 7.0 / 5.5). Checkpoint-to-checkpoint spread (6.3-8.7) is as big as the differences between runs now -> plateau; keep 11900 (best all-round). J1z queued after J1w: J1v recipe re-run from 11900, 1500 its.
- ~16:05 UTC: J1w final 14700 (VM 5.4/9.7) on Mac: moving rock 7.4, steps 2.33/s, slide 10.6 (deployed 7.2 / 1.14 / 7.0) -> not deployed. J1w done; J1z running.
- ~16:15 UTC: J1y 12950 (air time 0.1) Mac: moving rock 7.9, steps 2.08/s, slide 12.1 -> lower air-time reward did NOT reduce stepping; not deployed.
- ~16:45 UTC: J1w final 14899: Mac rocking moving 6.3 / standing 3.6 deg/s, drift 4.5 cm/s (deployed 7.2 / 4.3 / 5.5) but steps moving 1.96 vs 1.14/s, slide 10.5 vs 7.0 cm/s (quicker, shorter steps). Mixed -> kept as alternative models/locomotion/g1_velocity_arms_free_alt_J1w14899 (not deployed). Removed old g1_arms_free_candidate (J1p2 10450).
- ~17:25 UTC: overnight G1 steadiness search ended (all runs done, plateau). Last checks: J1x/J1y/J1z finals and J1y 14600
  (Mac moving rock 8.3, slide 14.2) don't beat J1v 11900. FINAL g1_arms_free = J1v model_11900. Alternative kept:
  g1_velocity_arms_free_alt_J1w14899 (less rocking/drift, quicker steps + more slide). VM: only the perception server running;
  steady-monitor stopped. What worked: torso roll/pitch rate penalty -0.05 -> -2.0, pelvis tilt -20, foot slip -1.0 always,
  turn tracking weight 3.0, 40 % in-place turns, stand-still penalty. Didn't: precise tracking (more rocking), stronger
  action-rate/joint-acc (VM-only gains), torso -3.0 (slow), air time 0.1 (no fewer steps). Next real step if needed:
  retrain G1 with Wisco-like higher joint damping / smaller action scale (needs sim + manifest gain support).
- ~18:50 UTC: user: g1_arms_free rocks much more in the ROS kitchen stack than in tools/play.sh. Probe (scripted stand/walk/stop/
  turn on /cmd_vel_teleop, pelvis gyro RMS): ROS standing/after-stop 17.6 deg/s vs plain MuJoCo 4.4-6.4 (6 runs); walking
  ~equal. Causes and fixes (hsim_sim):
  1. sensors.ImuModel tilt_gain 0.02 -> 0.001: the complementary filter (0.25 s time constant) followed the body's sway
     accelerations, tilt off by up to 7-10 deg; policy corrected a fake tilt (standing rock 4.6 -> 11.4 in plain MuJoCo
     with this IMU). Now ~0.5 deg error (realistic onboard AHRS).
  2. mujoco_sim lockstep (default on): physics waits at each 20 ms control boundary (+5 ms grace, 0.1 s timeout) for new
     /locomotion/joint_targets; catches back up to wall time after waits (backlog dropped only if > 100 ms behind). Under
     load the policy had missed ticks (gaps 50-95 ms, RTF 0.86) -> stale targets.
  Result, realistic EKF: standing rock 5.3 deg/s (was 17.6), walking 20.5 (plain 17-21), RTF 1.00, policy 50 Hz.
  Smoke test: g1_arms_free and wisco PASS in the default world. In robocasa/kitchen_l1_s1 the drive check moves only ~0.4 m
  (needs 0.5) with or without these changes (pre-existing; probably the room-facing spawn), contract checks pass.
- 2026-11-12: Wisco current-draw estimate for mech/elec: robots/wisco_v1/tools/current_estimate.py ->
  robots/wisco_v1/reports/current_draw/REPORT.md (+ plots, CSVs). Datasheet motor model (torque/current points, R_line,
  no-load current), loss terms fitted to the X8 load test (within 4 % above 5 N*m). Walking 0.5 m/s ~6.3 A mean at 48 V,
  1.0 m/s ~9.5 A; ankle X4-36 motors exceed peak torque/rated speed in plain MuJoCo (H3). Deployed Wisco policy
  (J3h 4650) was trained only up to 1.0 m/s forward (curriculum widens after 5k): 1.5 m/s step from standing falls,
  ramped it holds.
