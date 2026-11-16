# Phase A report: pre-training checks (2026-11-10)

Model: `model/wisco_v1_train.xml`, built from `robot.xml` (V1) by `build_training_model.py`.
Simulated in plain MuJoCo 3.11 with a copy of mjlab 1.6.0's actuator, collision and solver settings
(`sim_replica.py`). Raw numbers: `results.json`. Reproduce: `python run_checks.py`.

**Verdict: no blockers. Ready for Phase B.** One check (A4) fails its own pass rule on 5 non-critical joints; explained below.

| Check | Result | Key numbers |
|---|---|---|
| A1 model loads | PASS | 29 policy joints (head welded), 40.55 kg, all 5 task sensors, named collision geoms |
| A2 default pose | PASS | No self-contact; soles flat (0.000 deg); centre of mass 13 cm behind toes, 8 cm in front of heels; pelvis 0.688 m |
| A3 stand 10 s (zero action) | PASS | Stays up, tilt 6.4 deg, drift 3.4 cm, shaking gone (joint speed RMS 0.0002 rad/s). G1 falls over in the same test |
| A4 gain step response | FAIL (5/29) | All stable, no oscillation. Wrists slow (friction dead zone); torso roll settles in 0.415 s vs 0.4 s rule |
| A5 directions, mirror, range | PASS | Every joint moves the expected way; left/right mirror (cos >= 0.997); no blow-ups |
| A6 falls | PASS | Every fall caught by the 70 deg tilt rule in 0.6-1.4 s; 0.5 m/s push recovered, not flagged |
| A7 foot contacts | PASS | Feet carry 49.8 % / 50.2 % of 398 N |
| A8 sim sizing | PASS | Max 21 contacts, 113 constraint rows (G1 limit 300); 174x real time on one CPU core |

## Gains (mjlab G1 formula: kp = armature x (2 pi 10 Hz)^2, kd = 2 x 2 x armature x 2 pi 10 Hz)

| Joint group | Motor | Armature | kp | kd | Peak effort | Action scale (rad) | G1 equivalent kp |
|---|---|---|---|---|---|---|---|
| Hip pitch/roll, knee, torso yaw | X8-150 | 0.050 | 197 | 12.6 | 120 | 0.152 | 40-99 |
| Hip yaw, shoulder pitch/roll | X6-60 | 0.025 | 99 | 6.3 | 60 | 0.152 | 14-40 |
| Shoulder yaw, elbow, forearm yaw | X4-36 | 0.039 | 154 | 9.8 | 34 | 0.055 | 14 |
| Wrist pitch/roll | X4-10 | 0.004 | 15.8 | 1.0 | 10 | 0.158 | 17 |
| Ankle pitch | 2x X4-36 linkage | 0.126 | 499 | 31.8 | 86.6 | 0.043 | 28.5 |
| Ankle roll | 2x X4-36 linkage | 0.021 | 82 | 5.2 | 35.2 | 0.107 | 28.5 |
| Torso roll / pitch | 2x X6 linkage | 0.021 / 0.024 | 84 / 96 | 5.3 / 6.1 | 78 / 84 | 0.23 / 0.22 | 28.5 |

Our high gear ratios (X4-36 is 36:1) give much larger rotor inertia at the joint than the G1's motors,
so the same formula gives stiffer joints with smaller action scales, especially ankle pitch and the
X4-36 arm joints. Stable in every test. Watch in Phase C: if the policy can't move the arms or ankles
enough, raise those action scales first.

## Stand (A3): holding torque, top joints

| Joint | Holding torque | % of rated |
|---|---|---|
| Knees | 17.1 N*m | 40 % |
| Ankle pitch (joint) | 4.8 N*m | 18 % |
| Each ankle motor | 1.8 N*m | 17 % of X4-36 rated 10.5 |
| Wrist pitch | 0.6 N*m | 14 % |
| Shoulder pitch | 2.3 N*m | 12 % |

Knees sag 0.087 rad below target under the robot's weight (17 N*m / kp 197): expected with PD control,
the policy learns to aim slightly past.

## A4 failures, and why they don't block training
- **Wrists (4 joints):** settle in 0.7 s and stop 7-18 % short. Joint friction from the old URDF
  (0.8 N*m) against kp 15.8 makes a 0.05 rad dead zone. Wrists hardly matter for walking.
- **Torso roll:** settles in 0.415 s (rule 0.4 s). With the robot hung in the air, gravity pulls a
  0.2 rad lean to 0.35 rad (torso pitch: to 0.28 rad): the upper body is heavy for the waist gain.
  Fine while standing (A3); a Phase C knob if the torso wobbles when walking.

## Range sweep (A5): where joints stop short of their limit
| Joint | Limit | Reached | Cause |
|---|---|---|---|
| Hip roll inward | -0.611 | -0.446 | Foot meets the other knee (legs crossing): physical, fine |
| Shoulder roll inward | -0.436 | -0.172 (L) / -0.233 (R) | Elbow / upper arm hits the torso. Mech note (HARDWARE_ISSUES H9) |
| Wrist pitch up | 0.611 | 0.52 | Soft gain + friction + hand weight, no contact |
| Hip pitch back | -2.79 | -2.76 | Knee meets torso at -2.69 (extreme pose) |
| Shoulder yaw | 2.62 | 2.61 | Hand touches hip from 2.10 rad |

Note: a 2 mm pelvis / hip-pitch-link contact appears only in the hung-in-the-air test (welding the
pelvis to the world turns off MuJoCo's parent-child contact filter). The free-standing robot has none.

## Falls (A6)
| Case | Tilt rule fires at | Ends |
|---|---|---|
| Motors off (limp) | 0.82 s | pelvis 0.06 m, 86 deg |
| Push forward 1.5 m/s | 1.16 s | pelvis 0.09 m |
| Push sideways 1.0 m/s | 1.38 s | pelvis 0.20 m |
| Start tilted 25 deg | 0.64 s | pelvis 0.09 m |
| Push forward 0.5 m/s | never (recovers) | standing, 5.8 deg |

Same fall rule as the G1 (tilt > 70 deg) is enough; no height rule needed.

## Images
`default_pose.png` (front, side), `stand_after_10s.png`, `limp_collapse_end.png`.
G1 comparison numbers: `g1_baseline.json` (`python g1_baseline.py <mjlab G1 xml folder>`).
