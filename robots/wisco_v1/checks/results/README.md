# Phase A results

| Folder | Model | Actuator model | Command delay |
|---|---|---|---|
| `initial_model_2026-11-10/` | first version (old guessed damping 0.2-1.6) | position (MuJoCo PD) | 0 | 
| `position_delay0/` | datasheet motor values (now in model/robot_numbers.yaml) | position (MuJoCo PD) | 0 |
| `dc_delay0/` | datasheet motor values | **DC motor (torque-speed curve), used for training** | 0 |
| `dc_delay2_formula_gains/` | datasheet motor values, G1-formula gains (X8 rotor 1.2 kg*cm2) | DC motor | 10 ms |
| `dc_delay2/` | datasheet motor values, **RP1-style gains (current, robot_numbers.yaml)** | DC motor | **10 ms (worst case trained)** |

The full write-up of the checks is `initial_model_2026-11-10/REPORT.md`. Rerun: `python run_checks.py --actuator dc --delay 2`.

## What changed with the datasheet motor values (2026-11-10)

All configurations pass A1, A2, A3, A5, A6, A7, A8. A4 fails on the same 5 joints in every configuration
(4 wrists: friction dead zone; torso roll: slow settling). Nothing became unstable.

| | Initial | Datasheet + DC motor + 10 ms delay |
|---|---|---|
| Stand tilt after 10 s | 6.4 deg | 4.1 deg |
| Knee sag below target | 0.087 rad | 0.062 rad (X8 armature 0.050 -> 0.058, so kp 197 -> 228) |
| Joint shaking after 8 s (RMS speed) | 0.00019 rad/s | 0.00006 rad/s |
| Torso roll step: settle / overshoot | 0.41 s / 3 % | 0.74 s / 9 % (damping 0.85 -> 0.017: the old guess was hiding a lightly damped waist) |
| Ankle pitch step: settle | 0.22 s | 0.23 s |
| Falls caught by the 70 deg rule | all | all; 0.5 m/s push still recovered |

The DC motor model (mjlab `DcMotorActuatorCfg`: PD torque computed every 5 ms step, then clipped to
the torque-speed line) is stable with our stiff gains (ankle pitch kp 498) and a 10 ms delay, so the
mjlab task uses it.

## RP1-style gains (2026-11-10, current)

Gains now from `model/robot_numbers.yaml` (control.gains): hips/knee kp 120, hip yaw 50, ankles 35, waist 100,
arms (sim stand-in) 20-40; kd at damping ratio 0.5 on the home-pose joint inertia.

| Check | Formula gains (hips/knee 182, ankle pitch 498) | RP1-style gains |
|---|---|---|
| A3 stand 10 s, zero action | stands, tilt 8.6 deg | **falls (95 deg after ~2 s)** |
| A4 step responses | all stable; 7 joints slow/overshoot | all stable; 10 joints flagged (hips settle 0.46-0.49 s, ankle pitch reaches 69 %, torso yaw/roll, wrists) |
| A6 falls caught by the 70 deg rule | all; 0.5 m/s push recovered | all; 0.5 m/s push now falls (it falls without a push too) |
| A7 foot load split | 50 / 50 % | 44 / 44 % (measured on a robot that is already falling) |
| A1, A2, A5, A8 | pass | pass |

**Why it no longer stands with zero action (expected, not a bug):** standing still with zero action needs passive
ankle stiffness of about m*g*h/2 = 40 kg x 9.81 x 0.75 m / 2 = ~150 N*m/rad per ankle (inverted pendulum).
Tested: ankle kp 35, 60, 80, 120 all fall; removing ankle friction doesn't help; raising waist, knee or hip gains doesn't help;
1.6x all leg/waist gains still falls. The formula's ankle pitch 498 was above the threshold. The G1 with mjlab's own gains
(ankle 28.5) also falls in this test (`initial_model_2026-11-10/g1_baseline.json`). With moderate gains the policy balances
actively, as on every RL-walking robot.

**Flag: ankle pitch dead zone.** Datasheet ankle friction (2.9 N*m: two X4-36 backdrive torques through the linkage) against
kp 35 leaves a ~0.08 rad (5 deg) dead zone; a 0.2 rad step reaches 69 %. The policy can compensate (ankle action scale is
0.62 rad and friction is randomized +/-30 %), but if the Wisco policy shows ankle chatter or poor foot placement, the first
knobs are ankle kp (e.g. 60-80) or a measured (lower) ankle friction from the bench.
