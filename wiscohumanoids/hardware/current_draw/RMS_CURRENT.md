# RMS current (Wisco humanoid, sim estimate)

Pulled from the same runs as REPORT.md (40.6 kg robot, 48 V battery, 12 s per test, first 1.5 s skipped). RMS is the number to use for heating and for continuous ratings (wires, fuses, battery continuous C-rate, motor rated current).

## 1. Battery (DC bus) current at 48 V

| Test | What | Mean A | **RMS A** | Peak 100 ms A | Peak 10 ms A |
|---|---|---|---|---|---|
| stand | standing still, arms at default | 1.3 | **1.3** | 1.4 | 1.4 |
| walk_0.25 | walk forward 0.25 m/s | 4.5 | **4.7** | 8.4 | 10.9 |
| walk_0.50 | walk forward 0.5 m/s (typical indoor) | 6.3 | **6.6** | 12.3 | 17.0 |
| walk_0.75 | walk forward 0.75 m/s | 7.8 | **8.4** | 17.1 | 24.8 |
| walk_1.00 | walk forward 1.0 m/s | 9.5 | **10.3** | 21.3 | 27.6 |
| backward_0.4 | walk backward 0.4 m/s | 5.5 | **5.7** | 7.8 | 9.3 |
| sidestep_0.3 | sidestep 0.3 m/s | 4.2 | **4.8** | 9.4 | 11.5 |
| turn_0.8 | turn in place 0.8 rad/s | 4.6 | **4.8** | 7.1 | 10.6 |
| walk_turn | walk 0.5 m/s while turning 0.4 rad/s | 6.5 | **6.8** | 10.3 | 16.2 |
| walk_0.50_arms | walk 0.5 m/s with arms moving to random targets | 7.3 | **7.6** | 12.1 | 17.8 |
| walk_0.50_payload | walk 0.5 m/s with 1 kg in each hand | 6.6 | **6.9** | 13.3 | 17.9 |
| stand_pushes | standing, 0.5 m/s shove every 2 s (random direction) | 2.7 | **3.0** | 9.7 | 13.0 |
| walk_pushes | walk 0.5 m/s, 0.5 m/s shove every 2 s | 6.6 | **7.0** | 15.2 | 17.3 |
| stop_start | kitchen-like: walk 0.4 / stop / turn / stop, repeated | 3.6 | **4.3** | 11.2 | 15.4 |

Highest RMS: 10.3 A (walk_1.00). With +30 % margin: 13.4 A continuous.

## 2. Per motor phase current, A RMS

Rated = the motor's rated (continuous) phase current. Over 100 % means it will heat up past its rating if run like that for a long time.

| Motor | Type | Rated A | stand | walk_0.50 | walk_1.00 | turn_0.8 | walk_0.50_arms | walk_pushes | Worst (test) | Worst % of rated |
|---|---|---|---|---|---|---|---|---|---|---|
| left_hip_pitch | X8-150 | 17.6 | 1.4 | 3.5 | 4.8 | 5.6 | 5.3 | 3.5 | 5.6 (turn_0.8) | 32 % |
| left_hip_roll | X8-150 | 17.6 | 2.5 | 6.7 | 7.2 | 6.0 | 6.9 | 7.6 | 9.3 (backward_0.4) | 53 % |
| left_knee | X8-150 | 17.6 | 2.3 | 7.9 | 6.3 | 8.5 | 8.6 | 7.4 | 8.6 (walk_0.50_arms) | 49 % |
| right_hip_pitch | X8-150 | 17.6 | 0.3 | 2.6 | 4.8 | 6.0 | 4.6 | 2.7 | 6.0 (turn_0.8) | 34 % |
| right_hip_roll | X8-150 | 17.6 | 2.4 | 7.6 | 7.9 | 6.1 | 6.7 | 7.6 | 10.2 (sidestep_0.3) | 58 % |
| right_knee | X8-150 | 17.6 | 3.8 | 8.7 | 7.5 | 6.5 | 8.3 | 7.8 | 9.4 (walk_turn) | 53 % |
| torso_yaw | X8-150 | 17.6 | 0.0 | 2.4 | 3.8 | 1.2 | 3.4 | 2.5 | 3.8 (walk_1.00) | 22 % |
| left_hip_yaw | X6-60 | 9.5 | 0.7 | 2.7 | 2.5 | 2.8 | 3.3 | 2.9 | 3.3 (walk_0.50_arms) | 35 % |
| right_hip_yaw | X6-60 | 9.5 | 0.6 | 3.6 | 3.0 | 2.6 | 2.8 | 3.3 | 3.6 (walk_0.50) | 38 % |
| left_shoulder_pitch | X6-60 | 9.5 | 0.5 | 0.9 | 1.4 | 0.7 | 1.3 | 0.9 | 1.4 (walk_1.00) | 14 % |
| left_shoulder_roll | X6-60 | 9.5 | 0.5 | 0.5 | 0.5 | 0.6 | 1.8 | 0.4 | 1.8 (walk_0.50_arms) | 19 % |
| right_shoulder_pitch | X6-60 | 9.5 | 0.5 | 1.0 | 1.3 | 0.8 | 1.5 | 1.0 | 1.5 (walk_0.50_arms) | 16 % |
| right_shoulder_roll | X6-60 | 9.5 | 0.5 | 0.7 | 0.6 | 0.5 | 2.4 | 0.8 | 2.4 (walk_0.50_arms) | 25 % |
| left_shoulder_yaw | X4-36 | 6.1 | 0.2 | 0.1 | 0.1 | 0.2 | 1.0 | 0.1 | 1.0 (walk_0.50_arms) | 16 % |
| left_elbow | X4-36 | 6.1 | 0.5 | 0.6 | 0.6 | 0.5 | 0.9 | 0.6 | 1.0 (walk_0.50_payload) | 17 % |
| left_forearm_yaw | X4-36 | 6.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.5 | 0.0 | 0.5 (walk_0.50_arms) | 8 % |
| right_shoulder_yaw | X4-36 | 6.1 | 0.2 | 0.1 | 0.1 | 0.1 | 0.9 | 0.1 | 0.9 (walk_0.50_arms) | 15 % |
| right_elbow | X4-36 | 6.1 | 0.5 | 0.5 | 0.6 | 0.5 | 0.9 | 0.5 | 0.9 (walk_0.50_payload) | 16 % |
| right_forearm_yaw | X4-36 | 6.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.5 | 0.0 | 0.5 (walk_0.50_arms) | 8 % |
| left_wrist_pitch | X4-10 | 7.8 | 0.9 | 1.0 | 0.9 | 1.0 | 1.0 | 1.0 | 2.2 (walk_0.50_payload) | 28 % |
| left_wrist_roll | X4-10 | 7.8 | 0.1 | 0.0 | 0.1 | 0.1 | 1.4 | 0.0 | 1.4 (walk_0.50_arms) | 17 % |
| right_wrist_pitch | X4-10 | 7.8 | 0.9 | 1.0 | 0.9 | 1.0 | 1.2 | 1.0 | 2.1 (walk_0.50_payload) | 27 % |
| right_wrist_roll | X4-10 | 7.8 | 0.1 | 0.0 | 0.0 | 0.0 | 1.0 | 0.1 | 1.0 (walk_0.50_arms) | 13 % |
| left_ankle_motor_1 | X4-36 | 6.1 | 2.1 | 4.5 | 5.8 | 1.9 | 3.8 | 4.7 | 5.8 (walk_1.00) | 95 % |
| left_ankle_motor_2 | X4-36 | 6.1 | 1.5 | 4.1 | 6.8 | 4.5 | 3.7 | 6.4 | 6.8 (walk_1.00) | 112 % ⚠ |
| right_ankle_motor_1 | X4-36 | 6.1 | 1.5 | 4.5 | 5.6 | 4.3 | 5.0 | 4.6 | 5.6 (walk_1.00) | 92 % |
| right_ankle_motor_2 | X4-36 | 6.1 | 0.4 | 4.4 | 8.3 | 3.8 | 4.3 | 4.5 | 8.3 (walk_1.00) | 136 % ⚠ |
| waist_motor_1 | X6-60 | 9.5 | 0.7 | 4.5 | 4.0 | 3.7 | 5.3 | 4.5 | 5.3 (walk_0.50_arms) | 56 % |
| waist_motor_2 | X6-60 | 9.5 | 0.9 | 4.7 | 5.0 | 3.7 | 5.4 | 4.6 | 5.4 (walk_0.50_arms) | 57 % |

## Notes

- Motor current RMS is phase current (what heats the winding), not battery current. Battery RMS is in section 1.
- Uses datasheet torque-to-current numbers. Bench tests (in progress) suggest the X4-36 may need about 20 % more current per N·m than the datasheet says; if confirmed, X4-36 rows (ankles, arms) go up by about that much.
- All tests and every motor for every test are in the appendix of REPORT.md / REPORT.pdf.
