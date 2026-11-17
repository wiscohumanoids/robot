# Sim-to-real for walking

How a walking policy trained in MuJoCo (mjlab) gets onto the robot, and how the simulator is made to match the robot
first. The steps follow published practice for the same problem: ETH's PACE for actuator identification, Menlo's
Asimov for timing, and RoboParty's open RP1 stack for the gains and the drive-side PD. The arm's own version of this,
already used on arm01, is `ros2_ws/src/wisco_ethercat_ros2_hw/arm_stack/` (`arm pace`).

| Step | What | Where | State |
|---|---|---|---|
| 1. Model the robot | masses, inertias, joint limits, motor datasheet values (armature, friction, torque-speed), drive gains | `robots/wisco_v1/model/robot_numbers.yaml` → `build_training_model.py` | ✅ from CAD and datasheets |
| 2. Identify the actuators | gantry chirps on the robot, replayed in the model, CMA-ES fit per joint (PACE) | `pace.py` (below) | ✅ fit verified on simulated logs; 🟡 robot recording needs the legs in mode 5 |
| 3. Put the fit in the model | fitted armature, friction, damping into `robot_numbers.yaml` (source: the fit file), delay into `delay.command_delay_ms`; rebuild the training model | by hand, reviewed | ⬜ after step 2 on the robot |
| 4. Train | randomization narrowed around the fitted values (PACE trains with little or none); command delay sampled over the measured range | `training/train.sh wisco walk` | ✅ with datasheet values |
| 5. Check in sim, the robot's way | the policy through ros2_control and a forward position controller, no lockstep, as on the robot | `sim_stack.launch.py backend:=ros2_control` | ✅ G1 and Wisco walk |
| 6. Real-time computer | PREEMPT_RT, isolated cores, the `update_rate` ramp | `infra/rk3588/` | 🟡 script ready, not run on the robot |
| 7. First steps | on the gantry, then lowered; true base pose from mocap compared with the commanded velocity | `mocap.launch.py` + `mocap_check.py` (`wisco_ethercat_ros2_hw`) | ⬜ |

## Actuator identification (`pace.py`)

The robot hangs on a gantry with its feet clear. Each joint in turn follows a logarithmic chirp around the standing
pose (0.2 → 3 Hz, up to 0.3 rad, faded in and out), while the others hold. The joint commands and measured
positions are logged. The fit replays the same commands in the trained MuJoCo model with the base fixed. The drive's
PD is the model's position actuator with the logged kp / kd, which is also what the drives run in mode 5. For each
joint, CMA-ES searches **armature, Coulomb friction, viscous damping, encoder bias and command delay** to minimize the
RMS position error. Every candidate in a generation runs in one threaded `mujoco.rollout` call.

```bash
S=assets/robots/wisco_v1/mjlab_1.6.0/wisco_velocity_flat.xml
python training/sim2real/pace.py plan   --scene $S --out plan.npz --joints '^(left|right)_' --seconds 12
#   robot: play plan.npz on the gantry at its rate, record q_meas, save as log.npz (format in pace.py)
python training/sim2real/pace.py fit    --scene $S --log log.npz --out fitted.yaml
python training/sim2real/pace.py report --scene $S --log log.npz --params fitted.yaml   # RMS before / after
```

`synth` plays a plan in the model with chosen "true" parameters, which is the sim twin of the gantry test. On
simulated logs the fit recovers the true values to within the encoder noise. On two Wisco joints with 0.5 mrad of
noise, the knee came out at armature 0.0904 (true 0.09), friction 1.20 N·m (1.2), damping 0.609 (0.6), bias 0.0100
rad (0.01) and delay 7.9 ms (8). The residual was 0.5 mrad, about 28 s for both joints on 8 cores. `test_pace.py`
checks the same on a small model.

What it needs before it runs on the robot: the legs under ros2_control in mode 5 (position + kp + kd), so the plan can
be played through the same forward position controller the sim backend uses, and a gantry.

## Practices taken from published work

- **Identify, then train without heavy randomization (PACE).** A compact per-joint model (armature, friction,
  damping, bias, delay) fitted from a few minutes of in-air data closed most of the gap on several legged robots.
  Training then needs little dynamics randomization.
- **Model the timing (Asimov).** Menlo reports zero-shot walking only after the simulator reproduced their control
  loop's delays and observation staleness. Here the delay comes from the fit, and the training config samples it per
  episode (`robot_cfg.py`).
- **PD in the drive, at the drive's rate (RP1, MyActuator mode 5).** The policy sends position targets at 50 Hz, and
  the drives close the loop with fixed kp / kd. Training applies the same law at the physics step, and
  `backend:=ros2_control` runs it through the robot's controller path.
- **Measure the truth independently.** Mocap gives the base pose without the state estimator, so velocity tracking on
  the robot can be compared with sim.

## Sources

- Bjelonic, Tischhauser, Hutter, "Towards Bridging the Gap: Systematic Sim-to-Real Transfer for Diverse Legged
  Robots" (PACE), arXiv:2509.06342, 2025.
- Menlo Research, Asimov zero-shot sim-to-real: https://menlo.ai/research/zero-shot-sim2real-asimov and
  https://menlo.ai/research/noise-is-all-you-need
- RoboParty RP1 (open hardware and software humanoid): https://github.com/Roboparty/roboto_origin
- `vrpn_mocap` (ROS 2 VRPN client): https://index.ros.org/p/vrpn_mocap/
