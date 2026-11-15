# 0004 — v1 policy ignores small commands; Nav2 must command above the deadband

**Finding (2026-11-08):** `g1_velocity_flat_v1` stands still for small commands: pure turns below ~0.6 rad/s, pure sidesteps below ~0.4 m/s, slow walking (≤0.2 m/s) with a turn. Larger commands track, but turning in place undershoots (1.0 → 0.56–0.72 rad/s); sidestep 0.6 → 0.54 m/s and turning while walking track well. Reproduced identically in mjlab's own environment, so it's the policy, not our deployment.

**Cause:** reward balance. Velocity tracking uses a wide tolerance (exp kernel, std 0.5 m/s and 0.71 rad/s), so standing still while asked for 0.3 m/s still earns ~70% of the tracking reward, while stepping costs foot-clearance, swing-height, slip, and posture penalties.

**Decision**
1. Keep v1. Walking, teleop, and the Phase 1 milestone are unaffected.
2. Phase 3: configure Nav2 to stay outside the deadband: rotate-to-heading at ≥ 0.8 rad/s, minimum translational speed ~0.3 m/s while moving, and don't rely on tiny corrective commands near the goal (use a goal tolerance instead).
3. If precise low-speed control is needed, retrain with a tighter tracking std (e.g. 0.25) and/or higher tracking weight. Do it in the same run as the arms-free policy Phase 5 needs (decision 0002), so the GPU is used once.

**Found in Nav2 tuning (2026-11-08):** acceleration limits that ramp from the robot's *measured*
velocity deadlock with the deadband: a standing robot is asked for 0.1 rad/s, ignores it, keeps
standing, and is asked for 0.1 rad/s again. RPP's `max_angular_accel` is therefore set high (20);
smoothing is left to the velocity smoother, which ramps from the previous *command* (open loop).
