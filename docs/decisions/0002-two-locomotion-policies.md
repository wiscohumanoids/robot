# 0002 — Train a velocity policy and a tracking policy

**Context:** The brief names mjlab's `Mjlab-Tracking-Flat-Unitree-G1` (BeyondMimic-style motion tracking). That policy follows one mocap clip. Its command input is the reference motion, not a velocity, so it can't consume `/cmd_vel`. The whole stack (teleop, Nav2) needs a policy that walks wherever `/cmd_vel` says.

**Decision**
1. `Mjlab-Velocity-Flat-Unitree-G1` is the policy behind `/cmd_vel`. It's the one the stack depends on.
2. `Mjlab-Tracking-Flat-Unitree-G1` is trained on LAFAN1 clips as a showcase of BeyondMimic-style tracking (and later "skills" like a dance or a wave). It plugs into the same deploy pipeline because it has the same action contract (joint position targets, same PD gains).
3. Possible later step: make tracking drive `/cmd_vel` by choosing or blending LAFAN1 walking clips based on the commanded velocity. Not in Phase 0.

**Open for Phase 5:** Both stock policies control all 29 joints, arms included. Manipulation needs the arms free, so before Phase 5 we retrain the velocity policy with the arm joints removed from its actions, and random arm motion applied during training so the legs learn to balance whatever the arm does. That's a task-config change in mjlab, not a new algorithm.
