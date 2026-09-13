# Integration points — exactly where each team plugs in

This document is written for the three teams working in the *other* repos.
Each section says: what you own, what file/topic/message to target, and what
you must NOT break.

---

## Locomotion (`berkeley_humanoid` -> `locomotion_runner`)

**Target file:** `src/locomotion_runner/locomotion_runner/locomotion_node.py`,
inside the block literally marked:

```python
# === REAL POLICY GOES HERE (load ONNX) ===
```

**What you need to produce:** `policy.onnx`, exported from a trained
`G1Env`/PPO checkpoint (`berkeley_humanoid/train.py` + `play.py`). This repo
does not ship a checkpoint or an export script — that conversion
(`stable_baselines3` `PPO.policy` -> ONNX, plus freezing `VecNormalize`'s
running mean/var into constant scale/offset tensors baked into the ONNX
graph or applied just before/after inference) is `berkeley_humanoid`'s
responsibility, not this repo's.

**The two mismatches you must resolve before the drop-in works:**

1. **DOF count: 29 (yours) vs. 23 (canonical, this repo).** Your model's
   observation assumes 29 joints; `PolicyObservation` here carries 23 (see
   `ARCHITECTURE.md` "Reconciliation with berkeley_humanoid" for why this is
   believed to be a safe truncation, not a retrain). Concretely, when
   building the 29-dim vector your ONNX graph expects from the 23-dim
   `PolicyObservation` this repo hands you:
   - Indices 0-11 (both legs) and index 12 (`waist_yaw_joint`) map 1:1,
     canonical index == your index.
   - Insert zero-position/zero-velocity placeholders at your indices for
     `waist_roll_joint`, `waist_pitch_joint` (right after waist_yaw in your
     29-DOF ordering — confirm exact position against your own
     `g1_env.py` docstring, don't assume ours matches your insertion point).
   - Canonical indices 13-17 (left arm: shoulder pitch/roll/yaw, elbow, wrist
     roll) map to your left-arm indices; insert zero placeholders at your
     `left_wrist_pitch_joint`/`left_wrist_yaw_joint` slots.
   - Same pattern, mirrored, for the right arm (canonical 18-22).
   - When reading the 29-dim action your policy outputs, **drop** the 6
     entries corresponding to those same joints before publishing
     `JointTargets` (which is 23-wide).
   - **Verify, don't assume:** confirm at runtime (e.g. a one-time assertion
     in your adapter) that the dropped 6 action entries are actually
     near-zero for a representative rollout. If a future checkpoint starts
     producing nonzero corrections there, the truncation is no longer safe
     and this repo's 23-DOF hardware genuinely cannot execute that policy
     without either retraining on 23-DOF or adding those 4 extra DOF back
     into `g1_description`.
2. **The policy is not a standalone joint-target generator.** Per the earlier
   `berkeley_humanoid` audit, `G1Env.step()` adds the policy's [-1,1] output
   as a *correction* on top of a hard-coded nominal crouch pose, further
   summed with a hard-coded sinusoidal gait for the leg pitch joints during
   "walk" mode. **That nominal-pose and gait math is not part of the ONNX
   graph** — it lives in Python in `g1_env.py`'s `step()`. You must port that
   exact logic into `locomotion_runner`'s ONNX slot alongside the
   `session.run()` call, or the raw policy output will not produce sensible
   joint targets.

**What you must not change:** `PolicyObservation`'s field layout/order, the
canonical joint order, or the node's 50 Hz timer rate, without updating
`humanoid_interfaces` and `ARCHITECTURE.md` in the same change.

**Command input:** `locomotion_runner` already receives `VelocityCommand`
(`vx`,`vy`,`vyaw`) on `/cmd_vel` and puts it in `PolicyObservation.cmd_*`.
Your G1Env currently only has a binary stand/walk flag — you'll need to
either retrain with a real velocity-conditioned command, or (as an interim
step) threshold `vx`/`vyaw` magnitude into your existing binary flag inside
the adapter. Document whichever you choose in a comment at the call site.

---

## Manipulation (`lerobot_alohamini` -> `manipulation_runner`)

**Target file:** `src/manipulation_runner/manipulation_runner/manipulation_node.py`,
which already implements the `ExecuteManipulation` action server interface
(goal: `task`, `object_id`, `target_pose`; feedback: `phase`, `progress`;
result: `success`, `message`) — currently a stub that reports
`phase="approaching"` -> `"grasping"` -> `"retracting"` on a timer and
returns success. Replace the body of the goal-execution callback with a real
call into your inference loop.

**What you need to produce:** a callable inference step equivalent to
`evaluate_bi.py`'s `observation -> action` loop, wrapped so it can be driven
from a ROS2 action server callback instead of a standalone script.

**Concrete wiring gaps you own:**
1. **Transport.** `lerobot_alohamini` talks to hardware over a bespoke ZMQ
   PUSH/PULL link (laptop <-> Jetson), not ROS2. Either (a) run your existing
   `LeKiwiClient`/`LeKiwiHost` processes as-is and have `manipulation_runner`
   act purely as a ROS2-facing proxy that opens the same ZMQ sockets
   internally, or (b) port the Feetech-bus driver logic in
   `src/lerobot/robots/alohamini/lekiwi.py` into a real `ros2_control`
   hardware component so it can share `controller_manager` with the rest of
   this stack. (a) is far less work and is the recommended starting point;
   (b) is the "real" long-term answer if manipulation ever needs to be
   torque-coordinated with `wbc_stub`/`lowlevel_control` rather than
   independently teleoperated.
2. **Joint order/count mismatch.** Your `LeKiwi` observation/action space is
   16-dim (2x6 arm joints + 3 base velocities + 1 lift height) and is *not*
   the G1's 23-DOF canonical order — AlohaMini is a different physical robot
   (bimanual arm + mobile base) than the G1 humanoid this repo's
   `g1_description`/`JointTargets`/`JointCommand` model. **Do not try to
   force AlohaMini's 16-DOF action space into this repo's 23-DOF
   `JointTargets` message directly.** If/when AlohaMini's arms are mounted
   onto (or replace the arms of) a G1-class humaonid body, that requires a
   real remapping exercise (which physical joints correspond to which
   canonical indices) that has not been done and is out of scope for this
   repo's current scaffold. Until then, treat `manipulation_runner` as
   commanding whatever arm hardware is actually attached, publishing
   `JointTargets` only for the joint indices that exist on that hardware, and
   coordinate with the low-level team on what "the arm indices" concretely
   are for your specific robot.
3. **No IK today, either side.** Neither `lerobot_alohamini` (per its own
   `placo` dependency being unused) nor this repo's `wbc_stub` does
   end-effector-space planning. If `target_pose` (a Cartesian goal) is
   populated in an `ExecuteManipulation` goal, `manipulation_runner` is
   currently expected to have already resolved that into joint-space targets
   upstream (e.g. inside your ported inference loop, which is
   trained/behaves in joint space already per the earlier audit) — this repo
   does not provide an IK solver.
4. **Cameras.** `lerobot_alohamini`'s 4 camera streams are coded but
   disabled by default. If your policy needs them, you are responsible for
   publishing them as standard `sensor_msgs/Image`/`CameraInfo` topics
   yourself — no perception package exists in this repo to consume or
   republish them.

---

## Low-level / EtherCAT (this repo's own `ethercat_bridge`)

This is scaffolded, not delegated to another repo — see
`src/ethercat_bridge/README.md` for the full TODO list. Summary of what "done"
looks like:

1. Fill in `config/ethercat_slaves.yaml` with real `vendor_id`/`product_id`
   values from each drive's datasheet (placeholders are marked
   `0x00000000  # TODO`).
2. Implement the marked TODOs in
   `src/ethercat_bridge/src/ethercat_hardware_interface.cpp`:
   `on_configure()` (EtherCAT master init + slave scan),
   `on_activate()`/`on_deactivate()` (CiA402 Controlword state-machine
   transitions — see `RESEARCH_NOTES.md` for the transition table and its
   confirmed/unconfirmed parts), `read()` (consume TxPDO), `write()` (produce
   RxPDO).
3. Confirm the Statusword decode table against real hardware (flagged as an
   open gap in `RESEARCH_NOTES.md` — do not trust the generic CiA402 table
   blindly).
4. Swap the `hardware_plugin` xacro arg from
   `mujoco_ros2_control/MujocoSystem` to
   `ethercat_bridge/EthercatHardwareInterface` in
   `bringup/lowlevel_test.launch.py` and confirm a real command produces real
   motion, at which point the milestone this repo was built around is done.
5. Set up `PREEMPT_RT`, CPU isolation, and `SCHED_FIFO`/`mlockall` on the
   actual control computer — not part of this repo, tracked so it isn't
   forgotten (see `RESEARCH_NOTES.md` §4).

**What you must not change:** the `effort`/`position`/`velocity` interface
contract in `g1_description`'s `<ros2_control>` block — everything above
`ethercat_bridge` in the stack is written against exactly that interface and
would need to change in lockstep if you do.

---

## Safety

**Interface:** `humanoid_interfaces/SafetyStatus` on `/safety_status`,
published today by `src/safety/safety/safety_node.py` (currently a
software-only stub: it always reports `is_safe=true` unless you manually
publish a `estop_active=true` message for testing, or a future firmware
bridge does it for real).

**What the (separate, out-of-scope-here) wireless E-stop firmware project
needs to do:** publish `humanoid_interfaces/SafetyStatus` with
`estop_active=true` the instant the physical button is pressed, over
whatever transport bridges firmware to ROS2 (a micro-ROS agent, a serial
bridge node, etc. — not specified here, that's the firmware team's
integration choice). `safety_node.py` does not care how the message arrives,
only that it arrives on `/safety_status`. On receipt, `safety_node` sets
`is_safe=false` and `lowlevel_control`'s `JointImpedanceController` (which
subscribes to the same topic) zeroes its torque output and refuses to
re-enable until a subsequent message reports `is_safe=true`. See
`src/safety/README.md`.
