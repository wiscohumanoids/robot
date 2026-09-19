# Integration points — exactly where each team plugs in

Written for every team. Each section says: what you own, which stub you
replace, what interface you must honor, and what you must NOT break.

**The workflow is the same for every team** (details in
[`CONTRIBUTING.md`](CONTRIBUTING.md), the interfaces in
[`INTERFACE_CONTRACT.md`](INTERFACE_CONTRACT.md), where things stand in
[`STATUS.md`](STATUS.md)):

```bash
ros2 launch bringup full_stack.launch.py <your_layer>:=external   # the rest of the stack, as stubs
ros2 run <your_pkg> <your_node>                                    # your real node replaces your stub
ros2 run bringup check_contract.py                                 # did you honor the contract?
```

You never need another team's real node to develop yours, only their stub. If
you need an interface to change, that is a reviewed contract change, not a
private edit -- see CONTRIBUTING.md.

| Team / area | Stub you replace | Section |
|---|---|---|
| Planning | `task_planner`, `behavior_tree`, (`speech_input`) | [Task planning](#task-planning-task_planner-behavior_tree-speech_input) |
| Perception | `perception`, `slam` | [Perception and SLAM](#perception-and-slam-perception-slam) |
| State estimation | `state_estimation` | [State estimation](#state-estimation-state_estimation) |
| Navigation | `nav` | [Navigation](#navigation-nav) |
| Locomotion | `locomotion`, `wbc` | [Locomotion](#locomotion-berkeley_humanoid---locomotion_runner) |
| Manipulation | `manipulation` | [Manipulation](#manipulation-lerobot_alohamini---manipulation_runner) |
| Low-level | `ethercat_bridge` (scaffold) | [Low-level / EtherCAT](#low-level--ethercat-this-repos-own-ethercat_bridge) |
| Safety | `estop_bridge` (missing) | [Safety](#safety) |

---

## Task planning (`task_planner`, `behavior_tree`, `speech_input`)

**You own:** turning an intent into behavior. **You replace:**
`task_planner_stub` (canned plan) and `behavior_tree_stub` (sequential executor).

- **`task_planner`**: subscribe `/user_intent` (`std_msgs/String`) and
  `/object_poses`; publish `humanoid_interfaces/SkillSequence` on
  `/skill_sequence`. Skills use the
  [skill vocabulary](INTERFACE_CONTRACT.md#skill-vocabulary) (`navigate_to`,
  `pick`, `place`, `handover`, ...). Adding a skill name is an interface change.
- **`behavior_tree`** (BehaviorTree.CPP): subscribe `/skill_sequence`; be an
  action **client** of `navigate_to_pose` (`nav2_msgs/NavigateToPose`) and
  `execute_manipulation` (`ExecuteManipulation`); publish `/task_status`
  (`std_msgs/String`, e.g. `RUNNING 2/3 pick` / `SUCCEEDED` / `FAILED ...`) --
  `bringup/smoke_task.py` reads it.
- **`speech_input`** does not exist. It owns `/user_intent`; until then publish by hand:
  `ros2 topic pub --once /user_intent std_msgs/msg/String "{data: 'pick up the cube'}"`.

**Do not:** publish velocity commands yourself (go through `navigate_to_pose`);
publish `/cmd_vel` at all (it belongs to `cmd_vel_mux`).
**Test yours:** `full_stack.launch.py sim:=false planner:=external` (no MuJoCo
needed), then `ros2 run bringup smoke_task.py`.

## Perception and SLAM (`perception`, `slam`)

**You own:** what the robot sees and where it is on the map. **You replace:**
`perception_stub` (a hardcoded cube) and `slam_stub` (empty map, identity TF).

- **`perception`** publishes `humanoid_interfaces/ObjectPoseArray` on
  `/object_poses` at >= 15 Hz (nominal 30), **in the `map` frame** -- do the TF
  transform yourself so consumers never need to -- with stable `object_id`s.
- **`slam`** owns `/map` (`nav_msgs/OccupancyGrid`, transient-local) and the TF
  edge **`map -> odom`** (drift correction). It must **not** publish
  `odom -> base_link`; that belongs to state estimation.
- **Camera frame:** add the camera's mounting as a static edge in
  `g1_description`'s URDF (there is a `d455_link` mesh but no camera frame is
  wired into the model yet) and add it to the TF table in the contract.
- Camera drivers publish standard `sensor_msgs/Image`/`CameraInfo`; those topics
  are not in the contract yet -- add them when you have real ones.

**Test yours:** `full_stack.launch.py perception:=external slam:=external sim:=false`.

## State estimation (`state_estimation`)

**You own:** the robot's pose and velocity estimate. **You replace:**
`state_estimation_stub` (dead-reckons `/cmd_vel`, assuming perfect tracking).

- Publish `/robot_pose` (`geometry_msgs/PoseStamped`, frame `map`) at >= 50 Hz
  (nominal 100) and the TF edge **`odom -> base_link`** -- you are the only
  publisher of both. (The IEKF from its own repo, or `robot_localization`.)
- **Also needed by locomotion:** `PolicyObservation.base_linear_velocity` has no
  real-hardware source and is zero today. Coordinate with the locomotion team on
  how that velocity reaches `locomotion_runner` (a new contract entry).
- Inputs are yours to choose (IMU via `/robot_state`, joint kinematics, vision);
  add any new topic you need to the contract.

## Navigation (`nav`)

**You own:** getting the robot to a goal pose. **You replace:** `nav_stub` (a
go-to-point controller with no obstacle avoidance).

- Serve `navigate_to_pose` (`nav2_msgs/NavigateToPose`) -- the stub already uses
  Nav2's own action name and type, so the behavior tree's client won't change.
- **Remap Nav2's velocity output to `/cmd_vel_nav`** (not `/cmd_vel`).
  `cmd_vel_mux` forwards it. *(Exact Nav2 launch/param wiring for the remap has
  not been written or verified.)*
- Consume `/map` (from `slam`), `/robot_pose` and TF (`map -> odom -> base_link`).
  Nav2's robot base frame is `base_link`.
- A humanoid is not a diff-drive base: Nav2's controller output is only a
  velocity request to `locomotion_runner`, which decides whether the robot can
  follow it. Tune velocity/acceleration limits to what the gait can do.

---

## Locomotion (`berkeley_humanoid` -> `locomotion_runner`)

**Also yours:** `wbc` (`wbc_stub`, the 500 Hz priority merge of locomotion and manipulation targets into `JointCommand`) -- replace with a real whole-body controller behind the same `JointTargets in -> JointCommand out` contract.

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
canonical joint order, or the node's 50 Hz timer rate, without a contract change
(`interface_contract.yaml`, `humanoid_interfaces`, and the docs -- see CONTRIBUTING.md).

**Command input:** `locomotion_runner` receives a standard `geometry_msgs/Twist` on
`/cmd_vel` (published only by `cmd_vel_mux`, 50 Hz; uses `linear.x`, `linear.y`,
`angular.z`) and puts it in `PolicyObservation.cmd_*`. Your G1Env currently only
has a binary stand/walk flag — you'll need to either retrain with a real
velocity-conditioned command, or (as an interim step) threshold `vx`/`vyaw`
magnitude into your existing binary flag inside the adapter. Document whichever
you choose in a comment at the call site.

**Swap in your runner:** `full_stack.launch.py locomotion:=external`, or keep
this node and pass `policy_onnx_path:=...` to use the ONNX slot.

---

## Manipulation (`lerobot_alohamini` -> `manipulation_runner`)

**Target file:** `src/manipulation_runner/manipulation_runner/manipulation_node.py`,
which already implements the `ExecuteManipulation` action server interface
(goal: `task`, `object_id`, `target_pose`; feedback: `phase`, `progress`;
result: `success`, `message`; **called by the `behavior_tree`** for every
non-navigation skill, with `task` = the skill name) — currently a stub that reports
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
   yourself (and adding them to the contract). `perception_stub` fabricates a
   cube and does not consume or republish any camera stream; coordinate with
   the perception team on who owns the camera drivers.

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
published **only by `src/safety/safety/safety_node.py`** at 100 Hz. It combines
two inputs: `/manual_estop` (`std_msgs/Bool`) and a NaN/Inf check of
`/robot_state`.

**What the (separate, out-of-scope-here) wireless E-stop firmware project
needs to do:** publish `std_msgs/Bool` **on `/manual_estop`**, `true` the instant
the physical button is pressed, over whatever transport bridges firmware to ROS2
(micro-ROS agent, serial bridge node, etc. — the firmware team's choice; the node
that does this is the contract's `estop_bridge`, status MISSING).

**Do NOT publish `SafetyStatus` on `/safety_status` yourself.** It has one owner.
`safety_node` publishes `is_safe: true` at 100 Hz whenever it sees no fault, so a
second publisher's `estop_active: true` would be interleaved with those messages
and `lowlevel_control` would flicker back to "safe" between them. (An earlier
version of this document told the firmware to do exactly that; it was wrong.)

On `estop_active` (via `/manual_estop`), `safety_node` sets `is_safe: false` and
`lowlevel_control`'s `JointImpedanceController` (subscribed to `/safety_status`)
zeroes its torque and refuses to re-enable until `is_safe` is true again. See
`src/safety/README.md`.

**Known gaps for whoever owns safety** (also in STATUS.md): `lowlevel_control`
treats "no `/safety_status` ever received" as safe and never times out a stale
`/joint_command`, so a crashed `safety` node or WBC does not zero torque. Add
heartbeat/staleness timeouts before any hardware run. The software E-stop path is
not a substitute for a hardware E-stop.
