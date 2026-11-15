# Gripper slipping investigation — 2026-09-26

Implementation follow-up: the recommendation was subsequently implemented and
validated in [GRIPPER_IMPLEMENTATION_20260926.md](GRIPPER_IMPLEMENTATION_20260926.md).
The investigation below describes the state and experiments before that change.

**Recommendation:** replace the grasp weld with ordinary contact physics, interpolate arm commands at the physics rate, and raise `impratio` from 10 to 100 to suppress remaining slow creep. Keep the existing 3 N actuator limit initially. The principal failure is acceleration caused by stepped position commands, not inadequate static friction or too few contact points.

This is an experimentally tested recommendation, not a production migration. Only the investigation tool and reports were added. The scene, simulator, collector, training, and rollout behavior remain unchanged.

## Evidence at a glance

Tested the actual Panthera scene with MuJoCo **3.13.0**, a **2 ms** physics step, and **30 Hz** commands. There were **137 targeted runs plus 100 randomized three-stack trials**, excluding the initial duplicate screening runs. All completed without MuJoCo warnings or nonfinite states.

| Controller / physics | Lift slip, seated 55° | Lift slip, seated 30° | Lift slip, shallow 30° | Released three-stack success |
|---|---:|---:|---:|---:|
| Current weld | 0.360 mm | 0.310 mm | 0.287 mm | 16/20 |
| Remove weld only | 10.395 mm | 10.951 mm | 16.939 mm | 5/20 |
| Remove weld; 6 N actuator | 6.581 mm | 4.871 mm | 11.037 mm | Not tested |
| Remove weld; interpolate commands | 0.181 mm | 0.256 mm | 0.258 mm | 16/20 |
| Interpolate; `impratio=100` | 0.179 mm | 0.255 mm | 0.258 mm | 16/20 |
| Interpolate; `impratio=100`; 6 N | 0.180 mm | 0.255 mm | 0.258 mm | 16/20 |

Slip is the change in object position **in the wrist frame**, relative to the end of closing. It is not inferred from the latch flag. This avoids interpreting wrist rotation as object translation. A successful stack must be released, remain stable for one second, and meet the production planner's 12 mm horizontal alignment limit.

## 1. What the current assist actually does

`sim/make_mjcf.py:add_grasp_contacts` creates an inactive six-degree-of-freedom weld between each cube and `link6`. `sim/panthera_env.py:_update_grasp` activates it when:

- the commanded opening is below 25%;
- the cube touches both finger pads;
- the jaw closing axis is within 25° of a cube axis.

The capture pose becomes the weld reference. Once active, it only releases when commanded opening exceeds 70%. Contact loss, actual jaw separation, contact load, payload, and acceleration do not break it. Its softness permits small errors; it does not impose a friction-limited grasp wrench or simulate deforming rubber.

**Direct reproduction:** after lifting, command opening to 65%. The fingers separate to approximately +25.81/-25.79 mm. Both pad contact counts and normal forces become zero. The weld still holds the cube at **z=192.49 mm**. With the proposed no-weld controller the cube falls to the table at **z=72.48 mm**. See `gripper_partial_open.json`.

The weld also holds a 10× payload and a cube with a 0.3 N actuator in this benchmark. The proposed controller at the unchanged 3 N limit drops the overload and fails to hold with the weak actuator. These negative controls matter: reliable nominal grasps should not require unlimited attachment forces.

## 2. The dominant problem is the command waveform

The arm has stiff position servos, e.g. joint2 `kp=5000`, with torque limits. The collector, ACT rollout, pi0.5 evaluator, and RL environment set a complete new arm target and then run approximately 16–17 physics steps with that target unchanged. Teleoperation similarly holds each target over its variable-duration update interval.

A smooth sequence of 30 Hz waypoints is therefore a staircase at the actuator input. The existing IK joint-step limit bounds the size of a target jump; it does not distribute that jump across the intervening physics steps.

Measured TCP accelerations during the same one-second, 120 mm lift:

| Grasp setup | Stepped target peak | Interpolated target peak |
|---|---:|---:|
| Seated 55° | 67.61 m/s² | 1.16 m/s² |
| Seated 30° | 55.26 m/s² | 1.11 m/s² |
| Shallow 30° | 68.97 m/s² | 1.12 m/s² |

The shake peaks similarly fall from approximately **99–104 m/s²** to **9.6–10.3 m/s²**. These are finite differences of TCP position sampled every physics step; they quantify simulated motion, not measured hardware acceleration. See `gripper_acceleration.json`.

The ablation changes only the waveform between consecutive arm targets:

```python
for j in range(1, physics_steps + 1):
    q_servo = q_previous + (q_next - q_previous) * j / physics_steps
    sim.set_arm_ctrl(q_servo)
    sim.step(1)
```

All no-weld candidates use the same incoming commands per targeted scenario. There is no object-position correction, teleportation after initialization, force injection, attachment, or increased grip strength in the interpolation-only candidate. Gripper commands retain their original timing.

Across seven nominal scenarios—55° and 30° seating, shallow seating, an 8 mm lateral offset, cube yaw errors of 30° and 45°, and a sudden lift target—interpolation retains all seven cubes after the shake. The largest final displacement from the captured closing pose is **2.84 mm**, versus **17.86 mm** for doubling the force alone and **31.25 mm** for increasing sliding friction to 2 alone. Removing the weld alone retains only two of seven.

Here “retained” means the cube is still at least 40 mm above its initial height and both pads carry more than 0.01 N at the end of recovery. Retention is weaker than accurate placement; the displacement measurements are reported separately. A skewed grasp can legitimately hold a light cube, so a fixed angular rejection threshold is not a physical success criterion.

## 3. The nominal clamp-force description is wrong

`GRIPPER_FORCE=3` is documented as “per-finger.” In the model there is **one actuator on the left slide** and an equality coupling the right slide to it. Measured steady loads are approximately **1.5 N at each pad**, with the actuator saturated at -3 N. A -6 N limit produces approximately 3 N per pad.

The 45.56 g cube weighs approximately **0.447 N**. With sliding friction 0.8 and two 1.5 N contacts, the simple translational friction budget is approximately **2.4 N**. That comfortably exceeds static weight but corresponds to only about **42.9 m/s²** upward acceleration after gravity, before accounting for contact geometry, moments, or intermittent contact. Measured stepped-command acceleration peaks exceed this rough bound.

Doubling force helps, but it treats the acceleration symptom and still leaves multiple millimeters of lift slip. There is no verified hardware specification here establishing that 6 N is the intended actuator limit. Interpolation works at the existing force and is the better first change.

## 4. Remaining creep and misleading friction settings

Interpolation solves the large motion-induced slip. It does not remove MuJoCo's regularized-contact creep. In a 60-second seated 55° hold:

| No-weld configuration | Wrist-relative drift during hold |
|---|---:|
| Stepped commands, `impratio=10` | 0.418 mm |
| Interpolated commands, `impratio=10` | 0.300 mm |
| Interpolated commands, `impratio=100` | **0.035 mm** |

The scene already uses the Newton solver, elliptic cones, and `implicitfast`. Increasing `impratio` hardens friction relative to the normal contact constraint; it does not raise the friction coefficient. MuJoCo explicitly distinguishes slow regularization creep from inadequate friction force, geometry, and vibration. Its current guide recommends investigating these separately. [MuJoCo: preventing slip](https://mujoco.readthedocs.io/en/stable/modeling.html#preventing-slip).

Two additional source-code traps were confirmed:

- `PAD_FRICTION="2.0 0.2 0.01"` does **not** govern the explicit pad–cube pairs. Their effective coefficients are **`0.8 0.8 0.01 0.001 0.001`**. Increasing the geom coefficient alone would not change these contacts.
- Comments say MuJoCo uses the per-axis *minimum* of geom friction. For equal-priority dynamically generated contacts, it uses the **maximum**. Actual cube–table contacts are **`0.8 0.8 0.08 0.004 0.004`**, despite the table's sliding coefficient being 0.3. A table with truly lower friction needs explicit pairs or deliberate priority settings. This is a separate modeling correction, not required for the demonstrated interpolation improvement. [MuJoCo: contact parameters](https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters).

## 5. Alternatives evaluated

| Alternative | Finding |
|---|---|
| Raise `impratio` to 100 or 1000 alone | Reduces creep; does not remove large lift/shake slip. 1000 is unnecessary for the proposed first change. |
| `noslip_iterations=3` or `10` alone | Does not rescue the stepped-command failures; each retained only 1/7 nominal cases. Not a replacement for a controller with feasible accelerations. |
| Pad-pair impedance 0.999 | Retained 0/7 under the stepped trajectory; “make contacts harder” is not sufficient. |
| 6 N actuator alone | Retained 7/7, but worst final displacement was 17.86 mm. |
| Sliding friction 2 alone | Retained 7/7, but worst final displacement was 31.25 mm; requires material justification. |
| Interpolation alone | Retained 7/7; lift slip under 0.26 mm for the three standard setups; matched the weld in stacking. |
| Interpolation plus `impratio=100` | Same stack success; much lower long-hold creep. Preferred candidate. |

NoSlip is a friction-only postprocessing solver, not a weld; it could be considered later for stricter creep requirements. MuJoCo notes additional cost, possible multi-contact instability, and ill-defined inverse dynamics. [MuJoCo: softness and slip](https://mujoco.readthedocs.io/en/stable/overview.html#softness-and-slip).

The current box pads already generate multiple contacts—up to six per pad in the measured closing snapshots. There is no evidence here that a single-point convex-mesh contact is the remaining bottleneck. The earlier replacement of concave finger collision hulls with flat pads remains sensible. New pad geometry, compliant surface meshes, adhesion actuators, or a different simulator would add substantial modeling work without addressing the demonstrated input discontinuity. Adhesion also models a different gripping mechanism from a passive parallel jaw.

## 6. Randomized stacking validation

The harness uses the existing `Planner.move`, `hold`, `tick`, IK, randomized objects, fixed arm start, motion speed distribution, and red-on-green / blue-on-red order. Its stack driver mirrors `Planner.run`, replacing its weld-dependent acquisition assertion with bilateral contact. No success flags are fabricated. Seeds **0–19** are all included without rejection/resampling.

The weld, interpolation, interpolation + `impratio=100`, and interpolation + `impratio=100` + 6 N each succeeded on exactly **16/20 seeds**. They share failures on **4, 5, 9, 11**, all triggering arm tracking-error checks. This does not establish that every such failure is pure kinematic unreachability; collision or planning can also affect tracking. Removing the weld without interpolation succeeds on **5/20**.

The production planner was also run independently on seeds 0, 4, and 9: the success and failure stages matched the harness's weld baseline. No broad claim of a 100% robust stacker is justified by these tests. The evidence supports replacing grasp assistance without losing the baseline's success on this sample.

## 7. Concrete production design

1. **Introduce one shared arm-command interpolation path.** Give it the destination joint target and the actual number of physics steps for the current control interval. Interpolate from the previous applied command, not the measured joint state. Use it consistently in teleop, collection, ACT/pi0.5 rollout, and RL. Preserve `PhysicsClock`'s 16/17-step scheduling; do not silently change the 30 Hz action contract.
2. **Keep all object welds inactive in the new mode.** Retain legacy assisted mode only for explicitly identified old recordings/checkpoints. The new mode should have no hidden object-holding forces.
3. **Set `impratio=100`; retain current pair friction, force limits, timestep, and solver initially.** Separate later calibration from this controller correction. Add explicit velocity/acceleration limits for adversarially discontinuous policy commands: the benchmark's abrupt-target case still used the existing 0.08 rad-per-command IK limit, and linear interpolation is not a universal acceleration bound.
4. **Replace latch-dependent grasp metrics.** Provide a shared read-only per-object grasp-state API using bilateral *loaded* contacts, closing intent, and persistence. Validate a lift through elevation and relative-motion stability over time. Do not use the metric to apply forces. Mere collision records can include ineffective contacts, and a latch bit does not prove a physical grasp.
5. **Version controller/physics semantics in datasets and checkpoints.** Save high-level action targets separately from the applied servo trajectory when needed. Initialize interpolation state on reset and replay, account for its control-interval delay, and make legacy playback explicit. Existing demonstrations were collected with welds and stepped targets; do not silently claim they were generated under the new dynamics.

Latch dependencies currently exist in `tools/collect_scripted.py`, `tools/validate_scripted_dataset.py`, `rollout_act.py`, `tools/evaluate_pi05.py`, `train_act_rl.py`, and several ACT diagnostics and tests. Removing the weld without updating those consumers would report false grasp failures and alter reward/observation semantics. This is why changing only XML would be incomplete.

Before making the new mode the default, extend validation to learned-policy rollouts, variable teleop frame times, rotations while carrying, more seeds, timestep sensitivity, and the actual intended hardware grip-force calibration. The 2.3.7 environment is for the separate ACT reference simulator and was not used for these Panthera results.

## Reproduction and artifacts

Run from the repository root using `python` or `.venv-act/bin/python` (both reported MuJoCo 3.13.0 here). The tool changes model parameters only in memory.

```sh
OPENBLAS_NUM_THREADS=1 python tools/investigate_gripper.py \
  --variants weld contacts impratio100 impratio1000 noslip3 noslip10 pad_impedance force6 friction2 impratio100_force6 \
  --scenarios seated55 seated30 shallow30 offset8 yaw30 yaw45 jump55 heavy55 weak55 \
  --output reports/gripper_ablation.json

OPENBLAS_NUM_THREADS=1 python tools/investigate_gripper.py \
  --variants interpolate interpolate_force6 interpolate_impratio100 interpolate_impratio100_force6 \
  --scenarios seated55 seated30 shallow30 jump55 yaw30 yaw45 \
  --output reports/gripper_interpolation.json

OPENBLAS_NUM_THREADS=1 python tools/investigate_gripper.py \
  --variants interpolate interpolate_force6 interpolate_impratio100 interpolate_impratio100_force6 \
  --scenarios offset8 heavy55 weak55 --output reports/gripper_interpolation_limits.json

OPENBLAS_NUM_THREADS=1 python tools/investigate_gripper.py \
  --variants contacts interpolate --scenarios seated55 seated30 shallow30 \
  --physics-trace --hold-seconds 2 --output reports/gripper_acceleration.json

OPENBLAS_NUM_THREADS=1 python tools/investigate_gripper.py \
  --variants contacts interpolate interpolate_impratio100 --scenarios seated55 \
  --hold-seconds 60 --output reports/gripper_long_hold.json

OPENBLAS_NUM_THREADS=1 python tools/investigate_gripper.py \
  --variants weld interpolate_impratio100 --scenarios partial_open55 \
  --output reports/gripper_partial_open.json

OPENBLAS_NUM_THREADS=1 python tools/investigate_gripper.py \
  --variants weld contacts interpolate interpolate_impratio100 interpolate_impratio100_force6 \
  --stack-seeds 20 --output reports/gripper_stacks.json
```

The `force6` variants explicitly override the weak-force scenario; they are not weak-grip negative controls. Long-hold drift for a cube already dropped is not meaningful; inspect phase heights and loaded-contact fractions alongside drift. The seven-case retention results exclude overload, weak-force, and partial-opening controls. These are deterministic diagnostic cases, not statistically representative real-world success estimates.
