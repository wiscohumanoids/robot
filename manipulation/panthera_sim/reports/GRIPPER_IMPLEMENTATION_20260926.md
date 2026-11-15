# Physical gripper implementation — 2026-09-26

The default `PantheraSim` now uses **contact-v2**:

- Arm position commands interpolate from the previously applied command to the next endpoint over the actual physics steps in each control interval. Zero-step updates preserve the previous applied command. Reset and explicit snapshot restoration initialize that state.
- The gripper's existing 3 N actuator limit and pad-pair friction are unchanged. `impratio` is 100 in both the generator and generated robot XML.
- Object grasp welds are never activated in this mode. The inactive XML constraints and historical implementation remain available only through explicit `weld-v1` mode.
- `grasp_flags()` requires at least 0.01 N on both pads for 20 ms. Contact loss clears the flag immediately. An open command alone does not count as physical release. These read-only measurements never apply forces.
- Collection, ACT/pi0.5 evaluation, RL observations/rewards, and diagnostics use the shared grasp-state API. Existing sustained elevation and released-stack checks remain in place.

New episodes, rendered sources, dataset exports, and checkpoint deployment contracts carry `simulation_dynamics`. Export rejects mixed versions. Untagged historical recordings replay as `weld-v1`; old recordings/checkpoints are not rewritten. New rollouts default to contact-v2, with `--dynamics weld-v1` for explicit comparisons. ACT/pi0.5 reports distinguish runtime from training dynamics. RL resumes cannot silently cross physics versions; loading old actor weights as a fresh run remains supported.

The runtime bundler includes the new shared dynamics module. Its existing check against training-scene hashes still prevents accidentally claiming a changed simulator matches an old dataset. Existing published runtime archives are not updated by this local code change.

## Validation

**Production physics benchmark:** seven nominal scenarios all retained the block after lifting and shaking. A 10× payload and a 0.3 N weak-gripper case failed to hold, as expected. Opening to 65% released the block even though the historical weld remained active at that command. No MuJoCo warnings occurred.

| Scenario | Lift slip | Displacement after shake/recovery |
|---|---:|---:|
| Seated 55° | 0.179 mm | 1.388 mm |
| Seated 30° | 0.254 mm | 2.064 mm |
| Shallow 30° | 0.258 mm | 2.865 mm |
| Sudden lift target, with existing IK step limit | 0.654 mm | 1.868 mm |

The seated 55° 60-second hold drift was **0.0345 mm**.

**Production collector:** `Planner(seed, arm_start='fixed').run()` succeeded on **16/20 seeds (0–19)**, matching the earlier weld baseline. Failures on 4, 5, 9, and 11 were arm-tracking checks. No run activated a grasp weld or emitted a MuJoCo warning. This is a regression sample, not a guarantee of arbitrary-pose or learned-policy success.

**Historical replay:** existing `data/scripted_stack/episode_0000` and `episode_0001` replayed successfully using their inferred `weld-v1` mode, with both manipulated blocks recognized and a released final stack sustained for more than one second.

**Evaluation integration:** the actual pi0.5 `run_episode` loop replayed newly generated scripted commands for seed 230927, rendered both cameras, detected grasps/lifts, and recognized the released three-stack at step 675. Its trace had physical grasp flags and no active grasp welds. This exercises the evaluation pipeline, not learned pi0.5 weights.

**Automated tests:**

```sh
OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl .venv-act/bin/python -m pytest -q tests
# 61 passed, 1 skipped
```

The skip requires the notebook's pinned LeRobot pi0.5 environment. After tightening the release metric to depend entirely on actual contact load, the 14 gripper and scripted-collection tests were rerun and passed. Tests cover interval interpolation, pending targets over zero physics steps, reset/snapshot initialization, legacy stepped commands, low-slip lifts/shakes, overloaded/weak grasps, partial opening, contact persistence/loss, dynamics metadata, mixed-version rejection, and native recording/replay.

The base Python environment cannot collect the complete training suite because it lacks LeRobot/h5py and has incompatible Torch/Torchvision packages; the configured `.venv-act` environment passes it. An additional learned ACT rollout was attempted but could not load a model: the local ACT checkpoint directories retain metadata but contain no `model.safetensors`. No replacement weights were downloaded and no learned-policy performance claim is made.

## Reproduce the production benchmark

```sh
OPENBLAS_NUM_THREADS=1 python tools/investigate_gripper.py \
  --variants production \
  --scenarios seated55 seated30 shallow30 offset8 yaw30 yaw45 jump55 heavy55 weak55 partial_open55 \
  --output reports/gripper_implementation.json

OPENBLAS_NUM_THREADS=1 python tools/investigate_gripper.py \
  --variants production --scenarios seated55 --hold-seconds 60 \
  --output reports/gripper_production_long_hold.json
```

Other artifacts: `gripper_production_stacks.json` contains the direct production collector results; `gripper_evaluator_smoke.json` records the evaluation-loop check. The original investigation's experimental variants explicitly use historical stepped/weld settings so they remain valid comparisons after the production default changes.

This correction preserves the action representation and control frequency. Linear interpolation is not a universal acceleration/jerk bound for arbitrarily discontinuous policy commands; actuator limits and physical contact still allow blocks to drop when motion or load exceeds the grip.
