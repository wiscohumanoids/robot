# ACT grasp failures — investigation, 23 September 2026

The current ACT policy learns useful approach motions but lacks reliable grasp precision and recovery. The failure persists in the completed 100,000-step model. Short replanning is especially bad; executing longer chunks helps motion advance but exposes inaccurate future targets. The current evidence does **not** identify a single fatal tensor, units, or simulator bug in the corrected 30 Hz pipeline.

## Confirmed deployment results

These results use the existing `tools/evaluate_act.py`, which launches the normal `rollout_act.py` in a fresh process for each scene. Episodes last 15 simulated seconds at 30 Hz. Seeds begin at 20261201 and were not the eight checkpoint-selection seeds. Both checkpoints use their saved 30-action execution queue.

| Checkpoint | Scenes | Ever grasped a block | Released three-stack held for 0.5 s |
| --- | ---: | ---: | ---: |
| Default `panthera_stack_30hz`, 28,000 steps | 16 | 7/16 | 0/16 |
| `panthera_stack_30hz_rollout_selected/best_rollout`, 100,000 steps | 32 | 19/32 | 1/32 |

Thus the default model never grasped anything in 9/16 scenes; the latest selected model never grasped anything in 13/32. “Ever grasped” is permissive: it does not establish a successful sustained pickup. These sample sizes estimate reliability; they do not establish that rare successes are impossible.

A separate, instrumented comparison uses the same first 16 seeds, fresh environments and rendering at every control tick. Its stricter pickup metric requires **the same cube to remain simultaneously grasped and more than 25 mm above its initial table height for 15 consecutive ticks**.

| Latest checkpoint execution | Sustained pickup | Stable three-stack |
| --- | ---: | ---: |
| Execute all 30 actions, then replan | 8/16 | 1/16 |
| Execute 3 actions, then replan | 0/16 | 0/16 |
| Temporal ensemble, coefficient 0.01 | 8/16 | 0/16 |

The execution comparison is strong evidence that frequent replanning aggravates failure. It does not establish that a one-second queue solves grasping. A saved [normal-rollout failure video](../outputs/act/investigation_20260923/failed_pickup_seed20261201.mp4) shows the 100k model failing to pick up a block.

## 1. Prediction error is large relative to the movement being controlled

I sampled 1,024 frames from training demonstrations and 1,024 from held-out demonstrations for each of the 28k and 100k models. Inference uses observations only and the deployed zero latent. Predicted and demonstrated joint targets are converted to end-effector poses using the same robot's forward kinematics. Padded future targets are excluded.

For the 100k checkpoint:

| Measurement | Training demonstrations | Held-out demonstrations |
| --- | ---: | ---: |
| First target position error, mean | 14.5 mm | 18.2 mm |
| First target position error, 95th percentile | 26.7 mm | 35.2 mm |
| Target position error at one second, mean | 20.5 mm | 59.5 mm |
| First action joint MAE | 1.57° | 2.22° |
| First action gripper MAE | 0.56 mm | 1.30 mm |

The demonstrated first movement is only **6.0 mm on average**, and cubes are **45 mm wide**. These are target-pose errors, not measured simulator tracking errors or a calibrated grasp-tolerance threshold. Nevertheless, the scale is large enough to explain missed alignment and poorly timed closing.

On held-out frames with appreciable demonstrated joint motion, 31% of first predicted joint displacements have a negative dot product with the demonstrated displacement. Copying the current joints has a smaller one-step joint error (0.92°) than the model (2.22°). Holding still is not a useful policy; this comparison shows how weak aggregate imitation error is as a measure of useful short-step control.

The loss trains absolute joint targets and averages normalized L1 over seven coordinates and the entire chunk. It contains no explicit grasp-contact objective or Cartesian precision term. Repeatedly deploying its first few actions can therefore repeatedly apply an inaccurate local target. This mechanism is consistent with the short-queue failure observed above; the results do not prove that a particular alternative representation will fix it.

![ACT target error by prediction horizon](../outputs/act/investigation_20260923/target_error_evidence.png)

## 2. More training mostly improves familiar trajectories

From 28k to 100k, one-second target error falls from **37.8 to 20.5 mm on training frames**, but only **64.9 to 59.5 mm on held-out frames**. At 100k the held-out future error is almost three times the training error. This is a substantial generalization gap, not simply a run that needed its remaining optimizer steps.

The dataset contains 213 demonstrations and 59,211 frames: about 33 minutes of simulation, with 192 episodes used for training and 21 held out. Almost all demonstrations are clean completions: 212/213 end in the geometric three-stack configuration. Of 213 episodes, 204 contain exactly two gripper-close threshold crossings; only nine contain more. There is relatively little repeated-attempt behavior in this measure.

Poor coverage of states reached after a missed grasp is a likely contributor: the policy's own positioning errors take it away from demonstrated trajectories. The dataset audit supports this hypothesis, but an intervention with corrective demonstrations is needed to establish its causal importance. Future-action disagreement can also reflect valid alternate plans; the closed-loop failures are the reason the imitation-error gap matters here.

## 3. Evaluation and defaults obscure how weak the policy remains

- The default checkpoint path still points to the **28k run**, not the newer rollout-selected model. This explains some default-command failures, but the 100k failures show it is not the whole explanation.
- The original run stopped at 28k because held-out imitation loss plateaued. Its best imitation checkpoint was 18k. The later run reached 100k; simply removing early stopping did not produce reliable stacking.
- Checkpoint selection uses only **eight scenes**. The 100k selection report had 7/8 grasps; the new deployment test had 19/32. A small selection set is insufficient to certify reliability.
- `grasp_and_lift` in the current evaluator means “ever grasped AND ever lifted during the episode.” Those events need not overlap or involve the same cube. `lifted` can also count a knocked-up cube.
- `two_stacked` checks relative cube-center geometry and records whether that geometry occurred at any tick. It does not require release or a stable supporting contact. Training ranks this permissive milestone ahead of grasp/lift when full-stack rates tie.

These metrics do not cause action errors, but they can make a weak model look better and distort checkpoint selection. The released, sustained full-stack measure is more meaningful; it remained only 1/32 in the new deployment test.

## 4. Pipeline and simulator checks narrow the cause

- Current source/render provenance and converter hashes pass for all 213 episodes.
- Across **237 sampled Parquet rows spanning every data file**, state and successor-action values exactly match the intended rendered trajectory. Both cached camera images match their source JPEG pixels exactly.
- Saved checkpoint normalization means and standard deviations match the dataset, including state, action and both cameras. Gripper units are consistently metres, converted to simulator opening through division by 0.04.
- Replaying demonstrated controls at deployment frequency picked up and lifted a block in **32/32 episodes**; **30/32** ended in the geometric three-stack configuration. Replay joint tracking errors are small relative to policy target errors. Final replay geometry is not the same metric as released, sustained rollout success.
- All 213 recordings lack measured finger positions and simulation timestamps, so reconstruction remains a real limitation. Successful control replay argues against it being the dominant explanation for failed first pickups.
- On **63 held-out poses**, cached training images versus freshly rendered images gave nearly identical average target error: **18.53 versus 18.39 mm**. The image change shifted predictions by 4.53 mm on average, but did not reveal a systematic camera/encoding failure. Shuffling images worsens prediction error, confirming the model does use visual input.
- All **14 existing tests** pass.

### Evaluation reproducibility caveat

An initial diagnostic rendered only on policy queries and reused its simulator. A normal rollout of a supposedly identical seed behaved differently. I checked this rather than accepting the initial rates.

Two environments driven through identical physics states produced slightly different pixels with different renderer histories; a query-time wrist comparison differed by only 0.25/255 average intensity, yet policy actions subsequently diverged. This is evidence of sensitivity near contact, not proof of the underlying graphics mechanism. The diagnostic now renders every tick and creates a fresh environment per trial. Final deployment rates above come from the normal rollout command in separate processes. Tiny rendering differences can still change individual outcomes.

The earlier `random_queue30`, `replanning`, `demo_queue30`, `validation_best_queue30`, and `anchor_ablation` directories are exploratory artifacts, **not the final evaluation tables**. A pilot that anchored each chunk's first arm target to the current joints achieved 7/16 sustained pickups versus 8/16 for the corresponding unmodified pilot. It provides no evidence that a simple inference-time offset correction fixes the model. The final protocol is in `confirmed_modes` and the two deployment-evaluation directories.

## Recommended next work, in order

1. **Make pickup the next acceptance test.** Use simultaneous, same-cube sustained grasp/lift, and require release plus stability for two-stack credit. Evaluate at least 32 independent scenes during development and a larger untouched set before calling a checkpoint reliable. Keep checkpoint-selection and final-test seeds separate.
2. **Train and evaluate first-pickup completion explicitly.** Use the existing demonstrations through their first successful lift, then collect corrective demonstrations from the failed approach and premature/late-closure states actually visited by the policy. This isolates the failure the user sees before optimizing the entire stacking sequence.
3. **Run a controlled action-representation experiment.** Train relative-to-current arm-action chunks, preserving the same data split and simulator contract, and compare both first-step Cartesian error and actual pickups. Consider extra weight on the immediately executed actions or a Cartesian auxiliary loss. A post-hoc chunk offset is not a substitute for retraining and did not help the pilot.
4. **Then test observation history and visual robustness separately.** The model currently sees one image pair, joint positions and commanded gripper opening, without velocity/history. History may resolve motion/phase ambiguity; modest image perturbations may improve robustness. Both are hypotheses to measure, not established fixes.

Extending the same imitation run or starting RL from it is not yet supported as the best next step: the immediate issue is inaccurate, poorly generalizing grasp control.

## Artifacts and reproduction

[Machine-readable summary](../outputs/act/investigation_20260923/summary.json) includes environment versions, checkpoint hashes, final deployment rates, mode counts and offline evidence. [Diagnostic tool](../tools/investigate_act.py) provides data audit/replay, offline physical-error measurements and instrumented rollout comparisons. Camera and plotting probes are saved alongside the output files.

Run from `/home/zeb/Desktop/RobotArmLearning`:

```bash
.venv-act/bin/python tools/investigate_act.py data \
  --output outputs/act/investigation_repeat

.venv-act/bin/python tools/investigate_act.py offline \
  --checkpoints outputs/act/panthera_stack_30hz \
    outputs/act/panthera_stack_30hz_rollout_selected/best_rollout \
  --output outputs/act/investigation_repeat

.venv-act/bin/python tools/evaluate_act.py \
  --checkpoint outputs/act/panthera_stack_30hz_rollout_selected/best_rollout \
  --episodes 32 --seed 20261201 \
  --output outputs/act/investigation_repeat/deployment

.venv-act/bin/python tools/investigate_act.py rollout \
  --checkpoints outputs/act/panthera_stack_30hz_rollout_selected/best_rollout \
  --episodes 16 --seed 20261201 --modes queue30 queue3 ensemble \
  --output outputs/act/investigation_repeat/modes
```
