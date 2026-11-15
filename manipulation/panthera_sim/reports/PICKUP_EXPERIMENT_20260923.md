# First-pickup ACT experiment

Launched 23 September 2026. The experiment tests whether predicting arm targets relative to the chunk's starting joint pose improves first-pickup reliability over a matched absolute-target ACT model.

## Fixed protocol

| Setting | Both runs |
| --- | --- |
| Demonstrations | Same 213 recordings, cropped through the first sustained lift |
| Frames | 14,353 training; 1,545 validation |
| Split | Original 192/21 episode split, seed 20260922 |
| Initialization | Fresh ACT with ImageNet ResNet-18; same random seed |
| Training budget | 30,000 updates × batch 12, approximately 25 passes over the pickup windows |
| Control | 30 Hz; 30-action chunks; execute 30 actions per query |
| Offline validation | All held-out pickup frames every 2,000 updates, including physical target error |
| Checkpoint selection | Sustained pickup on 32 fixed scenes every 10,000 updates; capped pickup time breaks ties |
| Final test | 64 different fixed scenes per selected model, 8 seconds per scene |
| Scheduling | Absolute baseline first, then relative targets, on the same GPU |

The arm target in the relative run is `future_command[:6] - observed_joints[:6]`, anchored to the observation at the start of the entire chunk. During deployment the starting joint pose is added to **every action in that chunk before queueing**. Gripper commands stay absolute in metres. Each representation's action scaling is fitted only on valid training targets. Targets after the pickup crop boundary are padded and masked; stacking actions do not enter the pickup objective.

Because the legacy recordings have no grasp-latch flags, demonstration cropping uses a closed gripper command and the same cube elevated by more than 25 mm for 15 frames. Evaluation is stricter: the same cube must be actively grasped and elevated simultaneously for 15 consecutive control ticks. A tossed block or separate grasp/lift events cannot satisfy this metric.

No new expert corrective demonstrations are fabricated. Evaluation saves physical state traces for failed sustained pickups so subsequent corrective demonstrations can start from observed failure states.

## Progress and results

- [Run status](../outputs/act/pickup_ablation_20260923/run_status.json)
- [Absolute training log](../outputs/act/pickup_ablation_20260923/absolute/train.log)
- [Saved protocol and source hashes](../outputs/act/pickup_ablation_20260923/protocol.json)
- [Launch record](../outputs/act/pickup_ablation_20260923/launch.json)

On completion, `comparison.json` in the run directory contains each selected model's final-test rates, its physical validation errors, and paired scene outcomes. Checkpoints are stored under each representation's `best_rollout/`. Final-test scenes are not used to pick another checkpoint. Compare physical errors and sustained pickup rates across representations; their normalized training losses use different action scales and are not directly comparable.

The runner records a failed phase and the corresponding log if any child command fails. It retains two periodic resumable checkpoints per run, plus the best imitation and best rollout checkpoints.

## Validation before launch

All 21 tests passed, including crop-boundary masking, training-only normalization, query-time anchoring, temporal averaging of reconstructed absolute targets, and same-cube sustained pickup. A two-update end-to-end smoke experiment completed both representations, checkpoint reload, selection evaluation, final testing, and paired comparison generation.

The setup also fixed a cache initialization issue: Hugging Face's custom column transform was decoding every image while LeRobot built an index mapping. Cached image columns are now removed before that mapping is built; the existing decoded image cache is unchanged.

## Run a separate repetition

From the repository root, using a new output directory:

```bash
.venv-act/bin/python tools/run_pickup_experiment.py \
  --output outputs/act/pickup_ablation_repeat \
  --steps 30000 --selection-episodes 32 --test-episodes 64
```

Use `rollout_act.py` and `tools/evaluate_act.py` for the new relative checkpoints; deployment reads the saved action representation and reconstructs the correct absolute joint targets. The existing RL trainer explicitly rejects relative checkpoints because its action-head update path does not implement that representation.
