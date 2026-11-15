# ACT single-IK-demo overfit diagnostic — 23 September 2026

**The reliable-overfit sanity check did not pass.** The final model completed the training scene once in the training process, but failed both fresh-process reload evaluations from the same initial physics state. This establishes neither a specific implementation bug nor an inability to eventually fit the demo. It establishes that this run is not a reliable memorization success.

## Protocol

- Existing scripted IK data: `outputs/lerobot/panthera_scripted_stack_30hz`, episode index 0, native `data/scripted_stack/episode_0000`, generator seed 230925.
- One full demonstration, 994 observation/action pairs, about 33.1 seconds. Gradient updates use only this demo. Existing full-dataset normalization is preserved.
- Fresh standard ACT, two 256×256 cameras, absolute joint/gripper targets, 30-action chunks and execution queue at 30 Hz. ImageNet backbone initialization; no existing ACT weights at the start.
- Initial physics state regenerated with the original hashed IK generator, stopped immediately after the first recorded tick. All recorded fields match exactly, including arm/finger positions and velocities, object poses, controls and simulation time. Regeneration retains unrecorded simulator internals too.
- Recorded successor-action replay completes a released three-stack. The diagnostic checks all loaded states/actions against the rendered trajectory. Original `CachedLeRobotDataset` successor actions and padding also match at frames 0, 200, 500 and 993.
- Offline measurements use observations only, evaluation mode, and zero latent. Final measurements cover all 994 training observations, excluding padding.
- Policy rollouts allow 1,192 ticks / 39.73 seconds, 20% longer than the demo. Success requires a released three-stack held for 0.5 seconds. Same-cube grasp/lift must persist for 0.5 seconds for pickup credit.

## Training and results

| Run | Updates | Settings | Full-demo mean / p95 first-target error | Full-stack result |
| --- | ---: | --- | --- | --- |
| Standard | 0–3,000 | LR 1e-5, dropout 0.1, AMP, VAE KL weight 10 | 18.74 / 30.64 mm | Failed |
| Overfit continuation | 3,001–5,000 | LR 3e-5, dropout 0, FP32, same VAE KL weight | 7.92 / 16.53 mm | In-process success; both reloads failed |

A separate attempted continuation at LR 1e-4, dropout 0, AMP hit a nonfinite loss shortly after update 3,100. It was abandoned; the FP32 continuation starts from the preserved standard 3,000-step checkpoint, not the failed attempt.

The final normalized chunk L1 is 0.05249, first-action joint MAE 0.893 degrees, and first-action gripper MAE 0.786 mm. The successful rollout reaches the sustained-stack criterion at 35.2 seconds, slightly longer than the original demo. Both failed reloads grasp/lift red and leave red on green, but never grasp blue.

The final checkpoint therefore has **one successful attempt and two failed attempts on the same scene**. These are repeated runs of one scene, not three independent demonstrations or a generalization estimate.

## Why 5,000 updates are concerning, but not a root-cause proof

At batch size 12, 5,000 updates amount to approximately 60 passes over the 994-frame demo. That is a substantial sanity-test budget. Yet the final policy still has nearly 8 mm average target error on the observations it was trained on; accurate trajectory fitting has not been achieved.

The first 3,000 updates retained normal training regularization and a low learning rate. At update 3,000, minibatch action L1 was 0.0946 and weighted KL was 0.3538. This scalar-loss comparison does not establish which term dominates parameter gradients. Disabling dropout, changing precision and raising LR together improved fitting, but these simultaneous changes do not isolate the cause. The final training loss was still improving. A clean deterministic fixed-batch/zero-latent optimization test is the next way to isolate the optimizer/objective from rollout control, rather than increasing the dataset.

Reloaded trajectories expose additional brittleness: their first 150 actions and resulting joint states match the successful in-process run exactly. At tick 150 (5 seconds), the first differing action changes by only 7.7486e-6 in maximum absolute coordinate difference. The grasp flags eventually disagree at tick 565 (18.83 seconds). Reloads also differ from one another starting at tick 150. The source of that tiny perturbation has not been isolated between rendering and numerical inference; it is not evidence of a wrong initial physics state or wholesale checkpoint-loading mismatch.

Saved and live images at the initial state align visually. Mean RGB differences are 1.69/255 for shoulder and 1.32/255 for wrist. This limited check rules out an obvious initial camera framing mismatch; it does not rule out later rendering sensitivity.

## Artifacts

- Reusable diagnostic: [tools/overfit_act.py](../tools/overfit_act.py)
- Standard-run results: [result.json](../outputs/act/ik_overfit_20260923/result.json)
- Continuation results: [result.json](../outputs/act/ik_overfit_20260923_fp32/result.json)
- Combined final summary: [summary.json](../outputs/act/ik_overfit_20260923_fp32/summary.json)
- Successful in-process video: [rollout_005000.mp4](../outputs/act/ik_overfit_20260923_fp32/rollout_005000.mp4)
- Reload failure: [rollout.mp4](../outputs/act/ik_overfit_20260923_fp32/reload_1/rollout.mp4)
- Second reload failure: [rollout.mp4](../outputs/act/ik_overfit_20260923_fp32/reload_2/rollout.mp4)
- Final checkpoint: `outputs/act/ik_overfit_20260923_fp32/checkpoint`.

The 18 existing ACT tests pass. The new harness was exercised through training, recorded-action replay, complete offline evaluation and fresh-process checkpoint evaluation. Existing full-dataset training and its checkpoints were not modified.

## Reproduction

Use a fresh output directory for each run:

```bash
OPENBLAS_NUM_THREADS=1 .venv-act/bin/python tools/overfit_act.py \
  --output outputs/act/ik_overfit_repeat --steps 3000

OPENBLAS_NUM_THREADS=1 .venv-act/bin/python tools/overfit_act.py \
  --resume outputs/act/ik_overfit_repeat/checkpoint \
  --output outputs/act/ik_overfit_repeat_fp32 \
  --dropout 0 --lr 0.00003 --no-amp --steps 5000

OPENBLAS_NUM_THREADS=1 .venv-act/bin/python tools/overfit_act.py \
  --evaluate-only outputs/act/ik_overfit_repeat_fp32/checkpoint \
  --output outputs/act/ik_overfit_repeat_reload
```
