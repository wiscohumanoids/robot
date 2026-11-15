# ACT IK overfit: controlled root-cause tests

The single-demo failure has identifiable causes, rather than evidence of broken ACT backpropagation. The optimizer update size materially limited training precision, the scripted data contains timing-dependent labels that a single observation cannot uniquely determine, and multisampled rendering caused otherwise identical evaluations to diverge.

The strongest controlled fitting result is a learning-rate comparison: from identical weights, with the same training observations, minibatch order, loss and 1,000 additional updates, LR 3e-6 produces **3.15 mm** mean first-target error versus **9.44 mm** at LR 3e-5. The lower-rate checkpoint completed the stack in its in-process evaluation and both fresh-process reloads. These are three executions of one training scene, not three independent scenes or a generalization estimate.

This diagnoses the one-demo sanity test. It does not establish every cause of failure in the separate 1,000-demo policy.

![Controlled evidence](../outputs/act/ik_root_cause/evidence.png)

## 1. Optimization was preventing a sufficiently precise fit

All full-demo probes start from the same 5,000-update, dropout-free checkpoint from the earlier investigation. They use one IK demonstration, 994 frames, batch 12, FP32, seed 20260924, fresh AdamW, weight decay 1e-4, and the same absolute 30-action representation. The clean LR comparison trains the deployed zero-latent path directly with the stock padded L1 reduction. There are no new observations, demonstrations, architecture changes, or simulator changes in that comparison.

| Controlled continuation | Updates | Full-demo chunk L1 | Mean / p95 first-target error | Default-renderer stack completions |
| --- | ---: | ---: | ---: | ---: |
| Zero latent, LR 3e-5 | 1,000 | 0.04955 | 9.44 / 21.57 mm | 2/3 |
| Zero latent, LR 3e-6 | 1,000 | 0.01981 | 3.15 / 7.30 mm | 3/3 |
| Original VAE objective, LR 3e-6 | 500 | 0.02239 | 3.57 / 8.90 mm | 0/1 |

The high-rate model failed in-process but succeeded in both reloads. Therefore, high LR does not make task completion impossible; the rigorously isolated result is its substantially poorer fit. Three attempts are insufficient to certify reliability or a statistically meaningful success-rate difference. The VAE run used only half as many updates; its task result is not a matched objective comparison. At 250 and 500 updates, its fitting error closely matches the corresponding low-rate zero-latent run. Removing the VAE is not required to obtain that precision improvement.

The first experiment retained dropout 0.1 and LR 1e-5 for 3,000 updates, then **I raised the continuation LR to 3e-5**. That was an overfit-debugging choice, not the LR of the user's separate full-dataset run. The controlled result shows this continuation rate was too coarse for fine fitting at these weights. It does not prove that the original LR 1e-5 is always wrong. The production trainer currently uses constant AdamW learning rates without a decay schedule: [train_act.py](../train_act.py:358).

A fixed batch of 12 distinct frames also rules out a generally broken optimizer/gradient path. With the original VAE objective, mean target error fell from 6.37 to 1.66 mm in 100 updates and 1.28 mm in 500. Zero-latent training also fits this batch, reaching chunk L1 0.00677 in 500 updates. These probes continue existing weights; they are not claims about randomly initialized fixed-batch convergence.

## 2. The IK controller uses hidden timing state

The generator chooses a duration even for zero-distance moves, interpolates by loop index, and appends nine settling ticks to every move. Holds use the same timed move function. See [collect_scripted.py](../tools/collect_scripted.py:79).

ACT receives one pair of images, six measured joint angles and the commanded gripper opening. It receives neither the generator's stage/elapsed-time state nor observation history.

In episode 0, frames **17 and 39 have bit-for-bit identical seven-dimensional state and both camera images**, but the required future joint targets differ by as much as **0.28728 rad / 16.46 degrees**. Even the first target differs by 0.002088 rad. A deterministic observation-only function cannot return both target chunks for this same input.

For this pair, the minimum possible normalized L1 is **0.0803656**: half the mean absolute distance between the two target chunks. Training only this pair reaches **0.0808377**, close to the mathematical floor. Adding a diagnostic elapsed-time feature permits fitting below that floor: L1 **0.01837 at 100 updates**, **0.00990 at 400**, and **0.01267 at 500**. This is a causal demonstration that missing information, not insufficient model capacity, prevents fitting these conflicting targets.

The effective diagnostic time feature is a learned 512-dimensional contribution to the existing robot-state token. Its new projection uses LR 1e-3 while existing ACT parameters retain LR 3e-5. It starts at zero, preserving initial predictions. The pair's phase is normalized over those two frames. We also tested slower conditioning through the latent token and a low-rate state projection; those did not learn the distinction quickly. Merely adding a poorly trained time channel is not itself a fix.

Exact contradictory observation groups occur in all five examined IK demos:

| Episode | Frames | Frames in contradictory groups |
| --- | ---: | ---: |
| 0 | 994 | 24 |
| 1 | 717 | 21 |
| 2 | 942 | 21 |
| 3 | 747 | 21 |
| 4 | 724 | 14 |

These exact collisions alone impose a whole-episode L1 floor of only **0.000939** for episode 0. They do **not** explain the entire observed ~0.05 loss. Near-identical observations around later waits also ask for substantially different futures, but those do not give the same strict mathematical lower bound.

The full-demo time-conditioned probe reached 6.73 mm mean target error and completed the stack in-process and in two fresh-process default-renderer evaluations (**3/3**). Its ordinary no-time, high-rate control reached 9.44 mm and completed **2/3**. This supports the relevance of phase information to execution, while the small number of rollouts does not isolate the contribution to every failure.

Elapsed time normalized by this demo's length is privileged diagnostic information. It is **not** a ready general-purpose deployment feature: varying speeds, pauses and recovery break that fixed schedule. A lasting data/observation correction would make controller progress observable, e.g. appropriate recurrent history/controller state, or state-driven demonstrations that avoid unobservable arbitrary waits. Such a correction still needs replay and closed-loop validation.

## 3. Renderer nondeterminism explains the reload divergence

A separate probe drives two fresh simulators with exactly the same saved action sequence. Joint states, velocities and full qpos remain bit-for-bit identical. With the default four-sample renderer, some RGB values differ by one integer intensity level. Feeding the same image tensor through ACT repeatedly gives identical predictions; feeding the differing rendered images changes predictions by up to 2.48e-5 in this probe.

Setting `sim.model.vis.quality.offsamples = 0` removes the differences in the probe. Two independent full rollouts then have **bit-for-bit identical actions, joint states, object positions and grasp flags for all 1,192 ticks**. Both fail, confirming that rendering variation was not the sole cause of the policy error.

In the original 5,000-step checkpoint, the first action difference between the successful run and failed reload was only 7.75e-6 at five seconds. By six seconds it was 0.00198, and by seven seconds 0.00844, before the first grasp. The learned closed-loop behavior amplifies small input differences.

Disabling multisampling changes image appearance relative to the existing training images. The low-rate and phase-conditioned checkpoints both failed their one no-MSAA evaluation despite succeeding under the training renderer's default setting. Thus **do not treat turning off MSAA at inference as a performance fix**. It is a repeatability diagnostic; production rendering changes should be matched in training/export and revalidated.

## What this establishes

- ACT can fit selected observations accurately; gradients, basic target indexing and action replay work.
- The previous one-demo test remained underfit in part because of its optimization settings. Lowering the continuation LR yields a large controlled precision gain and a checkpoint that completed all three checked default-renderer executions.
- Exact memorization of all recorded timed chunks is mathematically impossible with the current observation contract. More parameters or more updates cannot resolve identical inputs with different labels.
- Tiny renderer differences caused the apparent reload nondeterminism, and the learned controller amplified them.
- These are demonstrated mechanisms and contributors, not proof that one setting explains every failure of the separate large-dataset policy. The current full-dataset run's saved selection report remains 0/8 full stacks.

No production dataset, simulation defaults, or full-dataset training checkpoint was changed. Diagnostic tools and artifacts were added. The 21 ACT tests, including new contradiction/loss-bound tests, pass.

## Artifacts and reproduction

- [Machine-readable summary](../outputs/act/ik_root_cause/summary.json)
- [Working low-rate rollout](../outputs/act/ik_root_cause/full_low_lr/zero/rollout.mp4)
- [Fresh low-rate reload](../outputs/act/ik_root_cause/low_lr_reload_1/rollout.mp4)
- [Low-rate checkpoint](../outputs/act/ik_root_cause/full_low_lr/zero/checkpoint/config.json)
- [Five-demo audit](../outputs/act/ik_root_cause/observability_5_demos.json)
- [Optimizer/conditioning probes](../tools/diagnose_act_overfit.py)
- [Observation audit](../tools/audit_act_observability.py)
- [Renderer/physics probe](../tools/probe_act_determinism.py)
- [Saved-checkpoint evaluator](../tools/probe_act_reload.py)

Use fresh output paths when repeating:

```bash
OPENBLAS_NUM_THREADS=1 .venv-act/bin/python tools/diagnose_act_overfit.py \
  --scope full --objectives zero --lr 0.000003 --steps 1000 \
  --output outputs/act/root_repeat_low_lr

OPENBLAS_NUM_THREADS=1 .venv-act/bin/python tools/probe_act_reload.py \
  --checkpoint outputs/act/ik_root_cause/full_low_lr/zero/checkpoint \
  --output outputs/act/root_repeat_reload

OPENBLAS_NUM_THREADS=1 .venv-act/bin/python tools/audit_act_observability.py \
  --dataset outputs/lerobot/panthera_scripted_stack_30hz \
  --episodes 0 1 2 3 4 --output outputs/act/root_repeat_audit.json
```
