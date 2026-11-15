# IK ACT continuation with the successful overfit settings

The 100,000-step full scripted-IK model is being continued for 20,000 additional
updates in a separate output directory. This tests whether the settings that
improved the single-demo fit also improve the full-dataset model. The original
checkpoint and its results are preserved.

The source run's selection evaluation completed 0/8 stacks, with 6/8 sustained
pickups. Its full held-out normalized chunk L1 was 0.0489773. The single-demo
root-cause report found 3.15 mm mean first-target error and three successful
executions of one scene with the selected settings; that result does not establish
generalization across the full dataset.

| Setting | Continuation |
| --- | --- |
| Source | `outputs/act/panthera_scripted_stack_30hz`, step 100,000 |
| Target | Step 120,000 (20,000 additional updates) |
| Data | Same 900 training / 100 validation IK episodes |
| Batch | 12 |
| Objective | Observation-only zero-latent path; stock padded L1 reduction |
| Optimizer | Fresh AdamW moments, LR 3e-6 for both backbone and other parameters, weight decay 1e-4 |
| Precision / dropout | FP32 / 0 |
| Actions | Absolute targets, chunk 30, execution queue 30, 30 Hz |
| Renderer | Existing defaults |
| Split seed | 20260922, preserving the source split; the single-demo probe used 20260924 |
| Checkpoints | Every 1,000 updates; retain two periodic checkpoints |
| Validation | 3,072 evenly sampled held-out frames every 5,000 updates; full held-out pass at the end |
| Rollouts | Eight original selection seeds starting 20261001, 40 seconds each, every 5,000 updates |

The trainer now supports explicit learning-rate and dropout overrides, fresh
optimizer continuation, and the zero-latent training objective. Learning-rate
overrides are applied after restoring optimizer state, and the selected objective
is saved in resumable training state. The zero-latent path calls the model with
autograd enabled and omits action targets from its inputs. All 22 ACT tests pass,
including a real ACT test checking deployment-equivalent predictions, padding,
and gradients through the deployed action path.

The run is managed by user service `robotarm-act-ik-low-lr-20260923.service`.
Its launch record contains the full command, environment, source checkpoint hash,
code hashes, and baseline selection metrics.

- [Launch record](../outputs/act/panthera_scripted_stack_30hz_low_lr_20260923/launch.json)
- [Training log](../outputs/act/panthera_scripted_stack_30hz_low_lr_20260923/training.log)
- [Effective settings](../outputs/act/panthera_scripted_stack_30hz_low_lr_20260923/training_settings.json)

Selection rollouts are a small, reused development set. Improvement in their
results would warrant a separate held-out scene evaluation before claiming
reliability. No continuation rollout result is available at launch.

Startup verified through step 100,100: the service is active, losses are finite,
and the recent training loss is 0.0310 at approximately 4.21 updates/second.
This is training progress only; the first scheduled rollout evaluation is at
step 105,000.

## Early pickup inspection at step 102,000

In response to reported pickup failures, two explicit 40-second deployments of
`checkpoints/step_00102000` used the normal 30-action queue and default renderer.
Seed 7 reached sustained pickup at 15.87 seconds; seed 8 at 16.77 seconds.
Both picked up blue rather than starting with red as in the demonstrations;
neither achieved two-stack geometry or a complete stack. Both ended holding blue.
These two inspected scenes do not establish pickup reliability.

Seed 8 repeatedly issued closing commands before reaching a cube. At the first
command crossing below 10 mm (2.97 seconds), the measured grip site was about
58 mm from the nearest cube's grasp position as defined by the scripted planner.
It only obtained the blue grasp on a later approach. This is direct evidence of
poor coordination between reaching and closing in this rollout, not proof of the
training-level cause. Videos and physical traces are in
`outputs/act/panthera_scripted_stack_30hz_low_lr_20260923/pickup_check_102000/`.

## Stopped by user

Stopped the service at the user's request. Last logged update: 103,800; latest saved resumable checkpoint: 103,000. Checkpoints and logs are retained. The planned 105,000-step selection evaluation and final validation were not reached.
