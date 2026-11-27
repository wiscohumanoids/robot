# Wisco AMP (clean-room prototype)

Clean-room Adversarial Motion Prior training for the Wisco V1 walking policy.

**Status:** prototype. It has not been trained on the GPU machine yet. The
discriminator and runner are deliberately separate from the velocity task so
the current policy keeps working while this is evaluated.

## What it does

```text
reference .npz clips
  → AMP features (joints, root velocities, gravity, body positions)
  → discriminator
  → style reward
  → velocity task reward + style reward
  → PPO
```

The actor still observes the normal velocity task and still exports to ONNX +
manifest. AMP exists only during training.

## Files

| File | What |
|---|---|
| `motion.py` | Load retargeted `.npz` clips and validate shapes. |
| `features.py` | AMP feature vector from a clip or from a live mjlab env. |
| `discriminator.py` | MLP and least-squares AMP loss with gradient penalty. |
| `runner.py` | PPO runner that adds style reward and trains the discriminator. |
| `env_cfg.py` | Wraps the existing Wisco velocity env. |
| `rl_cfg.py` | PPO values plus AMP hyperparameters. |
| `train.py`, `play.py` | mjlab launchers. |

## Motion data

Use the existing tracking retargeter:

```bash
# LAFAN1 BVH -> Wisco CSV (local only; LAFAN1 is CC BY-NC-ND)
.venv-mjlab/bin/python training/tracking/wisco_v1/retarget_lafan1.py \
  --bvh dance1_subject2.bvh --out /tmp/dance1_subject2.csv

# CSV -> .npz in the robot training model
.venv-mjlab/bin/python training/tracking/motion_to_npz.py \
  --robot wisco --input /tmp/dance1_subject2.csv --output /tmp/walk.npz
```

Do not commit derived LAFAN1 clips.

The `.npz` must contain:

```text
joint_pos (T,J), joint_vel (T,J)
body_pos_w (T,B,3), body_quat_w (T,B,4) [wxyz]
body_lin_vel_w (T,B,3), body_ang_vel_w (T,B,3)
fps
```

Joint and body order is the training-model order, which is also `entity.data`
order in mjlab.

## Train

From `training/locomotion`, using the mjlab uv environment:

```bash
uv run python ../../training/locomotion/wisco_v1/wisco_amp/train.py \
  Wisco-V1-AMP-Velocity-Flat \
  --env.scene.num-envs 512 \
  --agent.amp.motion-files /tmp/walk.npz \
  --agent.amp.style-weight 0.1 \
  --agent.amp.style-warmup-iterations 100
```

Start with one walk clip and 512–1024 envs. Do not start with dance clips or
4096 envs.

## Play

```bash
uv run python ../../training/locomotion/wisco_v1/wisco_amp/play.py \
  Wisco-V1-AMP-Velocity-Flat \
  --checkpoint-file <path/to/model_*.pt>
```

`play` uses the normal velocity environment. AMP is not needed at inference.

## Export

The existing export path works unchanged:

```bash
tools/dev/vm.sh pull-policy <run-dir> <policy-id>
```

The ONNX actor + manifest are the deployment artifact. The discriminator is
training-only.

## What is deliberately missing

- No pretrained AMP checkpoint.
- No tuned style-weight schedule beyond warmup + linear ramp.
- No multi-clip curriculum.
- No GPU validation yet.
- No beyondAMP / AMP_mjlab / Cyclotron code copied.

## First evaluation

Compare against `wisco_v1_velocity_arms_free`:

- velocity tracking error,
- fall rate,
- foot slip,
- ankle RMS/peak and time above rated,
- style reward and discriminator accuracy,
- qualitative gait,
- MuJoCo sim2sim.

If style does not improve without hurting velocity or ankle load, stop.
