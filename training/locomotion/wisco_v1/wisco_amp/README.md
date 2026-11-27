# Wisco AMP with amp-rsl-rl

Thin AMP integration for Wisco V1. The discriminator, AMP PPO and runner come
from [`amp-rsl-rl`](https://github.com/gbionics/amp-rsl-rl) (BSD-3-Clause); this
directory only supplies:

- the `amp` observation group for the Wisco velocity env,
- the runner/dataset configuration,
- an `.npz` → `.npy` motion converter,
- task registration.

AMP is training-only. The actor observation and ONNX/manifest deployment path
are unchanged.

## Dependency upgrade

`amp-rsl-rl` needs `rsl-rl-lib >= 5.5.1`. `training/locomotion/pyproject.toml`
therefore pins mjlab main at a commit that uses rsl-rl 5.5.2, and adds
`amp-rsl-rl`.

This is a training-stack upgrade; run the normal mjlab smoke test before a long
Wisco run.

## Motion data

Start from the normal tracking `.npz`:

```bash
# LAFAN1 BVH -> Wisco CSV (local only; LAFAN1 is CC BY-NC-ND)
.venv-mjlab/bin/python training/tracking/wisco_v1/retarget_lafan1.py \
  --bvh walk1_subject1.bvh --out /tmp/walk1_subject1.csv

.venv-mjlab/bin/python training/tracking/motion_to_npz.py \
  --robot wisco --input /tmp/walk1_subject1.csv --output /tmp/walk.npz
```

Convert it to the amp-rsl-rl `.npy` format:

```bash
.venv-mjlab/bin/python training/locomotion/wisco_v1/wisco_amp/prepare_motion.py \
  --input /tmp/walk.npz --output ~/hsim_data/amp/walk.npy
```

Do not commit LAFAN1-derived clips.

## Train

From `training/locomotion`, using its uv environment:

```bash
uv run python wisco_v1/wisco_amp/train.py \
  Wisco-V1-AMP-Velocity-Flat \
  --env.scene.num-envs 512 \
  --agent.dataset.amp-data-path ~/hsim_data/amp \
  --agent.style-weight 0.2
```

Start with one walk clip and 512–1024 envs. Do not start with dance clips or
4096 envs.

## Play

```bash
uv run python wisco_v1/wisco_amp/play.py \
  Wisco-V1-AMP-Velocity-Flat \
  --checkpoint-file <path/to/model_*.pt> \
  --agent.dataset.amp-data-path ~/hsim_data/amp
```

mjlab's play path constructs the registered AMP runner before loading the actor,
so the dataset path is still required. Export for deployment remains the normal
ONNX + manifest path.

## AMP observation

`amp_obs.py` returns, in order:

```text
joint positions, joint velocities,
base linear velocity (body frame), base angular velocity (body frame)
```

This matches `amp-rsl-rl`'s expert-loader vector.

## Status

Prototype. Not yet run on the GPU machine. The first checks are:

1. mjlab smoke test with the upgraded dependency,
2. `prepare_motion.py` on one walk clip,
3. 512-env AMP run,
4. comparison against `wisco_v1_velocity_arms_free`.
