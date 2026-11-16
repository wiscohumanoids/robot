# training/locomotion (GPU VM)

uv project that pins mjlab 1.6.0 (`pyproject.toml`, `.python-version`); every mjlab job (walking, tracking, Wisco)
runs from this folder with its `.venv`. Phase 0 walkthrough (first G1 run, setup on the VM):
[docs/phase0_locomotion.md](../../docs/phase0_locomotion.md).

| Path | What |
|---|---|
| [`wisco_v1/`](wisco_v1/README.md) | **Wisco V1** walking task `Wisco-V1-Velocity-Flat` (arms-free recipe), VM launchers, work log |
| `hsim_tasks/` | G1 task `Hsim-G1-Velocity-Flat-ArmsFree` (arms-free recipe: legs + waist, scripted arms; every change vs mjlab's G1 task is marked `# ARMS-FREE`), custom terms (`mdp.py`), runner, train/play launchers |
| `scripts/train_velocity.sh` | G1 all-joints policy: mjlab's stock `Mjlab-Velocity-Flat-Unitree-G1` (NUM_ENVS, ITERS env vars) |
| `scripts/train_g1_velocity_arms_free.sh`, `_resume.sh` | G1 arms-free policy (`Hsim-G1-Velocity-Flat-ArmsFree`) |
| `scripts/smoke_test.sh` | 5 PPO updates on 256 envs, W&B offline. Proves the GPU stack works |
| `scripts/fetch_lafan1.sh`, `upload_motion.sh <clip>`, `train_tracking.sh <registry>` | G1 tracking via the W&B motion registry (see also `training/tracking/`) |
| `scripts/eval_commands.py` | scripted command tests of a checkpoint (deadband, turns), JSON out |
| `scripts/export_manifest.py <run> --policy-id <id>` | ONNX metadata → `models/locomotion/<id>/manifest.yaml`, plus a CPU onnxruntime check |

Policy recipes: **all-joints** (the policy drives every joint; manifest v1) and **arms-free** (legs + waist/torso,
arms and head held for manipulation; manifest v2). Outputs: `logs/rsl_rl/<experiment>/<timestamp>/` (checkpoints,
`<timestamp>.onnx`, `params/`), gitignored. Deployed policies: `models/locomotion/README.md`.
