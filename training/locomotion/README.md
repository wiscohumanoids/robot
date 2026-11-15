# training/locomotion (Phase 0, GPU VM)

uv project that pins mjlab 1.6.0. Full walkthrough: [docs/phase0_locomotion.md](../../docs/phase0_locomotion.md).

| Script | What it does |
|---|---|
| scripts/smoke_test.sh | 5 PPO updates on 256 envs, W&B offline. Proves the GPU stack works |
| scripts/train_velocity.sh | Policy A: `Mjlab-Velocity-Flat-Unitree-G1` (NUM_ENVS, ITERS env vars) |
| scripts/fetch_lafan1.sh | downloads Unitree's G1-retargeted LAFAN1 CSVs |
| scripts/upload_motion.sh <clip> | CSV → NPZ (50 Hz + forward kinematics) → W&B Motions registry |
| scripts/train_tracking.sh <registry> | Policy B: `Mjlab-Tracking-Flat-Unitree-G1` on one clip |
| scripts/export_manifest.py <run> --policy-id <id> | ONNX metadata → models/locomotion/<id>/manifest.yaml, plus a CPU onnxruntime check |

Outputs: `logs/rsl_rl/<experiment>/<timestamp>/` (checkpoints, `<timestamp>.onnx`, `params/`). Gitignored.
