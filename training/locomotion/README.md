# training/locomotion (GPU machine)

The uv project that pins mjlab 1.6.0 (`pyproject.toml`, `.python-version`). Every mjlab job (G1 and Wisco V1 walking,
motion tracking) runs from this folder with its `.venv`. Trained policies are exported as ONNX plus a
`manifest.yaml` and deployed from `models/locomotion/` (see [its README](../../models/locomotion/README.md)).

| Path | What |
|---|---|
| [`wisco_v1/`](wisco_v1/README.md) | **Wisco V1** walking task `Wisco-V1-Velocity-Flat` (arms-free recipe) and its GPU-machine scripts |
| `hsim_tasks/` | G1 task `Hsim-G1-Velocity-Flat-ArmsFree` (legs + waist, scripted arms; every change vs mjlab's G1 task is marked `# ARMS-FREE`), custom terms (`mdp.py`), safe arm-target ranges (`arm_ranges.py`), runner, train / play launchers; `tools/arm_contact_sweep.py` checks how often sampled arm targets touch the body |
| `scripts/train_velocity.sh` | G1 all-joints policy: mjlab's stock `Mjlab-Velocity-Flat-Unitree-G1` (NUM_ENVS, ITERS env vars) |
| `scripts/train_g1_velocity_arms_free.sh`, `_resume.sh` | G1 arms-free policy (`Hsim-G1-Velocity-Flat-ArmsFree`) |
| `scripts/smoke_test.sh` | 5 PPO updates on 256 envs, W&B offline: proves the GPU stack works |
| `scripts/fetch_lafan1.sh`, `upload_motion.sh <clip>`, `train_tracking.sh <registry>` | G1 motion tracking through the W&B motion registry (see also `training/tracking/`) |
| `scripts/eval_commands.py` | scripted command tests of a checkpoint (deadband, turns), JSON out |
| `scripts/export_manifest.py <run> --policy-id <id>` | ONNX metadata → `models/locomotion/<id>/manifest.yaml`, plus a CPU onnxruntime check |

Policy recipes: **all-joints** (the policy drives every joint; manifest v1) and **arms-free** (legs + waist, arms and
head held for manipulation; manifest v2). Outputs: `logs/rsl_rl/<experiment>/<timestamp>/` (checkpoints,
`<timestamp>.onnx`, `params/`), not kept in the repo.

Shared training numbers (mjlab's G1 setup, also used for Wisco V1): physics 5 ms (200 Hz), policy every 4 steps
(50 Hz); actor and critic MLPs 512-256-128 with ELU and observation normalization; PPO learning rate 1e-3 with an
adaptive schedule, 24 steps per env per update; checkpoint and ONNX every 50 updates.

## Run order on the GPU machine

All commands from `training/locomotion/` on the GPU machine unless marked laptop. Queued jobs can run the same steps
unattended (`infra/vm_jobs/`).

| # | Step | Command | Done when |
|---|---|---|---|
| 0 | Copy the repo | (laptop) `tools/dev/vm.sh sync` | repo at `~/robot` on the machine |
| 1 | Set up the machine | `infra/aws/gpu_vm/setup.sh` | torch and Warp see the GPU |
| 2 | Smoke test | `scripts/smoke_test.sh` | 5 PPO updates on 256 envs without errors |
| 3 | W&B login | `uv run wandb login` | |
| 4 | Train walking | `scripts/train_velocity.sh`, `scripts/train_g1_velocity_arms_free.sh` or `wisco_v1/vm/train_velocity.sh` | reward plateaus; `play` walks at every command |
| 5 | Motion data | `scripts/fetch_lafan1.sh` | `data/lafan1/g1/*.csv` exist |
| 6 | W&B registry | one time, in the W&B web UI: Registry → new registry `Motions`, artifact type `All Types` | registry exists |
| 7 | Convert a clip | `scripts/upload_motion.sh walk1_subject1` | NPZ in the registry, preview video looks right |
| 8 | Train tracking | `scripts/train_tracking.sh <entity>-org/wandb-registry-Motions/walk1_subject1` | the robot follows the clip without falling |
| 9 | Export | (laptop) `tools/dev/vm.sh pull-policy <run-dir> <policy-id>` (runs `export_manifest.py` on the machine) | ONNX + manifest in `models/locomotion/<id>/` |
| 10 | Stop the machine | AWS console or CLI | |

mjlab converts motion clips itself, so the BeyondMimic `csv_to_npz.py` step is not used (its body ordering differs).

**Watching a policy:** `uv run play <task> --wandb-run-path <entity>/mjlab/<run-id> --viewer viser` starts a web viewer
on the machine; open it from the laptop through an SSH tunnel (the port is printed on start). On the laptop,
`tools/sim/play_mjlab.py` plays an exported policy in plain MuJoCo.
