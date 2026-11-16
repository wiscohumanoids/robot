# Phase 0 — Locomotion training with mjlab

Goal: prove that mjlab trains the G1 on the g7e GPU VM, and leave Phase 0 with two exported ONNX policies plus manifests.

## What we train

| Policy | mjlab task | Command input | Used for |
|---|---|---|---|
| **A. Velocity** | `Mjlab-Velocity-Flat-Unitree-G1` | (vx, vy, yaw rate) | walking behind `/cmd_vel`: teleop + Nav2 |
| **B. Tracking** | `Mjlab-Tracking-Flat-Unitree-G1` | a LAFAN1 mocap clip | BeyondMimic-style tracking demo (walk, dance) |

Why two: see `decisions/0002-two-locomotion-policies.md`. Short version: a tracking policy replays a clip and can't follow `/cmd_vel`.

## How this differs from "regular PPO"

The learning algorithm is the same: **PPO**, from RSL-RL (`rsl-rl-lib`). What changes is everything around it.

| | Typical "regular PPO" project | mjlab |
|---|---|---|
| Simulation | a few to dozens of CPU envs (Gym + MuJoCo), ~1–10k steps/s | **4096 envs at once on the GPU** (MuJoCo-Warp), roughly 100k+ steps/s. Billions of steps in hours |
| Task definition | one hand-written `step()` / `reward()` | **manager-based config**: observations, rewards, terminations, randomization, and commands are separate terms you switch on/off in config (Isaac Lab API) |
| Value function | sees the same input as the policy | **asymmetric actor-critic**: the critic gets privileged sim info (true velocities, contact forces, body poses); the actor only gets what a real robot could sense |
| Robustness | none | sensor noise on every observation, random pushes, randomized friction/mass, so the policy doesn't overfit to one perfect sim |
| Action | often raw torques | **joint position offsets**: `target = default_pose + scale × action`, tracked by PD actuators with gains computed from real motor physics (rotor inertia, 10 Hz natural frequency) |
| Output | a `.pt` checkpoint | **ONNX exported on every checkpoint**, with joint order, gains, default pose, and observation layout embedded |
| Physics, train vs deploy | often different engines | the same MuJoCo physics in training (GPU) and deployment (CPU) → minimal sim-to-sim gap |

### Policy A — velocity (classic legged-robot RL)
- **Reward:** hand-designed terms. Follow the commanded velocity, plus penalties and shaping for foot air time, foot slip, joint limits, action smoothness, upright posture, and so on. This *is* reward engineering, but mjlab ships tuned terms for the G1, so we don't write them ourselves.
- **Commands:** a random (vx, vy, yaw rate) every 3–8 s, in the range vx, vy ∈ [-1, 1] m/s and yaw ∈ [-0.5, 0.5] rad/s.
- **Actor sees:** base linear + angular velocity, gravity direction, joint positions and velocities, last action, command.
- **Note:** the actor uses base linear velocity, which a real robot has to estimate. In Phase 1 the sim provides it (honest stub); in Phase 3 the IEKF provides it.

### Policy B — motion tracking (BeyondMimic)
- **No gait rewards at all.** The reward is how closely the robot matches the mocap clip: pelvis/torso position and orientation, 14 key body positions and orientations, and body velocities, each through an `exp(-error²/σ²)` kernel. The only other terms are small penalties for jerky actions, joint limits, and self-collision.
- **Reference data:** LAFAN1 human mocap, already retargeted to the G1 by Unitree (CSV, 30 fps). mjlab's `csv_to_npz` resamples it to 50 Hz, runs forward kinematics in MuJoCo, and uploads it to a W&B registry.
- **Episodes reset to random points in the clip**, so hard segments (turns, jumps) get practiced as much as easy ones.
- **Early termination** when the robot drifts too far from the reference (torso height off by 0.25 m, hands/feet off by 0.25 m, torso tilt too large).
- **One policy per clip.** Generalizing across many clips is a later extension, not Phase 0.

### Shared numbers (from mjlab's G1 config)
- Physics step 5 ms (200 Hz), policy every 4 steps → **50 Hz**.
- Actor and critic MLPs: 512-256-128, ELU activations, observation normalization on.
- PPO: learning rate 1e-3 with an adaptive schedule, 24 steps per env per update → ~98k samples per update at 4096 envs.
- Checkpoint + ONNX every 50 updates.

## Steps

All commands run on the GPU VM, from `training/locomotion/` unless noted.

| # | Step | Command | Done when |
|---|---|---|---|
| 0 | Push the repo | (laptop) `tools/sync_to_vm.sh hsim-gpu` | repo on the VM |
| 1 | Set up the VM | `infra/aws/gpu_vm/setup.sh` | driver ≥ 580, torch sees the GPU, Warp sees the GPU |
| 2 | Smoke test | `scripts/smoke_test.sh` | 5 PPO updates run on 256 envs without errors |
| 3 | W&B login | `uv run wandb login` | |
| 4 | Train velocity | `scripts/train_velocity.sh` | reward plateaus; `play` shows stable walking at all commands |
| 5 | Get LAFAN1 | `scripts/fetch_lafan1.sh` | `data/lafan1/g1/*.csv` exist |
| 6 | Create the W&B registry | one-time, in the W&B web UI (see below) | registry "Motions" exists |
| 7 | Convert a clip | `scripts/upload_motion.sh walk1_subject1` | NPZ in the registry, preview video looks right |
| 8 | Train tracking | `scripts/train_tracking.sh <entity>-org/wandb-registry-Motions/walk1_subject1` | robot follows the clip without falling |
| 9 | Export manifests | `scripts/export_manifest.py <run-dir> --policy-id <id>` | `models/locomotion/<id>/manifest.yaml` |
| 10 | Pull to the laptop / CPU VM | (laptop) `tools/pull_policy.sh <run-dir-on-gpu> <id>` | ONNX + manifest in `models/locomotion/<id>/` |
| 11 | Stop the GPU VM | AWS console / CLI | |

**W&B registry (step 6):** in W&B, open Registry → create a new registry named `Motions` with artifact type `All Types`. This is the "registry setup" step of the BeyondMimic README; skip its `csv_to_npz.py` command, because mjlab has its own converter (body ordering differs from Isaac Lab's).

**Watching the policy:** `uv run play <task> --wandb-run-path <entity>/mjlab/<run-id> --viewer viser` starts a web viewer on the VM. Open it from the laptop through an SSH tunnel (the port is printed on start).

## Phase 0 exit criteria — DONE (2026-11-08)

- [x] Velocity policy walks forward, backward, sideways, and turns without falling (sideways/turning need commands above a learned deadband, see decision 0004).
- [ ] Tracking policy: deferred, kept on the side (decision 0002).
- [x] ONNX loads in onnxruntime on CPU with the manifest's I/O sizes (0.01 ms per step).
- [x] Training cost recorded: 1 h 27 min, ~185k steps/s, 2.4 GB GPU memory (`models/locomotion/g1_velocity_all_joints/README.md`).
- [x] Beyond plan: the policy runs in plain MuJoCo on the Mac (`tools/play.sh`) and inside the ROS stack (`sim_stack.launch.py`), matching mjlab's own evaluation.

## Known risks

- **Blackwell support: resolved (2026-11-08).** Verified on the g7e: driver 595, torch 2.15 (CUDA 13.2), Warp 1.19, sm_120. The smoke test runs at ~15.5k steps/s with 256 envs; ONNX export and manifest work.
- **Shared GPU.** A pi0.5 LeRobot inference server (`/mnt/pi05`) also runs on this VM and uses ~10 GB of GPU memory. Training speed numbers are measured with it running.
- **LAFAN1 download.** The official Unitree Hugging Face repo has gone offline before; `fetch_lafan1.sh` accepts a mirror. LAFAN1 is CC BY-NC-ND (non-commercial).
- **ONNX metadata precision.** mjlab rounds the floats it writes into ONNX metadata to 3 decimals. The full training config (`env.yaml`) is copied next to the manifest, and the Phase 1 deploy node checks the manifest gains against the MuJoCo model.
