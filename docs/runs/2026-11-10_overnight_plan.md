# Overnight plan, 2026-11-10: parallel training + perception + manipulation

Coordinator: the main session. Workstreams are built on the Mac by parallel helpers, then run on the GPU VM.
Agreements this is built on: see the conversation summary in each helper's context; key ones restated below.

## Jobs
| ID | Job | Owner | Runs on VM |
|---|---|---|---|
| J1 | G1 velocity v2 (legs+waist, v2 recipe) | B | GPU, overnight |
| J2 | G1 tracking: LAFAN1 dance1, jumps1, fallAndGetUp1 (one policy per clip) | C | GPU, overnight |
| J3 | Wisco velocity v2 (legs+waist, v2 recipe, RP1-style gains, MIT-mode model) | A | GPU, overnight |
| J4 | Wisco tracking: LAFAN1 dance1 retargeted to Wisco | C | GPU, overnight |
| S1 | Perception server: Grounding DINO + SAM 2 + ZED X depth -> 3D | E | GPU service |
| M1 | Manipulation v1 (cups): scripted demos -> MimicGen -> GR00T N1.7 fine-tune -> VLA server | F1 + F2 | CPU then GPU |
| M2, M3 | v2 (many shapes), v3 (plates, wider reach) | F1 + F2 | after M1 |

## Folder ownership (only edit your own folders; ask the coordinator for anything else)
| Helper | Owns |
|---|---|
| A: Wisco model + velocity | `wiscohumanoids/` (except `wiscohumanoids/training/tracking/`) |
| B: G1 velocity v2 + deploy | `training/locomotion/hsim_tasks/` (new), `training/locomotion/scripts/` (new files only), `ros2_ws/src/hsim_locomotion*` / the locomotion deploy node, `tools/sim2sim_play.py`, `training/locomotion/scripts/export_manifest.py` |
| C: tracking | `training/tracking/` (new), `wiscohumanoids/training/tracking/` (new), `tools/showcase_play.py` (new) |
| D: VM infra | `infra/vm_jobs/` (new), `tools/vm_*.sh` (new), `infra/aws/gpu_vm/` |
| E: perception | `services/perception_server/`, the ROS perception package(s) in `ros2_ws/src/` (hsim_perception), its config/launch entries |
| F1: manipulation sim + data | `training/manipulation/sim/`, `training/manipulation/datagen/`, `assets/robots/g1_dex1/` (new) |
| F2: GR00T + VLA server | `training/manipulation/groot/`, `services/vla_server/`, BT/planner skill hook for manipulation (coordinate: `ros2_ws/src/hsim_behavior` changes go through the coordinator) |
Shared files (README.md, docs/architecture.md, interface_contract.yaml, launch files outside your package): propose the change in your report; the coordinator applies it.

## Local tools (Mac)
- `.venv-mjlab/` = Python 3.12 + mjlab 1.6.0 (CPU). mjlab builds and steps envs on CPU (G1 flat, 4 envs: ~28 s incl. kernel compile). Use it to check configs before the VM.
- `.venv/` = Python 3.14 + mujoco 3.11 (Phase A checks). Don't install into either without saying so in your report.
- No git. Don't commit.

## VM (hsim-gpu, 1x RTX PRO 6000 96 GB, 8 vCPU, 62 GB RAM, 259 GB free on persistent /)
Layout (all persistent EBS; never use /opt/dlami/nvme for anything that must survive a stop):
| Path | What |
|---|---|
| `~/humanoid-sim-stack/` | code, synced from the Mac. Mac is the source of truth. |
| `~/hsim_runs/<job_id>/` | job outputs: logs, checkpoints, videos, reports |
| `~/hsim_envs/<name>/` | Python envs that are not the mjlab one (perception, gr00t, datagen) |
| `~/hsim_data/` | datasets and caches (`~/hsim_data/lafan1`, `~/hsim_data/manip/<version>/`, `HF_HOME=~/hsim_data/hf-cache`) |
| `~/humanoid-sim-stack/training/locomotion/.venv` | existing mjlab 1.6.0 GPU env (torch 2.15 cu132) - reuse for all mjlab jobs |
Rules:
1. **Never touch `/mnt/pi05/` or the `panthera-inference` user service.** It is stopped for tonight; restart note: `~/hsim_runs/PI05_SERVER_NOTE.md`. Only the coordinator restarts it.
2. Don't kill or restart processes you didn't start. Long training jobs are launched only by the coordinator (via D's launcher).
3. Sync only your own folders to the VM: `rsync -az --exclude __pycache__ --exclude .venv <local dir>/ hsim-gpu:~/humanoid-sim-stack/<same dir>/` (no `--delete`).
4. GPU tests: short (< 10 min) and < 15 GB, in a tmux session named `hsim-<helper>-<what>`; kill your own session when done.
5. `uv` on the VM is `~/.local/bin/uv` (not on PATH for ssh commands). HF token is already installed on the VM (`~/.cache/huggingface/token`, account SpiderWolf6); never print or copy it.
6. Downloads go to `~/hsim_data/`; envs to `~/hsim_envs/`.

## v2 locomotion recipe (J1 and J3 use the same list)
1. Actions: legs + waist only (G1: 12 legs + 3 waist = 15; Wisco: 12 legs + torso_yaw/roll/pitch = 15). Arms are NOT actions;
   observations still include all joints (arm joint pos/vel are known on the robot).
2. Arms in sim: position-held (stand-in for the ROS controls stack) with targets resampled every 1-3 s: 30 % of resamples hold
   the default pose, 70 % pick uniform targets within 60 % of each arm joint's soft range around default; random move speed.
3. Hand payload: add 0-1.0 kg (uniform, per hand) to each hand/wrist end body at startup.
4. Velocity tracking tighter (decision 0004 deadband fix): linear std 0.5 -> 0.25 m/s, angular std 0.71 -> 0.35 rad/s.
5. Turn-in-place: make sure >= 15 % of command samples are pure yaw (zero linear) with |yaw| 0.3-1.5 rad/s; yaw range (-1.5, 1.5).
6. Command delay 0-10 ms (mjlab actuator delay_min_lag 0 / delay_max_lag 2 at 5 ms physics).
7. Pushes stronger: push velocity range x/y 0.5 -> 0.8 m/s (rest as G1 default).
8. Motor randomization: armature +/-10 % (Wisco: per motor_specs ranges), kp/kd +/-10 %. G1 masses are accurate: no mass DR
   for G1 (Menlo lesson); Wisco keeps its mass DR (0.85-1.20).
9. Same PPO settings as the G1 recipe. Save every 50 iterations. max_iterations: 20000 (stopped by the coordinator when good).

## Interfaces between workstreams
- A -> C: keep `wisco_mjlab.robot_cfg.get_wisco_robot_cfg()`, `ACTION_SCALE`, `PARAMS` and `model/robot_params.py:joint_params()` working (names stable).
- Each training job provides one launch script that runs from `~/humanoid-sim-stack/training/locomotion` with its `.venv`
  and takes `--log-root` (D's launcher passes `~/hsim_runs/<job_id>`): A: `wiscohumanoids/training/vm/train_velocity.sh`,
  B: `training/locomotion/scripts/train_g1_velocity_v2.sh`, C: `training/tracking/train_g1_tracking.sh <clip>`, `wiscohumanoids/training/tracking/train_wisco_tracking.sh <clip>`.
- F1 -> F2 raw episode format (F1 writes, F2 converts to LeRobot v2 for GR00T): one `episode_XXXXXX.npz` + `.json` per episode in
  `~/hsim_data/manip/<version>/raw/`. npz keys (T steps at 20 Hz): `ego_rgb` (T,H,W,3 uint8, head camera), optional
  `wrist_left_rgb`/`wrist_right_rgb`; `q_left_arm` (T,7), `q_right_arm` (T,7), `q_waist` (T,3), `grip_left` (T,), `grip_right` (T,)
  (0 = closed, 1 = open); `act_left_arm` (T,7), `act_right_arm` (T,7) (joint position targets), `act_grip_left` (T,), `act_grip_right` (T,);
  `obj_pose` (T,7). json: `instruction`, `object_category`, `object_name`, `success`, `fps`, `camera_intrinsics`, `seed`, `source` ("scripted"|"mimicgen").
  F2 may request additions in its report; the coordinator relays.

## Reporting (every helper, in its final message)
Files created/changed; how it was tested (commands + results); problems hit and fixes (with why); anything that deviates from the
agreements; open questions; exact command(s) for the coordinator to run on the VM. Add a short entry to the relevant log
(wiscohumanoids/training/WORK_LOG.md for A and C's Wisco part; docs/runs/2026-11-10_worklog_<helper>.md for others).
