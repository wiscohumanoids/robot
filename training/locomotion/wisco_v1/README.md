# Wisco V1 walking policy: training

Training a walking (velocity) policy for Wisco V1 with mjlab, the same way the G1 policies were trained. The robot
model, its numbers and the pre-training checks live in `robots/wisco_v1/` (read-only from here).

**Start here:** `WORK_LOG.md` (what was done, decisions, issues, run table) and `robots/wisco_v1/HARDWARE_ISSUES.md`
(problems for the mechanical team).

```
training/locomotion/wisco_v1/
  README.md              this file
  WORK_LOG.md            chronological work / observation log + training run table
  train.py, play.py      launchers: register the task, then run mjlab's own train / play
  wisco_mjlab/           the mjlab task: robot_cfg.py (DC motors + delay; reads robots/wisco_v1/model/), env_cfg.py
                         (arms-free recipe), arms_free_terms.py, rl_cfg.py, linkage_penalty.py, ankle_actuator.py;
                         registers Wisco-V1-Velocity-Flat
  vm/                    GPU VM scripts
    sync_to_vm.sh          copy the Wisco V1 folders to the VM checkout (robots/wisco_v1, this folder, training/tracking/wisco_v1)
    gpu_status.sh          GPU memory and processes (check before training)
    check_env.py           build the real mjlab env and check it (no training)
    smoke_test.sh          5 PPO updates on 256 robots
    train_velocity.sh      launcher: Wisco V1 velocity (15 actions), 4096 envs, 20000 iterations, --log-root
    resume_latest.sh       resume from a previous run's newest checkpoint into a new log root
    ankle_torque_check.py  per-motor ankle torque distribution for a checkpoint (peak / rated / RMS)
```

The policy drives 15 joints (legs 12 + waist 3) and holds the arms and head ("arms-free" recipe, manifest v2), so the
arms stay free for manipulation. Deployed policy: `models/locomotion/wisco_v1_velocity_arms_free/` (J3h model_4650,
trained up to 1.0 m/s forward).

## On the GPU VM

The mjlab environment is the uv project in `training/locomotion/` (its `.venv`); every command runs from there.

```
# Mac:
training/locomotion/wisco_v1/vm/sync_to_vm.sh            # or tools/sync_to_vm.sh hsim-gpu for the whole repo
# VM:
cd ~/humanoid-sim-stack/training/locomotion
wisco_v1/vm/gpu_status.sh
MUJOCO_GL=egl .venv/bin/python wisco_v1/vm/check_env.py 1024
wisco_v1/vm/smoke_test.sh
wisco_v1/vm/train_velocity.sh --log-root ~/hsim_runs/J3_wisco_v1_velocity_arms_free
```
Or directly: `uv run python wisco_v1/train.py Wisco-V1-Velocity-Flat --env.scene.num-envs 4096
--agent.max-iterations 1500 --log-root logs/rsl_rl`. Export a run to the stack with
`training/locomotion/scripts/export_manifest.py` (see `training/locomotion/README.md`).

## Changing the robot's numbers

Edit `robots/wisco_v1/model/robot_numbers.yaml`, rebuild the model and rerun the checks
(`robots/wisco_v1/README.md`, "Rebuild"), then retrain. Run names recorded before the V1 rename are listed in
`docs/migration/renames.md`.
