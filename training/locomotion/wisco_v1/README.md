# Wisco V1 walking policy: training

The mjlab task that trains Wisco V1's walking (velocity) policy, built the same way as the G1 task. The robot model,
its numbers and the pre-training checks are in `robots/wisco_v1/` (read-only from here); hardware findings from these
runs go to `robots/wisco_v1/hardware_observations.md`.

```
training/locomotion/wisco_v1/
  train.py, play.py      launchers: register the task, then run mjlab's own train / play
  wisco_mjlab/           the task (registers Wisco-V1-Velocity-Flat):
    task/
      env_cfg.py             assembles the task from env/
      rl_cfg.py              PPO settings
      env/
        recipe.py            tuning constants
        sensors.py           feet contact + non-arm self-collision sensors
        actions.py           policy joints and viewer body
        commands.py          velocity command sampling + command curriculum
        events.py            randomization, scripted arms, payloads, pushes
        rewards.py           reward terms and metrics
        curriculum.py        performance-gated difficulty
        play.py              evaluation overrides
    robot/
      robot_cfg.py           actuators (DC motor torque-speed limits + command delay), gains and limits from robots/wisco_v1/model/
      ankle_actuator.py      per-motor ankle torque limits through the rod linkage
      linkage_penalty.py     penalty for ankle motor torque above peak and sustained above rated
    terms/
      arms_free_terms.py     custom terms: in-place-turn commands, performance-gated curriculum, scripted arm motion,
                             hand payloads, self-collision and angular-momentum penalties, yaw-rate tracking
  vm/                    GPU machine scripts (run from training/locomotion)
    check_env.py           build the real environment and check it, no training
    smoke_test.sh          5 PPO updates on 256 robots
    train_velocity.sh      full training: 4096 envs, 20000 iterations, --log-root
    resume_latest.sh       resume from a previous run's newest checkpoint into a new log root
    ankle_torque_check.py  per-motor ankle torque distribution of a checkpoint (peak / rated / RMS)
```

The policy drives 15 joints (legs 12 + waist 3) and holds the arms and head (the arms-free recipe, manifest v2), so the
arms stay free for manipulation; training moves the arms randomly and adds hand payloads so the legs learn to balance
whatever the arms do. Deployed policy: `models/locomotion/wisco_v1_velocity_arms_free/` (trained up to 1.0 m/s forward).

## On the GPU machine

The mjlab environment is the uv project in `training/locomotion/` (its `.venv`); every command runs from there.

```
# laptop
tools/vm.sh sync
tools/vm.sh jobs launch wisco_velocity            # the queued job (infra/vm_jobs/jobs.yaml), or by hand on the machine:
# GPU machine
cd ~/robot/training/locomotion
MUJOCO_GL=egl .venv/bin/python wisco_v1/vm/check_env.py 1024
wisco_v1/vm/smoke_test.sh
wisco_v1/vm/train_velocity.sh --log-root ~/hsim_runs/wisco_velocity
```
Export a run to the stack with `tools/vm.sh pull-policy <run-dir> <policy-id>`.

## Changing the robot's numbers

Edit `robots/wisco_v1/model/robot_numbers.yaml`, rebuild the model and rerun the checks (`robots/wisco_v1/README.md`,
"Rebuild"), then retrain.
