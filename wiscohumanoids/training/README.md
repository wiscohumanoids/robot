# Wisco V2 walking policy: training workspace

Everything for training a walking (velocity) policy for the Wisco humanoid with mjlab, the same way the
Unitree G1 policy was trained. The robot model itself lives in `../robot/` (V2, read-only: never edited here).

**Start here:** `WORK_LOG.md` (what was done, decisions, issues, run table) and `HARDWARE_ISSUES.md`
(problems for the mechanical team).

```
training/
  README.md              this file
  WORK_LOG.md            chronological work / observation log + training run table
  HARDWARE_ISSUES.md     hardware findings and questions for the mech team (H1, H2, ...)
  model/                 the training model, built from ../robot/
    robot_numbers.yaml     THE numbers file: motors (datasheet values + sources), delay/bus, mass overrides,
                           gains + control modes, default pose, locomotion action joints, IMU + camera poses
    motor_datasheets/      the official MyActuator parameter sheets (images)
    robot_params.py        derives all joint values (gains, limits, friction, torque-speed) from the files above
    sync_robot_folder.py   regenerates ../robot/robot.urdf, robot.xml, model_report.json from robot/original_from_team + these numbers
    linkage_tools.py       ankle linkage analysis -> ankle_linkage.json (reachable range, Jacobian grids)
    build_training_model.py  ../robot/robot.xml -> wisco_v2_train.xml + train_model_info.json
    ankle_motor.py         ankle joint torque -> per-motor torque (numpy)
  phase_a/               pre-training checks on the Mac (plain MuJoCo, mjlab's actuator model)
    sim_replica.py         mjlab's actuators/collisions/solver settings in plain MuJoCo
    run_checks.py          checks A1-A8 -> results/
    g1_baseline.py         same tests on the Unitree G1, for reference
    results/README.md      Phase A results per configuration (initial report in results/initial_model_2026-11-10/)
  mjlab_task/            the mjlab task (Phase B)
    wisco_mjlab/           robot_cfg.py (DC motors + delay), env_cfg.py (v2 recipe), v2_terms.py, rl_cfg.py, linkage_penalty.py; registers Wisco-V2-Velocity-Flat
    train.py, play.py      launchers: register our task, then run mjlab's own train / play
  vm/                    GPU VM scripts (Phase B)
    sync_to_vm.sh          copy wiscohumanoids/ to the VM (only this folder)
    gpu_status.sh          GPU memory and processes (check before training; leave pi0.5 alone)
    check_env.py           build the real mjlab env and check it (no training)
    smoke_test.sh          5 PPO updates on 256 robots
    train_velocity.sh      J3 launcher: Wisco v2 velocity (15 actions), 4096 envs, 20000 iterations, --log-root
    resume_latest.sh       resume from a previous run's newest checkpoint into a new log root
    ankle_torque_check.py  per-motor ankle torque distribution for a checkpoint (peak / rated / RMS)
```

## Rebuild (Mac)
```
cd wiscohumanoids/training/model
../../../.venv/bin/python linkage_tools.py          # ~6 min, only if robot/linkage.json changes
../../../.venv/bin/python sync_robot_folder.py      # robot/ files (URDF, MJCF, report)
../../../.venv/bin/python build_training_model.py   # training model from robot/robot.xml
cd ../phase_a && ../../../.venv/bin/python run_checks.py --actuator dc --delay 2
```

## On the GPU VM (Phase B)
```
# Mac:
wiscohumanoids/training/vm/sync_to_vm.sh
# VM:
cd ~/humanoid-sim-stack/training/locomotion
../../wiscohumanoids/training/vm/gpu_status.sh
MUJOCO_GL=egl .venv/bin/python ../../wiscohumanoids/training/vm/check_env.py 1024
../../wiscohumanoids/training/vm/smoke_test.sh
../../wiscohumanoids/training/vm/train_velocity.sh --log-root ~/hsim_runs/J3_wisco_velocity_v2   # J3
```
Training (Phase C) uses the same launcher with more robots and iterations, e.g.
`uv run python ../../wiscohumanoids/training/mjlab_task/train.py Wisco-V2-Velocity-Flat --env.scene.num-envs 4096 --agent.max-iterations 1500 --log-root logs/rsl_rl`.

## Changing numbers when real data arrives
| New data | Edit | Then |
|---|---|---|
| Measured / CAD masses | `model/robot_numbers.yaml` -> `masses.overrides` | `sync_robot_folder.py`, `build_training_model.py`, rerun phase_a, retrain |
| Motor bench tests (inertia, friction, damping, torque-speed) | `model/robot_numbers.yaml` -> `motors` (value + source) | same |
| Measured delay | `robot_numbers.yaml` -> `delay` | same |
| Real driver PD gains | `robot_numbers.yaml` -> `control.gains` (kp/kd) | same |
| IMU / camera pose | `robot_numbers.yaml` -> `sensors` | same |
