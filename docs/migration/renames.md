# Renames at unification (2026-11-15)

The repos were merged into this one, the robot is named **Wisco V1** everywhere, and the two walking-policy recipes
got descriptive names. Everything in this repo uses the new names. **Artifacts created before 2026-11-15 outside the
repo (GPU VM folders under `~/hsim_runs/` and `training/locomotion/logs/`, W&B runs, exported ONNX file names) still
carry the old ones**; use this table to find them.

## Robot and folders

| Before | Now |
|---|---|
| Wisco V2 (robot name in docs, comments, generated model headers) | Wisco V1 |
| `wiscohumanoids/robot/` | `robots/wisco_v1/` |
| `wiscohumanoids/training/model/` | `robots/wisco_v1/model/` |
| `wiscohumanoids/training/model/wisco_v2_train.xml` | `robots/wisco_v1/model/wisco_v1_train.xml` |
| `wiscohumanoids/training/phase_a/` | `robots/wisco_v1/checks/` |
| `wiscohumanoids/training/HARDWARE_ISSUES.md` | `robots/wisco_v1/HARDWARE_ISSUES.md` |
| `wiscohumanoids/tools/current_estimate.py`, `wiscohumanoids/hardware/current_draw/` | `robots/wisco_v1/tools/`, `robots/wisco_v1/reports/current_draw/` |
| `wiscohumanoids/training/mjlab_task/` (+ `vm/`, `README.md`, `WORK_LOG.md`) | `training/locomotion/wisco_v1/` |
| `wiscohumanoids/training/tracking/` | `training/tracking/wisco_v1/` |
| `assets/robots/wisco/` | `assets/robots/wisco_v1/` |

## Recipes, tasks, experiments, policies

Recipe "v1" (the policy drives every joint) is now **all-joints**; recipe "v2" (legs + waist/torso, arms held for
manipulation) is now **arms-free**. The manifest *format* versions (`manifest_version: 1 / 2`) are unchanged.

| Kind | Before | Now |
|---|---|---|
| mjlab task | `Wisco-V2-Velocity-Flat` | `Wisco-V1-Velocity-Flat` |
| mjlab task | `Wisco-V2-Tracking-Flat` | `Wisco-V1-Tracking-Flat` |
| mjlab task | `Hsim-G1-Velocity-Flat-v2` | `Hsim-G1-Velocity-Flat-ArmsFree` |
| experiment name (rsl_rl log folder) | `wisco_v2_velocity`, `wisco_v2_tracking` | `wisco_v1_velocity`, `wisco_v1_tracking` |
| experiment name | `g1_velocity_v2` | `g1_velocity_arms_free` |
| policy folder / id | `wisco_velocity_v2` | `wisco_v1_velocity_arms_free` |
| policy folder / id | `g1_velocity_flat_v1` | `g1_velocity_all_joints` |
| policy folder / id | `g1_velocity_v2`, `g1_velocity_v2_alt_J1w14899` (id `g1_v2_cand_J1w_model_14899`) | `g1_velocity_arms_free`, `g1_velocity_arms_free_alt_J1w14899` (id `g1_arms_free_cand_J1w_model_14899`) |
| robot profile | `g1_v2` | `g1_arms_free` |
| ONNX / W&B names | `wisco_velocity_v2_J3h_model_4650`, `g1_velocity_v2_J1v_model_11900`, ... | `wisco_v1_velocity_arms_free_J3h_...`, `g1_velocity_arms_free_J1v_...` |
| VM job ids and run folders | `J3*_wisco_vel_v2`, `J1*_g1_vel_v2` | `J3*_wisco_v1_vel`, `J1*_g1_vel_arms_free` |
| files | `hsim_tasks/g1_velocity_v2.py`, `scripts/train_g1_velocity_v2{,_resume}.sh`, `wisco_mjlab/v2_terms.py` | `g1_velocity_arms_free.py`, `train_g1_velocity_arms_free{,_resume}.sh`, `arms_free_terms.py` |
| change markers in task configs | `# V2:`, `# V2 (n):` | `# ARMS-FREE:`, `# ARMS-FREE (n):` |

`training/locomotion/wisco_v1/vm/resume_latest.sh` finds runs under either experiment name.

## Kept as they are

- **"Legs V2 Full Assembly.step"**: the mechanical team's CAD file name (and the CAD export in
  `robots/wisco_v1/original_from_team/`, kept byte-for-byte as received).
- **MyActuator "EtherCAT protocol V2.0"**, **LeRobot v2** datasets, `manifest_version`, perception "scorer v2",
  manipulation dataset/object-set versions (`v1`, `v2`, `v3`): other things' versions, not the robot's.
- The GPU VM checkout directory, `~/humanoid-sim-stack` (`VM_REPO=` overrides it in `training/locomotion/wisco_v1/vm/sync_to_vm.sh`).

## Removed at unification

- G1 candidate checkpoints `g1_velocity_v2_J1j`, `_J1n`, `_J1r10150`, `_J1r11500` (superseded by the deployed
  `g1_velocity_arms_free`; still in the pre-unification backup repository).
- Placeholder packages `hsim_arm_control` (arm control will be its own package) and `hsim_localization`
  (localization lives in `hsim_navigation`).
- The stub scaffold and the planner/BT/Nav2 branch's packages: see `porting_backlog.md` and the tags
  `legacy-scaffold-v0`, `archive/task-planner`.
