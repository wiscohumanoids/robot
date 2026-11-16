# models/locomotion

Deployed walking policies. One folder per policy, written by `training/locomotion/scripts/export_manifest.py`;
the ROS node `locomotion` loads the folder its robot profile names (`ros2_ws/src/hsim_bringup/config/robots/`).

    <policy-id>/
      policy.onnx     the policy (tracked: under 1 MB each)
      manifest.yaml   deploy contract: joint order, kp/kd, default pose, action scale, observation terms, timing, I/O shapes
      env.yaml        mjlab's full environment config from the training run
      agent.yaml      PPO config from the training run
      README.md       training notes (where written): wall-clock, steps/s, evaluation

| Policy | Robot profile | Recipe | Drives | Run |
|---|---|---|---|---|
| `wisco_v1_velocity_arms_free` | `wisco` (default) | arms-free (manifest v2) | 15: legs + waist; arms and head held | J3h model_4650 (2026-11-10), trained up to 1.0 m/s forward |
| `g1_velocity_all_joints` | `g1` | all-joints (manifest v1) | all 29 | 2026-11-08, model_9999 (see its README) |
| `g1_velocity_arms_free` | `g1_arms_free` | arms-free (manifest v2) | 15: legs + waist; arms held | J1v model_11900 (2026-11-11, anti-rocking fine-tune) |
| `g1_velocity_arms_free_alt_J1w14899` | (none; swap in with `policy:=`) | arms-free | 15 | J1w model_14899: less rocking and drift, quicker steps and more foot slide |

Recipes: **all-joints** = the policy controls every joint (cannot be combined with manipulation); **arms-free** =
legs + waist/torso only, the other joints follow `/arm/joint_targets` or hold the default pose (decision 0002).
Superseded G1 checkpoints (J1j, J1n, J1r) are in the pre-unification backup repository, not here. Run and file names
recorded before the V1 rename: `docs/migration/renames.md`.
