# models/locomotion

Deployed walking policies. One folder per policy, written by `training/locomotion/scripts/export_manifest.py`;
the ROS node `locomotion` loads the folder its robot profile names (`ros2_ws/src/hsim_bringup/config/robots/`).

    <policy-id>/
      policy.onnx     the policy (tracked: under 1 MB each)
      manifest.yaml   deploy contract: joint order, kp/kd, default pose, action scale, observation terms, timing, I/O shapes
      env.yaml        mjlab's full environment config from the training run
      agent.yaml      PPO config from the training run
      README.md       training notes (where written): wall-clock, steps/s, evaluation

| Policy | Robot profile | Recipe | Drives | Notes |
|---|---|---|---|---|
| `wisco_v1_velocity_arms_free` | `wisco` (default) | arms-free (manifest v2) | 15: legs + waist; arms and head held | trained up to 1.0 m/s forward |
| `g1_velocity_all_joints` | `g1` | all-joints (manifest v1) | all 29 | mjlab's stock G1 task; see its README |
| `g1_velocity_arms_free` | `g1_arms_free` | arms-free (manifest v2) | 15: legs + waist; arms held | fine-tuned against rocking |
| `g1_velocity_arms_free_alt` | (none; swap in with `policy:=`) | arms-free | 15 | a sibling run: less rocking and drift, quicker steps, more foot slide |

Recipes: **all-joints** = the policy controls every joint (cannot be combined with manipulation); **arms-free** =
legs + waist/torso only; the other joints follow `/arm/joint_targets` or hold the default pose, so the arms stay free
for manipulation while the robot walks.

Swap a policy without changing the profile: `ros2 launch hsim_bringup sim_stack.launch.py policy:=/hsim/models/locomotion/<policy-id>`. Add one:
`tools/dev/vm.sh pull-policy <run-dir> <policy-id>` (export on the GPU machine and copy here), then point a robot profile at it.
