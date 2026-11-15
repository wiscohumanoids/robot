# models/locomotion

One folder per exported policy, written by `training/locomotion/scripts/export_manifest.py`:

    <policy-id>/
      policy.onnx     gitignored
      manifest.yaml   deploy contract: joint order, kp/kd, default pose, action scale, obs terms, timing, I/O shapes
      env.yaml        mjlab's full env config from the run
      agent.yaml      PPO config from the run
      README.md       training notes: wall-clock, steps/s, eval results

Planned: g1_velocity_flat_v1 (drives /cmd_vel), g1_tracking_walk1_v1 (LAFAN1 demo).
