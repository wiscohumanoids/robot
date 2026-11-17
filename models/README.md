# models

Trained policies the stack runs, with the files that fix how they are run.

| Folder | What |
|---|---|
| [`locomotion/`](locomotion/README.md) | walking policies: one folder per policy (`policy.onnx` + `manifest.yaml` + the training configs), loaded by the ROS node `locomotion` |

Manipulation policies are not stored here: they are large and run on the GPU machine behind `services/vla_server/`
(checkpoints from `training/manipulation/` or `manipulation/`).
