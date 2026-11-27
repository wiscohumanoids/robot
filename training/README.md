# training (GPU machine)

Offline training. Each pipeline produces something the stack loads: walking policies go to `models/locomotion/`, VLA
checkpoints are served by `services/vla_server/`. Machines and access: `infra/README.md`; the job queue that runs these
on the GPU machine: `infra/vm_jobs/`.

**One entry point:** `training/train.sh <robot> <task> [args]` runs any of these jobs (`--help` lists them):
`wisco walk`, `wisco track <clip>`, `g1 walk`, `g1 walk-all-joints`, `g1 track <clip>`, `g1 manipulate <version>`,
`smoke`. The job queue (`infra/vm_jobs/jobs.yaml`) runs the same scripts unattended.

| Folder | What | Robots |
|---|---|---|
| [`locomotion/`](locomotion/README.md) | walking (velocity) policies with mjlab (RSL-RL 5.5.2) and RSL-RL PPO; also the Python environment every mjlab job runs in | Wisco V1 (`wisco_v1/`), G1 (`hsim_tasks/`, `scripts/`) |
| [`tracking/`](tracking/README.md) | motion tracking: one policy reproduces one human motion clip | Wisco V1 (`wisco_v1/`), G1 |
| [`manipulation/`](manipulation/README.md) | humanoid pick-and-place: scripted demonstrations → LeRobot dataset → GR00T fine-tune | G1 with Dex1 grippers |
| [`sim2real/`](sim2real/README.md) | making the simulator match the robot (PACE actuator identification) and the steps from a trained policy to first steps | Wisco V1 |

Arm manipulation training (ACT, pi0.5, GR00T on an arm profile) has its own entry point,
[`manipulation/finetune.sh`](../manipulation/README.md#fine-tuning-one-entry-point).

**From training to the robot:** a walking run is exported to ONNX + `manifest.yaml` (`tools/vm.sh pull-policy`), checked
in plain MuJoCo (`tools/play.sh`, sim-to-sim), then loaded by the ROS stack with `policy:=` or a robot profile.
