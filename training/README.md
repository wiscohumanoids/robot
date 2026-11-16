# training (GPU machine)

Offline training. Each pipeline produces something the stack loads: walking policies go to `models/locomotion/`, VLA
checkpoints are served by `services/vla_server/`. Machines and access: `infra/README.md`; the job queue that runs these
on the GPU machine: `infra/vm_jobs/`.

| Folder | What | Robots |
|---|---|---|
| [`locomotion/`](locomotion/README.md) | walking (velocity) policies with mjlab 1.6.0 and RSL-RL PPO; also the Python environment every mjlab job runs in | Wisco V1 (`wisco_v1/`), G1 (`hsim_tasks/`, `scripts/`) |
| [`tracking/`](tracking/README.md) | motion tracking: one policy reproduces one human motion clip | Wisco V1 (`wisco_v1/`), G1 |
| [`manipulation/`](manipulation/README.md) | humanoid pick-and-place: scripted demonstrations → LeRobot dataset → GR00T fine-tune | G1 with Dex1 grippers |

Arm-only manipulation training (the Panthera arm: ACT, pi0.5) lives with its pipelines in `manipulation/`.

**From training to the robot:** a walking run is exported to ONNX + `manifest.yaml` (`tools/vm.sh pull-policy`), checked
in plain MuJoCo (`tools/play.sh`, sim-to-sim), then loaded by the ROS stack with `policy:=` or a robot profile.
