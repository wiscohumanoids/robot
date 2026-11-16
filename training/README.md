# training (GPU VM, not ROS)

Offline training jobs. Each one produces artifacts that the stack loads: walking policies go to `models/locomotion/`,
VLA checkpoints are served by `services/vla_server/`. Machines and access: `docs/infra.md`; the overnight job runner:
`infra/vm_jobs/`.

| Folder | What | Robots |
|---|---|---|
| [`locomotion/`](locomotion/README.md) | walking (velocity) policies with mjlab 1.6.0 + RSL-RL PPO; the uv project every mjlab job runs in | G1 (`hsim_tasks/`, `scripts/`), Wisco V1 (`wisco_v1/`) |
| [`tracking/`](tracking/README.md) | motion tracking on LAFAN1 clips (BeyondMimic-style) | G1, Wisco V1 (`wisco_v1/`) |
| [`manipulation/`](manipulation/README.md) | standing pick-and-place: scripted demos → LeRobot → GR00T N1.6 | G1 + Dex1 grippers |

Arm-only manipulation training (Panthera arm: ACT, pi0.5, VLA-Adapter) lives with its pipelines in `manipulation/`.
