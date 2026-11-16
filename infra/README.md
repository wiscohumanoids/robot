# infra

How each machine is set up. Scripts are added only after setup has been done once by hand and works. Machines,
access and costs: `docs/infra.md`.

| Folder | What |
|---|---|
| [`docker/`](docker/README.md) | the ROS 2 Jazzy dev container (`tools/ros.sh` drives it): Nav2, RTAB-Map, BehaviorTree.CPP, MuJoCo 3.11, ONNX Runtime |
| `aws/gpu_vm/` | GPU VM setup and checks (`setup.sh`: driver, uv, EGL) |
| `aws/cpu_vm/` | notes for the (currently terminated) CPU VM |
| [`vm_jobs/`](vm_jobs/README.md) | overnight job runner for the GPU VM (`jobs.yaml`, `vmjobs.py`), evaluation monitor, morning report |

The robot's own computers (Jetson Thor, RK3588: PREEMPT_RT kernel, EtherCAT master) are not set up from here yet;
see `hardware/README.md`.
