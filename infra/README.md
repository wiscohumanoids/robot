# infra

How each development machine is set up, and the training job queue. The robot's own computers (RK3588, Jetson Thor or
Jetson Orin NX) are described in [`hardware/README.md`](../hardware/README.md).

| Folder | What |
|---|---|
| [`docker/`](docker/README.md) | the ROS 2 Jazzy development image; `tools/ros.sh` starts it with the repo mounted at `/hsim` |
| [`aws/gpu_vm/`](aws/gpu_vm/README.md) | the GPU training machine: requirements and `setup.sh` |
| [`robot_network/`](robot_network/README.md) | the robot's two computers on one cable: ROS 2 kept on it (Fast DDS allowlist, static peers), the RK3588's clock following the Jetson's (chrony) |
| [`rk3588/`](rk3588/README.md) | the robot's real-time computer: PREEMPT_RT kernel, core isolation, IRQ pinning, cyclictest, the `update_rate` ramp |
| [`vm_jobs/`](vm_jobs/README.md) | the job queue on the GPU machine: training runs, servers and pipeline steps, with status, resume and reports |

## Machines

| | GPU machine | Laptop |
|---|---|---|
| Instance | AWS g7e.2xlarge | any (macOS or Linux) |
| Hardware | 1× RTX PRO 6000 Blackwell (96 GB), 8 vCPU, 64 GiB RAM | |
| OS | Ubuntu 24.04 (AWS Deep Learning Base GPU AMI) | |
| Runs | mjlab training, LeRobot / GR00T fine-tuning, perception and VLA servers | editing, the ROS 2 stack in Docker, the MuJoCo viewer, teleop |
| On when | training jobs and server sessions | always |

mjlab simulates every environment on the GPU, so 8 vCPUs are enough. Requirements: NVIDIA driver 580 or newer (needed
for Blackwell), Python via `uv`, `MUJOCO_GL=egl` for headless rendering; `aws/gpu_vm/setup.sh` checks all of it.

## Access

Add a host alias to `~/.ssh/config` on the laptop; the scripts use `hsim-gpu` (override with `VM_HOST`):

```
Host hsim-gpu
  HostName <gpu-machine-public-dns>
  User ubuntu
  IdentityFile ~/.ssh/<key>.pem
```

Security groups only need SSH from your address. Service ports (perception 8100, VLA, Foxglove) are reached through SSH
tunnels, never opened publicly, e.g. `ssh -f -N -L 8100:localhost:8100 hsim-gpu`.

## Moving code and results (`tools/dev/vm.sh`)

- `tools/dev/vm.sh sync`: copy the repo to `~/robot` on the GPU machine (code only; `VM_REPO` changes the folder).
- `tools/dev/vm.sh pull-policy <run-dir> <id>`: export a trained walking policy and copy it into `models/locomotion/<id>/`.
- `tools/dev/vm.sh pull-results`: copy small results (ONNX, reports, plots, videos, newest checkpoint per run).
- `tools/dev/vm.sh jobs <command>`: the job queue (`vm_jobs/`).
- Experiment tracking: Weights & Biases (mjlab stores motion clips there for the tracking task); runs log offline by default.

## Cost rules

- The GPU machine costs about $3.4/hour on demand (about $1.5 spot). Stop it when a run ends.
- Training checkpoints every 50 iterations, so an interruption loses minutes; runs resume (`tools/dev/vm.sh jobs resume <job>`).
- Keep outputs on the EBS volume (`~/hsim_runs`, `~/hsim_data`), not the instance-store NVMe, which is wiped on stop.
