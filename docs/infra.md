# Infrastructure

## Machines

> **Current status (2026-11-15):** the CPU VM is terminated; the ROS stack runs in Docker on the Mac and training runs on
> the GPU VM. The robot's own computers (Jetson Thor, RK3588) are covered in `hardware/README.md`. The GPU VM keeps its
> checkout in `~/humanoid-sim-stack` (the folder name predates the unification; `tools/sync_to_vm.sh` syncs this repo there).

| | CPU VM | GPU VM | Laptop |
|---|---|---|---|
| Instance | c7a.48xlarge | g7e.2xlarge | Mac |
| Hardware | 192 vCPU (AMD EPYC), no GPU | 1× RTX PRO 6000 Blackwell Server Edition (96 GB), 8 vCPU, 64 GiB RAM, 1.9 TB local NVMe | |
| OS | Ubuntu 24.04 | Ubuntu 24.04 (AWS Deep Learning Base GPU AMI) | macOS |
| Runs | ROS 2 Jazzy, MuJoCo, Nav2, BT, ONNX inference | mjlab training, LeRobot fine-tuning, VLA + perception servers | editing, ROS 2 stack in Docker, MuJoCo viewer, Foxglove, keyboard teleop |
| On when | dev sessions | training jobs, manipulation/perception sessions | always |

## GPU VM requirements (mjlab)

- NVIDIA driver **580 or newer**. Blackwell needs it; mjlab's `cu126` fallback does not support Blackwell.
- Python 3.11–3.14, installed by `uv`.
- `MUJOCO_GL=egl` for headless rendering (motion conversion videos, training videos).
- `infra/aws/gpu_vm/setup.sh` checks all of this.

The 8 vCPUs are enough: mjlab simulates all environments on the GPU, so the CPU only drives the GPU.

## Access

Add host aliases to `~/.ssh/config` on the laptop. The scripts assume these names:

```
Host hsim-gpu
  HostName <gpu-vm-public-dns>
  User ubuntu
  IdentityFile ~/.ssh/<key>.pem
Host hsim-cpu
  HostName <cpu-vm-public-dns>
  User ubuntu
  IdentityFile ~/.ssh/<key>.pem
```

The security groups only need SSH from your IP. Service ports (VLA, perception, Foxglove) are reached through SSH tunnels, never opened publicly.

## Moving code and artifacts

- Code: `tools/sync_to_vm.sh hsim-gpu` (rsync of the repo, without build outputs or weights).
- Trained policies: `tools/pull_policy.sh <run-dir>` copies the ONNX plus manifest from the GPU VM into `models/locomotion/`.
- Experiment tracking: Weights & Biases. mjlab uses it to store motion clips and checkpoints, so it's required for the tracking task.

## Cost rules

- g7e.2xlarge on demand costs about $3.4/hr (spot about $1.5/hr). Check the AWS pricing page for your region.
- Stop the GPU VM when a training run ends. Training writes checkpoints every 50 iterations, so a spot interruption only loses minutes, and runs can resume (`--agent.resume True`).
- Keep training outputs on the EBS volume, not the 1.9 TB instance store: the instance store is wiped when the VM stops.
