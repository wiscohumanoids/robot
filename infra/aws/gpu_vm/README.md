# GPU VM — g7e.2xlarge

1× RTX PRO 6000 Blackwell (96 GB), 8 vCPU, 64 GiB RAM. Runs mjlab training, VLA fine-tuning, and the VLA + perception servers. Stop it when idle.

- AMI: AWS Deep Learning Base GPU AMI (Ubuntu 24.04), NVIDIA driver ≥ 580
- Root EBS ≥ 200 GB (the instance-store NVMe is wiped on stop)
- Setup: `infra/aws/gpu_vm/setup.sh` from the repo root (checks the driver, installs uv and mjlab, verifies torch and Warp on the GPU), then `tools/vm.sh sync` from the laptop
