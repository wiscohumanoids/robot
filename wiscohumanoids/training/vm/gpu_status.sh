#!/usr/bin/env bash
# Run on the VM before any training: shows GPU memory in use and by which processes (e.g. the pi0.5 server),
# so a training run never squeezes it. Read-only.
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
