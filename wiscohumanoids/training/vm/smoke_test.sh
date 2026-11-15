#!/usr/bin/env bash
# On the VM: 5 PPO updates on 256 robots for the Wisco v2 velocity task. Proves it builds and trains end to end.
# Outputs: ~/hsim_runs/wisco_smoke/
set -euo pipefail
NUM_ENVS=256 ITERS=5 "$(dirname "$0")/train_velocity.sh" --log-root "$HOME/hsim_runs/wisco_smoke" "$@"
echo "Smoke test passed. Outputs in ~/hsim_runs/wisco_smoke/"
