#!/usr/bin/env bash
# Build the robot workspace dev container. Run from the repo root:
#   ./docker/build.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
docker build -t wiscohumanoids/robot:latest -f "${SCRIPT_DIR}/Dockerfile" "${REPO_ROOT}"
