#!/usr/bin/env bash
# Build the robot workspace dev container (native architecture: works on
# x86_64 and Apple-silicon/arm64 hosts). Run from anywhere:
#   ./docker/build.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
docker build -t wiscohumanoids/robot:latest -f "${SCRIPT_DIR}/Dockerfile" "${REPO_ROOT}"
