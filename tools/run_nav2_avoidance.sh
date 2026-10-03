#!/usr/bin/env bash
# One command: run the Nav2 obstacle-avoidance tests in Docker and build an HTML replay.
#
#   tools/run_nav2_avoidance.sh                      # every default scenario (below)
#   tools/run_nav2_avoidance.sh zigzag obstacle_1@90 # chosen scenarios; <map>@<deg> also checks the goal heading
#
# Needs the dev image (./docker/build.sh) and Docker running. Each map runs in its own
# fresh container, so a leftover process from one run can never break the next.
# Output: reports/nav2_avoidance/report.html (open it in a browser), plus run_*.json
# and, for a failed map, logs/. Exits non-zero if any map fails.
set -uo pipefail
export PATH="$PATH:$HOME/.docker/bin"   # Docker Desktop's CLI is not always on PATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
IMAGE=wiscohumanoids/robot:latest
OUT="${REPO_ROOT}/reports/nav2_avoidance"
MAPS=("$@"); [ ${#MAPS[@]} -eq 0 ] && MAPS=(obstacle_1 obstacle_2 obstacle_3 zigzag unreachable obstacle_1@90 obstacle_1@180)

docker info >/dev/null 2>&1 || { echo "Docker is not running. Start Docker Desktop and retry." >&2; exit 2; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "Image $IMAGE not found. Run ./docker/build.sh first." >&2; exit 2; }

rm -rf "$OUT"; mkdir -p "$OUT"
MOUNTS=(-v "${REPO_ROOT}/src:/home/ros2_ws/src"
        -v robot_build:/home/ros2_ws/build -v robot_install:/home/ros2_ws/install -v robot_log:/home/ros2_ws/log
        -v "${OUT}:/out")

echo "== building packages in the container"
docker run --rm "${MOUNTS[@]}" "$IMAGE" bash -c \
  'cd /home/ros2_ws && colcon build --symlink-install --packages-select odom_adapter_stub nav2_config bringup 2>&1 | tail -3' \
  || { echo "build failed" >&2; exit 1; }

status=0
for map in "${MAPS[@]}"; do
  echo "== $map"
  docker run --rm "${MOUNTS[@]}" "$IMAGE" /home/ros2_ws/src/nav2_config/scripts/run_map_test.sh "$map" /out \
    || status=1
done

echo "== report"
python3 "${REPO_ROOT}/tools/nav2_avoidance_report.py" "$OUT" || status=1
[ $status -eq 0 ] && echo "ALL PASSED" || echo "SOME FAILED (see above and $OUT/logs)"
exit $status
