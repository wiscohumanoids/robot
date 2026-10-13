#!/usr/bin/env bash
# Integration tests: behavior tree + task planner + Nav2 working together, in Docker.
#
#   tools/run_integration_tests.sh                    # every scenario that needs no API key
#   tools/run_integration_tests.sh --llm              # ... plus the ones that call the real model (.env)
#   tools/run_integration_tests.sh nav_then_pick canned_queue_editing    # chosen scenarios
#   tools/run_integration_tests.sh --list             # names, layers and maps
#
# Layers: tree_nav2 (planner off; plans are published by the test, so tree + Nav2 + manipulation
# are checked deterministically), canned (planner on without a key: planner -> tree -> Nav2),
# llm (planner with the key in .env: real model calls, slower, costs money, not deterministic).
#
# Needs the dev image (./docker/build.sh) and Docker running. Each scenario runs in its own fresh
# container: stub stack + fake obstacle map + Nav2 + the scenario (src/integration_tests/scripts/).
# Output: reports/integration/<scenario>.json (timeline), logs/<scenario>/ for a failed one,
# output/<scenario>.txt (what the scenario printed). Exits non-zero if any scenario fails.
set -uo pipefail
export PATH="$PATH:$HOME/.docker/bin"   # Docker Desktop's CLI is not always on PATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
IMAGE=wiscohumanoids/robot:latest
OUT="${REPO_ROOT}/reports/integration"
SCRIPTS=/home/ros2_ws/src/integration_tests/scripts

LLM=0; LIST_ONLY=0; NAMES=()
for arg in "$@"; do
  case "$arg" in
    --llm) LLM=1 ;;
    --list) LIST_ONLY=1 ;;
    -h|--help) sed -n '2,17p' "${BASH_SOURCE[0]}"; exit 0 ;;
    -*) echo "unknown option $arg (try --help)" >&2; exit 2 ;;
    *) NAMES+=("$arg") ;;
  esac
done

docker info >/dev/null 2>&1 || { echo "Docker is not running. Start Docker Desktop and retry." >&2; exit 2; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "Image $IMAGE not found. Run ./docker/build.sh first." >&2; exit 2; }

mkdir -p "$OUT/output"
MOUNTS=(-v "${REPO_ROOT}/src:/home/ros2_ws/src"
        -v robot_build:/home/ros2_ws/build -v robot_install:/home/ros2_ws/install -v robot_log:/home/ros2_ws/log
        -v "${OUT}:/out")
ROS_ENV='source /opt/ros/humble/setup.bash; source /home/ros2_ws/install/setup.bash'

# The scenario table comes from the driver, inside the container (no ROS on the host).
TABLE=$(docker run --rm "${MOUNTS[@]}" "$IMAGE" bash -c "$ROS_ENV; python3 $SCRIPTS/integration_driver.py --list --llm") \
  || { echo "could not list the scenarios" >&2; exit 2; }

if [ $LIST_ONLY -eq 1 ]; then echo "$TABLE"; exit 0; fi

SELECTED=()
if [ ${#NAMES[@]} -eq 0 ]; then
  while read -r name layer map; do
    [ -z "$name" ] && continue
    [ "$layer" = llm ] && [ $LLM -eq 0 ] && continue
    SELECTED+=("$name $layer $map")
  done <<< "$TABLE"
else
  for want in "${NAMES[@]}"; do
    line=$(echo "$TABLE" | awk -v n="$want" '$1 == n')
    [ -z "$line" ] && { echo "unknown scenario '$want' (see --list)" >&2; exit 2; }
    [[ "$line" == *" llm "* ]] && [ $LLM -eq 0 ] && { echo "'$want' calls the real model: add --llm" >&2; exit 2; }
    SELECTED+=("$line")
  done
fi
if [ $LLM -eq 1 ]; then
  [ -f "${REPO_ROOT}/.env" ] || { echo "--llm needs a .env with the API key at the repo root" >&2; exit 2; }
fi

echo "== building packages in the container"
docker run --rm "${MOUNTS[@]}" "$IMAGE" bash -c \
  'cd /home/ros2_ws && colcon build --symlink-install --packages-select humanoid_interfaces behavior_tree task_planner perception_stub manipulation_runner odom_adapter_stub nav2_config bringup 2>&1 | tail -3' \
  || { echo "build failed" >&2; exit 1; }

status=0; RESULTS=()
for entry in "${SELECTED[@]}"; do
  read -r name layer map <<< "$entry"
  echo "== $name  [$layer, map $map]"
  extra=()
  [ "$layer" = llm ] && extra=(-v "${REPO_ROOT}/.env:/home/ros2_ws/.env:ro")
  started=$SECONDS
  docker run --rm "${MOUNTS[@]}" ${extra[@]+"${extra[@]}"} "$IMAGE" "$SCRIPTS/run_scenario.sh" "$name" "$layer" "$map" /out 2>&1 \
    | tee "$OUT/output/$name.txt" | grep -E '^(PASS|FAIL|TIMEOUT|   )'
  rc=${PIPESTATUS[0]}
  took=$((SECONDS - started))
  if [ $rc -eq 0 ]; then RESULTS+=("PASS  ${took}s  $name"); else RESULTS+=("FAIL  ${took}s  $name"); status=1; fi
done

echo
echo "== summary"
printf '%s\n' "${RESULTS[@]}"
[ $status -eq 0 ] && echo "ALL PASSED" || echo "SOME FAILED (details: $OUT/output, $OUT/logs)"
exit $status
