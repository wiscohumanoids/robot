#!/usr/bin/env bash
# Stress tests for the task planner, in Docker.
#
#   tools/run_planner_stress.sh                          # canned planner (no key): free, deterministic
#   tools/run_planner_stress.sh --llm                    # real model calls (.env): reduced set, costs money
#   tools/run_planner_stress.sh --soak 120               # soak for 120 s (default 45)
#   tools/run_planner_stress.sh burst,command_storm  # chosen scenarios
#
# What it does: starts the stub stack WITHOUT the behavior tree and plays the tree itself, so the
# tree can be made slow, silent, stale, hanging or failing on demand, then hits the planner with:
#   burst             200 intents in a burst while the tree is busy: all answered and queued in order, none rejected
#   command_storm     400 random submits / removes / clears / lists / garbage commands, each answered once
#   garbage_intents   blank, huge (20 kB), unicode, injection and JSON-looking text: one verdict each
#   misbehaving_tree  tree never starts / stale SUCCEEDED / hangs / always fails: resends, no plan storm
#   latency           QUEUED -> SENT over 50 sequential intents (p95 must stay under 2.5 s)
#   soak              random traffic for a while: memory growth, still serving at the end
# and, with --llm: llm_varied_requests, llm_queue_order, llm_replan_storm.
# After every scenario: the planner is alive, every published plan is valid (known skills and objects,
# goals inside the map, hand rules), and no plan was sent while the tree was busy.
#
# Needs the dev image (./docker/build.sh). Output: reports/stress/ (stress_results.json, output.txt,
# logs/ when something failed). Exits non-zero if any scenario fails.
set -uo pipefail
export PATH="$PATH:$HOME/.docker/bin"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
IMAGE=wiscohumanoids/robot:latest
OUT="${REPO_ROOT}/reports/stress"

LLM=0; SOAK=""; SCEN=""
while [ $# -gt 0 ]; do
  case "$1" in
    --llm) LLM=1 ;;
    --soak) shift; SOAK="${1:?--soak needs a number of seconds}" ;;
    -h|--help) sed -n '2,23p' "${BASH_SOURCE[0]}"; exit 0 ;;
    -*) echo "unknown option $1 (try --help)" >&2; exit 2 ;;
    *) SCEN="$1" ;;
  esac
  shift
done

docker info >/dev/null 2>&1 || { echo "Docker is not running. Start Docker Desktop and retry." >&2; exit 2; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "Image $IMAGE not found. Run ./docker/build.sh first." >&2; exit 2; }

rm -rf "$OUT"; mkdir -p "$OUT"
MOUNTS=(-v "${REPO_ROOT}/src:/home/ros2_ws/src"
        -v robot_build:/home/ros2_ws/build -v robot_install:/home/ros2_ws/install -v robot_log:/home/ros2_ws/log
        -v "${OUT}:/out")
ARGS=()
if [ $LLM -eq 1 ]; then
  [ -f "${REPO_ROOT}/.env" ] || { echo "--llm needs a .env with the API key at the repo root" >&2; exit 2; }
  MOUNTS+=(-v "${REPO_ROOT}/.env:/home/ros2_ws/.env:ro")
  ARGS+=(--llm)
fi
[ -n "$SOAK" ] && ARGS+=(--soak-seconds "$SOAK")
[ -n "$SCEN" ] && ARGS+=(--scenarios "$SCEN")

echo "== building packages in the container"
docker run --rm "${MOUNTS[@]}" "$IMAGE" bash -c \
  'cd /home/ros2_ws && colcon build --symlink-install --packages-select humanoid_interfaces behavior_tree task_planner perception_stub manipulation_runner bringup 2>&1 | tail -3' \
  || { echo "build failed" >&2; exit 1; }

echo "== stress run ($([ $LLM -eq 1 ] && echo 'LLM planner' || echo 'canned planner'))"
docker run --rm "${MOUNTS[@]}" "$IMAGE" /home/ros2_ws/src/integration_tests/scripts/run_stress.sh /out ${ARGS[@]+"${ARGS[@]}"} 2>&1 \
  | tee "$OUT/output.txt"
rc=${PIPESTATUS[0]}
[ $rc -eq 0 ] && echo "ALL STRESS SCENARIOS PASSED" || echo "SOME FAILED (details: $OUT)"
exit $rc
