#!/usr/bin/env bash
# Runs INSIDE the dev container: brings up the stack, a fake obstacle map and Nav2, then plays
# ONE integration scenario against it (integration_driver.py) and tears everything down.
#   run_scenario.sh <scenario> <layer> <map> <out_dir>
# layer = tree_nav2 (planner off, the driver publishes plans) | canned (planner without an API
# key) | llm (planner with the key from the mounted /home/ros2_ws/.env).
# Normally called by tools/run_integration_tests.sh, which starts a fresh container per scenario.
set -uo pipefail
SCENARIO=${1:?usage: run_scenario.sh <scenario> <layer> <map> <out_dir>}
LAYER=${2:?}
MAP=${3:?}
OUT=${4:?}
set +u; source /opt/ros/humble/setup.bash; source /home/ros2_ws/install/setup.bash; set -u
cd /home/ros2_ws
LOGS=/tmp/integration_logs; mkdir -p "$LOGS" "$OUT"
DRIVER=/home/ros2_ws/src/integration_tests/scripts/integration_driver.py

cleanup() { jobs -p | xargs -r kill 2>/dev/null; sleep 1; }
trap cleanup EXIT

wait_for() {  # wait_for <seconds> <description> <command...>
  local n=$1 what=$2; shift 2
  for _ in $(seq 1 "$n"); do "$@" >/dev/null 2>&1 && return 0; sleep 1; done
  echo "TIMEOUT waiting for: $what" >&2; return 1
}
fail() { mkdir -p "$OUT/logs/$SCENARIO"; cp "$LOGS"/*.log "$OUT/logs/$SCENARIO/" 2>/dev/null
         echo "FAIL  $SCENARIO: $1 (logs in reports/integration/logs/$SCENARIO)"; exit 1; }

PLANNER=true; [ "$LAYER" = tree_nav2 ] && PLANNER=false

# slam:=external because test_map.launch.py takes over /map and map -> odom.
ros2 launch bringup full_stack.launch.py sim:=false navigation:=external slam:=external planner:=$PLANNER \
  > "$LOGS/stack.log" 2>&1 &
wait_for 60 "/odom" timeout 3 ros2 topic echo /odom --once || fail "stub stack did not come up"

if [ "$LAYER" = llm ]; then
  wait_for 30 "planner on the LLM" grep -q "task_planner up: queued /user_intent -> LLM" "$LOGS/stack.log" \
    || fail "the planner is not in LLM mode (is .env mounted and does it hold the key?)"
elif [ "$LAYER" = canned ]; then
  wait_for 30 "planner on the canned fallback" grep -q "using the canned fallback planner" "$LOGS/stack.log" \
    || fail "the planner found an API key; the canned layer must run without one"
fi

ros2 launch nav2_config test_map.launch.py map:="$MAP" > "$LOGS/map.log" 2>&1 &
sleep 3
ros2 launch nav2_config nav2.launch.py > "$LOGS/nav2.log" 2>&1 &
wait_for 90 "Nav2 lifecycle active" grep -q "Managed nodes are active" "$LOGS/nav2.log" \
  || fail "Nav2 did not become active"

python3 "$DRIVER" --scenario "$SCENARIO" --out "$OUT/$SCENARIO.json" --stack-log "$LOGS/stack.log"
rc=$?
if [ $rc -ne 0 ]; then mkdir -p "$OUT/logs/$SCENARIO"; cp "$LOGS"/*.log "$OUT/logs/$SCENARIO/" 2>/dev/null; fi
exit $rc
