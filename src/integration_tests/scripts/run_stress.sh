#!/usr/bin/env bash
# Runs INSIDE the dev container: brings up the stub stack WITHOUT the behavior tree (the stress
# driver plays the tree) and runs stress_driver.py against the task planner.
#   run_stress.sh <out_dir> [stress_driver.py args...]      e.g. run_stress.sh /out --llm --soak-seconds 60
# The planner runs on its canned fallback unless /home/ros2_ws/.env (mounted by the host script)
# holds a key AND --llm is given. Normally called by tools/run_planner_stress.sh.
set -uo pipefail
OUT=${1:?usage: run_stress.sh <out_dir> [stress_driver.py args]}
shift
set +u; source /opt/ros/humble/setup.bash; source /home/ros2_ws/install/setup.bash; set -u
cd /home/ros2_ws
LOGS=/tmp/stress_logs; mkdir -p "$LOGS" "$OUT"

cleanup() { jobs -p | xargs -r kill 2>/dev/null; sleep 1; }
trap cleanup EXIT

wait_for() {
  local n=$1 what=$2; shift 2
  for _ in $(seq 1 "$n"); do "$@" >/dev/null 2>&1 && return 0; sleep 1; done
  echo "TIMEOUT waiting for: $what" >&2; return 1
}
fail() { mkdir -p "$OUT/logs"; cp "$LOGS"/*.log "$OUT/logs/" 2>/dev/null; echo "FAIL: $1 (logs in reports/stress/logs)"; exit 1; }

ros2 launch bringup full_stack.launch.py sim:=false behavior_tree:=false > "$LOGS/stack.log" 2>&1 &
planner_listening() { ros2 topic info /user_intent | grep -q "Subscription count: [1-9]"; }
wait_for 60 "the planner to listen on /user_intent" planner_listening || fail "the stack did not come up"

if [[ " $* " == *" --llm "* ]]; then
  grep -q "task_planner up: queued /user_intent -> LLM" "$LOGS/stack.log" \
    || fail "--llm given but the planner is not in LLM mode (is .env mounted and does it hold the key?)"
else
  grep -q "using the canned fallback planner" "$LOGS/stack.log" \
    || fail "the planner found an API key; the canned stress run needs it to be absent"
fi

python3 /home/ros2_ws/src/integration_tests/scripts/stress_driver.py --out "$OUT" --stack-log "$LOGS/stack.log" "$@"
rc=$?
if [ $rc -ne 0 ]; then mkdir -p "$OUT/logs"; cp "$LOGS"/*.log "$OUT/logs/" 2>/dev/null; fi
exit $rc
