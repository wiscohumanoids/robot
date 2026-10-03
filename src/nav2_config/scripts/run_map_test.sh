#!/usr/bin/env bash
# Runs INSIDE the dev container: brings up the stub stack, one fake obstacle map and
# Nav2, runs avoidance_test.py against it, then tears everything down.
#   run_map_test.sh <scenario> <out_dir>
# scenario = <map> or <map>@<goal heading in degrees>, e.g. zigzag, obstacle_1@90.
# Maps: obstacle_1 | obstacle_2 | obstacle_3 | zigzag | unreachable.
# Writes <out_dir>/run_<scenario>.json (the recorded run) and, on failure, <out_dir>/logs/.
# Normally called by tools/run_nav2_avoidance.sh, which starts a fresh container per map.
set -uo pipefail
SCENARIO=${1:?usage: run_map_test.sh <scenario> <out_dir>}
MAP=${SCENARIO%@*}
YAW_ARGS=(); [[ "$SCENARIO" == *@* ]] && YAW_ARGS=(--goal-yaw-deg "${SCENARIO#*@}")
NAME=${SCENARIO/@/_yaw}
OUT=${2:?usage: run_map_test.sh <map> <out_dir>}
set +u; source /opt/ros/humble/setup.bash; source /home/ros2_ws/install/setup.bash; set -u
LOGS=/tmp/nav2_test_logs; mkdir -p "$LOGS" "$OUT"

cleanup() { jobs -p | xargs -r kill 2>/dev/null; sleep 1; }
trap cleanup EXIT

wait_for() {  # wait_for <seconds> <description> <command...>
  local n=$1 what=$2; shift 2
  for _ in $(seq 1 "$n"); do "$@" >/dev/null 2>&1 && return 0; sleep 1; done
  echo "TIMEOUT waiting for: $what" >&2; return 1
}
fail() { mkdir -p "$OUT/logs"; cp "$LOGS"/*.log "$OUT/logs/" 2>/dev/null; echo "FAIL: $1 (logs in $OUT/logs)"; exit 1; }

# slam:=external because test_map.launch.py takes over /map and map -> odom.
ros2 launch bringup full_stack.launch.py sim:=false navigation:=external slam:=external \
  > "$LOGS/stack.log" 2>&1 &
wait_for 40 "/odom" timeout 3 ros2 topic echo /odom --once || fail "stub stack did not come up"

ros2 launch nav2_config test_map.launch.py map:="$MAP" > "$LOGS/map.log" 2>&1 &
sleep 3
ros2 launch nav2_config nav2.launch.py > "$LOGS/nav2.log" 2>&1 &
wait_for 60 "Nav2 lifecycle active" grep -q "Managed nodes are active" "$LOGS/nav2.log" \
  || fail "Nav2 did not become active"

ros2 run nav2_config avoidance_test.py --map "$MAP" "${YAW_ARGS[@]}" --save "$OUT/run_$NAME.json"
rc=$?
[ $rc -ne 0 ] && { mkdir -p "$OUT/logs"; cp "$LOGS"/*.log "$OUT/logs/" 2>/dev/null; }
exit $rc
