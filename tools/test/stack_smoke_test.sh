#!/usr/bin/env bash
# Runs inside the ROS container (via `tools/ros.sh test`).
# Launches a stack, checks it against the interface contract, then drives it.
#   STACK=stub (default) -> stub_stack.launch.py      STACK=sim -> sim_stack.launch.py (MuJoCo + policy)
#   ENV=test/table_room  -> load that environment and also check /map (contract phase 2)
#   ROBOT=g1             -> robot profile (default wisco: stereo camera, build_wisco/ envs; also g1, g1_arms_free)
#   BACKEND=ros2_control -> STACK=sim through the ros2_control backend (no ENV: it runs the bare training world)
#   BACKEND=ethercat_sim -> the same, one layer down: simulated EtherCAT drives under the robot's controller chain
set -uo pipefail
STACK="${STACK:-stub}"
LOG=/tmp/${STACK}_stack.log
# Own process group, so cleanup can stop the launcher AND every node it started.
# (Background jobs ignore SIGINT in non-interactive bash, and nodes inherit that,
# so a plain Ctrl-C to the launcher leaves orphaned nodes behind.)
# drive_check plays the teleop role itself, so the viewer's teleop bridge stays off
EXTRA=(); [[ $STACK == sim ]] && EXTRA=(viewer:=false)
PHASE=1; if [[ -n "${ENV:-}" ]]; then EXTRA+=(env:=$ENV); PHASE=2; fi
if [[ -n "${ROBOT:-}" ]]; then EXTRA+=(robot:=$ROBOT); fi
if [[ -n "${BACKEND:-}" ]]; then EXTRA+=(backend:=$BACKEND); fi
setsid ros2 launch hsim_bringup ${STACK}_stack.launch.py foxglove:=false "${EXTRA[@]}" > "$LOG" 2>&1 &
LAUNCH_PID=$!
cleanup() {
  kill -TERM -- -"$LAUNCH_PID" 2>/dev/null; sleep 2
  kill -KILL -- -"$LAUNCH_PID" 2>/dev/null; wait "$LAUNCH_PID" 2>/dev/null
}
trap cleanup EXIT
sleep 4

if ! kill -0 $LAUNCH_PID 2>/dev/null; then
  echo "Stack failed to start:"; cat "$LOG"; exit 1
fi

echo "== Contract check =="
ros2 run hsim_bringup contract_check --phase $PHASE; C=$?
echo "== Drive check =="
if [[ $STACK == sim ]]; then DRIVE=(--stop-tolerance 0.15); else DRIVE=(); fi
ros2 run hsim_bringup drive_check "${DRIVE[@]}"; D=$?

if (( C != 0 || D != 0 )); then
  echo "--- stack log (last 30 lines) ---"; tail -30 "$LOG"
  exit 1
fi
echo "STACK SMOKE TEST PASSED ($STACK${ENV:+, $ENV}${ROBOT:+, $ROBOT})"
