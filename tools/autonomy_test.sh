#!/usr/bin/env bash
# End-to-end autonomy checks (runs inside the container; from the Mac: tools/ros.sh autotest).
# Each check launches a fresh stack in the test room and grades with privileged ground truth.
set -uo pipefail
set +u; source /opt/ros/jazzy/setup.bash; source /hsim/ros2_ws/install/setup.bash; set -u
E=/hsim/tools/eval; FAILS=0
ONLY="${1:-}"   # optional: run only the checks whose name contains this
ROBOT="${ROBOT:-wisco}"   # robot profile: wisco (stereo camera, build_wisco/ envs) | g1 | g1_arms_free
SENSORS="${SENSORS:-}" # override the profile's sensors (e.g. SENSORS=stereo for the G1)
PERCEPTION="${PERCEPTION:-stub}"   # graded checks default to the stub (repeatable); PERCEPTION=server for the real one
export HSIM_BUILD_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('/hsim/ros2_ws/src/hsim_bringup/config/robots/$ROBOT.yaml'))['env_build_dir'])")
# SLAM checks follow the robot's sensors. A lidar robot (G1) sees all around from the start: slam_toolbox, goal sent
# directly. A stereo robot (Wisco V1, forward-only ZED X) uses RTAB-Map visual SLAM and first looks around, as in the
# real flow ("look_around, navigate_to: ..."): Nav2 does not plan into unexplored space in SLAM mode.
PROFILE_SENSORS=$(python3 -c "import yaml; print(yaml.safe_load(open('/hsim/ros2_ws/src/hsim_bringup/config/robots/$ROBOT.yaml'))['default_sensors'])")
if [[ "${SENSORS:-$PROFILE_SENSORS}" == stereo ]]; then SLAM_MAPPING=rtabmap; LOOK_FIRST=1; else SLAM_MAPPING=slam; LOOK_FIRST=0; fi
nav_slam_check() {   # nav_slam_check <x> <y> <yaw> --frame-offset ...: optional look-around, then the Nav2 goal
  if [[ $LOOK_FIRST == 1 ]]; then
    local r; r=$(python3 $E/run_plan.py "look_around" | tail -1)
    [[ "$r" == *SUCCESS* ]] || { echo "look_around before the goal: $r"; return 1; }
  fi
  python3 $E/nav_goal.py "$@"
}
export -f nav_slam_check; export E LOOK_FIRST
run() {   # run <name> <launch args> -- <check command>
  local name="$1"; shift; [[ -n "$ONLY" && "$name" != *"$ONLY"* ]] && return 0; local args=(); while [[ "$1" != "--" ]]; do args+=("$1"); shift; done; shift
  setsid ros2 launch hsim_bringup autonomy.launch.py viewer:=false robot:=$ROBOT ${SENSORS:+sensors:=$SENSORS} perception:=$PERCEPTION "${args[@]}" > "/tmp/auto_${name}.log" 2>&1 &
  local L=$!; sleep 20
  local out; out=$(timeout 300 "$@" 2>&1 | tail -1); local rc=$?
  kill -TERM -- -"$L" 2>/dev/null; sleep 3; kill -KILL -- -"$L" 2>/dev/null; wait "$L" 2>/dev/null
  local falls; falls=$(grep -c "robot fell" "/tmp/auto_${name}.log")
  # the verdict is the last "-> X" in the line (or a bare PASS / plan SUCCESS for checks without one)
  local verdict="${out##*-> }"
  [[ "$out" != *"-> "* ]] && verdict="$out"
  if [[ "$verdict" == PASS* || "$verdict" == *SUCCESS* || "$verdict" == "PASS" ]] && [[ $falls == 0 ]]; then echo "PASS  $name: $out"; else echo "FAIL  $name: $out (falls: $falls)"; FAILS=$((FAILS+1)); fi
}
run nav_known_map   env:=test/table_room mapping:=map  planner:=false -- python3 $E/nav_goal.py 1.0 -2.2 0
run nav_slam        env:=test/table_room mapping:=$SLAM_MAPPING planner:=false -- bash -c 'nav_slam_check 4.0 -2.2 0 --frame-offset -3.0 0.0 0'
run dropped_box     env:=test/table_room mapping:=map  planner:=false -- python3 $E/dropped_obstacle.py 1.0 0.0
run bt_plan         env:=test/table_room mapping:=map  planner:=false -- python3 $E/run_plan.py "navigate_to:table" "find_object:red block" "say:found it"
# an open viewer (idle keys) must not block navigation; its arrow keys must still take over
run viewer_idle     env:=test/table_room mapping:=map  planner:=false viewer:=true -- python3 $E/viewer_idle_nav.py 1.0 -2.2 0
run planner_flow    env:=test/table_room mapping:=map  planner:=false -- python3 $E/fake_claude_flow.py
# reach: navigate_to must end within arm's reach, square-on, on a side it can stand at (graded on true geometry)
run reach_table     env:=test/table_room mapping:=map  planner:=false -- bash $E/reach_plan.sh test/table_room table "navigate_to:table"
run reach_block     env:=test/table_room mapping:=map  planner:=false -- bash $E/reach_plan.sh test/table_room red_block "navigate_to:table" "find_object:red block" "navigate_to:red block"
run kitchen_fridge  env:=robocasa/kitchen_l1_s1 mapping:=map planner:=false -- bash $E/reach_plan.sh robocasa/kitchen_l1_s1 fridge_main_group "look_around" "navigate_to:fridge"
# SLAM: the fridge is seen before it is mapped; Nav2 must plan into not-yet-mapped space to reach it
run slam_fridge     env:=robocasa/kitchen_l1_s1 mapping:=$SLAM_MAPPING planner:=false -- bash $E/reach_plan.sh robocasa/kitchen_l1_s1 fridge_main_group "look_around" "navigate_to:fridge"
run kitchen_sink    env:=robocasa/kitchen_l1_s1 mapping:=map planner:=false -- bash $E/reach_plan.sh robocasa/kitchen_l1_s1 sink_main_group "look_around" "navigate_to:sink"
echo; [[ $FAILS == 0 ]] && echo "AUTONOMY TESTS PASSED ($ROBOT)" || echo "$FAILS AUTONOMY TEST(S) FAILED ($ROBOT)"
exit $FAILS
