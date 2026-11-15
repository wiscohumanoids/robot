#!/usr/bin/env bash
# Helper for the ROS 2 Jazzy container.
#   tools/ros.sh up      build the image (first time ~10-20 min) and start the container
#   tools/ros.sh shell   open a shell inside it (ROS + workspace already sourced)
#   tools/ros.sh build   colcon build the workspace
#   tools/ros.sh test [stub|sim] [env] [robot]  build, launch the stub (default) or real MuJoCo+policy stack
#                        (optionally in an environment, e.g. test/table_room, and for robot g1|wisco),
#                        then run the contract check + a drive test
#   tools/ros.sh autotest [name] [robot]  end-to-end autonomy checks (all, or only those whose name contains [name];
#                        robot g1 (default) | wisco)
#   tools/ros.sh stop    stop every running stack node in the container (keeps the container)
#   tools/ros.sh down    stop the container
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE=(docker compose -f "$ROOT/infra/docker/compose.yaml")
in_ros() { docker exec -i hsim-ros bash -lc "source /opt/ros/jazzy/setup.bash; cd /hsim/ros2_ws; $*"; }

case "${1:-}" in
  up)    "${COMPOSE[@]}" up -d --build ;;
  shell) docker exec -it hsim-ros bash ;;
  build) in_ros "colcon build --symlink-install --event-handlers console_direct-" ;;
  test)  "$0" stop >/dev/null; in_ros "colcon build --symlink-install && source install/setup.bash && STACK=${2:-stub} ENV=${3:-} ROBOT=${4:-} /hsim/tools/stack_smoke_test.sh" ;;
  stop)  # every node we or ROS started (anything from the workspace or /opt/ros), except the ros2 daemon.
          # The pattern text is in this shell's own command line, so exclude our own PID ($$).
          docker exec hsim-ros bash -c 'pkill -INT -f "[r]os2 launch"; sleep 2; \
            for p in $(pgrep -f "/hsim/ros2_ws/install/|/opt/ros/jazzy/lib/|ros2 launch|/hsim/tools/eval/"); do \
              [ "$p" != "$$" ] && kill -9 "$p" 2>/dev/null; done; \
            n=$(pgrep -f "/hsim/ros2_ws/install/|/opt/ros/jazzy/lib/" | grep -vxc "$$" || true); \
            if [ "$n" = 0 ]; then rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_* 2>/dev/null; fi; \
            echo "stack stopped ($n processes left; /dev/shm $(df -h /dev/shm | awk "NR==2{print \$5}") used)"' ;;
  autotest) "$0" stop >/dev/null; in_ros "colcon build --symlink-install >/dev/null && source install/setup.bash && ROBOT=${3:-g1} /hsim/tools/autonomy_test.sh ${2:-}" ;;
  down)  "${COMPOSE[@]}" down ;;
  *) sed -n '2,8p' "$0"; exit 1 ;;
esac
