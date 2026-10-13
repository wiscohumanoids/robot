#!/usr/bin/env bash
# Run the robot workspace dev container with YOUR working copy mounted, so edits
# on the host show up in the container immediately.
#
#   ./docker/run.sh                    # interactive shell, with the stub stack already
#                                      # running in the background (log: /tmp/stack.log).
#                                      # Do NOT run `ros2 launch bringup full_stack.launch.py`
#                                      # in that shell: every node would run twice.
#   START_STACK=0 ./docker/run.sh      # plain interactive shell, no stack: use this to
#                                      # launch the stack yourself
#   ./docker/run.sh <command...>       # run one command, e.g.
#   ./docker/run.sh ros2 launch bringup full_stack.launch.py sim:=false
#   ./docker/run.sh smoke              # build, start the stub stack (sim:=false), run
#                                      # smoke_task.py, stop the stack; exit 0 = SUCCEEDED
#
# Inside the container:
#   - Python edits are live (the image was built with --symlink-install).
#   - After changing C++, a .msg/.action, or package.xml:  colcon build --symlink-install
#   - build/ install/ log/ live in named Docker volumes so they persist between runs.
#     Reset them (e.g. after rebuilding the image) with:
#         docker volume rm robot_build robot_install robot_log
#
# GUI: on Linux with an X server, the MuJoCo viewer and rviz2 open on your
# display. On macOS (Docker Desktop) there is no X11 forwarding: run the sim
# headless:  ros2 launch bringup full_stack.launch.py headless:=true
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"

ARGS=(-it --rm
  -v "${REPO_ROOT}/src:/home/ros2_ws/src"
  -v robot_build:/home/ros2_ws/build
  -v robot_install:/home/ros2_ws/install
  -v robot_log:/home/ros2_ws/log)

# API keys etc. for the LLM planner: .env at the repo root is mounted read-only; the planner
# node reads it itself (llm_config.load_env). Not --env-file: docker rejects spaces around "=".
if [[ -f "${REPO_ROOT}/.env" ]]; then
  ARGS+=(-v "${REPO_ROOT}/.env:/home/ros2_ws/.env:ro")
fi

case "$(uname -s)" in
  Linux)
    ARGS+=(--network host)
    if [[ -n "${DISPLAY:-}" ]]; then
      xhost +local:docker >/dev/null 2>&1 || true
      ARGS+=(-e DISPLAY="${DISPLAY}" -v /tmp/.X11-unix:/tmp/.X11-unix:rw)
    else
      echo "note: no \$DISPLAY set, use headless:=true for the sim" >&2
    fi
    ;;
  Darwin)
    echo "note: macOS has no X11 forwarding here, use headless:=true for the sim" >&2
    ;;
esac

if [[ "${1:-}" == "smoke" ]]; then
  shift
  # Extra args go to full_stack.launch.py, e.g.  ./docker/run.sh smoke navigation:=external
  exec docker run "${ARGS[@]}" wiscohumanoids/robot:latest bash -c '
    set -o pipefail
    cd /home/ros2_ws
    echo "== building"
    colcon build --symlink-install --packages-select humanoid_interfaces behavior_tree task_planner bringup 2>&1 | tail -3 || exit 1
    source install/setup.bash
    echo "== starting the stack (log: /tmp/stack.log)"
    ros2 launch bringup full_stack.launch.py sim:=false "$@" > /tmp/stack.log 2>&1 &
    stack=$!
    trap "kill -INT $stack 2>/dev/null; wait $stack 2>/dev/null" EXIT
    for _ in $(seq 60); do   # wait until the planner is listening
      ros2 topic info /user_intent 2>/dev/null | grep -q "Subscription count: [1-9]" && break
      sleep 1
    done
    echo "== smoke_task"
    ros2 run bringup smoke_task.py || { echo "-- stack log (tail) --"; tail -40 /tmp/stack.log; exit 1; }
  ' smoke "$@"
fi

if [[ $# -gt 0 ]]; then
  exec docker run "${ARGS[@]}" wiscohumanoids/robot:latest "$@"
fi
if [[ "${START_STACK:-1}" == "0" ]]; then
  exec docker run "${ARGS[@]}" wiscohumanoids/robot:latest
fi
# Build what changed (fast when nothing did), start the headless stub stack in the
# background, then hand you a shell. Try it with:  ros2 run bringup smoke_task.py
exec docker run "${ARGS[@]}" wiscohumanoids/robot:latest bash -c '
  set -o pipefail
  cd /home/ros2_ws
  if ! colcon build --symlink-install --packages-select humanoid_interfaces behavior_tree task_planner bringup 2>&1 | tail -3; then
    echo "BUILD FAILED: the stack was NOT started. Run the colcon build again here to see the full output." >&2
    exec bash
  fi
  source install/setup.bash
  ros2 launch bringup full_stack.launch.py sim:=false > /tmp/stack.log 2>&1 &
  echo "stack starting in the background (pid $!, log: /tmp/stack.log), give it a few seconds"
  echo "WARNING: the stub stack is ALREADY RUNNING in this shell. Do not run"
  echo "  ros2 launch bringup full_stack.launch.py"
  echo "here: every node would run twice (two publishers per topic, which the contract forbids)."
  echo "To launch it yourself: stop this one first (kill -INT $!), or start the container"
  echo "with START_STACK=0 ./docker/run.sh"
  exec bash
'
