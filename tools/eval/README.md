# tools/eval

Graders and measurement tools for the running stack. These are the **only** code allowed to read simulator ground
truth (`/sim/ground_truth/*`, the env's true geometry); the robot stack never does. Most run inside the container
against a launched stack, and `tools/test/autonomy_test.sh` (`tools/ros.sh autotest`) calls them.

| File | What |
|---|---|
| `nav_goal.py X Y YAW` | send Nav2 a goal and grade where the robot really ended up |
| `dropped_obstacle.py` | drop a box in the robot's path mid-walk; pass if it walks around without touching it |
| `reach_check.py`, `reach_plan.sh ENV OBJECT <plan>` | after `navigate_to`, is the robot within arm's reach of the object, facing it |
| `run_plan.py "<skill:arg>" ...` | send a hand-written skill plan to the behavior tree (no LLM) and print progress |
| `fake_claude_flow.py` | planner end to end with a scripted stand-in for Claude (no API key) |
| `estop_check.py` | the e-stop must stop the robot while commands keep coming, and release must work |
| `viewer_idle_nav.py` | an open viewer must not block navigation, and its keys must still take over |
| `odometry_drift.py` | how far the robot's own pose estimate drifts from the truth |
| `steadiness.py` | sway and drift of a walking policy over a kitchen-like command script, in plain MuJoCo |
| `e2e_realistic.sh` | full stack with nothing privileged: SLAM, real perception on the GPU machine, behavior tree, walking |
