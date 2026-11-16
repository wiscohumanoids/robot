# hsim_behavior — task behavior tree (BehaviorTree.CPP v4, C++)

Node `behavior`, action `/behavior/execute_plan` (`hsim_interfaces/ExecutePlan`). Each plan becomes
`ReactiveSequence[IsSafe, Sequence[skills]]`, ticked at 20 Hz; Groot2 can watch it live on port 1667.

| Skill | Does |
|---|---|
| `navigate_to(target)` | ends **within arm's reach, facing the target** (see below) |
| `find_object(target)` | succeeds if remembered; otherwise turns around once and checks again |
| `search_for(target, hints)` | look around here, then at each hinted place, until the target is seen |
| `look_around` | full turn (Nav2 spin) |
| `say(text)` | `/robot/say` |
| `pick(X)` / `place(Y)` | one `/skills/manipulate` goal (`hsim_manipulation`, VLA); `pick` followed by `place` becomes a single pick-and-place goal |

### How navigate_to reaches a target
1. **Pick a side** it can stand at. It tries all four sides of the target's box (for a small object on
   furniture, the furniture's sides, nearest the object first). On the robot's own map (`/map`: known
   map or SLAM) it steps inward from each side to the real surface (perceived boxes are loose: a
   fridge's box sticks out past its doors) and drops sides with a wall or clutter where it would
   stand. A fridge against a wall between counters keeps one side: its door.
2. **Walk** there with Nav2, to a staging spot 0.65 m from the surface. Nav2 keeps its 0.3 m body
   radius off obstacles, so it can't bring the robot any closer itself.
3. **Step up** with its own closed-loop controller: square up, walk straight in at 0.3 m/s until the
   body center is 0.30 m from the surface (measured by the scan when it sees it: lidar on the G1, the
   depth-derived scan on Wisco; else by the pose estimate), then square up to the surface the scan sees (a line fit, so the pose estimate's heading
   error doesn't matter). The arms reach ~0.3 m into a surface from there (profile `reach_standoff`). Commands go on
   `/cmd_vel_approach`, a twist_mux input above Nav2 and below teleop; the e-stop lock still blocks it.

### Making room (`StepBack`, inserted before every motion skill; no-op when there is room)
Nav2 keeps a 0.3 m body radius off obstacles and refuses to plan or turn when the robot is closer, and a
humanoid drifts ~0.15–0.25 m while turning in place. So before a motion skill the tree:
1. walks straight back 0.3 m if it is still where navigate_to stepped it up to a surface;
2. steps directly away (walk or sidestep) from anything closer than 0.42 m;
3. before a turn in place (look_around, find_object, search_for): if anything is within 0.6 m, walks
   with Nav2 to the nearest open floor with 0.6 m all round, on its own map.

Graded on true geometry by `tools/eval/reach_plan.sh` / `reach_check.py` (autotest checks `reach_table`, `reach_block`,
`kitchen_fridge`, `slam_fridge`, `kitchen_sink`; run one with `tools/ros.sh autotest <name> [robot]`).

Failures return a plain-words reason, which the planner uses to replan. Try without the LLM:
`python3 /hsim/tools/eval/run_plan.py "navigate_to:table" "find_object:red block"`.
