# 0006 — Autonomy: Claude planner, two behavior trees, layered safety

**Decision**
1. **Planner = Claude** (`claude-opus-5-5`, effort `low`, structured outputs via `output_config.format`, server-side
   `fallbacks: "default"`). It gets the request + the robot's *own* object memory and returns skill names and object
   labels only, never coordinates. A deterministic check rejects plans that go to never-seen objects; failures (with a
   plain-words reason from the behavior tree) are fed back for up to 2 replans.
2. **Two behavior trees.** Ours (`hsim_behavior`, BehaviorTree.CPP v4) decides *what* to do; Nav2's own decides *how
   to get there* (replanning, recoveries). Our tree is `ReactiveSequence[IsSafe, Sequence[skills]]`, ticked at 20 Hz.
3. **Safety in two layers.** The `safety` node raises an e-stop (operator request, fall > 35°, battery < 5%, stale
   encoders/odometry). The BT's `IsSafe` halts the running skill; independently, a twist_mux **lock** at priority 255
   blocks every velocity command, so stopping does not depend on the tree behaving.
4. **Skills:** navigate_to, find_object, search_for (hints ranked by the planner), look_around, say. pick/place fail
   honestly until Phase 5.
5. **navigate_to means "within arm's reach, facing it".** Nav2 alone can't get there: it keeps its 0.3 m body radius
   off obstacles, so it stops ~0.6–0.8 m away. Nav2 walks to a staging spot on a side the robot's own map says it can
   stand at; the tree's own controller then steps in to 0.30 m (lidar-measured) on `/cmd_vel_approach`, and steps back
   before the next motion skill. Same shape as a mobile manipulator's "dock" / "undock". (Changing Nav2's footprint
   instead would change how it drives everywhere.) The same "make room" step also moves the robot off anything closer
   than 0.42 m, and to open floor before a turn in place: a humanoid drifts ~0.2 m while turning, which used to make
   Nav2's spin abort near counters (kitchen spawn points are often in corners).

**Consequences:** `ANTHROPIC_API_KEY` in the repo's `.env`. Without it the planner refuses requests; the BT, Nav2 and
everything below still work (`tools/eval/run_plan.py` sends hand-written plans).
