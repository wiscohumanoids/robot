# integration_tests

**Layer:** test harness, not part of the stack (no contract entry, no launch switch).
**Runs:** only in Docker. Nothing here needs ROS on the host.

Two scripts at the repo root drive everything (each needs `./docker/build.sh` once and Docker running):

```bash
tools/run_integration_tests.sh                 # tree + planner + Nav2, no API key needed
tools/run_integration_tests.sh --llm           # ... plus the scenarios that call the real model (.env)
tools/run_integration_tests.sh nav_then_pick   # chosen scenarios; --list shows them all
tools/run_planner_stress.sh                    # stress the task planner (canned planner, free)
tools/run_planner_stress.sh --llm --soak 120   # with the real model; soak for 120 s
```

Output goes to `reports/integration/` and `reports/stress/` (git-ignored). Both exit non-zero on a failure.
The comparison logic is plain Python and unit-tested (`python3 -m pytest src/integration_tests/test`),
including the check that every scenario below is in this README.

## Integration scenarios

Each runs in its own fresh container: stub stack + a fake obstacle map from `nav2_config` + Nav2 + the scenario.

**`tree_nav2`**: the planner is off; the test publishes `/skill_sequence` itself (tree, Nav2, manipulation):

| Scenario | Checks |
|---|---|
| `nav_reaches_goal` | `navigate_to (4, 0)` around a box: `RUNNING`, then `SUCCEEDED`, robot at the goal |
| `nav_then_pick` | both steps run, the hand ends up holding `cube_1` |
| `pick_and_place` | navigate, pick, navigate, place: the hand holds the cube and then lets go |
| `unreachable_retries_then_fails` | Nav2 aborts a goal in a sealed box: 3 attempts, then `FAILED 1/1 ... (attempt 3/3)` |
| `failed_plan_drops_rest_next_plan_runs` | plan A fails at step 1 and its pick never runs; plan B behind it still runs |
| `duplicate_plan_id_runs_once` | the same `plan_id` sent twice (a planner resend) runs once |
| `two_plans_run_in_order` | two plans queued back to back both run, in order |
| `unknown_skill_fails_fast` | an unknown skill fails at once, no retries |
| `empty_plan_is_ignored` | an empty plan changes nothing, the next plan runs |

**`canned`**: the planner runs without a key (its canned fallback): planner, tree, Nav2, manipulation, deterministic:

| Scenario | Checks |
|---|---|
| `canned_pick_the_cube` | "pick up the cube" ends with the cube in hand; planner says QUEUED, PLANNING, SENT, DONE in that order |
| `canned_pick_the_cup` | the intent text selects the object (`cup_1`) |
| `canned_queue_runs_in_order` | two intents typed together run one after the other; the second waits for the first |
| `canned_queue_editing` | while a plan runs: `list`, `remove 2`, `list`, `clear`, `list` act on the waiting intents only |
| `canned_queue_has_no_cap` | one plan running, 12 more intents: all 12 wait, in order, none is rejected |

**`llm`** (`--llm`): real model calls, slower, costs money, not deterministic:

| Scenario | Checks |
|---|---|
| `llm_go_to_xy` | "go to x=2 y=-3": one `navigate_to` there, robot arrives |
| `llm_pick_refuse_put_down` | pick the cup, pick the cube while holding it (refused, mentions the hand), put the cup down |
| `llm_impossible_requests` | nonsense, off-map goal and unknown object are all refused, nothing runs |
| `llm_replan_after_failure` | the model sends the robot into a sealed box: the tree fails it, the model gets exactly one replan |
| `llm_two_intents_in_order` | two typed intents are planned one at a time, in order |

Every check also requires `/task_status` to keep arriving (the 1 Hz heartbeat), every plan the planner
published to be valid (known skills and objects, goals inside the map, hand rules), and no Python
traceback or dead node in the stack log.

## Planner stress

The stack runs without the behavior tree and the stress driver plays the tree (`logic.FakeTree`: normal,
always-fail, hang, never-start, stale), so the tree can misbehave on demand. After every scenario the planner
must be alive, every plan valid, and no plan sent while the tree was busy.

| Scenario | What it does |
|---|---|
| `burst` | 200 intents in a burst while the tree is busy: every one is answered and queued in order, none rejected, `clear` removes them all |
| `command_storm` | 400 random submits, removes, clears, lists and garbage commands: each answered exactly once |
| `garbage_intents` | blank, 20 kB, unicode, SQL, JSON-looking, injection and `clear`/`list` as intent text: one verdict each |
| `misbehaving_tree` | never starts (3 sends then `NO_PLAN`), stale `SUCCEEDED`, hangs, always fails (at most one replan) |
| `latency` | `QUEUED` to `SENT` over 50 sequential intents; p95 must stay under 2.5 s |
| `soak` | random traffic for `--soak` seconds (default 45): memory growth under 40 MB, still planning at the end |
| `llm_varied_requests` | (`--llm`) sensible requests get plans, impossible ones are refused |
| `llm_queue_order` | (`--llm`) five queued intents are planned one at a time, in order |
| `llm_replan_storm` | (`--llm`) the tree fails everything: each intent is replanned at most once |

With `--llm` only the reduced set (`LLM_SCENARIOS` in `scripts/stress_driver.py`) runs, with smaller inputs.

## Layout

| Path | What |
|---|---|
| `integration_tests/logic.py` | Timeline, status readers, plan validation, `FakeTree` (no ROS, unit-tested) |
| `integration_tests/scenarios.py` | the scenario table and every check (no ROS, unit-tested) |
| `scripts/harness.py` | the ROS node: sends inputs, records a Timeline, optionally plays the fake tree |
| `scripts/integration_driver.py`, `scripts/stress_driver.py` | play one scenario / the stress set |
| `scripts/run_scenario.sh`, `scripts/run_stress.sh` | start the stack inside the container, then call the driver |

The scripts are run in place (no `colcon build` of this package is needed).
