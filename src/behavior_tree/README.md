# behavior_tree

**Layer:** 3 (task execution). **Status:** REAL (logic unit-tested; `smoke_task.py` passes on the stub stack in Docker; retry, timeout and failure paths have only been tested with the fake `ctx`, not live).
**Subscribes:** `/skill_sequence`.
**Action clients:** `navigate_to_pose` (`nav2_msgs/NavigateToPose`) for `navigate_to`;
`execute_manipulation` (`ExecuteManipulation`) for `pick`, `place`, `handover`.
**Publishes:** `/task_status` (`std_msgs/String`, on change + 1 Hz heartbeat), **the only publisher.**

## What it does

```
take the oldest plan from the queue (wait if there is none)
for each skill in the plan:
    up to 3 attempts, each limited to 60 s
    all attempts failed -> publish FAILED, drop the rest of that plan
all skills worked -> publish SUCCEEDED
next plan
```

- **Queue:** plans that arrive while it is busy wait their turn (FIFO, unbounded; in practice the task planner only sends a plan when the tree is idle, so at most one waits. The intent queue and its size cap live in the planner).
  An empty plan is ignored, and so is a plan whose `plan_id` was already seen (the
  planner resends until it sees the tree start; the last 64 ids are remembered).
- **Retry and timeout:** the same for every skill (`MAX_ATTEMPTS = 3` total attempts,
  `SKILL_TIMEOUT_S = 60` per attempt, in `bt_logic.py`). A timeout cancels the running action.
- **Unknown skill name:** fails at once with `unknown skill: <name>`, no retries.
- A failed plan does not stop the queue: the next plan runs.

`/task_status` reads `IDLE` (before the first plan), `RUNNING 2/3 pick`,
`RUNNING 1/2 navigate_to: attempt 2/3`, `SUCCEEDED`, or
`FAILED 1/2 navigate_to: timed out after 60 s (attempt 3/3)`. The last result stays up
until the next plan starts.

## Layout

| File | Role | ROS? |
|---|---|---|
| `skills.py` | One function per skill, the `SKILLS` switch table, `run_skill()` | no |
| `bt_logic.py` | Queue, the 3-attempt flow, status text, `BehaviorRunner` | no |
| `bt_node.py` | Subscription, publisher, the two action clients (`RosSkillContext`) | yes |

Skills reach the robot only through `ctx` (`navigate(pose, timeout_s)` and
`manipulate(task, object_id, pose, timeout_s)`, each returning an `Outcome`), so tests use a fake
one. **Adding a skill:** write a function in `skills.py`, add its name to `SKILLS`, and update the
contract's "Skill vocabulary" (a new skill name is an interface change).

```bash
python3 -m pytest tests src/*/test   # run from the repo root (see the root conftest.py)
ros2 launch bringup full_stack.launch.py sim:=false          # starts it (behavior_tree:=true is the default)
ros2 launch bringup full_stack.launch.py behavior_tree:=false  # run your own instead
ros2 run bringup smoke_task.py       # end-to-end check of the whole task layer
```

## Not done yet

A "stop" command that jumps the queue and cancels the running skill, and any
condition checks before a skill starts (e.g. safety status).
