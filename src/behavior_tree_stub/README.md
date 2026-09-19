# behavior_tree_stub

**Layer:** 3 -- task execution. **Status:** STUB.
**Subscribes:** `/skill_sequence`.
**Action clients:** `navigate_to_pose` (`nav2_msgs/NavigateToPose`) for
`navigate_to`; `execute_manipulation` (`ExecuteManipulation`) for every other skill.
**Publishes:** `/task_status` (`std_msgs/String`, on change + 1 Hz heartbeat) --
**the only publisher.**

## What is fake

It is not a behavior tree: it runs a plan's skills strictly in order on a worker
thread and stops at the first failure. A plan that arrives while it is busy is
ignored (logged), not queued. No retries, no conditions, no recovery.

`/task_status` reads like `IDLE`, `RUNNING 2/3 pick`, `SUCCEEDED`, or
`FAILED 1/2 navigate_to: goal rejected`.

## What replaces it

BehaviorTree.CPP with real condition/action nodes. Keep the same input, the same
two action clients, and `/task_status`. The dispatch rule lives in `bt_logic.py`
(unit-tested).

```bash
ros2 launch bringup full_stack.launch.py behavior_tree:=external
ros2 run bringup smoke_task.py       # end-to-end check of the whole task layer
```
