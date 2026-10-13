# task_planner

**Layer:** 2 (task planning). **Status:** real (LLM planner). Only the canned fallback used without an API key is a stub.
**Subscribes:** `/user_intent` (`std_msgs/String`), `/planner_command`, `/object_poses`, `/robot_pose`, `/task_status`, `/manipulation/held_object`.
**Publishes:** `/skill_sequence` (`humanoid_interfaces/SkillSequence`), one
message per planned intent, and `/planner_status` (`std_msgs/String`), what happened
to each intent. **The only publisher of both.**

## Queue

`/user_intent` text goes into a FIFO of **raw text** (`dispatcher.py`, cap
no size limit for now: `MAX_QUEUED` in `dispatcher.py` is `None`; give it a number and extra intents are rejected with `REJECTED ... queue full (N)`). Nothing is planned when
it is typed. The oldest intent is planned only when the behavior tree is idle
(`/task_status` is not `RUNNING ...` and the previous plan has finished), using the
robot pose and objects **at that moment**. A failed plan is logged and the next
intent is tried. The tree therefore never holds more than one waiting plan.

`/planner_status` lines (one per event): `QUEUED "<intent>": N waiting`,
`PLANNING "<intent>"`, `SENT "<intent>": N steps`, `NO_PLAN "<intent>": <reason>`
(the model refused, the plan was invalid, no objects seen...) and
`REJECTED "<intent>": queue full (N)` (only when a cap is set), plus the outcome: `DONE` (only when the
tree reported `SUCCEEDED`), `RETRYING` (the tree
failed, with its failure text, then `REPLANNING`) and `GAVE_UP` (with the reason: the tree's failure,
or that the tree stopped or went silent, see "Lost trees"). Watch it with `ros2 topic echo /planner_status`, or `ros2 run bringup watch_status.py`,
a print-only stand-in for the user-facing feedback (prints `/planner_status` and every
change of `/task_status`; replace it with the real UI).

## Replanning (LLM mode only)

When the tree reports `FAILED` for a plan we sent, the same intent gets **one**
replan call before the next queued intent: the prompt has the original request, the
old plan, where it failed (step, skill, reason, and that earlier steps succeeded),
and the robot and objects as they are now. The model may use any skills it likes
(`llm_planner.replan_with_llm`; the new plan is validated like any other). If the
new plan fails too, or no plan can be made, it is final (`GAVE_UP` / `NO_PLAN`).

### Editing the queue

`/planner_command` (`std_msgs/String`) edits the intents that are still waiting; the
running plan is never touched. Positions are 1-based, 1 = next to run.

```bash
ros2 topic pub --once /planner_command std_msgs/msg/String "{data: 'list'}"
ros2 topic pub --once /planner_command std_msgs/msg/String "{data: 'remove 2'}"
ros2 topic pub --once /planner_command std_msgs/msg/String "{data: 'clear'}"   # also drops a pending retry
```

Answers on `/planner_status`: `QUEUE: 1. "a" | 2. "b"` (or `QUEUE: empty`),
`REMOVED "b" (was #2)`, `REMOVE_FAILED: no intent #5 (queue has 3)`,
`CLEARED: 3 removed`, `COMMAND_FAILED "...": expected clear | remove <n> | list`. The
same operations exist as `Dispatcher.clear()`, `.remove(n)` and `.show()`.

## Lost plans

`/skill_sequence` is not delivered to a tree that has not subscribed yet. Every plan
carries a `plan_id`; if the tree has not gone `RUNNING` 2 s after a send, the same plan
is sent again (`RESENDING`), up to 3 sends in total, then the intent is given up
(`NO_PLAN ... behavior tree did not start the plan`). The tree remembers the last 64
ids and ignores a plan it has already seen, so a resend never runs twice.

## Lost trees

`DONE` means the tree said `SUCCEEDED`. If a plan we sent stops being `RUNNING` with anything else
that is not `FAILED` (e.g. `IDLE`, because the tree was restarted mid-plan), nobody knows what the
robot did: it is reported as `GAVE_UP "...": behavior tree stopped without a result (status: IDLE)`
and not replanned.

The tree re-sends `/task_status` every second. If none arrives for 5 s (`STATUS_TIMEOUT_S`) while a
plan is sent or running, the plan is given up (`GAVE_UP "...": no /task_status for 5 s ...`) and the
queue is no longer stuck behind it; nothing else is planned until a status is heard again.

## Hand state

`/manipulation/held_object` says what the hand holds (`""` = empty; unknown until the
first message). The prompt includes it, and a plan that is impossible for the hand
(pick while holding something, place or hand over what is not held) is rejected with
the reason (`skill_catalog.check_hand`). With an unknown hand nothing is rejected.

## LLM mode

Put `OPENAI_API_KEY` in `.env` at the repo root (`docker/run.sh`
mounts it into the container; `OPEN_AI_KEY` also works as the name). Each `/user_intent` then goes through
`llm_planner.py`: the prompt lists the skills from `skill_catalog.py` and the objects
perception sees, the model answers with a strict JSON plan, and every step is
validated (known skill, known object, goal inside the map) before publishing. The
object poses in the plan come from the world, never from the model. If the model
refuses or the plan is invalid, nothing is published and the reason is logged.
A transient API failure (HTTP 408/429/5xx, timeout, connection error) is retried, up to 3 requests in
total with 1 s and 2 s between them (`llm_client.py`); other errors fail at once (`NO_PLAN`).
Without a key the canned planner below is used.

## What is fake (canned planner, used without a key)

No LLM. Any intent becomes a canned two-step plan built by `plan_for_intent()`
(`planner_logic.py`, unit-tested):

1. `navigate_to` a point `standoff_m` (default 0.6 m) short of the chosen object;
2. the manipulation skill picked from the intent's verb, `pick` (default),
   `place`, or `handover`.

The object is the one whose label appears in the intent, else the first
reported. Like the LLM planner it refuses plans the hand makes impossible (put down or hand over while
the hand is empty, pick while holding something) when `/manipulation/held_object` is known. If perception has reported nothing yet, there is no plan (logged).

## What replaces it

The LLM + skill-library planner. Keep: subscribe `/user_intent`, publish
`SkillSequence` in the [skill vocabulary](../../INTERFACE_CONTRACT.md#skill-vocabulary).
Adding a skill name is an interface change (behavior tree must learn it).

```bash
ros2 launch bringup full_stack.launch.py planner:=false
ros2 topic pub --once /user_intent std_msgs/msg/String "{data: 'pick up the cube'}"
ros2 topic echo /skill_sequence
```
