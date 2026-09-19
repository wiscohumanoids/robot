# task_planner_stub

**Layer:** 2 -- task planning. **Status:** STUB.
**Subscribes:** `/user_intent` (`std_msgs/String`), `/object_poses`.
**Publishes:** `/skill_sequence` (`humanoid_interfaces/SkillSequence`), one
message per accepted intent -- **the only publisher.**

## What is fake

No LLM. Any intent becomes a canned two-step plan built by `plan_for_intent()`
(`planner_logic.py`, unit-tested):

1. `navigate_to` a point `standoff_m` (default 0.6 m) short of the chosen object;
2. the manipulation skill picked from the intent's verb -- `pick` (default),
   `place`, or `handover`.

The object is the one whose label appears in the intent, else the first
reported. If perception has reported nothing yet, there is no plan (logged).

## What replaces it

The LLM + skill-library planner. Keep: subscribe `/user_intent`, publish
`SkillSequence` in the [skill vocabulary](../../INTERFACE_CONTRACT.md#skill-vocabulary).
Adding a skill name is an interface change (behavior tree must learn it).

```bash
ros2 launch bringup full_stack.launch.py planner:=external
ros2 topic pub --once /user_intent std_msgs/msg/String "{data: 'pick up the cube'}"
ros2 topic echo /skill_sequence
```
