"""High-level skills for the behavior tree (no rclpy, unit-testable).

Each skill is one function: (ctx, skill, timeout_s) -> Outcome. A skill runs
once; retries and status reporting belong to bt_logic. `ctx` is the only way a
skill reaches the robot, so tests pass a fake one.

Adding a skill: write the function (or reuse `manipulate`), add its name to
SKILLS, and update INTERFACE_CONTRACT.md ("Skill vocabulary"): a new skill name
is an interface change.
"""
from dataclasses import dataclass
from typing import Any, Callable, Dict, Protocol


@dataclass
class Skill:
    name: str
    object_id: str = ''
    pose: Any = None      # opaque here; the ROS wrapper converts it (PoseStamped)


@dataclass
class Outcome:
    ok: bool
    reason: str = ''      # why it failed; '' on success
    retryable: bool = True  # False: another attempt cannot help (e.g. unknown skill)


class SkillContext(Protocol):
    """What a skill may ask of the robot. Both calls block until the action
    finishes or `timeout_s` passes; on timeout they cancel the goal and return
    Outcome(False, 'timed out after N s'). They never raise."""

    def navigate(self, pose: Any, timeout_s: float) -> Outcome: ...

    def manipulate(self, task: str, object_id: str, pose: Any, timeout_s: float) -> Outcome: ...


def navigate_to(ctx: SkillContext, skill: Skill, timeout_s: float) -> Outcome:
    return ctx.navigate(skill.pose, timeout_s)


def manipulate(ctx: SkillContext, skill: Skill, timeout_s: float) -> Outcome:
    """pick, place, handover: same action, the skill name is the task name."""
    return ctx.manipulate(skill.name, skill.object_id, skill.pose, timeout_s)


SkillFn = Callable[[SkillContext, Skill, float], Outcome]

# The switch: skill name -> function.
SKILLS: Dict[str, SkillFn] = {
    'navigate_to': navigate_to,
    'pick': manipulate,
    'place': manipulate,
    'handover': manipulate,
}


def run_skill(ctx: SkillContext, skill: Skill, timeout_s: float) -> Outcome:
    """One attempt of one skill. Unknown names and crashes become failed Outcomes."""
    fn = SKILLS.get(skill.name)
    if fn is None:
        return Outcome(False, f'unknown skill: {skill.name}', retryable=False)
    try:
        return fn(ctx, skill, timeout_s)
    except Exception as exc:  # a skill bug must fail the task, not kill the loop
        return Outcome(False, f'error: {exc}')
