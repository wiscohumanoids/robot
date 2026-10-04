"""Catalog of every skill the robot can run (no rclpy, unit-testable).

This is what the planner is allowed to plan with: one entry per skill, saying what
it does and what arguments it needs. The behavior tree has the matching table of
functions (`behavior_tree/skills.py: SKILLS`); a test keeps the two name lists equal.

Adding a skill: add a SkillSpec here, add the function to SKILLS in the behavior
tree, and update INTERFACE_CONTRACT.md ("Skill vocabulary") and Skill.msg.
"""
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str          # one line, also shown to the LLM
    needs_object: bool        # object_id must name a known object
    needs_pose: bool          # a target pose (map frame) is required
    pose_hint: str = ''       # what the pose means for this skill
    requires: str = ''        # what must be true before it runs (informational)
    effect: str = ''          # what is true after it succeeds (informational)
    hand: str = ''            # 'take' (needs an empty hand, then holds object_id) or
                              # 'release' (needs to hold object_id, then empty); '' = no effect


CATALOG: Dict[str, SkillSpec] = {s.name: s for s in (
    SkillSpec(
        name='navigate_to',
        description='Walk to a point in the map.',
        needs_object=False, needs_pose=True,
        pose_hint='where to stand, map frame',
        requires='a clear path exists',
        effect='robot is at the target pose'),
    SkillSpec(
        name='pick',
        description='Pick up an object with the hand.',
        needs_object=True, needs_pose=False,
        pose_hint='optional: where the object is, map frame',
        requires='object within reach, hand empty',
        effect='object is in the hand', hand='take'),
    SkillSpec(
        name='place',
        description='Put the held object down.',
        needs_object=True, needs_pose=False,
        pose_hint='optional: where to put it, map frame',
        requires='holding the object',
        effect='object is at the target, hand empty', hand='release'),
    SkillSpec(
        name='handover',
        description='Hand the held object to a person.',
        needs_object=True, needs_pose=False,
        pose_hint='optional: where the person is, map frame',
        requires='holding the object, person in reach',
        effect='person has the object, hand empty', hand='release'),
)}


def names() -> List[str]:
    return list(CATALOG)


def get(name: str):
    """The SkillSpec, or None if there is no such skill."""
    return CATALOG.get(name)


def check_step(step: dict) -> List[str]:
    """Problems with one plan step {name, object_id, x, y, z}; [] if it is valid
    for the catalog. Does not look at the world (whether the object exists)."""
    spec = CATALOG.get(step.get('name'))
    if spec is None:
        return [f"unknown skill: {step.get('name')!r}"]
    problems = []
    if spec.needs_object and not step.get('object_id'):
        problems.append(f'{spec.name} needs an object_id')
    if not spec.needs_object and step.get('object_id'):
        problems.append(f'{spec.name} takes no object_id')
    if spec.needs_pose and not all(k in step for k in ('x', 'y')):
        problems.append(f'{spec.name} needs a target pose (x, y)')
    return problems


def describe() -> str:
    """The catalog as text, for an LLM prompt."""
    lines = []
    for s in CATALOG.values():
        args = []
        if s.needs_object:
            args.append('object_id')
        if s.needs_pose:
            args.append('pose')
        lines.append(f'- {s.name}({", ".join(args)}): {s.description}'
                     f' Needs: {s.requires}. Result: {s.effect}.')
    return '\n'.join(lines)


def check_hand(plan: List[dict], held) -> List[str]:
    """Walk the plan with the hand state and report impossible steps.
    held: id of the object in the hand, '' if empty, None if unknown. With an unknown
    hand, nothing can be said until a step in the plan fixes it (a pick means we hold
    that object afterwards)."""
    problems = []
    state = held
    for number, step in enumerate(plan, start=1):
        spec = CATALOG.get(step.get('name'))
        if spec is None or not spec.hand:
            continue
        who = step.get('object_id', '')
        if spec.hand == 'take':
            if state:
                problems.append(f'step {number} ({spec.name}): hand is already holding {state}')
            state = who
        elif spec.hand == 'release':
            if state is not None and state != who:
                have = f'holding {state}' if state else 'hand is empty'
                problems.append(f'step {number} ({spec.name}): not holding {who} ({have})')
            state = ''
    return problems
