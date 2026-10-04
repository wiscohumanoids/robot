"""LLM planner: intent text + known objects -> a validated list of skills
(no rclpy, unit-testable; the network call is injectable).

    "Can you pick up the cup"  ->  navigate_to(x, y) then pick(cup_1)
    "go to x=2 y=-3"           ->  navigate_to(2, -3)

The LLM only decides WHICH skills, in what order, on which object or point. This
module then fills in everything the skills need (an object's pose comes from the
world, not from the LLM), validates every step against the skill catalog and the
world, and returns the same plan dicts the canned planner does:
{name, object_id, x, y, z}. The behavior tree's skill functions take it from there.
"""
import math
import re
from dataclasses import dataclass, field
from typing import List

from task_planner import skill_catalog
from task_planner.llm_client import LLMError, chat_json

MAP_MIN_M, MAP_MAX_M = -5.0, 5.0      # the 10 m x 10 m map, centred on the origin
EDGE_MARGIN_M = 0.3                   # keep goals this far from the map edge
MAX_STEPS = 10
STANDOFF_M = 0.6

SCHEMA = {
    'type': 'object',
    'additionalProperties': False,
    'required': ['feasible', 'reason', 'steps'],
    'properties': {
        'feasible': {'type': 'boolean'},
        'reason': {'type': 'string'},
        'steps': {
            'type': 'array',
            'items': {
                'type': 'object',
                'additionalProperties': False,
                'required': ['skill', 'object_id', 'x', 'y'],
                'properties': {
                    'skill': {'type': 'string', 'enum': skill_catalog.names()},
                    'object_id': {'type': 'string'},
                    'x': {'type': ['number', 'null']},
                    'y': {'type': ['number', 'null']},
                },
            },
        },
    },
}


@dataclass
class PlanResult:
    plan: List[dict] = field(default_factory=list)
    error: str = ''          # why there is no plan; '' when there is one

    @property
    def ok(self):
        return bool(self.plan) and not self.error


def system_prompt(standoff=STANDOFF_M):
    return f"""You are the task planner of a humanoid robot. Turn the user's request into a short ordered list of skills.

Skills (use only these):
{skill_catalog.describe()}

World:
- Positions are metres in the map frame. The map is 10 m x 10 m, x and y each from {MAP_MIN_M:g} to {MAP_MAX_M:g}, origin at the centre. The request tells you where the robot is right now and which way it faces.
- You are given the objects the robot can currently see, with ids and positions. Refer to objects only by those ids.

Rules:
- navigate_to takes x and y (and no object_id). pick, place and handover take an object_id and leave x and y null.
- To pick up an object, first navigate_to a point {standoff:g} m short of it (same y, x minus {standoff:g}), then pick it.
- To place or hand over something the robot is not holding yet, pick it up first (navigate, pick, then place/handover). To put an object somewhere, navigate to that spot first.
- The request tells you what the hand holds right now (empty, holding an object, or unknown). That is current and authoritative. pick needs an empty hand; place and handover need the hand to hold that object. If the hand holds something else, place or hand it over first. Never pick when already holding.
- For "go to x, y" requests use navigate_to with exactly those coordinates.
- If the request cannot be done (unknown object, coordinates outside the map, not a robot task), set feasible to false, give a short reason, and return no steps.
- Never invent object ids or skills. Use at most {MAX_STEPS} steps."""


def describe_objects(objects):
    if not objects:
        return 'Visible objects: none.'
    lines = [f"- {o['object_id']} ({o['label']}) at x={o['x']:.2f}, y={o['y']:.2f}" for o in objects]
    return 'Visible objects:\n' + '\n'.join(lines)


def describe_robot(robot):
    """robot: {'x', 'y', 'yaw'} (radians, 0 = facing +x) or None if unknown."""
    if not robot:
        return 'Robot position: unknown.'
    degrees = math.degrees(robot['yaw'])
    return (f"Robot now: x={robot['x']:.2f}, y={robot['y']:.2f}, "
            f'facing {degrees:.0f} degrees (0 = +x, 90 = +y).')


def describe_hand(held):
    """held: object id in the hand, '' if empty, None if unknown."""
    if held is None:
        return 'Hand: unknown.'
    return f'Hand: holding {held}.' if held else 'Hand: empty.'


def build_messages(intent, objects, standoff=STANDOFF_M, robot=None, held=None):
    world = f'{describe_robot(robot)}\n{describe_hand(held)}\n{describe_objects(objects)}'
    return [
        {'role': 'system', 'content': system_prompt(standoff)},
        {'role': 'user', 'content': f'{world}\n\nRequest: {intent}'},
    ]


_FAILURE = re.compile(r'^FAILED (\d+)/(\d+) (\S+?)(?:: (.*))?$')


def parse_failure(status):
    """'FAILED 2/3 pick: timed out after 60 s (attempt 3/3)' ->
    {'index': 2, 'total': 3, 'skill': 'pick', 'reason': '...'}; None if the status
    has no step in it (e.g. 'FAILED: unexpected error')."""
    match = _FAILURE.match(status.strip())
    if not match:
        return None
    index, total, skill, reason = match.groups()
    return {'index': int(index), 'total': int(total), 'skill': skill, 'reason': reason or ''}


def describe_plan(plan):
    lines = []
    for number, step in enumerate(plan, start=1):
        if step['object_id']:
            what = f"{step['name']} {step['object_id']}"
        else:
            what = f"{step['name']} x={step['x']:.2f}, y={step['y']:.2f}"
        lines.append(f'{number}. {what}')
    return '\n'.join(lines)


def describe_failure(status):
    parsed = parse_failure(status)
    if parsed is None:
        return f'The plan failed: {status}'
    done = parsed['index'] - 1
    text = (f"The plan failed at step {parsed['index']} of {parsed['total']} "
            f"({parsed['skill']}): {parsed['reason'] or 'no reason given'}.")
    if done:
        text += f' Steps 1 to {done} succeeded, so their results still hold.'
    return text + ' The failed step and the steps after it did not (fully) happen.'


REPLAN_RULES = """A previous plan for this request failed. Make ONE new complete plan, starting from the current situation (the robot and objects may have moved), that still achieves the original request. You may use any skills in a different order or approach; do not simply repeat the failed plan unless nothing else makes sense. If there is no sensible way to still achieve the request, set feasible to false and explain why."""


def build_replan_messages(intent, old_plan, failure, objects, standoff=STANDOFF_M, robot=None,
                          held=None):
    world = f'{describe_robot(robot)}\n{describe_hand(held)}\n{describe_objects(objects)}'
    user = (f'{world}\n\nOriginal request: {intent}\n\n'
            f'Previous plan:\n{describe_plan(old_plan)}\n\n{describe_failure(failure)}\n\n{REPLAN_RULES}')
    return [
        {'role': 'system', 'content': system_prompt(standoff)},
        {'role': 'user', 'content': user},
    ]


def _inside_map(x, y):
    low, high = MAP_MIN_M + EDGE_MARGIN_M, MAP_MAX_M - EDGE_MARGIN_M
    return low <= x <= high and low <= y <= high


def build_plan(reply, objects, held=None):
    """Validate the LLM's JSON reply and turn it into plan dicts.
    Returns a PlanResult; any problem means no plan (a half-valid plan is never run)."""
    if not isinstance(reply, dict):
        return PlanResult(error='LLM reply was not an object')
    if not reply.get('feasible'):
        return PlanResult(error=f"cannot do that: {reply.get('reason') or 'no reason given'}")
    steps = reply.get('steps')
    if not isinstance(steps, list) or not steps:
        return PlanResult(error='LLM gave no steps')
    if len(steps) > MAX_STEPS:
        return PlanResult(error=f'plan too long ({len(steps)} steps, limit {MAX_STEPS})')

    by_id = {o['object_id']: o for o in objects}
    plan = []
    for number, raw in enumerate(steps, start=1):
        name = raw.get('skill')
        object_id = raw.get('object_id') or ''
        step = {'name': name, 'object_id': object_id}
        x, y = raw.get('x'), raw.get('y')
        if x is not None and y is not None:
            step['x'], step['y'] = float(x), float(y)
        problems = skill_catalog.check_step(_with_world_pose(step, by_id))
        if problems:
            return PlanResult(error=f'step {number} ({name}): {problems[0]}')
        if object_id and object_id not in by_id:
            return PlanResult(error=f'step {number} ({name}): unknown object {object_id!r}')
        spec = skill_catalog.get(name)
        if spec.needs_pose:
            if not _inside_map(step['x'], step['y']):
                return PlanResult(error=f"step {number} ({name}): ({step['x']:g}, {step['y']:g}) is outside the map")
        plan.append(_with_world_pose(step, by_id, final=True))
    problems = skill_catalog.check_hand(plan, held)
    if problems:
        return PlanResult(error=problems[0])
    return PlanResult(plan=plan)


def _with_world_pose(step, by_id, final=False):
    """Skills that act on an object get the object's pose from the world, not from the LLM.
    With final=True the missing numbers are also defaulted to 0."""
    out = dict(step)
    obj = by_id.get(step['object_id'])
    if obj is not None:
        out['x'], out['y'], out['z'] = obj['x'], obj['y'], obj['z']
    if final:
        out.setdefault('x', 0.0)
        out.setdefault('y', 0.0)
        out.setdefault('z', 0.0)
    return out


def plan_with_llm(intent, objects, config, standoff=STANDOFF_M, robot=None, held=None,
                  **client_kwargs):
    """Intent -> PlanResult. Never raises: LLM/network problems come back as .error."""
    try:
        reply = chat_json(config, build_messages(intent, objects, standoff, robot, held), SCHEMA, **client_kwargs)
    except LLMError as exc:
        return PlanResult(error=str(exc))
    return build_plan(reply, objects, held)


def replan_with_llm(intent, old_plan, failure, objects, config, standoff=STANDOFF_M, robot=None,
                    held=None, **client_kwargs):
    """Failed plan -> one new PlanResult. Never raises (errors come back as .error)."""
    try:
        reply = chat_json(
            config, build_replan_messages(intent, old_plan, failure, objects, standoff, robot, held),
            SCHEMA, **client_kwargs)
    except LLMError as exc:
        return PlanResult(error=str(exc))
    return build_plan(reply, objects, held)
