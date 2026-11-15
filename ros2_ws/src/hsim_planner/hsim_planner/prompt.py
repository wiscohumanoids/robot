"""What the planner tells Claude, and the exact shape of the answer it must give back."""

import json

SKILLS = ['navigate_to', 'find_object', 'search_for', 'look_around', 'pick', 'place', 'say']

SYSTEM_PROMPT = """You are the task planner of a Unitree G1 humanoid robot in a room. You turn a person's
request into a short sequence of skills for the robot's behavior tree. You never control motion
directly and you never output coordinates: positions come from the robot's own perception and map.

Skills (use object labels exactly as they appear in the robot's memory):
- navigate_to(target): walk up to a remembered object and stop within arm's reach of it, facing it, on a
  side the robot can stand at (a fridge: its open front; a block on a table: the table edge nearest the
  block). "Go to X", "in front of X" and "next to X" all mean navigate_to(X). Fails if X is not in memory.
- find_object(target): check memory for the object; if missing, turn around once to look. Use it after
  walking somewhere the object is likely to be (e.g. on a table) before going to it or picking it up.
- search_for(target, hints): look around, then visit each hint (remembered furniture, best first) and look
  around there, until the object is seen. Use when you do not know where the object is.
- look_around(): turn in place once to see more of the room.
- pick(target) / place(target): hand skills, done standing at a counter or table. Always navigate_to the object
  first (that brings the robot to arm's reach, facing the surface). For "put X on/in Y" plan pick(X) then place(Y)
  right after each other: they run as one motion ("pick up the X and put it on the Y"), so Y must be within reach
  of the same spot (same counter). If Y is somewhere else, say that the robot can't carry things between places yet.
- say(text): tell the person something short.

The robot only knows what it has seen. Its memory lists objects with label, position (meters, map frame),
how confident it is, how long ago it was seen, and what they are on. Small objects are only visible up
close, so an object the person mentions may be missing from memory even if it is there.

How to plan:
- Object in memory: navigate_to it (then pick/place if asked).
- Object not in memory but its likely place is known (e.g. "the block on the table" and a table is known):
  navigate_to(the place), then find_object(object), then continue.
- Location unknown: search_for(object, hints = known furniture where it plausibly is, most likely first).
- Ambiguous or impossible request (e.g. two equally matching objects, or something the robot cannot do
  even in principle): set understood=false, ask or explain in `reply`, and return no skills.
- If a previous attempt failed (you will be told why), plan around the failure using the updated memory,
  or explain in `reply` why it cannot be done and return no skills.
Keep plans short. `reasoning` is one sentence for the logs. `reply` is one sentence for the person.
Fill unused fields with "" or []."""

PLAN_SCHEMA = {
    'type': 'object',
    'properties': {
        'understood': {'type': 'boolean'},
        'reasoning': {'type': 'string'},
        'reply': {'type': 'string'},
        'skills': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'name': {'type': 'string', 'enum': SKILLS},
                    'target': {'type': 'string'},
                    'hints': {'type': 'array', 'items': {'type': 'string'}},
                    'text': {'type': 'string'},
                },
                'required': ['name', 'target', 'hints', 'text'],
                'additionalProperties': False,
            },
        },
    },
    'required': ['understood', 'reasoning', 'reply', 'skills'],
    'additionalProperties': False,
}


def scene_text(summary: dict | None) -> str:
    """The robot's memory, compact and readable for the model."""
    if not summary:
        return 'Robot memory: not available yet (the robot has not started perceiving).'
    lines = [f"Robot pose (map frame): {summary.get('robot')}"]
    objs = summary.get('objects', [])
    if not objs:
        lines.append('Objects seen so far: none.')
    else:
        lines.append('Objects seen so far (best estimate per object):')
        best = {}
        for o in objs:                      # keep the most confident entry per label
            if o['label'] not in best or o['confidence'] > best[o['label']]['confidence']:
                best[o['label']] = o
        for o in sorted(best.values(), key=lambda o: o['label']):
            on = f", on {', '.join(o['on'])}" if o.get('on') else ''
            lines.append(f"- {o['label']} ({o['id']}) at {o['position']}, size {o['size']}, "
                         f"confidence {o['confidence']}, seen {o['seen_ago_s']} s ago{on}")
    return '\n'.join(lines)


def user_message(command: str, summary: dict | None, history: list[dict]) -> str:
    parts = [f'Request: "{command}"', '', scene_text(summary)]
    for h in history:
        parts += ['', f"Previous attempt: {json.dumps(h['skills'])}", f"It failed: {h['failure']}"]
    return '\n'.join(parts)


def known_labels(summary: dict | None) -> set[str]:
    return {o['label'].lower() for o in (summary or {}).get('objects', [])}


def check_plan(plan: dict, summary: dict | None) -> str | None:
    """Reject plans that go to things the robot has never seen without looking for them first.
    Returns a reason, or None if the plan is fine."""
    known = known_labels(summary)
    found_later = set()
    for i, s in enumerate(plan['skills']):
        t = s['target'].lower()
        if s['name'] in ('find_object', 'search_for'):
            found_later.add(t)
        if s['name'] == 'navigate_to' and t and t not in known and t not in found_later:
            return (f'step {i} navigate_to("{s["target"]}"): the robot has not seen a "{s["target"]}". '
                    f'Known objects: {sorted(known) or "none"}. Go somewhere it is likely to be and '
                    f'find_object it first, or search_for it.')
    return None
