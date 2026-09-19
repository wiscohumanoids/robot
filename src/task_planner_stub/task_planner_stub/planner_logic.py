"""Pure planning logic for task_planner_stub (no rclpy, unit-testable anywhere).

This is the canned "planner". The real one (LLM call + skill library) replaces
plan_for_intent(); the node around it and the messages in and out stay.
"""

# Verbs that select the manipulation skill. First match wins; default is "pick".
_TASK_KEYWORDS = (
    ('handover', 'handover'),
    ('hand over', 'handover'),
    ('place', 'place'),
    ('put', 'place'),
    ('drop', 'place'),
    ('pick', 'pick'),
    ('grab', 'pick'),
    ('get', 'pick'),
)


def choose_task(intent):
    text = intent.lower()
    for keyword, task in _TASK_KEYWORDS:
        if keyword in text:
            return task
    return 'pick'


def choose_object(intent, objects):
    """objects: list of dicts {object_id, label, x, y, z}. Prefer one whose
    label appears in the intent text, else the first. None if there are none."""
    if not objects:
        return None
    text = intent.lower()
    for obj in objects:
        if obj['label'].lower() in text:
            return obj
    return objects[0]


def plan_for_intent(intent, objects, standoff=0.6):
    """Return a list of skill dicts {name, object_id, x, y, z}, or [] if no plan
    is possible (e.g. perception has reported no objects yet).

    Plan: walk to `standoff` metres short of the object along +x, then run the
    manipulation task on it. Coordinates are in the `map` frame.
    """
    obj = choose_object(intent, objects)
    if obj is None:
        return []
    return [
        {'name': 'navigate_to', 'object_id': '', 'x': obj['x'] - standoff, 'y': obj['y'], 'z': 0.0},
        {'name': choose_task(intent), 'object_id': obj['object_id'],
         'x': obj['x'], 'y': obj['y'], 'z': obj['z']},
    ]
