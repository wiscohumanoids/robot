"""What the hand is holding (no rclpy, unit-testable).

State is the held object's id, or '' for an empty hand. STUB rule: a goal that
succeeds changes it (pick -> holding that object, place/handover -> empty); a goal
that fails or is canceled leaves it as it was, because the stub has no sensor to
tell. A real manipulation node should publish what it actually senses instead.
"""
EMPTY = ''


def next_hand(held, task, object_id, success):
    """The hand state after a manipulation goal finished."""
    if not success:
        return held
    if task == 'pick':
        return object_id
    if task in ('place', 'handover'):
        return EMPTY
    return held
