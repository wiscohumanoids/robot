"""Planner logic that doesn't need the API: plan checks and the memory summary."""
from hsim_planner.prompt import check_plan, scene_text, user_message

MEMORY = {'robot': {'x': 0, 'y': 0, 'yaw_deg': 0}, 'objects': [
    {'id': 'table_1', 'label': 'table', 'position': [2, 0, 0.37], 'size': [1.2, 0.8, 0.75], 'confidence': 0.99,
     'seen_ago_s': 0.5, 'times_seen': 50, 'on': []},
    {'id': 'table_2', 'label': 'table', 'position': [2.4, 0, 0.37], 'size': [1.2, 0.8, 0.75], 'confidence': 0.4,
     'seen_ago_s': 9, 'times_seen': 2, 'on': []}]}


def plan(*steps):
    """A plan made of (skill, target) steps."""
    return {'understood': True, 'reasoning': '', 'reply': '', 'skills': [
        {'name': n, 'target': t, 'hints': [], 'text': ''} for n, t in steps]}


def test_known_target_ok():
    assert check_plan(plan(('navigate_to', 'table')), MEMORY) is None


def test_unseen_target_rejected_with_reason():
    why = check_plan(plan(('navigate_to', 'red block')), MEMORY)
    assert why and 'has not seen' in why and 'table' in why


def test_unseen_target_ok_after_finding_it():
    assert check_plan(plan(('navigate_to', 'table'), ('find_object', 'red block'), ('navigate_to', 'red block')), MEMORY) is None


def test_scene_keeps_best_entry_per_label():
    text = scene_text(MEMORY)
    assert 'table_1' in text and 'table_2' not in text


def test_failure_history_reaches_the_model():
    msg = user_message('pick up the red block', MEMORY, [{'skills': [{'name': 'pick'}], 'failure': 'no hands'}])
    assert 'It failed: no hands' in msg and 'pick up the red block' in msg
