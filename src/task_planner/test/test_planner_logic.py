from task_planner.planner_logic import choose_object, choose_task, plan_for_intent

CUBE = {'object_id': 'cube_1', 'label': 'cube', 'x': 1.5, 'y': 0.3, 'z': 0.8}
BALL = {'object_id': 'ball_1', 'label': 'ball', 'x': 2.0, 'y': -0.5, 'z': 0.8}


def test_no_objects_means_no_plan():
    assert plan_for_intent('pick up the cube', []) == []


def test_default_plan_is_navigate_then_pick():
    plan = plan_for_intent('do something', [CUBE], standoff=0.6)
    assert [s['name'] for s in plan] == ['navigate_to', 'pick']
    assert plan[0]['x'] == 1.5 - 0.6 and plan[0]['y'] == 0.3
    assert plan[1]['object_id'] == 'cube_1'


def test_place_verb_selects_place():
    assert choose_task('put the cube down') == 'place'
    assert choose_task('hand over the cube') == 'handover'
    assert choose_task('anything else') == 'pick'


def test_label_in_intent_selects_object():
    assert choose_object('grab the ball', [CUBE, BALL]) is BALL
    assert choose_object('grab it', [CUBE, BALL]) is CUBE


# ---------------------------------------------------------------- the checks the LLM path also makes
from task_planner.planner_logic import plan_checked  # noqa: E402


def test_checked_plan_for_a_pick_needs_no_hand_state():
    for held in (None, ''):
        plan, error = plan_checked('pick up the cube', [CUBE], held=held)
        assert error == '' and [s['name'] for s in plan] == ['navigate_to', 'pick']


def test_checked_plan_rejects_a_pick_while_holding_something():
    plan, error = plan_checked('pick up the cube', [CUBE], held='ball_1')
    assert plan == [] and 'already holding ball_1' in error


def test_checked_plan_rejects_a_place_or_handover_with_an_empty_hand():
    for intent in ('put the cube down', 'hand over the cube'):
        plan, error = plan_checked(intent, [CUBE], held='')
        assert plan == [] and 'not holding cube_1 (hand is empty)' in error, intent


def test_checked_plan_allows_a_place_of_the_held_object_or_an_unknown_hand():
    assert plan_checked('put the cube down', [CUBE], held='cube_1')[1] == ''
    assert plan_checked('put the cube down', [CUBE], held=None)[1] == ''


def test_checked_plan_with_no_objects_says_so():
    assert plan_checked('pick up the cube', []) == ([], 'perception has reported no objects yet')
