from task_planner_stub.planner_logic import choose_object, choose_task, plan_for_intent

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
