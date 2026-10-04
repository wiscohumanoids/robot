import pytest

from task_planner import skill_catalog as cat
from task_planner.planner_logic import plan_for_intent

CUBE = {'object_id': 'cube_1', 'label': 'cube', 'x': 2.0, 'y': 0.0, 'z': 0.0}


def test_catalog_lists_the_four_skills():
    assert cat.names() == ['navigate_to', 'pick', 'place', 'handover']


def test_entries_are_keyed_by_their_own_name_and_documented():
    for name, spec in cat.CATALOG.items():
        assert spec.name == name and spec.description and spec.requires and spec.effect


def test_get_known_and_unknown():
    assert cat.get('pick').needs_object
    assert cat.get('dance') is None


@pytest.mark.parametrize('step', [
    {'name': 'navigate_to', 'object_id': '', 'x': 1.0, 'y': 2.0},
    {'name': 'pick', 'object_id': 'cube_1'},
    {'name': 'place', 'object_id': 'cube_1', 'x': 1.0, 'y': 1.0},
    {'name': 'handover', 'object_id': 'cube_1'},
])
def test_valid_steps_have_no_problems(step):
    assert cat.check_step(step) == []


@pytest.mark.parametrize('step, fragment', [
    ({'name': 'dance'}, 'unknown skill'),
    ({}, 'unknown skill'),
    ({'name': 'pick', 'object_id': ''}, 'needs an object_id'),
    ({'name': 'pick'}, 'needs an object_id'),
    ({'name': 'navigate_to', 'object_id': 'cube_1', 'x': 1.0, 'y': 1.0}, 'takes no object_id'),
    ({'name': 'navigate_to', 'x': 1.0}, 'needs a target pose'),
    ({'name': 'navigate_to'}, 'needs a target pose'),
])
def test_invalid_steps_are_reported(step, fragment):
    assert any(fragment in p for p in cat.check_step(step))


def test_describe_mentions_every_skill_with_its_arguments():
    text = cat.describe()
    for name in cat.names():
        assert f'- {name}(' in text
    assert '- navigate_to(pose)' in text and '- pick(object_id)' in text


def test_the_current_canned_planner_only_makes_valid_steps():
    for intent in ('pick up the cube', 'put it down', 'hand over the cube', 'x'):
        for step in plan_for_intent(intent, [CUBE]):
            assert cat.check_step(step) == [], (intent, step)


# ---------------------------------------------------------------- hand state
def pick(o='cup_1'):
    return {'name': 'pick', 'object_id': o}


def release(name, o='cup_1'):
    return {'name': name, 'object_id': o}


NAV = {'name': 'navigate_to', 'object_id': '', 'x': 1.0, 'y': 1.0}


def test_hand_effects_are_declared_for_the_manipulation_skills_only():
    assert {n: cat.get(n).hand for n in cat.names()} == {
        'navigate_to': '', 'pick': 'take', 'place': 'release', 'handover': 'release'}


@pytest.mark.parametrize('plan, held', [
    ([pick()], ''),
    ([NAV, pick(), NAV, release('place')], ''),
    ([release('place')], 'cup_1'),
    ([release('handover')], 'cup_1'),
    ([release('place', 'ball_1'), NAV, pick(), release('handover')], 'ball_1'),   # swap objects
    ([pick(), release('place')], None),             # unknown hand, the plan fixes it
    ([release('place')], None),                     # unknown hand: nothing to object to
    ([NAV], 'cup_1'),
])
def test_possible_plans_have_no_hand_problems(plan, held):
    assert cat.check_hand(plan, held) == []


@pytest.mark.parametrize('plan, held, fragment', [
    ([pick()], 'ball_1', 'step 1 (pick): hand is already holding ball_1'),
    ([pick()], 'cup_1', 'already holding cup_1'),
    ([release('place')], '', 'not holding cup_1 (hand is empty)'),
    ([release('place')], 'ball_1', 'not holding cup_1 (holding ball_1)'),
    ([pick(), pick('ball_1')], '', 'step 2 (pick): hand is already holding cup_1'),
    ([pick(), release('place'), release('handover')], '', 'step 3 (handover)'),
    ([pick(), release('place', 'ball_1')], None, 'not holding ball_1 (holding cup_1)'),
])
def test_impossible_plans_are_reported(plan, held, fragment):
    problems = cat.check_hand(plan, held)
    assert problems and fragment in problems[0]
