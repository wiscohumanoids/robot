import pytest

from manipulation_runner.hand_logic import EMPTY, next_hand


@pytest.mark.parametrize('held, task, obj, success, expected', [
    (EMPTY, 'pick', 'cup_1', True, 'cup_1'),
    ('cup_1', 'place', 'cup_1', True, EMPTY),
    ('cup_1', 'handover', 'cup_1', True, EMPTY),
    (EMPTY, 'pick', 'cup_1', False, EMPTY),          # failed pick: nothing in the hand
    ('cup_1', 'place', 'cup_1', False, 'cup_1'),     # failed place: still holding (stub cannot sense)
    ('cup_1', 'handover', 'cup_1', False, 'cup_1'),
    ('cup_1', 'dance', 'x', True, 'cup_1'),          # unknown task changes nothing
    ('cup_1', 'pick', 'ball_1', True, 'ball_1'),     # stub: a second pick just swaps the object
])
def test_next_hand(held, task, obj, success, expected):
    assert next_hand(held, task, obj, success) == expected


def test_a_pick_place_cycle_ends_with_an_empty_hand():
    held = EMPTY
    for task in ('pick', 'place'):
        held = next_hand(held, task, 'cup_1', True)
    assert held == EMPTY
