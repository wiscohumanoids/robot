import numpy as np
import pytest

from sim.mega_bloks import PITCH_M
from sim.mega_bloks_grid import plan_grid
from sim.mega_bloks_stack import STACK_CENTER_GAP_M


def studs(x, y, yaw=0., z=.1):
    delta = np.array([np.cos(yaw), np.sin(yaw), 0.]) * PITCH_M / 2
    return np.array([np.array([x, y, z]) + sign * delta for sign in (-1, 1)])


def test_anchor_grid_uses_latest_stud_row_and_actual_support():
    blocks = {i: studs(.2 + i * .08, .15, .7) for i in range(5)}
    base, first, second, third = plan_grid(blocks, 1, first_source=2)
    assert base.source == 2 and base.support is None
    assert [first.support, second.support, third.support] == [1, first.source, second.source]
    assert {base.source, first.source, second.source, third.source} == {0, 2, 3, 4}
    assert [p.cell[2] for p in (base, first, second, third)] == list(range(4))
    anchor = studs(.235, .105, .81)
    shift = base.cell[0] * np.array([np.cos(.81), np.sin(.81)]) * PITCH_M
    target = base.target_studs(anchor[::-1])
    np.testing.assert_allclose(target[:, :2], anchor[:, :2] + shift)
    support = studs(.238, .110, .85, z=.143)
    np.testing.assert_allclose(second.target_studs(anchor, support[::-1]),
                               support + [0., 0., STACK_CENTER_GAP_M])


def test_base_avoids_stationary_blocks_in_destination():
    blocks = {'anchor': studs(.23, .15), 'pick': studs(.34, .25),
              'obstacle': studs(.23 + 2 * PITCH_M, .15)}
    base, top = plan_grid(blocks, 'anchor', first_source='pick')
    assert base.cell == (-2, 0, 0)
    assert top.source == 'obstacle'


def test_two_blocks_fit_as_one_tower():
    blocks = {0: studs(.2, .1), 1: studs(.23, .2)}
    move, = plan_grid(blocks, 0)
    assert (move.source, move.support, move.cell) == (1, 0, (0, 0, 1))
    with pytest.raises(ValueError, match='distinct'):
        plan_grid(blocks, 0, first_source=0)


def test_planner_tries_another_side_if_preflight_rejects_closest_cell():
    rows = {0: studs(.2, .1), 1: studs(.3, .12), 2: studs(.33, .25)}
    base, top = plan_grid(rows, 0, reachable=lambda source, row: row.mean(axis=0)[0] <= .201)
    assert base.cell == (-2, 0, 0)
    assert top.support == 0


def test_grid_coordinates_measure_unaligned_blocks_in_anchor_frame():
    from sim.mega_bloks_grid import grid_coordinates

    yaw = .6
    anchor = studs(.23, .15, yaw)
    shift = np.array([np.cos(yaw) + 2*np.sin(yaw), np.sin(yaw) - 2*np.cos(yaw), 0.]) * PITCH_M
    shift[2] = STACK_CENTER_GAP_M
    measured = grid_coordinates({0: anchor[::-1], 1: anchor + shift}, 0)
    np.testing.assert_allclose(measured[0], [0., 0., 0.])
    np.testing.assert_allclose(measured[1], [1., -2., 1.])
