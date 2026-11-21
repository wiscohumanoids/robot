import numpy as np
import pytest

from sim.mega_bloks import PITCH_M
from sim.mega_bloks_blueprint import blueprint_coordinates, load_blueprint, plan_blueprint
from sim.mega_bloks_stack import STACK_CENTER_GAP_M


def row(n, center, theta=0.):
    offsets = (np.arange(n) - (n - 1) / 2) * PITCH_M
    return np.asarray(center) + np.outer(offsets, [np.cos(theta), np.sin(theta), 0.])


def castle_rows():
    return {k: row(n, [x, y, .1]) for k, n, x, y in [
        ("anchor", 2, .2, .1), ("wall", 2, .15, .27),
        ("beam", 4, .32, .12), ("crown", 2, .33, .26)]}


def test_castle_bridge_tracks_contacts_on_two_supports():
    rows = castle_rows()
    wall, beam, crown = plan_blueprint(rows, "anchor", "castle", first_source="wall")
    assert wall.source == "wall" and not wall.supports
    assert beam.source == "beam" and beam.supports == ("anchor", "wall")
    assert crown.supports == ("beam",)
    base = rows["anchor"][::-1]
    wall_row = wall.target_studs(base)
    ideal = beam.target_studs(base)
    np.testing.assert_allclose(ideal[0], base[-1] + [0, 0, STACK_CENTER_GAP_M])
    np.testing.assert_allclose(ideal[-1], wall_row[0] + [0, 0, STACK_CENTER_GAP_M])
    # Move the far wall sideways. The crossbeam must translate and rotate,
    # using the contacting end stud on each wall even if their order flips.
    wall_row[:, 0] += .004
    aligned = beam.target_studs(base, {"anchor": base, "wall": wall_row[::-1]})
    np.testing.assert_allclose(aligned[[0, -1], :2],
                               [base[-1, :2], wall_row[0, :2]], atol=.00005)
    np.testing.assert_allclose(np.linalg.norm(np.diff(aligned, axis=0), axis=1), PITCH_M)
    top = crown.target_studs(base, {"beam": aligned[[1, 3, 0, 2]]})
    np.testing.assert_allclose(top, aligned[1:3] + [0, 0, STACK_CENTER_GAP_M])


def test_one_stud_contact_balances_a_perpendicular_three_stud_cap():
    theta = .55
    base = row(2, [.23, .15, .11], theta)
    step, = plan_blueprint({0: base, 1: row(3, [.33, .25, .11])}, 0, "one-stud")
    result = step.target_studs(base)
    np.testing.assert_allclose(result[1], base[0] + [0, 0, STACK_CENTER_GAP_M])
    assert abs((result[-1] - result[0]) @ (base[-1] - base[0])) < 1e-12
    shifted = step.target_studs(base, {0: base[::-1] + [.003, .001, -.002]})
    np.testing.assert_allclose(shifted, result + [.003, .001, -.002])


def test_custom_blueprint_rejects_collision_and_floating_row():
    obj = load_blueprint("one-stud")
    assert load_blueprint(obj) == obj
    obj["blocks"][1]["cell"] = [0, 1, 1]
    with pytest.raises(ValueError, match='support'):
        load_blueprint(obj)
    obj["blocks"][1]["cell"] = [0, 0, 0]
    with pytest.raises(ValueError, match='Overlapping'):
        load_blueprint(obj)


def test_pyramid_assigns_reachable_inventory_and_supports_both_base_blocks():
    rows = {k: row(2, [x, y, .1]) for k, x, y in [(2, .2, .1), (3, .3, .2), (4, .21, .3)]}
    wall, cap = plan_blueprint(rows, 2, "pyramid", reachable=lambda k, t: k != 3 or t[0, 2] > .13)
    assert wall.source == 4 and cap.source == 3 and cap.supports == (2, 4)
    with pytest.raises(ValueError, match='reachable'):
        plan_blueprint(rows, 2, "pyramid", first_source=3, reachable=lambda k, t: k != 3)


def test_custom_blueprint_uses_reversed_four_stud_anchor():
    custom = {"blocks": [{"cell": [0, 0, 0], "length": 4},
                         {"cell": [3, 0, 1], "length": 2, "turn": 2}]}
    base = row(4, [.2, .15, .1])
    step, = plan_blueprint({"base": base[[2, 0, 3, 1]], "cap": row(2, [.3, .1, .1])}, "base", custom)
    np.testing.assert_allclose(step.target_studs(base[::-1]), base[[3, 2]] + [0, 0, STACK_CENTER_GAP_M])


def test_blueprint_inventory_limits():
    with pytest.raises(ValueError, match="two to six"):
        load_blueprint({"blocks": [{"length": 2, "cell": [0, 0, i]} for i in range(7)]})
    with pytest.raises(ValueError, match="length"):
        load_blueprint({"blocks": [{"length": 2, "cell": [0, 0, 0]}, {"length": 1, "cell": [0, 0, 1]}]})


def test_offset_two_stud_row_can_cantilever_on_one_end_stud():
    custom = {"blocks": [{"cell": [0, 0, 0], "length": 2},
                         {"cell": [1, 0, 1], "length": 2}]}
    base = row(2, [.21, .15, .1])
    move, = plan_blueprint({1: base, 2: row(2, [.32, .23, .1])}, 1, custom)
    assert move.supports == (1,)
    np.testing.assert_allclose(move.target_studs(base), base + [PITCH_M, 0, STACK_CENTER_GAP_M])


def test_measured_center_coordinates_use_first_anchor_stud_as_origin():
    base = row(4, [.25, .11, .1], theta=.5)
    axis = np.array([np.cos(.5), np.sin(.5), 0.])
    across = np.array([-axis[1], axis[0], 0.])
    center = base[0] + PITCH_M * (2 * axis - 3 * across) + [0, 0, STACK_CENTER_GAP_M]
    grid = blueprint_coordinates({"base": base[[2, 3, 0, 1]], "cap": row(3, center)}, "base")
    np.testing.assert_allclose(grid["base"], [1.5, 0, 0], atol=1e-12)
    np.testing.assert_allclose(grid["cap"], [2, -3, 1], atol=1e-12)
