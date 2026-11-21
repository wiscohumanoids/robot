"""Exercise aligned pickups, releases and load-bearing stud contacts in MuJoCo."""

import numpy as np
import pytest

from sim.mega_bloks import PITCH_M
from sim.mega_bloks_grid import plan_grid
from sim.mega_bloks_stack import STACK_CENTER_GAP_M, build_stack_scene
from sim.mega_bloks_stack_motion import StackMotion
from sim.panthera_env import PantheraSim
from tools.dino_pick_targets import jaw_axes
from tools.dino_stack_score import StackObserver


def test_midpoint_observation_during_a_seated_stack(tmp_path):
    scene, blocks = build_stack_scene(tmp_path / 'scene.xml', seed=7)
    sim = PantheraSim(scene)
    sim.reset(randomize=False)
    sim.step(400)
    bodies = [sim.data.body(block['body_name']) for block in blocks]
    source, base = [body.xpos.copy() for body in bodies]
    source_studs, base_studs = [
        np.array([sim.data.geom(stud).xpos.copy() for stud in block['stud_geoms']])
        for block in blocks
    ]
    ticks = []

    def track(_motion, phase):
        ticks.append((phase, sim.ee_pos(), sim.data.time))

    motion = StackMotion(sim, on_step=track)
    yaw = motion.choose_stack_yaw(source, source_studs, base, base_studs, jaw_axes(source_studs))
    assert yaw is not None
    capture_times = []

    def refine_pick(_center, heading):
        assert sim.data.time - capture_times[0] >= .5
        return source, heading

    error = np.array([.002, -.001, .001])
    assert motion.pick(source + error, yaw, refine=refine_pick,
                       observe_approach=lambda: capture_times.append(sim.data.time))

    samples = {}

    def capture(phase):
        samples[phase] = (sim.ee_pos(), np.linalg.norm(sim.dq))

    assert motion.stack(source, source_studs, base + error, base_studs + error,
                        observe=capture, refine=lambda plan: plan.release - error)
    assert list(samples) == ['lift', 'transfer_mid', 'transfer_above']
    start, _ = samples['lift']
    middle, speed = samples['transfer_mid']
    end, _ = samples['transfer_above']
    np.testing.assert_allclose(middle[:2], (start + end)[:2] / 2, atol=.01)
    assert speed > .1  # rad/s at the moving wrist-camera capture
    # Each vertical leg begins while the wrist is already moving downward.
    # A stopped hover or preinsertion waypoint would give zero displacement.
    for phase in ('descend', 'preinsert', 'insert'):
        i = next(i for i, sample in enumerate(ticks) if sample[0] == phase)
        assert ticks[i - 1][1][2] - ticks[i][1][2] > .0001

    sim.step(400)
    np.testing.assert_allclose(bodies[0].xpos - bodies[1].xpos,
                               [0., 0., STACK_CENTER_GAP_M], atol=.002)


def test_grid_places_a_base_and_two_tower_levels(tmp_path):
    scene, blocks = build_stack_scene(tmp_path / 'grid.xml', seed=0, count=4)
    sim = PantheraSim(scene)
    sim.reset(randomize=False)
    sim.step(400)
    motion = StackMotion(sim)

    def studs(index):
        return np.array([sim.data.geom(name).xpos.copy() for name in blocks[index]['stud_geoms']])

    def center(index):
        return sim.data.body(blocks[index]['body_name']).xpos.copy()

    anchor = 0
    anchor_center = center(anchor)
    initial_rows = {index: studs(index) for index in range(len(blocks))}
    placements = plan_grid(initial_rows, anchor)
    for move in placements:
        source = center(move.source)
        row = studs(move.source)
        support_row = studs(move.support) if move.support is not None else None
        target_row = move.target_studs(studs(anchor), support_row)
        target_center = target_row.mean(axis=0) + source - row.mean(axis=0)
        yaw = motion.choose_placement_yaw(source, row, target_center, target_row, jaw_axes(row))
        assert yaw is not None
        assert motion.pick(source, yaw)
        assert motion.place_aligned(source, row, target_center, target_row)
        sim.step(200)
        np.testing.assert_allclose(center(move.source), target_center, atol=.003)

    np.testing.assert_allclose(center(anchor), anchor_center, atol=.002)
    for move in placements:
        expected = move.target_studs(initial_rows[anchor]).mean(axis=0)
        np.testing.assert_allclose(studs(move.source).mean(axis=0), expected, atol=.005)


def _row(sim, block):
    return np.array([sim.data.geom(name).xpos.copy() for name in block['stud_geoms']])


def _place_row(motion, block, target):
    """Isolate IK and contact dynamics with measured simulation poses."""
    sim = motion.sim
    row = _row(sim, block)
    source = sim.data.body(block['body_name']).xpos.copy()
    desired = target.mean(axis=0) + (source - row.mean(axis=0))
    # Camera detections can list studs in a different order between planning
    # and release, including swapping which row end is seen first.
    yaw = motion.choose_placement_yaw(source, np.roll(row, 1, axis=0), desired, target, jaw_axes(row))
    assert yaw is not None
    assert motion.pick(source, yaw)
    assert motion.place_aligned(source, row[::-1], desired, target)
    sim.step(400)
    actual = _row(sim, block)
    error = min(np.linalg.norm(order - target, axis=1).max() for order in (actual, actual[::-1]))
    assert error < .005


def test_rotated_four_stud_bridge_remains_on_both_pillars_under_load(tmp_path):
    scene, blocks = build_stack_scene(tmp_path / 'gate.xml', seed=0, lengths=(2, 2, 4, 2))
    sim = PantheraSim(scene)
    sim.reset(randomize=False)
    sim.step(400)
    motion = StackMotion(sim)
    observer = StackObserver(sim.model, sim.data, blocks)
    anchor, post, beam, cap = blocks
    base = _row(sim, anchor)
    along = base[1] - base[0]
    across = np.array([-along[1], along[0], 0.]) / np.linalg.norm(along[:2])

    _place_row(motion, post, base + 3 * PITCH_M * across)
    bridge = np.linspace(_row(sim, anchor)[0], _row(sim, post)[0], num=4)
    _place_row(motion, beam, bridge + [0., 0., STACK_CENTER_GAP_M])
    _place_row(motion, cap, _row(sim, beam)[1:3] + [0., 0., STACK_CENTER_GAP_M])

    supports = observer._snapshot().supports
    for pillar in (anchor, post):
        assert ('bench', pillar['body_name']) in supports
        assert (pillar['body_name'], beam['body_name']) in supports
    assert (beam['body_name'], cap['body_name']) in supports


@pytest.mark.parametrize(("anchor_length", "support_stud"), [(2, 0), (4, 2)])
def test_three_stud_row_crosses_on_one_seated_socket(tmp_path, anchor_length, support_stud):
    scene, blocks = build_stack_scene(tmp_path / 'crossing.xml', seed=0, lengths=(anchor_length, 3))
    sim = PantheraSim(scene)
    sim.reset(randomize=False)
    sim.step(400)
    anchor, top = blocks
    base = _row(sim, anchor)
    along = base[1] - base[0]
    across = np.array([-along[1], along[0], 0.]) / np.linalg.norm(along[:2])
    target = np.array([base[support_stud] + i * PITCH_M * across for i in (-1, 0, 1)])
    _place_row(StackMotion(sim), top, target + [0., 0., STACK_CENTER_GAP_M])

    supports = StackObserver(sim.model, sim.data, blocks)._snapshot().supports
    assert (anchor['body_name'], top['body_name']) in supports
    assert ('bench', top['body_name']) not in supports
    np.testing.assert_allclose(_row(sim, anchor), base, atol=.004)
