"""Verify two-block association using only calibrated image detections."""

import numpy as np

from tools.dino_stack_track import BlockMap, StackFrame, StackTracker
from test_dino_pick_track import OVERHEAD, SECOND_WRIST, WRIST, camera_at, detection


SOURCE = np.array([[.24, .07, .13], [.272, .07, .13]])
TARGET = np.array([[.25, .175, .13], [.282, .175, .13]])


def frames(source=SOURCE, target=TARGET):
    """Project an upright pair at a height unknown to the tracker."""
    blocks = [(source, "red"), (target, "blue")]
    return [StackFrame(camera, tuple(detection(camera, points, color=color)
                                     for points, color in blocks if points is not None))
            for camera in (OVERHEAD, WRIST, SECOND_WRIST)]


def test_recovers_both_positions_from_multiview_detections():
    top, *sides = frames()
    tracker = StackTracker(top, *top.detections)
    result = tracker.observe(top, sides)
    np.testing.assert_allclose(result.source.studs_xyz, SOURCE)
    np.testing.assert_allclose(result.target.studs_xyz, TARGET)
    assert result.target.views == 3


def test_remeasures_moved_target_beside_held_source():
    top, *sides = frames()
    tracker = StackTracker(top, *top.detections)
    initial = tracker.observe(top, sides)

    moved = TARGET + [.008, -.006, .005]
    held = moved + [0., -.018, .04]
    top, *sides = frames(held, moved)
    result = tracker.observe(top, sides, source_held=True)

    assert result.source is None
    assert tracker.latest.source is initial.source
    np.testing.assert_allclose(result.target.studs_xyz, moved)


def test_uses_both_wrists_when_overhead_only_detects_one_stud():
    top, *sides = frames()
    tracker = StackTracker(top, *top.detections)
    source, target = top.detections
    partial = detection(OVERHEAD, TARGET[:1], color=target.color, bbox=target.bbox_xyxy)

    result = tracker.observe(StackFrame(OVERHEAD, (source, partial)), sides)

    np.testing.assert_allclose(result.target.studs_xyz, TARGET)
    assert result.target.views == 2


def test_retains_last_pose_when_current_detections_cannot_confirm_target():
    top, *sides = frames()
    tracker = StackTracker(top, *top.detections)
    initial = tracker.observe(top, sides)
    top, *sides = frames(target=TARGET[:1])

    result = tracker.observe(top, sides)

    assert result.target is None
    assert tracker.latest.target is initial.target

def test_passive_map_refines_three_blocks_from_successive_wrist_positions():
    third = SOURCE + [.075, -.042, 0.]
    objects = [(SOURCE, "red"), (TARGET, "blue"), (third, "red")]

    def frame(camera, visible):
        return StackFrame(camera, tuple(detection(camera, objects[i][0], color=objects[i][1])
                                        for i in visible))

    top = frame(OVERHEAD, [0, 1, 2])
    scene = BlockMap(top)
    first = scene.observe(top, [frame(WRIST, [1, 2, 0])])
    assert set(first) == {0, 1, 2}
    second = scene.observe(frame(OVERHEAD, [2, 1, 0]), [frame(SECOND_WRIST, [0, 2, 1])])
    for block_id, (studs, _) in enumerate(objects):
        assert second[block_id].views == 3
        np.testing.assert_allclose(second[block_id].studs_xyz, studs)

    third_view = frame(camera_at([.2, .22, .45]), [1, 2])
    current = scene.observe(frame(OVERHEAD, [2, 1]), [third_view], excluded=(0,))
    assert set(current) == {1, 2}
    assert scene.latest[0] is second[0]


def test_placed_block_requires_measurement_near_new_grid_location():
    top, *sides = frames()
    scene = BlockMap(top)
    scene.observe(top, sides)
    goal = TARGET + [0., 0., .038]
    scene.expect_placement(0, goal)
    assert 0 not in scene.observe(top, sides)
    assert 0 not in scene.latest

    placed = goal + [.003, -.002, .001]
    top, *sides = frames(placed, None)
    current = scene.observe(top, sides, excluded=(1,))
    np.testing.assert_allclose(current[0].studs_xyz, placed)


def test_map_leaves_competing_claims_unresolved():
    top, *sides = frames()
    scene = BlockMap(StackFrame(top.camera, (top.detections[0],) * 2))
    assert scene.observe(top, sides) == {}
    assert scene.latest == {}


def test_mixed_rows_preserve_lengths_and_positions_with_shuffled_stud_order():
    lengths = (2, 3, 4)
    colors = ("red", "blue", "green")
    rows = [np.array([.29, .055 * i, .13]) + np.arange(n)[:, None] * [.024, .021, 0]
            for i, n in enumerate(lengths)]

    def frame(camera, indices):
        return StackFrame(camera, tuple(detection(camera, row[order], color=color)
                                        for row, order, color in zip(rows, indices, colors)))

    top = frame(OVERHEAD, ([1, 0], [1, 0, 2], [2, 0, 3, 1]))
    side = frame(SECOND_WRIST, ([0, 1], [2, 1, 0], [1, 3, 2, 0]))
    scene = BlockMap(top)
    current = scene.observe(top, [side])
    assert [len(current[i].studs_xyz) for i in current] == list(lengths)
    for i, row in enumerate(rows):
        np.testing.assert_allclose(current[i].xyz, row.mean(axis=0))


def test_recovers_four_stud_row_when_wrist_only_sees_middle_and_last_studs():
    row = np.array([.24, .12, .13]) + np.arange(4)[:, None] * [.024, .021, 0]
    block = detection(OVERHEAD, row[[1, 3, 0, 2]])
    top = StackFrame(OVERHEAD, (block,))
    side = StackFrame(SECOND_WRIST, (detection(SECOND_WRIST, row[[1, 3]]),))
    scene = BlockMap(top)
    pose = scene.observe(top, [side])[0]
    np.testing.assert_allclose(pose.studs_xyz, row[[1, 3, 0, 2]])
    scene.expect_placement(0, row)
    hidden = StackFrame(OVERHEAD, (detection(OVERHEAD, row[1:3]),))
    side = StackFrame(SECOND_WRIST, (detection(SECOND_WRIST, row[1:3]),))
    assert not scene.observe(hidden, [side])


def test_recovers_full_row_from_one_wrist_when_overhead_only_sees_the_middle():
    row = np.array([.24, .12, .13]) + np.arange(4)[:, None] * [.024, .021, 0]
    top = StackFrame(OVERHEAD, (detection(OVERHEAD, row[1:3]),))
    side = StackFrame(SECOND_WRIST, (detection(SECOND_WRIST, row),))
    pose = BlockMap(top).observe(top, [side])[0]
    np.testing.assert_allclose(pose.studs_xyz, row)


def test_single_stud_confirmed_against_full_wrist_row_triangulates_height():
    row = np.array([.24, .12, .13]) + np.arange(3)[:, None] * [.032, 0., 0.]
    bbox = detection(OVERHEAD, row).bbox_xyxy
    top = StackFrame(OVERHEAD, (detection(OVERHEAD, row[:1], bbox=bbox),))
    wrist = StackFrame(SECOND_WRIST, (detection(SECOND_WRIST, row),))
    pose = BlockMap(top).observe(top, [wrist])[0]
    np.testing.assert_allclose(pose.studs_xyz, row)


def test_confirmed_row_length_cannot_grow_after_inventory():
    top, *sides = frames()
    scene = BlockMap(top)
    scene.observe(top, sides)
    scene.confirm_lengths()
    extra = np.vstack([TARGET, TARGET[-1] + [.032, 0., 0.]])
    top, *sides = frames(target=extra)
    assert 1 not in scene.observe(top, sides)
    np.testing.assert_allclose(scene.latest[1].studs_xyz, TARGET)
