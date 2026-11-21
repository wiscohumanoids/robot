"""Verify calibrated RGB correspondence and triangulation without a support plane."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perception.pickup import BlockDetection
from tools.dino_pick_camera import CameraCalibration
from tools.dino_pick_elevated import FixedCameraTrack
from tools.dino_pick_track import BlockTrack, ImageAnchor


CENTER = np.array([.31, .12, .18])
STUDS = CENTER + [[-.032, 0, 0], [0, 0, 0], [.032, 0, 0]]


def camera_at(position, *, focal=500.):
    position = np.asarray(position, dtype=float)
    forward = CENTER - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 1, 0])
    right /= np.linalg.norm(right)
    return CameraCalibration(position, right, np.cross(right, forward), forward,
                             focal, np.array([255.5, 255.5]))


OVERHEAD = camera_at([.31, .12, .81])
WRIST = camera_at([.16, .12, .44])
SECOND_WRIST = camera_at([.40, .23, .47])


def detection(camera, studs=STUDS, *, shift=(0, 0), color="red", bbox=None):
    pixels = np.array([camera.project(stud) for stud in studs]) + shift
    if bbox is None:
        bbox = (*np.floor(pixels.min(axis=0) - 10), *np.ceil(pixels.max(axis=0) + 10))
    return BlockDetection(tuple(pixels.mean(axis=0)), color, 1., bbox,
                          tuple(map(tuple, pixels)), 1000)


@pytest.mark.parametrize("height_delta", [-.10, 0, .08])
def test_estimates_elevated_or_lowered_stud_height_and_fuses_two_wrist_views(height_delta):
    studs = STUDS + [0, 0, height_delta]
    track = BlockTrack(OVERHEAD, detection(OVERHEAD, studs))

    first = track.update(WRIST, [detection(WRIST, studs[::-1])])
    second = track.update(SECOND_WRIST, [detection(SECOND_WRIST, studs[[1, 2, 0]])])

    assert first.status == second.status == "tracked"
    assert (first.views, second.views) == (2, 3)
    np.testing.assert_allclose(first.xyz, studs.mean(axis=0), atol=1e-10)
    np.testing.assert_allclose(second.studs_xyz, studs, atol=1e-10)


def test_rejects_weak_parallax():
    nearby = camera_at(OVERHEAD.position + [.003, 0., 0.])
    track = BlockTrack(OVERHEAD, detection(OVERHEAD))
    result = track.update(nearby, [detection(nearby)])
    assert (result.status, result.xyz, result.views) == ("insufficient_parallax", None, 1)


def test_rejects_competing_associations_and_does_not_commit_them():
    track = BlockTrack(OVERHEAD, detection(OVERHEAD))
    result = track.update(WRIST, [detection(WRIST), detection(WRIST, shift=(0, .4))])
    assert (result.status, result.xyz, result.views) == ("target_ambiguous", None, 1)


def test_recovers_after_bad_current_frame_without_reusing_its_pixels():
    track = BlockTrack(OVERHEAD, detection(OVERHEAD))
    first = track.update(WRIST, [detection(WRIST)])
    lost = track.update(SECOND_WRIST, [detection(SECOND_WRIST, shift=(0, 20))])
    recovered = track.update(SECOND_WRIST, [detection(SECOND_WRIST)])
    assert first.status == recovered.status == "tracked"
    assert (lost.status, lost.xyz, lost.views) == ("target_lost", None, 2)
    np.testing.assert_allclose(recovered.xyz, CENTER, atol=1e-10)
    assert recovered.views == 3


def test_rejects_clipped_or_incomplete_regions():
    track = BlockTrack(OVERHEAD, detection(OVERHEAD))
    result = track.update(WRIST, [detection(WRIST, bbox=(0, 120, 310, 300)),
                                 detection(WRIST, STUDS[:2])], frame_shape=(512, 512))
    assert (result.status, result.xyz) == ("target_lost", None)


def test_ignores_extra_stud_peak_that_fails_geometry():
    track = BlockTrack(OVERHEAD, detection(OVERHEAD))
    points = np.vstack((STUDS, CENTER + [.009, .009, 0]))
    result = track.update(WRIST, [detection(WRIST, points)])
    assert result.status == "tracked"
    np.testing.assert_allclose(result.xyz, CENTER, atol=1e-10)


def test_rejects_one_spurious_overhead_peak_and_keeps_the_verified_grid():
    square = CENTER + [[-.016, -.016, 0], [.016, -.016, 0],
                       [-.016, .016, 0], [.016, .016, 0]]
    overhead_studs = np.vstack((square, CENTER))
    track = BlockTrack(OVERHEAD, detection(OVERHEAD, overhead_studs))
    first = track.update(WRIST, [detection(WRIST, square)])
    second = track.update(SECOND_WRIST, [detection(SECOND_WRIST, square[::-1])])
    assert first.status == second.status == "tracked"
    assert (first.views, second.views) == (2, 3)
    np.testing.assert_allclose(second.studs_xyz, square, atol=1e-10)


def test_same_camera_cannot_increase_view_count():
    track = BlockTrack(OVERHEAD, detection(OVERHEAD))
    for _ in range(3):
        result = track.update(WRIST, [detection(WRIST)])
        assert (result.status, result.views) == ("tracked", 2)


def test_rejects_wrist_pattern_matching_same_color_overhead_neighbor_at_another_height():
    single = CENTER[None, :]
    # The wrist sees both world points on the same ray, but overhead separates them.
    other = WRIST.position + 1.2 * (single - WRIST.position)
    reference = detection(OVERHEAD, single)
    neighbor = detection(OVERHEAD, other)
    track = BlockTrack(OVERHEAD, reference, neighbors=[reference, neighbor])
    result = track.update(WRIST, [detection(WRIST, single)])
    assert (result.status, result.reason) == (
        "target_ambiguous", "wrist_region_matches_overhead_neighbor"
    )


def test_refresh_replaces_stale_views_when_the_confirmed_block_moves():
    track = BlockTrack(OVERHEAD, detection(OVERHEAD))
    track.update(WRIST, [detection(WRIST)])
    assert track.update(SECOND_WRIST, [detection(SECOND_WRIST)]).views == 3
    moved = STUDS + [.018, -.012, .010]

    refresh = track.refresh(OVERHEAD, [detection(OVERHEAD, moved[::-1])], WRIST,
                            [detection(WRIST, moved)])

    assert (refresh.status, refresh.reason, refresh.views) == (
        "tracked", "moved_target_reacquired", 2
    )
    np.testing.assert_allclose(refresh.xyz, moved.mean(axis=0), atol=1e-10)
    next_view = track.update(SECOND_WRIST, [detection(SECOND_WRIST, moved)])
    assert next_view.views == 3
    np.testing.assert_allclose(next_view.xyz, refresh.xyz, atol=1e-10)


def test_static_refresh_keeps_the_established_observations():
    track = BlockTrack(OVERHEAD, detection(OVERHEAD))
    track.update(WRIST, [detection(WRIST)])
    refresh = track.refresh(OVERHEAD, [detection(OVERHEAD)], SECOND_WRIST,
                            [detection(SECOND_WRIST)])
    assert (refresh.status, refresh.reason, refresh.views) == ("tracked", "matched_stud_pattern", 3)
    np.testing.assert_allclose(refresh.xyz, CENTER, atol=1e-10)


def test_refresh_rejects_two_nearby_same_color_matches_without_committing_either():
    single = CENTER[None, :]
    track = BlockTrack(OVERHEAD, detection(OVERHEAD, single))
    track.update(WRIST, [detection(WRIST, single)])
    first = single + [0, -.026, 0]
    second = single + [0, .026, 0]
    result = track.refresh(OVERHEAD, [detection(OVERHEAD, first), detection(OVERHEAD, second)],
                           SECOND_WRIST, [detection(SECOND_WRIST, first),
                                          detection(SECOND_WRIST, second)])
    assert (result.status, result.reason, result.xyz, result.views) == (
        "target_ambiguous", "fresh_pair_has_competing_targets", None, 2
    )
    result = track.refresh(OVERHEAD, [detection(OVERHEAD, single)], SECOND_WRIST,
                           [detection(SECOND_WRIST, single)])
    np.testing.assert_allclose(result.xyz, CENTER, atol=1e-10)
    assert result.views == 3


def test_refresh_requires_confirmation_in_both_images():
    track = BlockTrack(OVERHEAD, detection(OVERHEAD))
    track.update(WRIST, [detection(WRIST)])
    result = track.refresh(OVERHEAD, [], SECOND_WRIST, [detection(SECOND_WRIST)])
    assert (result.status, result.xyz, result.views) == ("target_lost", None, 2)


def test_refresh_does_not_substitute_a_previously_seen_neighbor():
    single = CENTER[None, :]
    nearby = single + [0, .035, 0]
    neighbor = detection(OVERHEAD, nearby)
    track = BlockTrack(OVERHEAD, detection(OVERHEAD, single), neighbors=[neighbor])
    assert track.update(WRIST, [detection(WRIST, single)]).status == "tracked"
    result = track.refresh(OVERHEAD, [neighbor], SECOND_WRIST,
                           [detection(SECOND_WRIST, nearby)])
    assert (result.status, result.xyz, result.views) == ("target_ambiguous", None, 2)


def anchor_frame(*positions):
    image = np.full((256, 256, 3), 130, dtype=np.uint8)
    for x, y in positions:
        image[y:y + 18, x:x + 25] = [225, 18, 20]
        image[y + 3:y + 12, x + 4:x + 9] = [175, 12, 14]
        image[y + 4:y + 10, x + 17:x + 22] = [255, 32, 34]
    return image


def test_image_anchor_translates_an_offset_anchor_without_new_detections():
    initial = anchor_frame((100, 95))
    anchor = np.array([115.3, 105.7])
    block = BlockDetection((112, 104), "red", 1., (99, 94, 126, 114), (), 450)
    track = ImageAnchor(initial, block, anchor)
    np.testing.assert_allclose(track.locate(anchor_frame((115, 88))), anchor + [15, -7])
    assert track.confidence > .95


def test_image_anchor_rejects_substantial_occlusion():
    initial = anchor_frame((100, 95))
    block = BlockDetection((112, 104), "red", 1., (99, 94, 126, 114), (), 450)
    track = ImageAnchor(initial, block, (112, 104))
    current = anchor_frame((105, 90))
    current[89:110, 105:119] = 0
    assert track.locate(current) is None
    assert track.confidence == 0.


def test_image_anchor_rejects_two_indistinguishable_candidates():
    initial = anchor_frame((100, 95))
    block = BlockDetection((112, 104), "red", 1., (99, 94, 126, 114), (), 450)
    track = ImageAnchor(initial, block, (112, 104))
    assert track.locate(anchor_frame((78, 95), (122, 95))) is None
    assert track.confidence == 0.


def test_fixed_camera_track_matches_visible_studs_after_an_extra_peak_disappears():
    original = BlockDetection((20, 30), "blue", 1., (8, 18, 42, 48),
                              ((26, 24), (32, 35), (12, 27)), 500)
    moved = BlockDetection((35, 35), "blue", 1., (23, 23, 57, 53),
                           ((47, 40), (41, 29), (51, 26)), 500)
    np.testing.assert_allclose(FixedCameraTrack._shift(original, moved), (15, 5))
