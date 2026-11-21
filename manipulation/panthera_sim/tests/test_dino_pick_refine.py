"""Exercise cross-view stud matching and the gates before descent."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perception.pickup import BlockDetection
from tools.dino_pick_refine import refine_grasp


class Camera:
    def __init__(self, pixels_per_meter):
        self.pixels_per_meter = pixels_per_meter

    def on_plane(self, xy, z):
        return np.r_[np.asarray(xy) / self.pixels_per_meter, z]

    def project(self, xyz):
        return np.asarray(xyz[:2]) * self.pixels_per_meter


OVERHEAD = Camera(500)
WRIST = Camera(1000)
SOURCE = np.array([.300, .120])
STUDS = np.array([[.268, .120], [.300, .120], [.332, .120]])


def detection(camera, studs, *, bbox=None, color="red"):
    pixels = [
        tuple(camera.project(xy)) for xy in studs
    ]
    center = tuple(np.mean(pixels, axis=0))
    if bbox is None:
        low, high = np.min(pixels, axis=0), np.max(pixels, axis=0)
        padding = .016 * camera.pixels_per_meter
        bbox = (*np.floor(low - padding), *np.ceil(high + padding))
    return BlockDetection(center, color, .99, bbox, tuple(pixels), 1000)


def refine(wrist, *, neighbors=(), overhead_block=None):
    return refine_grasp(
        WRIST, overhead_block or detection(OVERHEAD, STUDS), SOURCE, wrist,
        overhead=OVERHEAD, frame_shape=(512, 512), stud_z=.100, body_z=.070,
        neighbors=neighbors,
    )


def test_fuses_corresponding_studs_without_averaging_an_extra_peak():
    match = detection(WRIST, np.vstack((STUDS + [.006, -.002], [.320, .130])))
    result = refine([match])
    assert result.status == "wrist_fused"
    assert result.detection is match
    np.testing.assert_allclose(result.xy, SOURCE + [.003, -.001])


@pytest.mark.parametrize(("studs", "bbox", "reason"), [
    (STUDS[:1], (251, 100, 349, 140), "wrist_pattern_unconfirmed"),
    (STUDS, (0, 100, 349, 140), "clipped_wrist_region"),
])
def test_partial_or_clipped_wrist_keeps_overhead_center(studs, bbox, reason):
    result = refine([detection(WRIST, studs, bbox=bbox)])
    assert (result.status, result.reason) == ("overhead_fallback", reason)
    np.testing.assert_array_equal(result.xy, SOURCE)


def test_two_regions_covering_the_reprojected_center_prevent_pick():
    left = detection(WRIST, STUDS[:1], bbox=(248, 100, 310, 140))
    right = detection(WRIST, STUDS[2:], bbox=(291, 100, 352, 140))
    result = refine([left, right])
    assert result.status == "wrist_target_ambiguous"
    assert result.xy is None


def test_wrist_region_that_also_covers_another_source_prevents_pick():
    match = detection(WRIST, STUDS, bbox=(250, 100, 405, 140))
    other_xy = [.397, .120]
    other = detection(OVERHEAD, [other_xy])
    result = refine([match], neighbors=[(other, other_xy)])
    assert result.status == "wrist_target_ambiguous"
    assert result.xy is None


def test_overhead_region_with_an_incompatible_pattern_prevents_pick():
    merged = detection(OVERHEAD, np.vstack((STUDS, [.383, .120])))
    result = refine([detection(WRIST, STUDS)], overhead_block=merged)
    assert result.status == "wrist_target_ambiguous"
    assert result.xy is None
