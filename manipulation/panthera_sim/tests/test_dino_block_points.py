"""Verify that matched stud observations preserve their identity across cameras."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from tools.dino_block_points import flat_block_points
from tools.dino_pick_camera import CameraCalibration


def _camera(x):
    return CameraCalibration(np.array([x, 0., .50]), np.array([1., 0., 0.]),
                             np.array([0., 1., 0.]), np.array([0., 0., -1.]),
                             420., np.array([255.5, 255.5]))


def _detection(points):
    x0, y0 = points.min(axis=0) - 5
    x1, y1 = points.max(axis=0) + 5
    return SimpleNamespace(color='blue', bbox_xyxy=(x0, y0, x1, y1), studs_xy=points)


def test_resting_studs_use_observed_pixels_and_independent_world_points():
    overhead, wrist = _camera(.25), replace(_camera(.31), k1=.11)
    points = np.array([[.28, .04, .1], [.312, .04, .1]])
    source = _detection(np.array([overhead.project(p) for p in points]))
    measured = np.array([wrist.project(p) for p in points]) + np.array([4., -7.])
    target = _detection(measured[::-1])
    world, pixels = flat_block_points(overhead, [source], wrist, [target],
                                      frame_shape=(512, 512, 3), stud_z=.1)
    np.testing.assert_allclose(world, points)
    np.testing.assert_allclose(pixels, measured)


def test_ambiguous_color_and_partial_patterns_yield_empty_pairs():
    cam = _camera(.25)
    detection = _detection(np.array([[251., 242.], [285., 242.]]))
    partial = _detection(np.array([[251., 242.]]))
    for top, wrist in [([detection, detection], [detection]),
                       ([detection], [detection, detection]), ([detection], [partial])]:
        world, pixels = flat_block_points(cam, top, cam, wrist,
                                          frame_shape=(512, 512), stud_z=.1)
        assert world.shape == (0, 3)
        assert pixels.shape == (0, 2)


class _Detections:
    def detect(self, rgb):
        return [self.block]


def test_unknown_platform_height_comes_from_stud_spacing():
    from tools.dino_block_points import BlockSamples

    cam = _camera(.25)
    points = np.array([[.28, .04, .18], [.312, .04, .18]])
    measured = np.array([cam.project(p) for p in points])
    detector = _Detections()
    detector.block = _detection(measured)
    samples = BlockSamples(detector, flat_bench=False)
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    world, pixels, fit = samples.collect("left_wrist", image, np.zeros(3), np.eye(3),
                                         cam, image, cam, "survey_1")
    assert fit is None
    np.testing.assert_allclose(world, points)
    np.testing.assert_allclose(pixels, measured)


def test_moved_block_does_not_enter_stationary_bundle(monkeypatch):
    from tools import dino_block_points

    cam = _camera(.25)
    detector = _Detections()
    samples = dino_block_points.BlockSamples(detector, flat_bench=True)
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    solves = []

    def capture_fit(top, blocks, frames, mount, *, overhead_fixed):
        solves.append([frame.grip_position[0] for frame in frames])

    monkeypatch.setattr(dino_block_points, "fit_block_calibration", capture_fit)
    for idx in range(6):
        pixel_shift = 20. if idx > 1 else 0.
        detector.block = _detection(np.array([[250. + pixel_shift, 245.], [283. + pixel_shift, 246.]]))
        samples.collect("left_wrist", image, [idx * .01, 0, 0], np.eye(3),
                        cam, image, cam, "survey_1")
    np.testing.assert_allclose(solves, [[.02, .03, .04, .05]])
