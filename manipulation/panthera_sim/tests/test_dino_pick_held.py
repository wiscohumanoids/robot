"""Exercise held-block estimation with calibrated projections and grip feedback."""

from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import pytest
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perception.pickup import BlockDetection
from tools.dino_pick_camera import CameraCalibration
from tools.dino_pick_held import HeldBlockTracker, STUD_TO_BOTTOM_M, _model_vertices, correct_placement


CAMERA = CameraCalibration(
    position=np.array([.2, .02, .7]),
    right=np.array([1., 0., 0.]),
    up=np.array([0., 1., 0.]),
    forward=np.array([0., 0., -1.]),
    focal=450.,
    center=np.array([319.5, 239.5]),
)
GRIP_ROTATION = np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])
GRIP_POSITIONS = (
    np.array([.23, -.02, .14]),
    np.array([.30, -.02, .14]),
    np.array([.30, .06, .16]),
)
EXPECTED = np.array([.040, -.010, .004])
STUD_CENTER = EXPECTED + [.013, .008, -.006]


def detection(center, *, spacing=.032, color="red", axis=(0., 1., 0.)):
    center = np.asarray(center, dtype=float)
    pixel = CAMERA.project(center)
    half_span = np.asarray(axis) * spacing / 2
    studs = tuple(tuple(CAMERA.project(center + shift))
                  for shift in (-half_span, half_span))
    x, y = map(int, pixel)
    return BlockDetection(tuple(pixel), color, .99, (x - 21, y - 28, x + 21, y + 28), studs, 500)


def observe(tracker, grip_xyz, center, *, spacing=.032, extra=()):
    return tracker.observe(CAMERA, [detection(center, spacing=spacing), *extra],
                           grip_xyz, GRIP_ROTATION, frame_shape=(480, 640, 3))


def test_fits_moving_block_in_gripper_frame_and_corrects_placement():
    tracker = HeldBlockTracker("red", EXPECTED, 2)
    tabletop_block = detection([.34, .16, .10])
    for grip_xyz, angle in zip(GRIP_POSITIONS, (0., .2, -.15)):
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]) @ GRIP_ROTATION
        held = detection(grip_xyz + rotation @ STUD_CENTER, axis=rotation[:, 1])
        assert tracker.observe(CAMERA, [held, tabletop_block], grip_xyz, rotation,
                               frame_shape=(480, 640, 3))
    fit = tracker.fit([1., 0., 0.])
    assert fit.status == "held_tracking_applied"
    assert fit.observations == 3
    assert fit.reprojection_error_px < 1e-8
    np.testing.assert_allclose(fit.bottom_offset, STUD_CENTER - [STUD_TO_BOTTOM_M, 0., 0.])

    body_center = np.array([.33, .04, .07])
    nominal_bottom = EXPECTED - [STUD_TO_BOTTOM_M, 0., 0.]
    nominal_grip = body_center - [0., 0., .02] - GRIP_ROTATION @ nominal_bottom
    command = correct_placement(body_center, nominal_grip, GRIP_ROTATION, fit.bottom_offset)
    np.testing.assert_allclose(command, body_center - GRIP_ROTATION @ (STUD_CENTER - EXPECTED))


def test_rejects_repeated_pose_and_dropped_block():
    tracker = HeldBlockTracker("red", EXPECTED, 2)
    grip_xyz = GRIP_POSITIONS[0]
    center = grip_xyz + GRIP_ROTATION @ STUD_CENTER
    for _ in GRIP_POSITIONS:
        assert observe(tracker, grip_xyz, center)
    assert tracker.fit([1., 0., 0.]).reason == "insufficient_held_parallax"

    tracker = HeldBlockTracker("red", EXPECTED, 2)
    for grip_xyz in GRIP_POSITIONS:
        observe(tracker, grip_xyz, center)
    assert tracker.fit([1., 0., 0.]).bottom_offset is None


def test_rejects_slip_and_merged_patterns():
    for discrepancy, spacing, reason in (
            (.025, .032, "held_views_disagree"),
            (0., .082, "held_pattern_inconsistent")):
        tracker = HeldBlockTracker("red", EXPECTED, 2)
        for index, grip_xyz in enumerate(GRIP_POSITIONS):
            center = grip_xyz + GRIP_ROTATION @ STUD_CENTER
            if index == 1:
                center[1] += discrepancy
            assert observe(tracker, grip_xyz, center, spacing=spacing)
        fit = tracker.fit([1., 0., 0.])
        assert fit.reason == reason
        assert fit.bottom_offset is None


def test_rejects_ambiguous_view_and_excessive_correction():
    tracker = HeldBlockTracker("red", EXPECTED, 2)
    grip_xyz = GRIP_POSITIONS[0]
    center = grip_xyz + GRIP_ROTATION @ STUD_CENTER
    assert not observe(tracker, grip_xyz, center, extra=(detection(center + [.02, 0, 0]),))
    assert tracker.fit([1., 0., 0.]).observations == 0
    with pytest.raises(ValueError, match="placement limits"):
        correct_placement([.3, .0, .07], np.zeros(3), np.eye(3), np.zeros(3))


def test_fits_partial_silhouettes_without_stud_landmarks():
    tracker = HeldBlockTracker("blue", EXPECTED, 2, reference_grip_rotation=GRIP_ROTATION)
    geometry = _model_vertices(2, 1)
    angle = .2
    rotation = GRIP_ROTATION.T @ np.array([
        [1., 0., 0.], [0., np.cos(angle), -np.sin(angle)], [0., np.sin(angle), np.cos(angle)],
    ])
    body = STUD_CENTER - [STUD_TO_BOTTOM_M - .020, 0., 0.]
    for grip_xyz in GRIP_POSITIONS:
        corners = (geometry @ rotation.T + body) @ GRIP_ROTATION.T + grip_xyz
        pixels = np.array([CAMERA.project(point) for point in corners])
        polygon = pixels[ConvexHull(pixels).vertices]
        image = Image.new("RGB", (640, 480))
        draw = ImageDraw.Draw(image)
        draw.polygon([tuple(pixel) for pixel in polygon], fill=(25, 80, 230))
        draw.rectangle((polygon[:, 0].min(), polygon[:, 1].min(),
                        polygon[:, 0].min() + 6, polygon[:, 1].min() + 8), fill=(110, 110, 110))
        bbox = tuple(np.r_[np.floor(polygon.min(axis=0)), np.ceil(polygon.max(axis=0))].astype(int))
        match = BlockDetection(tuple(polygon.mean(axis=0)), "blue", .99, bbox, (), 500)
        rgb = np.asarray(image)
        assert tracker.observe(CAMERA, [match], grip_xyz, GRIP_ROTATION,
                               frame_shape=rgb.shape, rgb=rgb)
    result = tracker.fit([1., 0., 0.])
    assert result.reason == "consistent_held_silhouettes"
    np.testing.assert_allclose(result.bottom_offset, body - .020 * rotation[:, 2], atol=.009)
