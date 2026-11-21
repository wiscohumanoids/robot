"""Check camera calibration against independent synthetic DINO detections."""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perception.pickup import BlockDetection
from tools.dino_block_calibration import BlockCalibrationFrame, fit_block_calibration, stud_pattern_world
from tools.dino_pick_camera import CameraCalibration


def _camera(position, axes=np.diag([1., -1., -1.]), k1=0.):
    return CameraCalibration(np.asarray(position), axes[:, 0], -axes[:, 1], axes[:, 2],
                             365., np.array([255.5, 255.5]), k1=k1)


def _detection(camera, points, color):
    pixels = np.array([camera.project(point) for point in points])
    return BlockDetection(tuple(pixels.mean(axis=0)), color, .98, (0, 0, 512, 512),
                          tuple(map(tuple, pixels)), 100)


def _fixture(k1=0.):
    overhead = _camera([.22, 0., .49])
    true_mount = _camera([-.08, .02, .06],
                         Rotation.from_euler("xyz", [4., -7., 9.], degrees=True).as_matrix()
                         @ np.diag([1., -1., -1.]), k1)
    nominal = _camera([-.073, .013, .062],
                      Rotation.from_euler("xyz", [1., -4., 5.], degrees=True).as_matrix()
                      @ np.diag([1., -1., -1.]))
    blocks = [
        ("red", np.array([[.24, -.071, .105], [.272, -.071, .105], [.24, -.039, .105], [.272, -.039, .105]])),
        ("green", np.array([[.17, .042, .17], [.17, .074, .17], [.17, .106, .17]])),
    ]
    overhead_detections = tuple(_detection(overhead, xyz, color) for color, xyz in blocks)
    frames = []
    for position, angles in (
        ([.31, .00, .37], [3, 20, 9]),
        ([.20, .03, .38], [17, -10, -8]),
        ([.31, -.08, .35], [-13, 5, 19]),
        ([.37, .03, .41], [10, -15, 40]),
        ([.26, -.09, .46], [9, 12, 8]),
    ):
        grip = np.asarray(position)
        rotation = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
        axes = rotation @ np.column_stack((true_mount.right, -true_mount.up, true_mount.forward))
        camera = _camera(grip + rotation @ true_mount.position, axes, k1)
        # The order of detected studs can change between exposures.
        detections = tuple(_detection(camera, xyz[::-1], color) for color, xyz in blocks)
        frames.append(BlockCalibrationFrame(grip, rotation, detections))
    return overhead, overhead_detections, frames, nominal, true_mount


def test_fit_recovers_mount_from_shuffled_studs_on_multiple_surfaces():
    overhead, detections, frames, nominal, truth = _fixture()
    result = fit_block_calibration(overhead, detections, frames, nominal, overhead_fixed=True)
    assert result is not None
    assert result.rmse_px < .001
    assert result.views == 5
    assert result.blocks == 2
    np.testing.assert_allclose(result.mount.position, truth.position, atol=1.e-5)
    np.testing.assert_allclose(result.mount.forward, truth.forward, atol=1.e-5)
    np.testing.assert_allclose(result.overhead.position, overhead.position)


def test_fit_recovers_radial_distortion_with_off_axis_points():
    overhead, detections, frames, nominal, truth = _fixture(k1=.11)
    result = fit_block_calibration(overhead, detections, frames, nominal, overhead_fixed=True)
    assert result is not None
    assert result.rmse_px < .001
    np.testing.assert_allclose(result.mount.k1, truth.k1, atol=1.e-4)


def test_fit_rejects_no_motion_and_single_stud_landmarks():
    overhead, detections, frames, nominal, _ = _fixture()
    assert fit_block_calibration(overhead, detections, [frames[0]] * 4, nominal, overhead_fixed=True) is None
    one_stud = [replace(detection, studs_xy=detection.studs_xy[:1]) for detection in detections]
    assert fit_block_calibration(overhead, one_stud, frames, nominal, overhead_fixed=True) is None


def test_joint_fit_uses_grip_motion_to_anchor_overhead_world_pose():
    overhead, detections, frames, nominal, truth = _fixture()
    axes = Rotation.from_euler("xy", [2., -4.], degrees=True).as_matrix() @ np.diag([1., -1., -1.])
    overhead_prior = _camera(overhead.position + [.01, -.01, .015], axes)
    result = fit_block_calibration(overhead_prior, detections, frames, nominal)
    assert result is not None
    assert result.rmse_px < .001
    np.testing.assert_allclose(result.mount.position, truth.position, atol=1.e-5)
    np.testing.assert_allclose(result.overhead.position, overhead.position, atol=1.e-5)
    np.testing.assert_allclose(result.overhead.forward, overhead.forward, atol=1.e-5)


def test_fit_rejects_blocks_that_move_in_the_held_out_view():
    overhead, detections, frames, nominal, _ = _fixture()
    moved = tuple(
        replace(block, studs_xy=tuple((x + 20., y) for x, y in block.studs_xy))
        for block in frames[-1].detections
    )
    frames[-1] = replace(frames[-1], detections=moved)
    assert fit_block_calibration(overhead, detections, frames, nominal, overhead_fixed=True) is None

def test_four_stud_strip_constrains_camera_height():
    overhead = _camera([.22, 0., .49])
    xyz = np.array([.19, -.04, .15]) + np.arange(4)[:, None] * [.032, 0, 0]
    detection = _detection(overhead, xyz[[2, 0, 3, 1]], "blue")
    points, _ = stud_pattern_world(overhead, detection)
    np.testing.assert_allclose(points[points[:, 0].argsort()], xyz)
