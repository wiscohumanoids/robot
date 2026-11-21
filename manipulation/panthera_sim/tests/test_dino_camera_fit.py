"""Recover camera geometry from independently projected, matched 3D surface points."""

from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.dino_camera_fit import fit_camera


def observations(tilt, position, k1):
    # OpenCV looks along +Z with +Y down. These cameras stand above a 3D landmark cloud.
    rotation, _ = cv2.Rodrigues(np.array([np.pi - tilt, 0., 0.]))
    rng = np.random.default_rng(134)
    xy = rng.uniform([-.30, -.25], [.30, .25], size=(300, 2))
    world = np.c_[xy, rng.uniform(-.05, .08, len(xy))]
    matrix = np.array([[350., 0., 255.5], [0., 350., 255.5], [0., 0., 1.]])
    pixels, _ = cv2.projectPoints(world, rotation, -rotation @ position, matrix,
                                  np.array([k1, 0., 0., 0.]))
    pixels = pixels.reshape(-1, 2)
    visible = ((pixels > 1.) & (pixels < 511.)).all(axis=1)
    noise = rng.normal(scale=.7, size=pixels.shape)
    return world[visible], (pixels + noise)[visible]


@pytest.mark.parametrize("tilt,position,k1", [
    (.18, [.035, -.07, .48], .15),
    (.48, [-.025, -.21, .46], -.10),
    (.55, [-.03, -.18, .43], 0.),
])
def test_fit_perturbed_camera_with_false_matches(tilt, position, k1):
    world, pixels = observations(tilt, position, k1)
    pixels[::5] += [25., -12.]
    camera, rms = fit_camera(world, pixels, (512, 512, 3), 350.)
    assert rms < 1.3
    np.testing.assert_allclose(camera.position, position, atol=.002)
    assert abs(camera.k1 - k1) < .02
    recovered = [camera.on_plane(pixel, point[2])
                 for pixel, point in zip(pixels[1::5], world[1::5])]
    errors = np.linalg.norm(recovered - world[1::5], axis=1)
    assert np.quantile(errors, .95) < .004


def test_too_few_or_inconsistent_points_do_not_establish_calibration():
    world, pixels = observations(.4, [.02, -.14, .48], 0.)
    assert fit_camera(world[:8], pixels[:8], (512, 512), 350.) is None
    pixels = np.random.default_rng(34).permutation(pixels)
    assert fit_camera(world, pixels, (512, 512), 350.) is None


def test_collinear_points_do_not_establish_calibration():
    x = np.linspace(0., .2, 30)
    world = np.column_stack([x, x, 0. * x])
    pixels = 500. * world[:, :2] + 180.
    assert fit_camera(world, pixels, (512, 512), 350.) is None


def test_output_uses_the_common_mount_frame():
    world, pixels = observations(.48, [-.025, -.21, .46], -.10)
    angle = .83
    basis, _ = cv2.Rodrigues(np.array([angle, -.5 * angle, angle]))
    origin = np.array([.33, .12, .55])
    parent_points = (world - origin) @ basis
    camera, _ = fit_camera(world, pixels, (512, 512), 350.)
    mount, _ = fit_camera(parent_points, pixels, (512, 512), 350.)
    np.testing.assert_allclose(mount.position, (camera.position - origin) @ basis, atol=1e-5)
    np.testing.assert_allclose(mount.forward, camera.forward @ basis, atol=1e-5)
    assert abs(mount.k1 - camera.k1) < 1e-4
