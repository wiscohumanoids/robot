"""Use an RGB frame and its camera calibration to locate points on a plane."""

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    position: np.ndarray
    right: np.ndarray
    up: np.ndarray
    forward: np.ndarray
    focal: float
    center: np.ndarray
    k1: float = 0.0

    def project(self, xyz):
        """Project a world point in meters to zero-based pixel-center XY."""
        offset = np.asarray(xyz, dtype=float) - self.position
        distance = offset @ self.forward
        if distance <= 0:
            raise ValueError("World point must be in front of the camera")
        normalized = np.array([offset @ self.right, -offset @ self.up]) / distance
        return self.center + self.focal * normalized * (1 + self.k1 * (normalized @ normalized))

    def on_plane(self, xy, z):
        """Intersect a pixel ray with the horizontal world plane at z meters."""
        direction = self.ray(xy)
        if abs(direction[2]) < 1e-10:
            raise ValueError("Pixel ray is parallel to the plane")
        distance = (z - self.position[2]) / direction[2]
        if distance <= 0:
            raise ValueError("Plane intersection must be in front of the camera")
        return self.position + distance * direction

    def ray(self, xy):
        """Return a world direction through a zero-based pixel-center XY."""
        xy = np.asarray(xy, dtype=float)
        if self.k1:
            import cv2

            x, y = cv2.undistortPoints(xy.reshape(1, 1, 2), self.matrix,
                                       np.array([self.k1, 0., 0., 0.]))[0, 0]
        else:
            x, y = (xy - self.center) / self.focal
        return self.forward + x * self.right - y * self.up

    @property
    def matrix(self):
        cx, cy = self.center
        return np.array([[self.focal, 0., cx], [0., self.focal, cy], [0., 0., 1.]])


def prepare_rgb(sim, renderer, name):
    """Select a camera and exclude the invisible grip-site markers."""
    renderer.disable_depth_rendering()
    renderer.disable_segmentation_rendering()
    visibility = mujoco.MjvOption()
    visibility.sitegroup[:] = 0  # Grip-site and fingertip markers are invisible to a physical camera.
    renderer.update_scene(sim.data, camera=name, scene_option=visibility)


def read_camera(sim, renderer, name):
    """Return the named perspective camera's uint8 RGB frame and calibration."""
    prepare_rgb(sim, renderer, name)
    # MuJoCo's mono renderer uses the midpoint of the stereo eye cameras.
    view = mujoco.mjv_averageCamera(*renderer.scene.camera)
    if view.orthographic:
        raise ValueError(f"Camera {name!r} must use perspective projection")
    pixels_per_meter = renderer.height / (view.frustum_top - view.frustum_bottom)
    center = np.array([
        (renderer.width - 1) / 2 - view.frustum_center * pixels_per_meter,
        view.frustum_top * pixels_per_meter - .5,
    ])
    forward = np.array(view.forward, dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, view.up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    calibration = CameraCalibration(
        position=np.array(view.pos, dtype=float),
        right=right,
        up=up,
        forward=forward,
        focal=pixels_per_meter * view.frustum_near,
        center=center,
    )
    return renderer.render(), calibration
