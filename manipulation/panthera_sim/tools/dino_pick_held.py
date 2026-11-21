"""Measure the held block's support point from RGB views and gripper poses."""

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from perception.instances import PLASTIC_COLORS, plastic_colors
from sim.mega_bloks import BODY_HEIGHT_M, PITCH_M, SHAPES, STUD_HEIGHT_M, STUD_RADIUS_M
from tools.dino_pick_camera import CameraCalibration
from tools.dino_pick_refine import _coherent_studs

STUD_TO_BOTTOM_M = BODY_HEIGHT_M + STUD_HEIGHT_M / 2
_MAX_OBSERVATION_OFFSET_M = .042
_MIN_RAY_SINE = .06
_MAX_REPROJECTION_ERROR_PX = 3.
_MAX_XY_CORRECTION_M = .028
_MAX_Z_CORRECTION_M = .025
_LOOK_ANGLES = np.arange(16) * (np.pi / 8)
_IMAGE_DIRECTIONS = np.column_stack((np.cos(_LOOK_ANGLES), np.sin(_LOOK_ANGLES)))
_UP = np.array([0., 0., 1.])


@dataclass(frozen=True)
class HeldBlockFit:
    """The local block-bottom offset and the acceptance decision."""

    bottom_offset: np.ndarray | None
    status: str
    reason: str
    observations: int
    reprojection_error_px: float | None = None


@dataclass(frozen=True)
class _Observation:
    camera: CameraCalibration
    studs: np.ndarray
    pixel: np.ndarray
    grip_xyz: np.ndarray
    grip_rotation: np.ndarray
    origin: np.ndarray
    direction: np.ndarray


@dataclass(frozen=True)
class _Silhouette:
    origin: np.ndarray
    projection: np.ndarray
    translation: np.ndarray
    focal: float
    center: np.ndarray
    support: np.ndarray
    clipped: np.ndarray


def _model_vertices(nx, ny):
    length, width = (np.array([nx, ny]) * PITCH_M - .0004) / 2
    half_height = BODY_HEIGHT_M / 2
    body = np.array(list(product((-length, length), (-width, width), (-half_height, half_height))))
    rings = np.concatenate([
        np.column_stack((_IMAGE_DIRECTIONS * STUD_RADIUS_M * scale, np.full(16, z + half_height)))
        for z, scale in ((0., 1.), (STUD_HEIGHT_M - .0012, .94), (STUD_HEIGHT_M, .84))
    ])
    centers = np.array(list(product(range(nx), range(ny), (0,))), dtype=float)
    centers -= centers.mean(axis=0)
    return np.vstack((body, (centers[:, None] * PITCH_M + rings).reshape(-1, 3)))


_MODELS = {shape: _model_vertices(*shape) for shape in SHAPES}


def _silhouette(rgb, camera, detections, color, grip_xyz, rotation, expected):
    """Measure directional extrema inside the associated DINO region."""
    expected_world = grip_xyz + rotation @ expected
    try:
        expected_pixel = camera.project(expected_world)
    except ValueError:
        return None
    pixel_scale = camera.focal / ((expected_world - camera.position) @ camera.forward)
    nearby = []
    for detection in detections:
        if detection.color != color or detection.area_pixels < 80:
            continue
        lo, hi = np.asarray(detection.bbox_xyxy).reshape(2, 2)
        closest = np.clip(expected_pixel, lo, hi)
        if np.linalg.norm(closest - expected_pixel) <= _MAX_OBSERVATION_OFFSET_M * pixel_scale:
            nearby.append(detection)
    if not nearby:
        return None
    nearby.sort(key=lambda detection: detection.area_pixels, reverse=True)
    if len(nearby) > 1 and nearby[1].area_pixels >= .45 * nearby[0].area_pixels:
        return None
    x0, y0, x1, y1 = nearby[0].bbox_xyxy
    abs_color, saturated = plastic_colors(rgb[y0:y1, x0:x1])
    y, x = np.nonzero((abs_color == PLASTIC_COLORS.index(color)) & saturated)
    if len(x) < 80:
        return None
    pixels = np.column_stack((x + x0, y + y0))
    projections = pixels @ _IMAGE_DIRECTIONS.T
    support = np.quantile(projections, .997, axis=0)
    extrema = pixels[np.argmax(projections, axis=0)]
    # A direction pointing out of the image gives a lower bound on the true extent.
    limits = np.array(rgb.shape[1::-1]) - 3
    outward = np.where(_IMAGE_DIRECTIONS < -.05, extrema <= 1,
                       (_IMAGE_DIRECTIONS > .05) & (extrema >= limits))
    basis = np.column_stack((camera.right, -camera.up, camera.forward))
    return _Silhouette(rotation.T @ (camera.position - grip_xyz), rotation.T @ basis,
                       (grip_xyz - camera.position) @ basis, camera.focal, camera.center,
                       support, np.any(outward, axis=1))


def _fit_silhouettes(views, expected_stud, reference_rotation):
    """Fit the body center and orientation to partial RGB silhouettes."""
    nominal_body = expected_stud - (BODY_HEIGHT_M + STUD_HEIGHT_M) / 2 * reference_rotation[:, 2]
    projection = np.array([view.projection for view in views])
    translation = np.array([view.translation for view in views])
    focal = np.array([view.focal for view in views])
    image_center = np.array([view.center for view in views])
    observed = np.array([view.support for view in views])
    clipped = np.array([view.clipped for view in views])
    nominal_depth = translation[:, 2] + projection[:, :, 2] @ nominal_body
    pixel_to_mm = nominal_depth * 1000 / focal
    bounds = np.array([3., 3., 3., 1.2, 1.2, 8.])

    def pose(parameters):
        center = nominal_body + parameters[:3] * .01
        attitude = Rotation.from_euler("z", parameters[5]).as_matrix()
        tilt = Rotation.from_rotvec([*parameters[3:5], 0.]).as_matrix()
        return center, reference_rotation @ tilt @ attitude

    def errors(parameters, vertices):
        center, rotation = pose(parameters)
        points = vertices @ rotation.T + center
        projected = points @ projection + translation[:, None]
        projected = image_center[:, None] + (
            focal[:, None, None] * projected[:, :, :2] / projected[:, :, 2:]
        )
        delta = (projected @ _IMAGE_DIRECTIONS.T).max(axis=1) - observed
        # The jaws can hide edges, but they cannot enlarge the block's silhouette.
        return np.where(clipped, np.minimum(delta, 0.), np.where(delta < 0., delta, delta * .45))

    fits = []
    for vertices in _MODELS.values():
        def residuals(parameters):
            measured = errors(parameters, vertices) * pixel_to_mm[:, None]
            return np.r_[measured.ravel(), parameters[:3] * 1.5, parameters[3:5] * 3.]

        options = [
            least_squares(residuals, [0., 0., 0., 0., 0., yaw], bounds=(-bounds, bounds),
                          loss="soft_l1", f_scale=2., max_nfev=90)
            for yaw in np.arange(4) * (np.pi / 4)
        ]
        best = min(options, key=lambda option: option.cost)
        center, orientation = pose(best.x)
        bottom = center - BODY_HEIGHT_M / 2 * orientation[:, 2]
        delta = errors(best.x, vertices)
        fits.append((best.cost, bottom, delta))
    fits.sort(key=lambda fit: fit[0])
    cost, bottom, delta = fits[0]
    reprojection = float(np.sqrt(np.mean(delta ** 2)))
    typical_error = np.median(np.abs(delta * pixel_to_mm[:, None]), axis=1)
    if np.max(typical_error) > 5. or cost / delta.size > 6.:
        return None, "held_silhouettes_disagree", reprojection
    if any(c <= cost * 1.25 and np.linalg.norm(b - bottom) > .008 for c, b, _ in fits[1:]):
        return None, "held_shape_ambiguous", reprojection
    return bottom, "consistent_held_silhouettes", reprojection


class HeldBlockTracker:
    """Associate RGB observations near the gripper and fit a local bottom point."""

    def __init__(self, color, expected_stud_offset, stud_count, *, reference_grip_rotation=None):
        self.color = color
        self.expected = np.asarray(expected_stud_offset, dtype=float)
        self.stud_count = stud_count
        self._views = []
        self._silhouettes = []
        self._reference_rotation = (None if reference_grip_rotation is None
                                    else np.asarray(reference_grip_rotation, dtype=float).T)
        self._silhouette_fit = None

    def observe(self, camera, detections, grip_xyz, grip_rotation, *, frame_shape, rgb=None):
        """Store a complete stud pattern or a partial RGB silhouette and grip pose."""
        height, width = frame_shape[:2]
        grip_xyz = np.array(grip_xyz, dtype=float, copy=True)
        grip_rotation = np.array(grip_rotation, dtype=float, copy=True)
        origin = grip_rotation.T @ (camera.position - grip_xyz)
        new_silhouette = False
        if (rgb is not None and self.color in PLASTIC_COLORS
                and all(np.linalg.norm(origin - view.origin) >= .015 for view in self._silhouettes)):
            view = _silhouette(rgb, camera, detections, self.color, grip_xyz, grip_rotation, self.expected)
            if view is not None:
                if self._reference_rotation is None:
                    self._reference_rotation = grip_rotation.T
                self._silhouettes.append(view)
                self._silhouette_fit = None
                new_silhouette = True
        candidates = []
        for detection in detections:
            if detection.color != self.color or len(detection.studs_xy) != self.stud_count:
                continue
            x0, y0, x1, y1 = detection.bbox_xyxy
            if x0 <= 0 or y0 <= 0 or x1 >= width or y1 >= height:
                continue
            studs = np.asarray(detection.studs_xy, dtype=float)
            if not 1 <= len(studs) <= 4 or not np.isfinite(studs).all():
                continue
            pixel = studs.mean(axis=0)
            direction = grip_rotation.T @ camera.ray(pixel)
            direction /= np.linalg.norm(direction)
            along_ray = (self.expected - origin) @ direction
            distance = np.linalg.norm(self.expected - origin - along_ray * direction)
            if along_ray > 0 and distance <= _MAX_OBSERVATION_OFFSET_M:
                candidates.append(_Observation(camera, studs, pixel, grip_xyz, grip_rotation, origin, direction))
        if len(candidates) != 1:
            return new_silhouette
        self._views.append(candidates[0])
        return True

    def fit(self, up_in_gripper):
        """Estimate the center of the bottom face in the moving gripper frame.

        Silhouettes constrain the body pose even when the jaws obscure its
        studs. The landmark fit uses the supplied upright direction and the
        midpoint of the stud height to infer the bottom from a stud centroid.
        """
        count = len(self._views)
        silhouette_reason = None
        if len(self._silhouettes) >= 3:
            if self._silhouette_fit is None:
                rays = self.expected - np.array([view.origin for view in self._silhouettes])
                rays /= np.linalg.norm(rays, axis=1, keepdims=True)
                if np.max(np.linalg.norm(np.cross(rays[:, None], rays), axis=-1)) >= _MIN_RAY_SINE:
                    self._silhouette_fit = _fit_silhouettes(self._silhouettes, self.expected,
                                                            self._reference_rotation)
                else:
                    self._silhouette_fit = (None, "insufficient_held_parallax", None)
            bottom, silhouette_reason, error = self._silhouette_fit
            if bottom is not None:
                return HeldBlockFit(bottom, "held_tracking_applied", silhouette_reason,
                                    len(self._silhouettes), error)

        def reject(reason, error=None):
            return HeldBlockFit(None, "held_tracking_rejected", reason, count, error)

        if count < 3:
            if silhouette_reason is not None:
                return HeldBlockFit(None, "held_tracking_rejected", silhouette_reason,
                                    len(self._silhouettes), error)
            return reject("insufficient_held_views")
        origins = np.array([view.origin for view in self._views])
        directions = np.array([view.direction for view in self._views])
        if np.max(np.linalg.norm(np.cross(directions[:, None], directions), axis=-1)) < _MIN_RAY_SINE:
            return reject("insufficient_held_parallax")

        # Each projector removes the along-ray component. The solution minimizes
        # the sum of squared perpendicular distances to the moving-frame rays.
        projectors = np.eye(3) - directions[:, :, None] * directions[:, None, :]
        point, _, rank, _ = np.linalg.lstsq(
            projectors.reshape(-1, 3),
            (projectors @ origins[..., None]).reshape(-1),
            rcond=None,
        )
        if rank < 3 or np.linalg.norm(point - self.expected) > _MAX_OBSERVATION_OFFSET_M:
            return reject("implausible_held_offset")

        errors = []
        for view in self._views:
            try:
                projection = view.camera.project(view.grip_xyz + view.grip_rotation @ point)
            except ValueError:
                return reject("held_point_behind_camera")
            errors.append(np.linalg.norm(projection - view.pixel))
        error = float(max(errors))
        if error > _MAX_REPROJECTION_ERROR_PX:
            return reject("held_views_disagree", error)

        up = np.asarray(up_in_gripper, dtype=float)
        up_length = np.linalg.norm(up)
        if up.shape != (3,) or not np.isfinite(up_length) or up_length == 0:
            raise ValueError("Held-block up must be a finite, nonzero three-vector")
        up = up / up_length
        for view in self._views:
            # Intersect each pixel with the fitted stud plane to detect merged
            # or incomplete patterns that happen to share the right centroid.
            world_up = view.grip_rotation @ up
            world_point = view.grip_xyz + view.grip_rotation @ point
            rays = np.array([view.camera.ray(pixel) for pixel in view.studs])
            incidence = rays @ world_up
            if np.any(np.abs(incidence) < .1):
                return reject("held_pattern_edge_on", error)
            distances = ((world_point - view.camera.position) @ world_up) / incidence
            if np.any(distances <= 0) or not _coherent_studs(rays * distances[:, None]):
                return reject("held_pattern_inconsistent", error)
        return HeldBlockFit(point - STUD_TO_BOTTOM_M * up, "held_tracking_applied",
                            "consistent_held_views", count, error)


def correct_placement(body_center_xyz, grip_xyz, grip_rotation, bottom_offset):
    """Shift the nominal body-center command so the measured bottom hits the target."""
    body_center = np.asarray(body_center_xyz, dtype=float)
    target_bottom = body_center - [0., 0., BODY_HEIGHT_M / 2]
    predicted_bottom = grip_xyz + grip_rotation @ bottom_offset
    correction = target_bottom - predicted_bottom
    if (not np.isfinite(correction).all() or np.linalg.norm(correction[:2]) > _MAX_XY_CORRECTION_M
            or abs(correction[2]) > _MAX_Z_CORRECTION_M):
        raise ValueError("Held block correction exceeds the placement limits")
    return body_center + correction
