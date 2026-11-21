"""Track a block's stud pattern from calibrated overhead and wrist RGB detections."""

from dataclasses import dataclass
from itertools import combinations, permutations, product

import cv2
import numpy as np

from perception.instances import PLASTIC_COLORS, plastic_colors
from perception.pickup import BlockDetection
from tools.dino_pick_camera import CameraCalibration
from tools.dino_pick_refine import _coherent_studs, _region_fits_footprint

_MIN_PARALLAX = np.sin(np.deg2rad(5))
_MAX_REPROJECTION_PX = 6.
_AMBIGUITY_MARGIN_PX = 1.5
_MAX_STUD_HEIGHT_SPREAD_M = .008
_MAX_STUD_PEAKS = 8
_SAME_VIEW_DISTANCE_M = .001
_MAX_REFRESH_DISTANCE_M = .045
_MIN_MOTION_M = .005
_MIN_MOTION_PX = 3.


class ImageAnchor:
    """Track the translation of a selected block in images from one fixed camera."""

    def __init__(self, rgb, detection, anchor_xy):
        self._shape = rgb.shape
        self._anchor = np.asarray(anchor_xy, dtype=float).copy()
        self._color = detection.color
        self._template = None
        self.confidence = 0.
        height, width = rgb.shape[:2]
        x0, y0, x1, y1 = map(int, detection.bbox_xyxy)
        if (self._color not in PLASTIC_COLORS or not (0 < x0 < x1 < width and 0 < y0 < y1 < height)
                or self._anchor.shape != (2,) or not np.isfinite(self._anchor).all()):
            return

        foreground = self._foreground(rgb[y0:y1, x0:x1])
        count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground)
        if count <= 1:
            return
        largest = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()
        if stats[largest, cv2.CC_STAT_AREA] < 25:
            return
        ys, xs = np.nonzero(labels == largest)
        # Keep a narrow background ring to distinguish the silhouette from a
        # larger patch of the same color. Score RGB only on the block itself.
        left, top = max(0, x0 + xs.min() - 3), max(0, y0 + ys.min() - 3)
        right, bottom = min(width, x0 + xs.max() + 4), min(height, y0 + ys.max() + 4)
        self._origin = np.array([left, top])
        self._template = rgb[top:bottom, left:right].astype(np.float32) / 255
        self._mask = np.zeros(self._template.shape[:2], dtype=np.uint8)
        self._mask[y0 + ys - top, x0 + xs - left] = 1
        background = self._foreground(rgb[top:bottom, left:right]) == 0
        self._ring = cv2.dilate(self._mask, np.ones((5, 5), np.uint8)) * background

    def _foreground(self, rgb):
        colors, saturated = plastic_colors(rgb)
        return ((colors == PLASTIC_COLORS.index(self._color)) & saturated).astype(np.uint8)

    def locate(self, rgb):
        """Return the translated anchor in pixels, or None when the match is uncertain."""
        self.confidence = 0.
        if self._template is None or rgb.shape != self._shape or rgb.dtype != np.uint8:
            return None
        height, width = rgb.shape[:2]
        template_h, template_w = self._mask.shape
        radius = max(8, round(72 * min(height, width) / 512))
        left, top = np.maximum(0, self._origin - radius)
        right, bottom = np.minimum([width, height], self._origin + [template_w, template_h] + radius)
        search = rgb[top:bottom, left:right]
        colors = self._foreground(search).astype(np.float32)
        foreground = self._mask.astype(np.float32)
        visible = cv2.matchTemplate(colors, foreground, cv2.TM_CCORR) / foreground.sum()
        ring = self._ring.astype(np.float32)
        spill = cv2.matchTemplate(colors, ring, cv2.TM_CCORR) / max(1., ring.sum())
        error = cv2.matchTemplate(search.astype(np.float32) / 255, self._template,
                                  cv2.TM_SQDIFF, mask=self._mask)
        rmse = np.sqrt(np.maximum(0., error) / (3 * foreground.sum()))
        score = np.clip(visible, 0., 1.) - .45 * np.clip(spill, 0., 1.) - rmse
        y, x = np.unravel_index(score.argmax(), score.shape)
        best = score[y, x]
        if visible[y, x] < .76 or best < .60:
            return None
        separation = max(5, round(.30 * min(template_w, template_h)))
        score[max(0, y - separation):y + separation + 1,
              max(0, x - separation):x + separation + 1] = -np.inf
        if score.max() >= best - .08:
            return None
        self.confidence = float(best)
        return self._anchor + np.array([left + x, top + y]) - self._origin


@dataclass(frozen=True)
class TrackEstimate:
    """A verified wrist match and its estimate in world meters, when available."""

    xyz: np.ndarray | None
    studs_xyz: np.ndarray | None
    detection: BlockDetection | None
    status: str
    reason: str
    views: int
    reprojection_error_px: float | None


@dataclass(frozen=True)
class _Observation:
    camera: CameraCalibration
    pixels: np.ndarray

    @property
    def rays(self):
        x, y = ((self.pixels - self.camera.center) / self.camera.focal).T
        rays = (self.camera.forward + x[:, None] * self.camera.right
                - y[:, None] * self.camera.up)
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def _pixels(detection):
    return np.asarray(detection.studs_xy, dtype=float).reshape(-1, 2)


def _triangulate(observations, *, complete_pattern=True):
    """Intersect corresponding world rays by minimizing squared ray distances."""
    rays = np.array([view.rays for view in observations])
    parallax = np.linalg.norm(np.cross(rays[0], rays[1:]), axis=-1).max(axis=0)
    if np.any(parallax < _MIN_PARALLAX):
        return None
    origins = np.array([view.camera.position for view in observations])
    projectors = np.eye(3) - rays[..., :, None] * rays[..., None, :]
    rhs = np.einsum("vnij,vj->ni", projectors, origins)
    normal = projectors.sum(axis=0)
    points = np.linalg.solve(normal, rhs[..., None])[..., 0]

    # Fit one height for the horizontal pattern, then measure reprojection from that fit.
    if np.ptp(points[:, 2]) > _MAX_STUD_HEIGHT_SPREAD_M:
        return None
    # Eliminate the independent XY coordinates from the shared height equation.
    coupling = normal[:, :2, 2]
    horizontal = np.linalg.solve(normal[:, :2, :2], np.stack((rhs[:, :2], coupling), axis=-1))
    xy_at_zero, xy_slope = horizontal[..., 0], horizontal[..., 1]
    z = (rhs[:, 2] - (coupling * xy_at_zero).sum(axis=1)).sum() / (
        normal[:, 2, 2] - (coupling * xy_slope).sum(axis=1)
    ).sum()
    xy = xy_at_zero - xy_slope * z
    points = np.column_stack((xy, np.full(len(xy), z)))
    errors = []
    for view in observations:
        try:
            predicted = np.array([view.camera.project(point) for point in points])
        except ValueError:
            return None
        errors.extend(np.linalg.norm(predicted - view.pixels, axis=1))
    errors = np.asarray(errors)
    if errors.max() > _MAX_REPROJECTION_PX or (complete_pattern and not _coherent_studs(points[:, :2])):
        return None
    return points, np.sqrt(np.mean(errors ** 2))


def _clipped(detection, frame_shape):
    if frame_shape is None:
        return False
    height, width = frame_shape[:2]
    x0, y0, x1, y1 = detection.bbox_xyxy
    return x0 <= 0 or y0 <= 0 or x1 >= width or y1 >= height


def _epipolar_choices(reference, camera, pixels):
    """Find wrist peaks near each overhead ray's epipolar line."""
    normals = np.cross(reference.rays, reference.camera.position - camera.position)
    x = normals @ camera.right
    y = -normals @ camera.up
    z = normals @ camera.forward
    offsets = (pixels - camera.center) @ np.stack((x, y)) + camera.focal * z
    bound = 2 * _MAX_REPROJECTION_PX * np.hypot(x, y)
    return tuple(np.flatnonzero(choices) for choices in (np.abs(offsets) <= bound).T)


def _extends_pattern(anchor, subset, points):
    """Check whether a discarded overhead peak belongs to the reconstructed grid."""
    if len(points) == 4:
        return False
    for index in set(range(len(anchor.pixels))) - set(subset):
        try:
            extra = anchor.camera.on_plane(anchor.pixels[index], points[0, 2])
        except ValueError:
            continue
        if _coherent_studs(np.vstack((points[:, :2], extra[:2]))):
            return True
    return False


class BlockTrack:
    """Track one stationary block without a prior on its support height."""

    def __init__(self, overhead, block, *, neighbors=()):
        self._block = block
        self._neighbors = tuple(neighbor for neighbor in neighbors
                                if neighbor is not block and neighbor.color == block.color)
        self._views = [_Observation(overhead, _pixels(block))]
        self._points = None

    def _result(self, status, reason, *, match=None, points=None, error=None):
        return TrackEstimate(
            None if points is None else points.mean(axis=0),
            None if points is None else points.copy(),
            match, status, reason, len(self._views), error,
        )

    def _neighbor_matches(self, wrist, detection, max_error):
        wrist_pixels = _pixels(detection)
        overhead = self._views[0].camera
        for neighbor in self._neighbors:
            pixels = _pixels(neighbor)
            count = len(pixels)
            if not 1 <= count <= min(4, len(wrist_pixels)) or not np.isfinite(pixels).all():
                continue
            reference = _Observation(overhead, pixels)
            for order in permutations(range(len(wrist_pixels)), count):
                fit = _triangulate([reference, _Observation(wrist, wrist_pixels[list(order)])])
                if fit is None:
                    continue
                points, error = fit
                xy, z = points[:, :2], points[0, 2]
                if (error <= max_error
                        and _region_fits_footprint(overhead, neighbor, xy, z)
                        and _region_fits_footprint(wrist, detection, xy, z)):
                    return True
        return False

    def _candidates(self, camera, detections, same_view, count):
        anchor = self._views[0]
        candidates = []
        weak_geometry = False
        for detection in detections:
            pixels = _pixels(detection)
            if len(pixels) < count:
                continue
            choices = _epipolar_choices(anchor, camera, pixels)
            if self._points is not None:
                try:
                    expected = np.array([camera.project(point) for point in self._points])
                except ValueError:
                    continue
                distances = np.linalg.norm(expected[:, None] - pixels[None, :], axis=2)
                choices = tuple(indices[distances[i, indices] <= 2 * _MAX_REPROJECTION_PX]
                                for i, indices in enumerate(choices))
            for subset in combinations(range(len(anchor.pixels)), count):
                reference = _Observation(anchor.camera, anchor.pixels[list(subset)])
                for order in product(*(choices[i] for i in subset)):
                    if len(set(order)) != count:
                        continue
                    view = _Observation(camera, pixels[list(order)])
                    views = [reference, *self._views[1:], view]
                    if same_view not in (None, 0):
                        views.pop(same_view)
                    parallax = np.linalg.norm(np.cross(reference.rays, view.rays), axis=1)
                    weak_geometry |= bool(np.any(parallax < _MIN_PARALLAX))
                    fit = _triangulate(views)
                    if fit is None:
                        continue
                    points, error = fit
                    xy, z = points[:, :2], points[0, 2]
                    if (not _extends_pattern(anchor, subset, points)
                            and _region_fits_footprint(reference.camera, self._block, xy, z)
                            and _region_fits_footprint(camera, detection, xy, z)):
                        candidates.append((error, points, detection, views))
        return candidates, weak_geometry

    def update(self, camera, detections, *, frame_shape=None):
        """Check a wrist frame and fuse its matched stud pixels into the track.

        The first accepted wrist view selects the largest verified pattern from
        the overhead peaks. Each later view must match every selected stud. A
        rejected frame leaves the previous observations available for a later
        update, but supplies no position for the current frame.
        """
        reference = self._views[0]
        count = len(reference.pixels)
        if (self._block.color is None or not 1 <= count <= _MAX_STUD_PEAKS
                or not np.isfinite(reference.pixels).all()):
            return self._result("target_lost", "overhead_stud_pattern_unavailable")

        same_view = next((i for i, view in enumerate(self._views)
                          if np.linalg.norm(camera.position - view.camera.position)
                          < _SAME_VIEW_DISTANCE_M), None)
        if same_view == 0 and self._points is None:
            return self._result("insufficient_parallax", "wrist_view_too_close_to_overhead")
        detections = [d for d in detections if d.color == self._block.color
                      and not _clipped(d, frame_shape)
                      and 1 <= len(d.studs_xy) <= _MAX_STUD_PEAKS
                      and np.isfinite(_pixels(d)).all()]
        weak_geometry = False
        candidates = []
        # Keep a full pattern when possible. Allow one spurious overhead peak
        # when both views support a smaller, complete block footprint.
        minimum = max(1, count - 1) if self._points is None else count
        for size in range(min(4, count), minimum - 1, -1):
            candidates, weak = self._candidates(camera, detections, same_view, size)
            weak_geometry |= weak
            if candidates:
                break

        if not candidates:
            if weak_geometry and self._points is None:
                return self._result("insufficient_parallax", "stud_rays_have_small_crossing_angle")
            return self._result("target_lost", "no_consistent_wrist_stud_pattern")
        candidates.sort(key=lambda candidate: candidate[0])
        error, points, match, views = candidates[0]
        if any(other_error <= error + _AMBIGUITY_MARGIN_PX
               for other_error, _, _, _ in candidates[1:]):
            return self._result("target_ambiguous", "competing_wrist_stud_matches")
        if self._neighbor_matches(camera, match, error + _AMBIGUITY_MARGIN_PX):
            return self._result("target_ambiguous", "wrist_region_matches_overhead_neighbor")

        if same_view != 0:
            self._views = views
            self._points = points
        return self._result("tracked", "matched_stud_pattern", match=match, points=points, error=error)

    def _refresh_candidate(self, camera, detection, previous_center):
        pixel = np.asarray([detection.xy], dtype=float)
        if detection.color != self._block.color or not np.isfinite(pixel).all():
            return False
        offset = previous_center - camera.position
        ray, = _Observation(camera, pixel).rays
        return (offset @ ray > 0
                and np.linalg.norm(np.cross(offset, ray)) <= _MAX_REFRESH_DISTANCE_M)

    def _previous_neighbor_matches(self, xyz):
        overhead = self._views[0].camera
        try:
            pixel = overhead.project(xyz)
            previous = overhead.project(self._points.mean(axis=0))
        except ValueError:
            return True
        distance = np.linalg.norm(pixel - previous)
        return any(np.linalg.norm(pixel - neighbor.xy) <= distance + _AMBIGUITY_MARGIN_PX
                   for neighbor in self._neighbors)

    def refresh(self, overhead_camera, new_overhead_detections, wrist_camera,
                wrist_detections, *, frame_shape=None):
        """Verify the established target in a synchronized overhead and wrist pair.

        A moved target must retain its color and a compatible stud footprint near
        the previous position. Its estimate comes entirely from the new pair.
        A static match can contribute to the previous multiview estimate. Failed
        refreshes leave the tracker state as it was and return no grasp position.
        """
        if self._points is None:
            return self._result("target_lost", "refresh_requires_established_track")
        previous_center = self._points.mean(axis=0)
        overhead_blocks = tuple(new_overhead_detections)
        wrist_blocks = tuple(wrist_detections)
        candidates = []
        ambiguous = False
        for block in overhead_blocks:
            if not self._refresh_candidate(overhead_camera, block, previous_center):
                continue
            fresh = BlockTrack(overhead_camera, block, neighbors=overhead_blocks)
            estimate = fresh.update(wrist_camera, wrist_blocks, frame_shape=frame_shape)
            if estimate.status == "target_ambiguous":
                ambiguous = True
            if estimate.xyz is None or len(estimate.studs_xyz) != len(self._points):
                continue
            if np.linalg.norm(estimate.xyz - previous_center) > _MAX_REFRESH_DISTANCE_M:
                continue
            if self._previous_neighbor_matches(estimate.xyz):
                ambiguous = True
                continue
            candidates.append((fresh, estimate))

        if ambiguous or len(candidates) > 1:
            return self._result("target_ambiguous", "fresh_pair_has_competing_targets")
        if not candidates:
            return self._result("target_lost", "fresh_pair_does_not_confirm_target")
        fresh, estimate = candidates[0]
        count = len(self._points)
        # Stud indices can change between detector calls. Assign one old point
        # per new point before deciding whether the complete pattern shifted.
        best = min(permutations(range(count)), key=lambda order: np.linalg.norm(
            self._points[list(order)] - estimate.studs_xyz
        ))
        previous = self._points[list(best)]
        displacement = np.linalg.norm(previous - estimate.studs_xyz) / np.sqrt(count)
        if displacement >= _MIN_MOTION_M:
            try:
                image_shift = max(np.linalg.norm(np.array([view.camera.project(point)
                                                           for point in previous]) - view.pixels)
                                  / np.sqrt(count) for view in fresh._views)
            except ValueError:
                return self._result("target_lost", "previous_pattern_outside_fresh_view")
            if image_shift < _MIN_MOTION_PX:
                return self._result("target_lost", "fresh_pair_motion_unconfirmed")
            self._block = fresh._block
            self._neighbors = fresh._neighbors
            self._views = fresh._views
            self._points = fresh._points
            return self._result("tracked", "moved_target_reacquired", match=estimate.detection,
                                points=estimate.studs_xyz, error=estimate.reprojection_error_px)
        return self.update(wrist_camera, [estimate.detection], frame_shape=frame_shape)
