"""Refine a tabletop grasp by matching the overhead and wrist stud patterns."""

from dataclasses import dataclass
from itertools import product

import numpy as np

from perception.pickup import BlockDetection
from sim.mega_bloks import BODY_HEIGHT_M, PITCH_M, STUD_HEIGHT_M

_ASSOCIATION_RADIUS_M = .035
_ASSOCIATION_MARGIN_M = .012
_MAX_CORRECTION_M = .012
_STUD_TOLERANCE_M = .008

# Distances between every pair of studs for each supported complete footprint.
_STUD_SPACING = {
    1: (),
    2: (1,),
    3: (1, 1, 2),
    4: (1, 1, 1, 1, np.sqrt(2), np.sqrt(2)),
}
_FOOTPRINT = {1: (1, 1), 2: (2, 1), 3: (3, 1), 4: (2, 2)}


@dataclass(frozen=True)
class GraspRefinement:
    """The commanded XY in meters, wrist association, and decision for the trial log."""

    xy: np.ndarray | None
    detection: BlockDetection | None
    status: str
    reason: str


def _on_plane(camera, pixels, z):
    points = []
    for pixel in pixels:
        try:
            point = camera.on_plane(pixel, z)[:2]
        except ValueError:
            continue
        if np.isfinite(point).all():
            points.append(point)
    return np.asarray(points).reshape(-1, 2)


def _stud_spacing(points):
    i, j = np.triu_indices(len(points), k=1)
    return np.sort(np.linalg.norm(points[i] - points[j], axis=1))


def _four_stud_row(points):
    return len(points) == 4 and np.allclose(
        _stud_spacing(points), np.array([1, 1, 1, 2, 2, 3]) * PITCH_M,
        rtol=0, atol=_STUD_TOLERANCE_M,
    )


def _coherent_studs(points):
    count = len(points)
    return _four_stud_row(points) or count in _STUD_SPACING and np.allclose(
        _stud_spacing(points), np.asarray(_STUD_SPACING[count]) * PITCH_M,
        rtol=0, atol=_STUD_TOLERANCE_M,
    )


def _region_fits_footprint(camera, block, points, stud_z):
    """Compare the region with a 3D block envelope in the overhead frame."""
    nx, ny = (4, 1) if _four_stud_row(points) else _FOOTPRINT[len(points)]
    offsets = points[1:] - points[0]
    if not len(offsets):
        angles = np.arange(4) * np.pi / 8
    else:
        # A square uses an edge. A strip uses its longest stud baseline.
        lengths = np.linalg.norm(offsets, axis=1)
        x, y = offsets[lengths.argmin() if nx == ny else lengths.argmax()]
        angles = [np.arctan2(y, x)]
    center = points.mean(axis=0)
    height = stud_z - STUD_HEIGHT_M / 2
    corners = np.asarray(list(product(
        (-nx * PITCH_M / 2, nx * PITCH_M / 2),
        (-ny * PITCH_M / 2, ny * PITCH_M / 2),
        (height - BODY_HEIGHT_M, height),
    )))
    x0, y0, x1, y1 = block.bbox_xyxy
    for angle in angles:
        c, s = np.cos(angle), np.sin(angle)
        xy = center + corners[:, :2] @ np.array([[c, s], [-s, c]])
        try:
            pixels = [camera.project(np.r_[point, z]) for point, z in zip(xy, corners[:, 2])]
        except ValueError:
            continue
        # Semantic boundaries vary with lighting and can include stud pixels.
        if np.all(np.asarray([x1 - x0, y1 - y0]) <= 1.35 * np.ptp(pixels, axis=0) + 4):
            return True
    return False


def overhead_pattern_ok(camera, block, stud_z):
    """Reject regions whose overhead stud span suggests merged blocks."""
    points = _on_plane(camera, block.studs_xy, stud_z)
    return len(points) == len(block.studs_xy) and (
        _four_stud_row(points) or np.all(_stud_spacing(points) <= 2 * PITCH_M + 2 * _STUD_TOLERANCE_M)
    )


def _inside_region(pixel, detection):
    x, y = pixel
    x0, y0, x1, y1 = detection.bbox_xyxy
    return x0 <= x < x1 and y0 <= y < y1


def _project_centers(camera, xy, stud_z, body_z):
    pixels = []
    for z in (stud_z, body_z):
        try:
            pixels.append(camera.project(np.r_[xy, z]))
        except ValueError:
            continue
    return pixels


def _align_studs(overhead, wrist):
    """Fit a small tabletop translation with one wrist match per overhead stud."""
    if len(wrist) < len(overhead):
        return None

    # Each possible correspondence proposes a translation. Score its nearest
    # neighbors and reject a fit if any vertex is missing or assigned twice.
    choices = []
    for offset in (wrist[:, None] - overhead).reshape(-1, 2):
        if np.linalg.norm(offset) > _MAX_CORRECTION_M + _STUD_TOLERANCE_M:
            continue
        distances = np.linalg.norm(wrist[None, :] - (overhead + offset)[:, None], axis=-1)
        indices = distances.argmin(axis=1)
        if len(set(indices)) != len(overhead):
            continue
        shifts = wrist[indices] - overhead
        translation = shifts.mean(axis=0)
        shift_size = np.linalg.norm(translation)
        residuals = np.linalg.norm(shifts - translation, axis=1)
        if shift_size <= _MAX_CORRECTION_M and residuals.max() <= _STUD_TOLERANCE_M / 2:
            choices.append((np.mean(residuals ** 2), shift_size, translation))
    return min(choices, key=lambda choice: choice[:2])[2] if choices else None


def refine_grasp(wrist, overhead_block, source_xy, wrist_detections, *,
                 frame_shape, stud_z, body_z, neighbors=(), overhead=None):
    """Update a grasp from RGB detections and camera calibration.

    The frame shape starts with the wrist frame's height and width. Neighbors
    pair each other overhead detection with its estimated tabletop XY in meters.
    A wrist region must be associated with the selected block before the arm can
    descend. Partial or clipped wrist landmarks leave a coherent overhead estimate
    in place. A complete wrist pattern contributes half the matched translation.
    """
    source_xy = np.asarray(source_xy, dtype=float)
    if overhead is None:
        overhead = wrist
    if not overhead_pattern_ok(overhead, overhead_block, stud_z):
        return GraspRefinement(None, None, "wrist_target_ambiguous", "overhead_region_spans_blocks")
    reference = _on_plane(overhead, overhead_block.studs_xy, stud_z)
    overhead_reliable = (_coherent_studs(reference)
                         and _region_fits_footprint(overhead, overhead_block, reference, stud_z))

    expected = _project_centers(wrist, source_xy, stud_z, body_z)
    candidates = []
    for detection in wrist_detections:
        if detection.color != overhead_block.color:
            continue
        z = stud_z if detection.studs_xy else body_z
        positions = _on_plane(wrist, [detection.xy], z)
        if not len(positions):
            continue
        distance = np.linalg.norm(positions[0] - source_xy)
        covers_target = any(_inside_region(pixel, detection) for pixel in expected)
        if distance <= _ASSOCIATION_RADIUS_M or covers_target:
            candidates.append((distance, covers_target, detection, positions[0]))

    if not candidates:
        return GraspRefinement(None, None, "wrist_target_lost", "no_nearby_wrist_region")
    candidates.sort(key=lambda entry: (not entry[1], entry[0]))
    distance, covers_target, match, wrist_center = candidates[0]
    for other_distance, other_covers, _, _ in candidates[1:]:
        if (other_covers and covers_target
                or abs(other_distance - distance) < _ASSOCIATION_MARGIN_M):
            return GraspRefinement(None, None, "wrist_target_ambiguous", "competing_wrist_regions")
    for neighbor, xy in neighbors:
        if neighbor.color != overhead_block.color:
            continue
        neighbor_pixels = _project_centers(wrist, xy, stud_z, body_z)
        if (any(_inside_region(pixel, match) for pixel in neighbor_pixels)
                or np.linalg.norm(wrist_center - xy) <= distance + _ASSOCIATION_MARGIN_M):
            return GraspRefinement(None, None, "wrist_target_ambiguous", "competing_overhead_target")

    height, width = frame_shape[:2]
    x0, y0, x1, y1 = match.bbox_xyxy
    if x0 <= 0 or y0 <= 0 or x1 >= width or y1 >= height:
        if not covers_target or not overhead_reliable:
            return GraspRefinement(None, match, "wrist_target_lost", "clipped_wrist_region_off_target")
        return GraspRefinement(source_xy, match, "overhead_fallback", "clipped_wrist_region")

    observed = _on_plane(wrist, match.studs_xy, stud_z)
    if overhead_reliable:
        correction = _align_studs(reference, observed)
        if correction is not None:
            return GraspRefinement(source_xy + .5 * correction, match, "wrist_fused", "matched_stud_pattern")
    elif covers_target and distance <= _MAX_CORRECTION_M and _coherent_studs(observed):
        return GraspRefinement(source_xy + .5 * (wrist_center - source_xy), match,
                               "wrist_fused", "wrist_pattern_confirms_source")
    if covers_target and overhead_reliable:
        return GraspRefinement(source_xy, match, "overhead_fallback", "wrist_pattern_unconfirmed")
    return GraspRefinement(None, match, "wrist_target_lost", "grasp_center_unconfirmed")
