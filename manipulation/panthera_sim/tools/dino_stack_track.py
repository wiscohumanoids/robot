"""Associate and triangulate resting stud rows across calibrated RGB views."""

from collections import Counter
from copy import copy
from dataclasses import dataclass
from itertools import permutations

import numpy as np

from perception.pickup import BlockDetection
from tools.dino_block_calibration import stud_pattern_world
from tools.dino_pick_camera import CameraCalibration
from tools.dino_pick_refine import _coherent_studs, _four_stud_row, _region_fits_footprint
from tools.dino_pick_track import BlockTrack, TrackEstimate, _Observation, _clipped, _triangulate


@dataclass(frozen=True)
class StackFrame:
    """Detections and calibration from one RGB frame."""

    camera: CameraCalibration
    detections: tuple[BlockDetection, ...]
    shape: tuple[int, ...] | None = None


@dataclass(frozen=True)
class StackEstimate:
    """Current observation or the last accepted geometry for each block."""

    source: TrackEstimate | None = None
    target: TrackEstimate | None = None


def _region_center(block):
    x0, y0, x1, y1 = block.bbox_xyxy
    return np.array([(x0 + x1) / 2, (y0 + y1) / 2])


def _nearby_overhead(frame, selection, previous):
    if previous is None:
        expected = _region_center(selection)
        radius = 10.
    else:
        expected = frame.camera.project(previous)
        # Express a 25 mm window in the current overhead image.
        distance = (previous - frame.camera.position) @ frame.camera.forward
        radius = .025 * frame.camera.focal / distance
    return tuple(block for block in frame.detections
                 if block.color == selection.color and block.studs_xy
                 and (np.linalg.norm(_region_center(block) - expected) <= radius
                      or previous is not None and np.linalg.norm(block.xy - expected) <= radius))


def _partial_row(anchor, block, side, previous, min_count, max_count):
    """Use shared studs to measure row height, then intersect the remaining anchor rays."""
    count = len(block.studs_xy)
    if not max(3, min_count) <= count <= max_count:
        return None
    # Incomplete landmarks provide less evidence to distinguish overlapping colors.
    if sum(b.color == block.color for b in anchor.detections) != 1:
        return None
    matches = [d for d in side.detections if d.color == block.color]
    if len(matches) != 1:
        return None
    match, = matches
    pixels = np.asarray(match.studs_xy)
    if not 1 <= len(pixels) < count or not np.isfinite(pixels).all():
        return None
    anchor_xy = np.asarray(block.studs_xy)
    if not np.isfinite(anchor_xy).all():
        return None
    allowed = range(count)
    if len(pixels) == 1:
        # The full view supplies pitch and orientation. Resolve the visible
        # stud's index before triangulating its height with the partial view.
        pattern = stud_pattern_world(anchor.camera, block)
        if pattern is None:
            return None
        xyz, ordered_pixels = pattern
        try:
            errors = [np.linalg.norm(side.camera.project(point) - pixels[0]) for point in xyz]
        except ValueError:
            return None
        best, second = np.argsort(errors)[:2]
        if errors[best] > 10 or errors[second] - errors[best] < 5:
            return None
        allowed = [np.linalg.norm(anchor_xy - ordered_pixels[best], axis=1).argmin()]
    candidates = []
    for indices in permutations(allowed, len(pixels)):
        reference = _Observation(anchor.camera, anchor_xy[list(indices)])
        fit = _triangulate([reference, _Observation(side.camera, pixels)], complete_pattern=False)
        if fit is None:
            continue
        measured, error = fit
        try:
            points = np.array([anchor.camera.on_plane(xy, measured[0, 2]) for xy in anchor_xy])
        except ValueError:
            continue
        points[list(indices)] = measured
        xy = points[:, :2]
        if (not _coherent_studs(xy) or count == 4 and not _four_stud_row(xy)
                or not _region_fits_footprint(anchor.camera, block, xy, measured[0, 2])
                or previous is not None and np.linalg.norm(points.mean(axis=0) - previous) > .025):
            continue
        candidates.append(TrackEstimate(points.mean(axis=0), points, match, "tracked",
                                        "matched_partial_stud_row", 2, error))
    candidates.sort(key=lambda fit: fit.reprojection_error_px)
    if (not candidates or len(candidates) > 1
            and candidates[1].reprojection_error_px - candidates[0].reprojection_error_px < 1.5):
        return None
    return candidates[0]


def _fit_from_views(anchor, block, sides, previous, min_count, max_count):
    if _clipped(block, anchor.shape):
        return None
    candidates = []
    for first in sides:
        track = BlockTrack(anchor.camera, block, neighbors=anchor.detections)
        fit = track.update(first.camera, first.detections, frame_shape=first.shape)
        if (fit.xyz is None or not min_count <= len(fit.studs_xyz) <= max_count
                or len(fit.studs_xyz) == 4 and not _four_stud_row(fit.studs_xyz[:, :2])):
            partial = _partial_row(anchor, block, first, previous, min_count, max_count)
            if partial is not None:
                candidates.append((partial, None))
            continue
        for other in sides:
            if other is not first:
                refined = track.update(other.camera, other.detections, frame_shape=other.shape)
                if refined.xyz is not None:
                    fit = refined
        if previous is not None and np.linalg.norm(fit.xyz - previous) > .025:
            continue
        candidates.append((fit, track))
    if not candidates:
        return None
    return min(candidates, key=lambda pair: (-len(pair[0].studs_xyz), -pair[0].views, pair[0].reprojection_error_px))


def _overhead_confirms(overhead, block, fit):
    try:
        pixels = np.asarray([overhead.camera.project(point) for point in fit.studs_xyz])
    except ValueError:
        return False
    x0, y0, x1, y1 = block.bbox_xyxy
    distance = np.linalg.norm(pixels[:, None] - np.asarray(block.studs_xy)[None], axis=2)
    visible = distance.argmin(axis=0)
    if 2 <= len(visible) < len(pixels) and len(set(visible)) == len(visible):
        return distance.min(axis=0).max() <= 5
    return (distance.min() <= 5
            and np.all(pixels >= [x0 - 2, y0 - 2])
            and np.all(pixels < [x1 + 2, y1 + 2]))


def _fit_from_wrists(overhead, block, sides, previous, min_count, max_count):
    """Use the second wrist when the overhead detector misses part of the row."""
    fits = []
    for side in sides:
        others = tuple(other for other in sides if other is not side)
        for candidate in side.detections:
            if candidate.color != block.color or len(candidate.studs_xy) < 2:
                continue
            pair = _fit_from_views(side, candidate, others, previous, min_count, max_count)
            if pair is not None and _overhead_confirms(overhead, block, pair[0]):
                fits.append(pair)
            elif len(block.studs_xy) < len(candidate.studs_xy):
                partial = _partial_row(side, candidate, overhead, previous, min_count, max_count)
                if partial is not None and partial.detection is block:
                    fits.append((partial, None))
    if not fits:
        return None
    longest = max(len(pair[0].studs_xyz) for pair in fits)
    fits = [pair for pair in fits if len(pair[0].studs_xyz) == longest]
    first = min(fits, key=lambda pair: (-pair[0].views, pair[0].reprojection_error_px))
    if any(np.linalg.norm(other[0].xyz - first[0].xyz) > .006 for other in fits):
        return None
    return first


class BlockMap:
    """Track overhead selections by their initial detection index.

    The initial pixel selections belong to the fixed overhead camera. Its
    extrinsic calibration can improve as the robot moves.

    latest contains only measured stud poses. Held and covered blocks can be
    excluded from a capture while the other tracks continue to refine.
    """

    def __init__(self, overhead):
        self.selections = tuple(overhead.detections)
        self._expected = {}
        self._tracks = {}
        self._lengths = {}
        self.latest = {}

    def confirm_lengths(self):
        """Keep each inventoried block's stud count fixed through subsequent grasps."""
        if len(self.latest) != len(self.selections):
            raise ValueError("Every block needs a measured stud row before confirming lengths")
        self._lengths = {i: len(pose.studs_xyz) for i, pose in self.latest.items()}

    def expect_placement(self, block_id, expected_studs_xyz):
        """Forget the pre-grasp pose and search near the planned release."""
        if not 0 <= block_id < len(self.selections):
            raise KeyError(block_id)
        points = np.asarray(expected_studs_xyz, dtype=float)
        if points.shape not in ((2, 3), (3, 3), (4, 3)) or not np.isfinite(points).all():
            raise ValueError("Expected a row of two to four finite world-space stud positions")
        if block_id in self._lengths and len(points) != self._lengths[block_id]:
            raise ValueError("Placement length must match the confirmed stud count")
        self._expected[block_id] = points.copy()
        self.latest.pop(block_id, None)
        self._tracks.pop(block_id, None)

    def _observe_block(self, block_id, overhead, sides):
        selection = self.selections[block_id]
        last = self.latest.get(block_id)
        points = last.studs_xyz if last is not None else self._expected.get(block_id)
        previous = None if points is None else points.mean(axis=0)
        min_count = 2 if points is None else len(points)
        max_count = self._lengths.get(block_id, 4)
        candidates = []
        for block in _nearby_overhead(overhead, selection, previous):
            pair = _fit_from_views(overhead, block, sides, previous, min_count, max_count)
            count = len(pair[0].studs_xyz) if pair is not None else 0
            if (count < max_count
                    and any(len(d.studs_xy) > count for side in sides for d in side.detections
                            if d.color == block.color)):
                wrist_pair = _fit_from_wrists(overhead, block, sides, previous, max(min_count, count + 1), max_count)
                if wrist_pair is not None:
                    pair = wrist_pair
            if pair is not None:
                candidates.append((block, *pair))
        return candidates[0] if len(candidates) == 1 else None

    def observe(self, overhead, sides, *, excluded=()):
        """Return this capture's verified poses, retaining prior poses on misses."""
        sides = tuple(sides)
        proposals = {}
        for i in range(len(self.selections)):
            if i not in excluded:
                match = self._observe_block(i, overhead, sides)
                if match is not None:
                    proposals[i] = match
        claims = Counter(id(block) for block, _, _ in proposals.values())
        current = {}
        for i, (block, fit, track) in proposals.items():
            if claims[id(block)] > 1:
                continue
            previous = self.latest.get(i)
            # Fresh triangulation detects a bumped block. Fuse earlier rays
            # only while both sets of stud positions agree within 5 mm.
            if (previous is not None and self._tracks[i] is not None
                    and _stud_distance(previous, fit) < .005):
                candidate = self._tracks[i]
                fresh = fit
                for frame in sides:
                    trial = copy(candidate)
                    refined = trial.update(frame.camera, frame.detections, frame_shape=frame.shape)
                    if (refined.xyz is not None and refined.views >= fit.views
                            and _stud_distance(refined, fresh) < .005):
                        fit, track, candidate = refined, trial, trial
                    elif refined.xyz is not None:
                        break
            self.latest[i] = current[i] = fit
            self._tracks[i] = track
        return current


def _stud_distance(first, second):
    if len(first.studs_xyz) != len(second.studs_xyz):
        return np.inf
    distances = np.linalg.norm(first.studs_xyz[:, None] - second.studs_xyz[None], axis=2)
    return max(distances.min(axis=0).max(), distances.min(axis=1).max())


class StackTracker:
    """Two-block view of a shared map for a selected pickup and support."""

    def __init__(self, overhead, source, target):
        if source is target:
            raise ValueError("Select two distinct block detections")
        self._map = BlockMap(overhead)
        self._ids = tuple(next(i for i, b in enumerate(overhead.detections) if b is selection)
                          for selection in (source, target))

    def _pair(self, estimates):
        return StackEstimate(*(estimates.get(i) for i in self._ids))

    @property
    def latest(self):
        return self._pair(self._map.latest)

    def observe(self, overhead, sides, *, source_held=False):
        excluded = (self._ids[0],) if source_held else ()
        return self._pair(self._map.observe(overhead, sides, excluded=excluded))
