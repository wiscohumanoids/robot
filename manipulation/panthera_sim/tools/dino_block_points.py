"""Pair resting block studs across synchronized overhead and wrist observations."""

from collections import Counter, deque
from dataclasses import replace
from itertools import permutations

import numpy as np

from tools.dino_block_calibration import BlockCalibrationFrame, fit_block_calibration, stud_pattern_world
from tools.dino_pick_camera import CameraCalibration
from tools.dino_pick_refine import _coherent_studs, _region_fits_footprint
from tools.dino_pick_track import _clipped


def flat_block_points(overhead, overhead_detections, wrist, wrist_detections, *,
                      stud_z, frame_shape):
    """Return overhead-derived world XYZ and corresponding observed wrist pixels.

    Call on synchronized images before contact, when all blocks rest upright
    on the known plane and their visible stud centers lie near `stud_z` meters.
    Each camera must describe its own image, including any lens distortion.
    Only complete patterns of a color unique in both images contribute points.
    """
    return _block_points(overhead, overhead_detections, wrist, wrist_detections,
                         frame_shape=frame_shape, stud_z=stud_z)


def _block_points(overhead, overhead_detections, wrist, wrist_detections, *,
                  frame_shape, stud_z):
    overhead_colors = Counter(d.color for d in overhead_detections)
    wrist_colors = Counter(d.color for d in wrist_detections)
    targets = {d.color: d for d in wrist_detections if wrist_colors[d.color] == 1}
    world, observed = [], []
    for source in overhead_detections:
        if source.color is None or overhead_colors[source.color] != 1:
            continue
        target = targets.get(source.color)
        if (target is None or _clipped(target, frame_shape)
                or not 1 <= len(source.studs_xy) <= 4
                or len(source.studs_xy) != len(target.studs_xy)):
            continue
        try:
            if stud_z is None:
                pattern = stud_pattern_world(overhead, source)
                if pattern is None:
                    continue
                points, _ = pattern
            else:
                points = np.array([overhead.on_plane(p, stud_z) for p in source.studs_xy])
            predicted = np.array([wrist.project(p) for p in points])
        except ValueError:
            continue
        measured = np.asarray(target.studs_xy, dtype=float)
        if (not np.isfinite(predicted).all() or not np.isfinite(measured).all()
                or not _coherent_studs(points[:, :2])
                or not _region_fits_footprint(overhead, source, points[:, :2], points[0, 2])):
            continue

        # Unlabeled studs admit permutations. First check the pattern's shape
        # after subtracting the shift caused by the approximate wrist mount.
        candidates = []
        scale = min(frame_shape[:2]) / 512
        for order in permutations(range(len(points))):
            residual = measured[list(order)] - predicted
            shift = residual.mean(axis=0)
            shape_error = np.linalg.norm(residual - shift, axis=1).max()
            if shape_error <= 4 * scale and np.linalg.norm(shift) <= (24 if len(points) == 1 else 55) * scale:
                candidates.append((shape_error, order))
        candidates.sort()
        if not candidates or (len(candidates) > 1
                              and candidates[1][0] - candidates[0][0] < 2 * scale):
            continue
        world.extend(points)
        observed.extend(measured[list(candidates[0][1])])

    return np.asarray(world).reshape(-1, 3), np.asarray(observed).reshape(-1, 2)


def _unique_patterns(detections):
    """Map each unambiguous multi-stud block color to its image coordinates."""
    colors = Counter(block.color for block in detections)
    return {block.color: np.asarray(block.studs_xy) for block in detections
            if block.color is not None and colors[block.color] == 1 and 2 <= len(block.studs_xy) <= 4}


def _same_pixels(reference, current):
    if reference.shape != current.shape:
        return False
    # DINO occasionally reverses the order in which it detects a stud row.
    for order in permutations(range(len(reference))):
        if np.max(np.linalg.norm(reference - current[list(order)], axis=1)) <= 4.:
            return True
    return False


class BlockSamples:
    """Accumulate block observations at distinct wrist poses before grasping."""

    def __init__(self, detector, flat_bench):
        self.detector = detector
        self.flat_bench = flat_bench
        self._history = {}
        self._views = {}

    def collect(self, name, wrist_rgb, world_from_grip_origin, world_from_grip_rotation,
                wrist_mount_prior, top_raw_rgb, top_raw_camera, phase, stud_z=.10):
        """Return direct world/pixel pairs and an optional static-block bundle fit.

        Supply synchronized raw overhead and wrist images, the accepted overhead
        calibration, and the wrist mount estimate in grip coordinates. The caller
        samples only before grasping, while the blocks are resting upright.
        """
        if not (phase.startswith("survey") or phase in ("right_observer", "approach", "preshape")):
            return np.empty((0, 3)), np.empty((0, 2)), None

        origin = np.array(world_from_grip_origin, dtype=float)
        rotation = np.array(world_from_grip_rotation, dtype=float)
        history = self._history.setdefault(name, deque(maxlen=12))
        if history:
            last = history[-1][0]
            turn = np.trace(last.grip_rotation.T @ rotation)
            if np.linalg.norm(last.grip_position - origin) < .008 and turn > 1 + 2 * np.cos(np.deg2rad(4)):
                return np.empty((0, 3)), np.empty((0, 2)), None

        overhead = tuple(self.detector.detect(top_raw_rgb))
        observed = tuple(self.detector.detect(wrist_rgb))
        mount = wrist_mount_prior
        camera = CameraCalibration(origin + rotation @ mount.position, rotation @ mount.right,
                                   rotation @ mount.up, rotation @ mount.forward,
                                   mount.focal, mount.center, mount.k1)
        points, pixels = _block_points(top_raw_camera, overhead, camera, observed,
                                       frame_shape=wrist_rgb.shape, stud_z=stud_z if self.flat_bench else None)

        patterns = _unique_patterns(overhead)
        if not any(block.color in patterns and len(block.studs_xy) >= 2 for block in observed):
            return points, pixels, None

        history.append((BlockCalibrationFrame(origin, rotation, observed), patterns))
        self._views[name] = self._views.get(name, 0) + 1
        fit = None
        if len(history) >= 3 and self._views[name] % 3 == 0:
            frames = []
            for frame, old in history:
                # A nudge, occlusion, or competing same-colored block invalidates
                # that color for this frame. Other stationary blocks still count.
                stationary = {color for color in old.keys() & patterns.keys()
                              if _same_pixels(old[color], patterns[color])}
                compatible = tuple(block for block in frame.detections if block.color in stationary)
                if compatible:
                    frames.append(replace(frame, detections=compatible))
            if len(frames) >= 3:
                anchors = [block for block in overhead if block.color in patterns]
                fit = fit_block_calibration(top_raw_camera, anchors, frames, mount, overhead_fixed=True)
        return points, pixels, fit
