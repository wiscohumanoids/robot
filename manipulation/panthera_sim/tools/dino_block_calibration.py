"""Estimate camera mounts from DINO stud detections and measured arm poses."""

from dataclasses import dataclass
from itertools import permutations, product

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from perception.pickup import BlockDetection
from sim.mega_bloks import PITCH_M, SHAPES
from tools.dino_pick_camera import CameraCalibration


@dataclass(frozen=True)
class BlockCalibrationFrame:
    """Detections and measured world-from-grip pose for one wrist exposure."""

    grip_position: np.ndarray
    grip_rotation: np.ndarray
    detections: tuple[BlockDetection, ...]


@dataclass(frozen=True)
class BlockCalibrationResult:
    """World overhead pose and grip-relative wrist pose with pixel fit metrics."""

    overhead: CameraCalibration
    mount: CameraCalibration
    overhead_k1: float
    wrist_k1: float
    rmse_px: float
    views: int
    blocks: int
    validation_rmse_px: float | None


@dataclass
class _Track:
    shape: np.ndarray
    pose: np.ndarray
    overhead_pixels: np.ndarray
    wrist: list


def _axes(camera):
    return np.column_stack((camera.right, -camera.up, camera.forward))


def _distortion(camera):
    return getattr(camera, "k1", 0.)


def _project(xyz, position, axes, focal, center, k1):
    local = (xyz - position) @ axes
    depth = local[:, 2]
    normalized = local[:, :2] / np.maximum(depth, .01)[:, None]
    radii_squared = np.sum(normalized**2, axis=1, keepdims=True)
    return center + focal * normalized * (1 + k1 * radii_squared)


def _studs(shape, pose):
    x, y, z, yaw = pose
    c, s = np.cos(yaw), np.sin(yaw)
    xy = shape @ np.array([[c, s], [-s, c]]) + [x, y]
    return np.column_stack((xy, np.full(len(shape), z)))


def _pairs(points):
    a, b = np.triu_indices(len(points), k=1)
    return np.sort(np.linalg.norm(points[a] - points[b], axis=1))


def _seed_track(camera, detection):
    pixels = np.asarray(detection.studs_xy, dtype=float)
    count = len(pixels)
    if detection.color is None or count not in (2, 3, 4) or not np.isfinite(pixels).all():
        return None
    try:
        intercepts = np.array([camera.on_plane(pixel, 0.)[:2] for pixel in pixels])
    except ValueError:
        return None
    measured = _pairs(intercepts)
    if measured.min() < .005:
        return None
    # A four-stud strip and a square have different scale-independent pair distances.
    candidates = []
    for nx, ny in dict.fromkeys((*SHAPES, (4, 1))):
        if nx * ny == count:
            shape = PITCH_M * (np.array(list(product(range(nx), range(ny)))) - [(nx - 1) / 2, (ny - 1) / 2])
            nominal = _pairs(shape)
            scale = nominal @ measured / (measured @ measured)
            candidates.append((np.max(abs(scale * measured - nominal)), shape, scale))
    residual, shape, scale = min(candidates, key=lambda candidate: candidate[0])
    # Along rays from one center, moving the common plane scales all XY distances.
    height = camera.position[2] * (1 - scale)
    if not .04 <= height <= .32 or residual > .009:
        return None
    xy = camera.position[:2] + scale * (intercepts - camera.position[:2])
    center = xy.mean(axis=0)
    best = None
    for order in permutations(range(count)):
        matched = xy[list(order)] - center
        dot = np.sum(shape * matched)
        cross = np.sum(shape[:, 0] * matched[:, 1] - shape[:, 1] * matched[:, 0])
        yaw = np.arctan2(cross, dot)
        pose = np.array([*center, height, yaw])
        error = np.linalg.norm(_studs(shape, pose)[:, :2] - xy[list(order)])
        if best is None or error < best[0]:
            best = error, pose, pixels[list(order)]
    return _Track(shape, best[1], best[2], [])


def stud_pattern_world(camera, detection):
    """Infer an upright stud pattern's 3D positions from calibrated overhead RGB.

    Return corresponding world XYZ and pixel XY arrays for an isolated two-,
    three-, or four-stud block. A single stud cannot determine height.
    """
    track = _seed_track(camera, detection)
    if track is None:
        return None
    return _studs(track.shape, track.pose), track.overhead_pixels


def _matches(expected, frame, color):
    matches = []
    for index, detection in enumerate(frame.detections):
        pixels = np.asarray(detection.studs_xy, dtype=float)
        count = len(pixels)
        if detection.color != color or count < 2 or count > len(expected) or not np.isfinite(pixels).all():
            continue
        for assignment in permutations(range(len(expected)), count):
            residual = expected[list(assignment)] - pixels
            shift = residual.mean(axis=0)
            shape_error = np.sqrt(np.mean(np.sum((residual - shift)**2, axis=1)))
            offset = np.linalg.norm(shift)
            score = shape_error + .28 * offset + 2 * (len(expected) - count)
            if shape_error < 14 and offset < 80:
                matches.append((score, index, np.array(assignment), pixels))
    matches.sort(key=lambda match: match[0])
    if not matches or (len(matches) > 1 and matches[1][0] - matches[0][0] < 1.8):
        return None
    _, index, assignment, pixels = matches[0]
    return index, assignment, pixels


def _group_tracks(overhead_camera, overhead_detections, frames, mount):
    proposals = {}
    tracks = []
    for detection in overhead_detections:
        track = _seed_track(overhead_camera, detection)
        if track is None:
            continue
        tracks.append(track)
        xyz = _studs(track.shape, track.pose)
        for index, frame in enumerate(frames):
            grip_rotation = np.asarray(frame.grip_rotation)
            expected = _project(xyz, frame.grip_position + grip_rotation @ mount.position,
                                grip_rotation @ _axes(mount), mount.focal, mount.center, _distortion(mount))
            match = _matches(expected, frame, detection.color)
            if match is not None:
                region, assignment, pixels = match
                proposals.setdefault((index, region), []).append((track, assignment, pixels))
    for (index, _), candidates in proposals.items():
        if len(candidates) == 1:
            track, assignment, pixels = candidates[0]
            track.wrist.append((index, assignment, pixels))
    return [track for track in tracks if len(track.wrist) >= 2]


def _observable(jacobian):
    singular = np.linalg.svd(jacobian, compute_uv=False)
    return jacobian.shape[0] >= jacobian.shape[1] and singular[-1] > .18


def _camera(reference, position, axes, k1):
    return CameraCalibration(position, axes[:, 0], -axes[:, 1], axes[:, 2],
                             reference.focal, reference.center, k1=k1)


def fit_block_calibration(overhead_camera, overhead_detections, frames, nominal_mount, *,
                          overhead_fixed=False):
    """Fit upright stationary blocks and camera poses using only stud pixels and FK.

    Camera inputs supply starting poses and independently known focal lengths and
    principal points. The mount uses the grip frame. Pass overhead_fixed=True
    only when the overhead pose and distortion have already been calibrated.
    Return None when the captures lack correspondence, motion, or geometric rank.
    """
    frames = tuple(frames)
    tracks = _group_tracks(overhead_camera, overhead_detections, frames, nominal_mount)
    indices = sorted({index for track in tracks for index, _, _ in track.wrist})
    if len(indices) < 3:
        return None
    positions = np.asarray([frames[index].grip_position for index in indices])
    if np.linalg.norm(np.ptp(positions, axis=0)) < .05:
        return None

    mount_axes = _axes(nominal_mount)
    overhead_axes = _axes(overhead_camera)
    overhead_size = 0 if overhead_fixed else 6
    pose_start = 6 + overhead_size
    block_count = len(tracks)
    param_count = pose_start + 4 * block_count
    # Scale each parameter to a comparable physical displacement before checking rank.
    camera_scale = [.01, .01, .01, .07, .07, .07]
    scales = np.asarray(camera_scale * (1 + int(not overhead_fixed)) + [.01, .01, .01, .15] * block_count)
    limits = np.asarray([6, 6, 6, 4, 4, 4] * (1 + int(not overhead_fixed)) + [6, 6, 6, 3] * block_count)
    seed_poses = np.asarray([track.pose for track in tracks])

    def camera_pose(reference, axes, update):
        return reference.position + update[:3], Rotation.from_rotvec(update[3:6]).as_matrix() @ axes

    def unpack(parameters, fit_radial):
        updates = parameters[:param_count] * scales
        mp, ma = camera_pose(nominal_mount, mount_axes, updates[:6])
        op, oa = overhead_camera.position, overhead_axes
        if not overhead_fixed:
            op, oa = camera_pose(overhead_camera, overhead_axes, updates[6:pose_start])
        wk = _distortion(nominal_mount)
        ok = _distortion(overhead_camera)
        if fit_radial:
            wk += .1 * parameters[param_count]
            if not overhead_fixed:
                ok += .1 * parameters[param_count + 1]
        return mp, ma, wk, op, oa, ok, seed_poses + updates[pose_start:].reshape(-1, 4)

    def pixel_errors(parameters, fit_radial, held_view=None, *, validation=False):
        mp, ma, wk, op, oa, ok, poses = unpack(parameters, fit_radial)
        camera_views = {}
        for index in indices:
            frame = frames[index]
            rotation = np.asarray(frame.grip_rotation)
            camera_views[index] = frame.grip_position + rotation @ mp, rotation @ ma
        errors = []
        for track, pose in zip(tracks, poses):
            xyz = _studs(track.shape, pose)
            top = _project(xyz, op, oa, overhead_camera.focal, overhead_camera.center, ok)
            if not validation:
                errors.append(top - track.overhead_pixels)
            for index, assignment, pixels in track.wrist:
                if (index == held_view) != validation:
                    continue
                pos, axes = camera_views[index]
                observed = _project(xyz[assignment], pos, axes, nominal_mount.focal, nominal_mount.center, wk)
                errors.append(observed - pixels)
        return np.concatenate(errors).ravel()

    def solve(start, bounds, fit_radial, held_view=None):
        return least_squares(pixel_errors, start, args=(fit_radial, held_view), bounds=(-bounds, bounds),
                             loss="soft_l1", f_scale=2., max_nfev=100)

    def rms(errors):
        return float(np.sqrt(np.mean(np.sum(errors.reshape(-1, 2)**2, axis=1))))

    held_view = indices[-1] if len(indices) >= 4 else None
    fitted = solve(np.zeros(param_count), limits, False, held_view)
    if not fitted.success or not _observable(fitted.jac):
        return None

    validation_rmse = None
    use_radial = False
    bounds = limits
    if held_view is not None:
        validation_rmse = rms(pixel_errors(fitted.x, False, held_view, validation=True))
        radial_count = 1 if overhead_fixed else 2
        radial_bounds = np.r_[limits, np.full(radial_count, 3.)]
        radial = solve(np.r_[fitted.x, np.zeros(radial_count)], radial_bounds, True, held_view)
        if radial.success and _observable(radial.jac) and np.max(abs(radial.x[-radial_count:])) < 2.9:
            radial_rmse = rms(pixel_errors(radial.x, True, held_view, validation=True))
            if radial_rmse + .15 < .9 * validation_rmse:
                use_radial = True
                fitted, bounds, validation_rmse = radial, radial_bounds, radial_rmse
        if validation_rmse > 5.:
            return None
        fitted = solve(fitted.x, bounds, use_radial)
        if not fitted.success or not _observable(fitted.jac):
            return None

    distances = np.linalg.norm(fitted.fun.reshape(-1, 2), axis=1)
    rmse = rms(fitted.fun)
    if rmse > 4. or np.max(distances) > 9. or np.any(abs(fitted.x) > .98 * bounds):
        return None
    mp, ma, wk, op, oa, ok, poses = unpack(fitted.x, use_radial)
    for track, pose in zip(tracks, poses):
        xyz = _studs(track.shape, pose)
        if np.any((xyz - op) @ oa[:, 2] <= 0):
            return None
        for index, assignment, _ in track.wrist:
            frame = frames[index]
            view_pos = frame.grip_position + frame.grip_rotation @ mp
            view_forward = frame.grip_rotation @ ma[:, 2]
            if np.any((xyz[assignment] - view_pos) @ view_forward <= 0):
                return None
    for k1, camera in ((ok, overhead_camera), (wk, nominal_mount)):
        if k1 < 0 and np.linalg.norm(camera.center) / camera.focal >= 2 / (3 * np.sqrt(-3 * k1)):
            return None
    return BlockCalibrationResult(_camera(overhead_camera, op, oa, ok),
                                  _camera(nominal_mount, mp, ma, wk), ok, wk, rmse,
                                  len(indices), block_count, validation_rmse)
