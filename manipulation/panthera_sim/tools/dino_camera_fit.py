"""Fit a centered perspective camera to matched 3D robot landmarks."""

import cv2
import numpy as np
from scipy.optimize import least_squares

from tools.dino_pick_camera import CameraCalibration


def _project(world, pose, matrix, k1):
    distortion = np.array([k1, 0., 0., 0.])
    pixels, jacobian = cv2.projectPoints(world, pose[:3], pose[3:6], matrix, distortion)
    return pixels.reshape(-1, 2), jacobian


def _consensus(world, pixels, matrix):
    """Identify consistent 3D correspondences under plausible initial lens warps."""
    best = None
    for k1 in (0., -.20, .20):
        distortion = np.array([k1, 0., 0., 0.])
        ok, _, _, support = cv2.solvePnPRansac(
            world, pixels, matrix, distortion, flags=cv2.SOLVEPNP_AP3P,
            reprojectionError=3.5, iterationsCount=250,
        )
        if not ok or support is None or len(support) < 7:
            continue
        indices = support.ravel()
        ok, rot, trans = cv2.solvePnP(world[indices], pixels[indices], matrix, distortion,
                                      flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            continue
        pose = np.r_[rot.ravel(), trans.ravel()]
        predicted, _ = _project(world, pose, matrix, k1)
        errors = np.linalg.norm(predicted - pixels, axis=1)
        mask = errors < 4.
        if mask.sum() < max(7, .35 * len(world)):
            continue
        score = (mask.sum(), -np.median(errors[mask]))
        if best is None or score > best[0]:
            best = (score, mask, pose, k1)
    return None if best is None else best[1:]


def _holdout(pixels):
    # Sort by direction about the centroid to distribute held-out features
    # over the visible surface, even when matches arrive in scanline order.
    centered = pixels - np.mean(pixels, axis=0)
    angle = np.arctan2(centered[:, 1], centered[:, 0])
    count = max(3, len(pixels) // 4)
    rng = np.random.default_rng(0)
    groups = np.array_split(np.argsort(angle), count)
    return np.array([rng.choice(group) for group in groups])


def _refine(world, pixels, matrix, pose, radial):
    initial = np.r_[pose[:6], 0.] if radial else pose[:6]

    def evaluate(params):
        prediction, jacobian = _project(world, params, matrix, params[6] if radial else 0.)
        # OpenCV puts k1 after the six extrinsic and four intrinsic parameters.
        jacobian = jacobian[:, [0, 1, 2, 3, 4, 5, 10]] if radial else jacobian[:, :6]
        return (prediction - pixels).ravel(), jacobian

    limits = np.full(len(initial), np.inf)
    if radial:
        limits[-1] = .5
    fit = least_squares(lambda p: evaluate(p)[0], initial, jac=lambda p: evaluate(p)[1],
                        bounds=(-limits, limits), loss='soft_l1', f_scale=.8, x_scale='jac', max_nfev=40)
    return fit.x if fit.success else None


def _radial_is_observable(world, matrix, params, noise_px):
    # A small field of view lets pose changes mimic radial distortion. Keep
    # only the part of the k1 derivative independent of the six pose columns.
    _, jacobian = _project(world, params, matrix, params[6])
    pose_jac = jacobian[:, :6]
    radial_jac = jacobian[:, 10]
    equivalent_pose, *_ = np.linalg.lstsq(pose_jac, radial_jac, rcond=None)
    sensitivity = np.linalg.norm(radial_jac - pose_jac @ equivalent_pose)
    return abs(params[6]) * sensitivity > 3 * max(.5, noise_px)


def _rms(world, pixels, matrix, params):
    predicted, _ = _project(world, params, matrix, params[6] if len(params) == 7 else 0.)
    return np.sqrt(np.mean(np.sum((predicted - pixels) ** 2, axis=1)))


def _image_spread(pixels):
    # Minor-axis standard deviation detects thin or nearly collinear image coverage.
    return np.linalg.svd(pixels - pixels.mean(axis=0), compute_uv=False)[-1] / np.sqrt(len(pixels))


def fit_camera(points_world, pixels, image_shape, focal, *, fit_distortion=True):
    """Return (calibration, inlier RMS in pixels), or None for a weak view.

    points_world gives CAD landmarks in meters, paired with pixel-center XY.
    All landmarks and the resulting pose use the same coordinate frame.
    Set fit_distortion=False when the pixels have already been rectified. To
    pool exposures from a moving wrist, express each observation in that
    wrist's mount frame at exposure time. The sensor has square pixels, a
    centered principal point, and a known focal length in pixels. Off-axis features at different
    radii can also constrain the first radial distortion coefficient.
    """
    world = np.ascontiguousarray(points_world, dtype=np.float64)
    pixels = np.ascontiguousarray(pixels, dtype=np.float64)
    if (world.ndim != 2 or world.shape[1] != 3 or pixels.shape != (len(world), 2)
            or not np.isfinite(world).all() or not np.isfinite(pixels).all()
            or not np.isfinite(focal) or focal <= 0):
        return None
    # SIFT can assign several orientations to one point in the same image.
    _, unique = np.unique(np.rint(pixels / 2.), axis=0, return_index=True)
    world, pixels = world[unique], pixels[unique]
    if len(world) < 10:
        return None
    singular = np.linalg.svd(world - world.mean(axis=0), compute_uv=False)
    if singular[1] < max(1e-6, .001 * singular[0]):
        return None
    height, width = image_shape[:2]
    center = (np.array([width, height]) - 1.) / 2.
    matrix = np.array([[focal, 0., center[0]], [0., focal, center[1]], [0., 0., 1.]])
    hold = _holdout(pixels)
    train = np.ones(len(world), dtype=bool)
    train[hold] = False
    consensus = _consensus(world[train], pixels[train], matrix)
    if consensus is None:
        return None
    inliers, pose, seed_k1 = consensus
    train[np.flatnonzero(train)] = inliers
    min_spread = .035 * min(height, width)
    if _image_spread(pixels[train]) < min_spread:
        return None
    # The pose fit never saw these points. Use the seed to
    # screen false matches before comparing the two models on identical data.
    prediction, _ = _project(world[hold], pose, matrix, seed_k1)
    agree = np.linalg.norm(prediction - pixels[hold], axis=1) < 5.
    if agree.sum() < max(3, .35 * len(hold)):
        return None
    hold = hold[agree]
    if _image_spread(pixels[hold]) < min_spread * .5:
        return None
    pinhole = _refine(world[train], pixels[train], matrix, pose, radial=False)
    if pinhole is None:
        return None
    error = _rms(world[hold], pixels[hold], matrix, pinhole)
    radial = False
    radius = np.linalg.norm((pixels - center) / focal, axis=1)
    if (fit_distortion and len(hold) >= 4 and radius[train].max() > .35 and np.ptp(radius[train]) > .20
            and radius[hold].max() > .35):
        warped = _refine(world[train], pixels[train], matrix, pinhole, radial=True)
        if warped is not None:
            warp_error = _rms(world[hold], pixels[hold], matrix, warped)
            if (warp_error + .15 < error
                    and _radial_is_observable(world[train], matrix, warped, warp_error)):
                error, radial = warp_error, True
    if error > 2.8:
        return None

    train[hold] = True
    world, pixels = world[train], pixels[train]
    params = _refine(world, pixels, matrix, pinhole, radial)
    if params is None:
        return None
    rotation, _ = cv2.Rodrigues(params[:3])
    position = -rotation.T @ params[3:6]
    rms = _rms(world, pixels, matrix, params)
    k1 = float(params[6]) if radial else 0.
    # A negative cubic reaches its largest distorted radius at its turning point.
    # All sensor corners need a unique inverse on the branch through the origin.
    if (k1 < 0 and np.linalg.norm(center) / focal >= 2 / (3 * np.sqrt(-3 * k1))):
        return None
    if rms > 2.5 or np.min((world - position) @ rotation[2]) <= 0:
        return None
    camera = CameraCalibration(position=position, right=rotation[0], up=-rotation[1],
                               forward=rotation[2], focal=focal, center=center, k1=k1)
    return camera, float(rms)
