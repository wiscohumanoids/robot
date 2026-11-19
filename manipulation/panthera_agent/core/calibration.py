"""Calibration math for the rig (R0), separate from the capture tools so it can be tested on synthetic data.

- Intrinsics from ChArUco views (`calibrate_intrinsics`).
- Hand-eye for a wrist camera (`hand_eye`): the board lies still on the table, the arm moves; each view pairs the
  flange pose (forward kinematics) with the board's pose in the camera (solvePnP). OpenCV's calibrateHandEye solves
  for the camera on the flange.
- The fixed overhead camera from the same board (`fixed_camera_pose`): the wrist calibration gives the board's pose
  in the arm's base frame, the overhead camera sees the board, so its pose in the base frame follows.
- The table plane from touched points (`fit_plane`).
"""
from __future__ import annotations

import cv2
import numpy as np


def make_board(squares=(7, 5), square_m=0.04, marker_m=0.03, dictionary=cv2.aruco.DICT_5X5_100):
    """The ChArUco board (print it at exactly this square size)."""
    d = cv2.aruco.getPredefinedDictionary(dictionary)
    return cv2.aruco.CharucoBoard(squares, square_m, marker_m, d)


def detect_board(board, gray: np.ndarray):
    """(object points (N, 3), image points (N, 2)) of the board's corners found in an image, or None."""
    det = cv2.aruco.CharucoDetector(board)
    corners, ids, _, _ = det.detectBoard(gray)
    if ids is None or len(ids) < 6:
        return None
    obj, img = board.matchImagePoints(corners, ids)
    return obj.reshape(-1, 3), img.reshape(-1, 2)


def calibrate_intrinsics(views: list[tuple[np.ndarray, np.ndarray]], size: tuple[int, int], model: str = 'pinhole'):
    """K, distortion and RMS reprojection error (pixels) from (object, image) point pairs of several views."""
    obj = [o.astype(np.float32).reshape(-1, 1, 3) for o, _ in views]
    img = [i.astype(np.float32).reshape(-1, 1, 2) for _, i in views]
    if model == 'fisheye':
        K, D = np.eye(3), np.zeros(4)
        flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
        rms, K, D, _, _ = cv2.fisheye.calibrate(obj, img, size, K, D, flags=flags)
        return K, D.ravel(), rms
    rms, K, D, _, _ = cv2.calibrateCamera(obj, img, size, None, None)
    return K, D.ravel(), rms


def board_in_camera(obj: np.ndarray, img: np.ndarray, K: np.ndarray, dist: np.ndarray, model='pinhole') -> np.ndarray:
    """4x4 pose of the board in the camera (board -> camera)."""
    if model == 'fisheye':
        und = cv2.fisheye.undistortPoints(img.reshape(-1, 1, 2).astype(np.float64), K, dist[:4], P=K)
        ok, rvec, tvec = cv2.solvePnP(obj, und, K, None)
    else:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist)
    if not ok:
        raise RuntimeError('solvePnP failed')
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    return T


def hand_eye(T_base_flange: list[np.ndarray], T_cam_board: list[np.ndarray]) -> np.ndarray:
    """4x4 camera pose on the flange (camera -> flange) from paired views (Park's method; needs >= 3 views with
    rotations about different axes, more is better)."""
    R_g = [T[:3, :3] for T in T_base_flange]
    t_g = [T[:3, 3] for T in T_base_flange]
    R_t = [T[:3, :3] for T in T_cam_board]
    t_t = [T[:3, 3] for T in T_cam_board]
    R, t = cv2.calibrateHandEye(R_g, t_g, R_t, t_t, method=cv2.CALIB_HAND_EYE_PARK)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t.ravel()
    return T


def board_in_base(T_base_flange: list[np.ndarray], T_flange_cam: np.ndarray, T_cam_board: list[np.ndarray]):
    """The (still) board's pose in the base frame, averaged over the views, and the spread (m) of its origin."""
    poses = [Tf @ T_flange_cam @ Tb for Tf, Tb in zip(T_base_flange, T_cam_board)]
    t = np.array([P[:3, 3] for P in poses])
    R = average_rotation([P[:3, :3] for P in poses])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t.mean(axis=0)
    return T, float(np.linalg.norm(t - t.mean(axis=0), axis=1).max())


def fixed_camera_pose(T_base_board: np.ndarray, T_cam_board: np.ndarray) -> np.ndarray:
    """A fixed camera's pose in the base frame (camera -> base), from the board seen by it and known in the base."""
    return T_base_board @ np.linalg.inv(T_cam_board)


def average_rotation(Rs: list[np.ndarray]) -> np.ndarray:
    """The rotation closest to the mean of several (chordal L2 mean)."""
    U, _, Vt = np.linalg.svd(np.sum(Rs, axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Plane through touched points: (unit normal pointing up, offset d with n.p = d)."""
    p = np.asarray(points, float)
    c = p.mean(axis=0)
    n = np.linalg.svd(p - c)[2][-1]
    if n[2] < 0:
        n = -n
    return n, float(n @ c)
