"""Calibration math on synthetic data: a known camera on a known flange, a still board, recovered."""
import cv2
import numpy as np

from core.calibration import (board_in_base, board_in_camera, detect_board, fit_plane, fixed_camera_pose, hand_eye,
                              make_board)


def pose(rx, ry, rz, t):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.array([rx, ry, rz], float))[0]
    T[:3, 3] = t
    return T


def test_hand_eye_and_fixed_camera_recover_the_truth():
    rng = np.random.default_rng(0)
    T_flange_cam = pose(0.1, -0.3, 1.4, [0.05, 0.0, 0.06])
    T_base_board = pose(0.0, 0.0, 0.3, [0.4, 0.05, 0.05])
    T_base_flange, T_cam_board = [], []
    for _ in range(12):
        t = [0.3 + rng.uniform(0, .2), rng.uniform(-.1, .1), 0.25 + rng.uniform(0, .1)]
        Tf = pose(*rng.uniform(-0.6, 0.6, 3), t)
        T_base_flange.append(Tf)
        T_cam_board.append(np.linalg.inv(Tf @ T_flange_cam) @ T_base_board)
    est = hand_eye(T_base_flange, T_cam_board)
    assert np.allclose(est, T_flange_cam, atol=1e-6)
    board, spread = board_in_base(T_base_flange, est, T_cam_board)
    assert np.allclose(board, T_base_board, atol=1e-6) and spread < 1e-6
    T_base_overhead = pose(np.pi, 0.0, 0.0, [0.4, 0.0, 0.9])
    seen = np.linalg.inv(T_base_overhead) @ T_base_board
    assert np.allclose(fixed_camera_pose(board, seen), T_base_overhead, atol=1e-6)


def test_board_detection_and_pnp_on_a_rendered_board():
    board = make_board()
    img = board.generateImage((1400, 1000), marginSize=40)
    found = detect_board(board, img)
    assert found is not None and len(found[0]) > 20
    obj, pts = found
    K = np.array([[1000.0, 0, 700], [0, 1000.0, 500], [0, 0, 1]])
    T = board_in_camera(obj.astype(np.float64), pts.astype(np.float64), K, np.zeros(5))
    assert np.isfinite(T).all() and T[2, 3] > 0


def test_plane_fit():
    pts = np.array([[0.3, 0.1, 0.050], [0.5, -0.1, 0.051], [0.4, 0.2, 0.0505], [0.45, 0.0, 0.0506]])
    n, d = fit_plane(pts)
    c = pts.mean(axis=0)
    z_at_centre = (d - n[0] * c[0] - n[1] * c[1]) / n[2]
    assert n[2] > 0.99 and abs(z_at_centre - c[2]) < 1e-9
