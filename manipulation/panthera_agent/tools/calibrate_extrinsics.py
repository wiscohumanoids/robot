#!/usr/bin/env python3
"""R0: the wrist camera on the flange (hand-eye), the centre camera in the arm's base frame, and the table plane,
from one board lying still on the table. Needs both cameras' intrinsics first (tools/calibrate_intrinsics.py).

    python tools/calibrate_extrinsics.py --board-x 0.38 --board-y 0.0 --board-thickness 0.003

Lay the ChArUco board flat near (board-x, board-y) in the arm's base frame, where the centre camera sees all of it.
The arm (slowly, inside rig.yaml's safety box) visits poses around the board, tilting and turning the wrist camera
over it; at each it records the flange pose (forward kinematics) and the board's pose in the wrist image. Then:

- hand-eye (OpenCV, Park's method) gives the wrist camera on the flange;
- the board's pose in the base frame follows; its spread across poses is the calibration's consistency check;
- the centre camera sees the board, so its pose in the base frame follows too;
- the board's plane, less its thickness, is the table plane (rig.yaml `table.z`; a tilt over 0.5 deg is reported).

NOT YET RUN ON HARDWARE.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.build import load_config  # noqa: E402
from core.calibration import (board_in_base, board_in_camera, detect_board, fixed_camera_pose, hand_eye,  # noqa: E402
                              make_board)
from core.geometry import gripper_rotation  # noqa: E402


def view_poses(board_xy, n=14, rng=None):
    """TCP targets above the board with varied yaw, pitch and height (the wrist camera looks at the TCP)."""
    rng = rng or np.random.default_rng(0)
    out = []
    for k in range(n):
        yaw = np.deg2rad(rng.uniform(-35, 35)) + np.arctan2(board_xy[1], board_xy[0])
        pitch = np.deg2rad(rng.uniform(40, 70))
        p = np.array([board_xy[0] + rng.uniform(-0.05, 0.05), board_xy[1] + rng.uniform(-0.05, 0.05),
                      rng.uniform(0.12, 0.22)])
        out.append((p, gripper_rotation(yaw, pitch)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='config/rig.yaml')
    ap.add_argument('--board-x', type=float, required=True)
    ap.add_argument('--board-y', type=float, required=True)
    ap.add_argument('--board-thickness', type=float, default=0.003)
    ap.add_argument('--square', type=float, default=0.04)
    ap.add_argument('--poses', type=int, default=14)
    args = ap.parse_args()
    from robots.real_robot import RealRobot
    cfg = load_config(args.config)
    board = make_board(square_m=args.square, marker_m=args.square * 0.75)
    robot = RealRobot(cfg, calibrating=True)
    cal_path = {n: ROOT / c['calibration'] for n, c in cfg['cameras'].items()}
    cal = {n: json.loads(p.read_text()) for n, p in cal_path.items()}

    def board_pose(name, rgb):
        found = detect_board(board, cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        if not found:
            return None
        c = cal[name]
        return board_in_camera(found[0].astype(np.float64), found[1].astype(np.float64), np.array(c['K']),
                               np.array(c['dist']), c.get('model', 'pinhole'))

    flanges, boards = [], []
    try:
        for p, R in view_poses((args.board_x, args.board_y), args.poses):
            r = robot.move_linear(p, R)
            if not r.ok:
                print(f'skip {p.round(3)}: {r.reason}')
                continue
            T = board_pose('wrist', robot.observe('wrist').rgb)
            if T is None:
                print(f'skip {p.round(3)}: board not seen')
                continue
            flanges.append(robot.flange_pose())
            boards.append(T)
            print(f'view {len(boards)} at {p.round(3)}')
        robot.home()
    finally:
        robot.stop()
    if len(boards) < 6:
        raise SystemExit(f'only {len(boards)} usable views; move the board or widen the safety box')
    T_flange_cam = hand_eye(flanges, boards)
    T_base_board, spread = board_in_base(flanges, T_flange_cam, boards)
    print(f'hand-eye: board origin consistent to {spread * 1000:.1f} mm across {len(boards)} views')
    T_overhead_board = board_pose('overhead', robot.observe('overhead').rgb)
    if T_overhead_board is None:
        raise SystemExit('the centre camera does not see the whole board; move it and run again')
    T_base_overhead = fixed_camera_pose(T_base_board, T_overhead_board)
    normal = T_base_board[:3, 2]
    tilt = np.rad2deg(np.arccos(abs(normal[2])))
    table_z = float(T_base_board[2, 3]) - args.board_thickness
    cal['wrist']['T_flange_cam'] = T_flange_cam.tolist()
    cal['overhead']['T_base_cam'] = T_base_overhead.tolist()
    for n, p in cal_path.items():
        cal[n]['hand_eye_spread_m'] = spread
        p.write_text(json.dumps(cal[n], indent=2))
    print(f'centre camera at {T_base_overhead[:3, 3].round(3)} m; table z {table_z:.4f} m, tilt {tilt:.2f} deg'
          + ('  (over 0.5 deg: level the table or model the tilt)' if tilt > 0.5 else ''))
    print(f'set table.z: {table_z:.4f} in {args.config}; then run tools/touch_test.py')


if __name__ == '__main__':
    main()
