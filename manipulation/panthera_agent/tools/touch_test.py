#!/usr/bin/env python3
"""R1: does a pixel become the right place on the table? A printed ArUco marker (id 0, from the board's dictionary)
lies on the table; the centre camera finds it, the ray through its centre meets the table plane, and the arm brings
its TCP straight above that point. Measure the horizontal miss between the jaws' centre and the marker's centre.

    python tools/touch_test.py --print-marker marker.png          # print it (any size)
    python tools/touch_test.py --height 0.02                      # TCP 2 cm above the predicted point

Target: under 3-5 mm before any wrist refine (README, R1). A larger miss means the centre camera's calibration
(tools/calibrate_extrinsics.py) or the table height is off. NOT YET RUN ON HARDWARE.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.build import load_config  # noqa: E402
from core.geometry import gripper_rotation, pixels_to_plane  # noqa: E402

DICTIONARY = cv2.aruco.DICT_5X5_100


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='config/rig.yaml')
    ap.add_argument('--height', type=float, default=0.02, help='TCP height above the predicted point, metres')
    ap.add_argument('--print-marker', type=pathlib.Path)
    args = ap.parse_args()
    d = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    if args.print_marker:
        cv2.imwrite(str(args.print_marker), cv2.aruco.generateImageMarker(d, 0, 600))
        print(args.print_marker)
        return
    from robots.real_robot import RealRobot
    cfg = load_config(args.config)
    robot = RealRobot(cfg)
    try:
        obs = robot.observe('overhead')
        corners, ids, _ = cv2.aruco.ArucoDetector(d).detectMarkers(cv2.cvtColor(obs.rgb, cv2.COLOR_RGB2GRAY))
        if ids is None or 0 not in ids.ravel():
            raise SystemExit('marker 0 not seen by the centre camera')
        centre = corners[list(ids.ravel()).index(0)].reshape(4, 2).mean(axis=0)
        p = pixels_to_plane(obs.camera, centre[None], robot.table_z)[0]
        print(f'predicted marker centre: x={p[0] * 100:.1f} y={p[1] * 100:.1f} cm')
        target = p + np.array([0.0, 0.0, args.height])
        R = gripper_rotation(float(np.arctan2(p[1], p[0])), np.deg2rad(cfg['grasp']['pitch']))
        above = target + np.array([0.0, 0.0, 0.08])
        for q in (above, target):
            r = robot.move_linear(q, R)
            if not r.ok:
                raise SystemExit(f'move failed: {r.reason}')
        input('Measure the miss between the jaws\' centre and the marker\'s centre, then press Enter to go home. ')
        robot.move_linear(above, R)
        robot.home()
    finally:
        robot.stop()


if __name__ == '__main__':
    main()
