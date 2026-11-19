#!/usr/bin/env python3
"""R0: a camera's intrinsics from views of the ChArUco board (core/calibration.py `make_board`: print it, measure
one square, pass --square if it is not 40 mm).

    python tools/calibrate_intrinsics.py overhead --config config/rig.yaml          # live: SPACE capture, q finish
    python tools/calibrate_intrinsics.py wrist --images captures/wrist/             # from saved images
    python tools/calibrate_intrinsics.py --print-board board.png                    # the board to print

Hold the board at different distances, tilts and positions, filling the corners of the image too (20+ views). Use
--fisheye if the RMS error stays above ~0.5 px with the normal model. Writes K and distortion into the camera's
calibration file (rig.yaml `cameras.<name>.calibration`), keeping any pose already there; the poses come from
tools/calibrate_extrinsics.py.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.build import load_config  # noqa: E402
from core.calibration import calibrate_intrinsics, detect_board, make_board  # noqa: E402


def capture_live(device: str, width: int, height: int, board) -> list:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    views = []
    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f'no frame from {device}')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found = detect_board(board, gray)
        shown = frame.copy()
        if found:
            for p in found[1]:
                cv2.circle(shown, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)
        cv2.putText(shown, f'{len(views)} views  SPACE capture  q finish', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 255), 2)
        cv2.imshow('calibrate', shown)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and found:
            views.append(found)
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return views


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('camera', nargs='?')
    ap.add_argument('--config', default='config/rig.yaml')
    ap.add_argument('--images', type=pathlib.Path, help='calibrate from saved images instead of the live camera')
    ap.add_argument('--square', type=float, default=0.04, help='printed square size, metres')
    ap.add_argument('--fisheye', action='store_true')
    ap.add_argument('--print-board', type=pathlib.Path, help='write the board image and exit')
    args = ap.parse_args()
    board = make_board(square_m=args.square, marker_m=args.square * 0.75)
    if args.print_board:
        cv2.imwrite(str(args.print_board), board.generateImage((2100, 1500), marginSize=60))
        print(f'{args.print_board}: print without scaling, then measure a square (pass --square)')
        return
    if not args.camera:
        ap.error('name a camera')
    cfg = load_config(args.config)
    cam = cfg['cameras'][args.camera]
    size = (int(cam['width']), int(cam['height']))
    if args.images:
        views = []
        for p in sorted(args.images.glob('*')):
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None and (found := detect_board(board, img)):
                views.append(found)
                size = img.shape[::-1]
    else:
        devices = json.loads((ROOT.parent / 'panthera_teleop' / 'config' / 'recording.json').read_text())['cameras']
        views = capture_live(devices[cam['device_key']], *size, board)
    if len(views) < 10:
        raise SystemExit(f'only {len(views)} usable views; take at least 10 (20+ is better)')
    model = 'fisheye' if args.fisheye else 'pinhole'
    K, dist, rms = calibrate_intrinsics(views, size, model)
    path = ROOT / cam['calibration']
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text()) if path.exists() else {}
    data.update(name=args.camera, width=size[0], height=size[1], K=K.tolist(), dist=dist.tolist(), model=model,
                rms_px=rms, views=len(views))
    path.write_text(json.dumps(data, indent=2))
    print(f'{args.camera}: RMS {rms:.3f} px over {len(views)} views ({model}) -> {path}')


if __name__ == '__main__':
    main()
