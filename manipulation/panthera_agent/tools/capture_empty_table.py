#!/usr/bin/env python3
"""R0, after calibration: photograph the empty table with the overhead camera, for the candidate points.

    python tools/capture_empty_table.py [--robot rig|sim]

The numbered points the LLM chooses from (propose_points, perception/candidates.py) treat as background whatever
still looks like the empty table at the same place, which handles uneven light and any table colour. Clear the
table (fixtures that never move may stay), send the arm home, run this. Saves memory/empty_overhead.png (the
seen-objects folder: PANTHERA_AGENT_MEMORY) and a few frames' median, so a flicker or a passing hand does not end up
in it. Retake it when the light or the camera changes. The sim renders its own and needs none.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from core.build import load_config  # noqa: E402
from perception.memory import ObjectMemory  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--robot', choices=['rig', 'sim'], default='rig')
    ap.add_argument('--frames', type=int, default=7)
    a = ap.parse_args()
    if a.robot == 'rig':
        from robots.real_robot import RealRobot
        robot = RealRobot(load_config('config/rig.yaml'))
        r = robot.home()
        if not r.ok:
            sys.exit(f'could not go home first: {r.reason}')
        frames = [robot.observe('overhead').rgb for _ in range(a.frames)]
    else:
        from robots.sim_robot import SimRobot
        frames = [SimRobot(load_config()).empty_table_image('overhead')]
    img = np.median(np.stack(frames), axis=0).astype(np.uint8)
    folder = ObjectMemory().folder
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'empty_overhead.png'
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f'saved {path} ({img.shape[1]} x {img.shape[0]}, median of {len(frames)} frames)')


if __name__ == '__main__':
    main()
