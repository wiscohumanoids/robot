#!/usr/bin/env python3
"""S0: save what each sim camera sees (plain, with the table grid, and the ground-truth masks), to compare with the
rig's cameras when choosing the sim cameras' placement and field of view.

    .venv/bin/python tools/render_views.py --seed 3                 # -> runs/views_seed3/
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import cv2

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.build import load_config, make_segmenter  # noqa: E402
from perception.overlay import draw_instances, draw_table_grid  # noqa: E402
from robots.sim_robot import SimRobot  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--config', default='config/sim.yaml')
    args = ap.parse_args()
    cfg = load_config(args.config)
    robot = SimRobot(cfg, seed=args.seed)
    seg = make_segmenter('sim_truth', cfg)
    out = ROOT / 'runs' / f'views_seed{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    for cam in robot.camera_names():
        obs = robot.observe(cam)
        found = seg.segment(obs, 'block')
        images = {'plain': obs.rgb, 'masks': draw_instances(obs.rgb, [(k + 1, i.mask) for k, i in enumerate(found)])}
        if cam == 'overhead':
            t = cfg['table']
            images['grid'] = draw_table_grid(obs.rgb, obs.camera, t['z'], t['x'], t['y'])
        for kind, img in images.items():
            cv2.imwrite(str(out / f'{cam}_{kind}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        K = obs.camera.K
        print(f'{cam}: {obs.camera.width}x{obs.camera.height}, f {K[0, 0]:.0f} px, at {obs.camera.position.round(3)}')
    print(out)


if __name__ == '__main__':
    main()
