#!/usr/bin/env python3
"""S2: the segmenters side by side on rendered sim images, scored against MuJoCo's ground-truth masks.

    tools/tunnel.sh &                                                    # both VM servers on 8100 / 8101
    .venv/bin/python tools/compare_segmenters.py --seeds 5               # -> runs/segcompare_<time>/

For each seed (a random layout of the three cubes), each camera (the overhead one, and the wrist one from a
pre-grasp pose over the red cube) and each query ("red cube", ..., "cube"), every segmenter's masks are matched to
the true ones (IoU >= 0.5). Reported per segmenter and query: found (recall), wrong (false positives: another cube,
or nothing), mask IoU, the position error that matters for grasping (the top-face centre from the mask through
the same geometry the skills use, against the one from the true mask), and the server's time per request. The
overlays (true outline white, found ones coloured) are saved for each image.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robots.sim_robot import SimRobot  # noqa: E402  (sets MUJOCO_GL first)

from core.build import SEGMENTERS, load_config, load_dotenv, make_segmenter  # noqa: E402
from core.geometry import footprint, gripper_rotation  # noqa: E402
from perception.overlay import draw_instances  # noqa: E402

QUERIES = ['red cube', 'green cube', 'blue cube', 'cube']


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def centre(mask, camera, cfg):
    """The top-face centre (x, y) of a cube on the table, from its mask (as the skills measure it)."""
    z0, h = float(cfg['table']['z']), float(cfg['objects']['cube']['height'])
    fp = footprint(mask, camera, z0 + h, z0)
    return None if fp is None else fp.center[:2]


def views(robot: SimRobot, cfg):
    """(camera name, observation): the overhead view from the start pose, the wrist view over the red cube."""
    yield 'overhead', robot.observe('overhead')
    p = robot.truth_objects()['cube_red'][0]
    R = gripper_rotation(float(np.arctan2(p[1], p[0])), np.deg2rad(cfg['grasp']['pitch']))
    pre = np.array([p[0], p[1], p[2] + cfg['objects']['cube']['height'] / 2 + cfg['grasp']['pregrasp_height']])
    if robot.move_linear(pre, R).ok:
        yield 'wrist', robot.observe('wrist')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--segmenters', nargs='+', default=['dino_sam2', 'sam3'], choices=[s for s in SEGMENTERS
                                                                                       if s != 'sim_truth'])
    ap.add_argument('--queries', nargs='+', default=QUERIES)
    args = ap.parse_args()
    load_dotenv()
    cfg = load_config()
    truth_seg = make_segmenter('sim_truth', cfg)
    segs = {n: make_segmenter(n, cfg) for n in args.segmenters}
    out = ROOT / 'runs' / f'segcompare_{time.strftime("%Y%m%d_%H%M%S")}'
    out.mkdir(parents=True)
    stats = defaultdict(lambda: defaultdict(list))         # (segmenter, camera, query) -> metric -> values
    for seed in range(args.seeds):
        robot = SimRobot(cfg, seed=seed)
        for cam, obs in views(robot, cfg):
            for q in args.queries:
                truth = truth_seg.segment(obs, q)
                for name, seg in segs.items():
                    t0 = time.perf_counter()
                    pred = seg.segment(obs, q)
                    wall = (time.perf_counter() - t0) * 1000
                    s = stats[(name, cam, q)]
                    s['server_ms'].append(seg.last_timing.get('total', np.nan))
                    s['wall_ms'].append(wall)
                    used = set()
                    for t in truth:
                        best = max(((iou(t.mask, p.mask), k) for k, p in enumerate(pred) if k not in used),
                                   default=(0.0, None))
                        s['truth'].append(1)
                        if best[0] >= 0.5:
                            used.add(best[1])
                            s['found'].append(1)
                            s['iou'].append(best[0])
                            ct, cp = centre(t.mask, obs.camera, cfg), centre(pred[best[1]].mask, obs.camera, cfg)
                            if ct is not None and cp is not None:
                                s['pos_mm'].append(float(np.linalg.norm(ct - cp)) * 1000)
                    s['wrong'].append(len(pred) - len(used))
                    img = draw_instances(obs.rgb, [(k + 1, p.mask) for k, p in enumerate(pred)])
                    for t in truth:
                        cs, _ = cv2.findContours(t.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        cv2.drawContours(img, cs, -1, (255, 255, 255), 1)
                    cv2.imwrite(str(out / f'seed{seed}_{cam}_{q.replace(" ", "_")}_{name}.jpg'),
                                cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f'seed {seed} done', flush=True)
    rows = []
    for (name, cam, q), s in sorted(stats.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0])):
        n_truth = len(s['truth'])
        rows.append(dict(segmenter=name, camera=cam, query=q, found=f'{len(s["found"])}/{n_truth}',
                         wrong=int(np.sum(s['wrong'])),
                         iou=round(float(np.mean(s['iou'])), 3) if s['iou'] else None,
                         pos_mm_median=round(float(np.median(s['pos_mm'])), 2) if s['pos_mm'] else None,
                         pos_mm_max=round(float(np.max(s['pos_mm'])), 2) if s['pos_mm'] else None,
                         server_ms=round(float(np.nanmedian(s['server_ms'])), 0),
                         wall_ms=round(float(np.median(s['wall_ms'])), 0)))
    (out / 'summary.json').write_text(json.dumps(rows, indent=2))
    head = f'{"camera":9} {"query":11} {"segmenter":10} {"found":>6} {"wrong":>5} {"IoU":>6} {"pos mm":>13} ' \
           f'{"server ms":>9} {"wall ms":>8}'
    print(head)
    for r in rows:
        pos = f'{r["pos_mm_median"]} / {r["pos_mm_max"]}' if r['pos_mm_median'] is not None else '-'
        print(f'{r["camera"]:9} {r["query"]:11} {r["segmenter"]:10} {r["found"]:>6} {r["wrong"]:>5} '
              f'{r["iou"] if r["iou"] is not None else "-":>6} {pos:>13} {r["server_ms"]:>9.0f} {r["wall_ms"]:>8.0f}')
    print(f'(pos mm: median / max; overlays and summary.json in {out})')


if __name__ == '__main__':
    main()
