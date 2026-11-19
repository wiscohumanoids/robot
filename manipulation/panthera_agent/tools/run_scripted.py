#!/usr/bin/env python3
"""S1-S3: scripted pick-and-place through the agent's skills, no LLM. Measures each stage against the sim's ground
truth: the overhead estimate, the wrist-refined estimate, and the final stack.

    .venv/bin/python tools/run_scripted.py --seeds 20                                  # S1: ground-truth masks
    .venv/bin/python tools/run_scripted.py --seeds 20 --segmenter server               # S2: the perception server
    .venv/bin/python tools/run_scripted.py --seeds 20 --camera-error 0.005 1.0         # S3: overhead off 5 mm / 1 deg

Tasks: stack_red_on_green (pick red, place on green) and stack_three (red on green, then blue on red).
Results: runs/scripted_<time>/results.jsonl (one line per seed) and a summary on stdout.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.build import load_config, make_segmenter, make_skills  # noqa: E402
from core.camera import pose_error  # noqa: E402
from core.tasks import TASKS  # noqa: E402
from robots.sim_robot import SimRobot  # noqa: E402

PLANS = {'stack_red_on_green': [('red cube', 'green cube')],
         'stack_three': [('red cube', 'green cube'), ('blue cube', 'red cube')]}


def nearest_truth(truth: dict, xy: np.ndarray) -> tuple[str, float]:
    """The true object nearest to an estimate, and the horizontal distance (m)."""
    name = min(truth, key=lambda n: np.linalg.norm(truth[n][0][:2] - xy))
    return name, float(np.linalg.norm(truth[name][0][:2] - xy))


def run_seed(task, seed, config, segmenter_name, camera_error, rng):
    """One scripted episode; returns its record."""
    robot = SimRobot({**config, 'scene': task.scene}, seed=seed)
    if camera_error:
        robot.camera_errors['overhead'] = pose_error(camera_error[0], camera_error[1], rng)
    skills = make_skills(robot, make_segmenter(segmenter_name, config), config)
    rec = dict(seed=seed, task=task.name, steps=[], ok=True)
    for moved, support in PLANS[task.name]:
        step = dict(move=moved, onto=support)
        a, b = skills.find(moved), skills.find(support)
        world = skills.world.objects
        mine = [f for f in world.values() if f.query == moved]
        theirs = [f for f in world.values() if f.query == support]
        if not mine or not theirs:
            step['error'] = f'find: {a.message} / {b.message}'
            rec['steps'].append(step)
            rec['ok'] = False
            break
        obj, sup = mine[-1], theirs[-1]
        truth = robot.truth_objects()
        step['find_error_mm'] = {q: round(nearest_truth(truth, f.top[:2])[1] * 1000, 2) for q, f in
                                 ((moved, obj), (support, sup))}
        pick = skills.pick(obj.number)
        step['pick'] = pick.message
        step['refine_mm'] = pick.details.get('refine', {}).get('corrections_mm')
        if not pick.ok:
            rec['ok'] = False
            rec['steps'].append(step)
            break
        skills._current(sup)                      # the estimate place_on will use (with the overhead correction)
        step['place_estimate_error_mm'] = round(nearest_truth(truth, sup.top[:2])[1] * 1000, 2)
        place = skills.place_on(sup.number)
        step['place'] = place.message
        rec['steps'].append(step)
        if not place.ok:
            rec['ok'] = False
            break
        skills.home()
    rec['metrics'] = task.check(robot.truth_objects())
    rec['success'] = bool(rec['ok'] and rec['metrics']['success'])
    rec['overhead_correction_mm'] = (skills.overhead_correction()[:2] * 1000).round(2).tolist()
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--task', choices=sorted(PLANS), default='stack_red_on_green')
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--first-seed', type=int, default=0)
    ap.add_argument('--segmenter', default='sim_truth', choices=['sim_truth', 'server'])
    ap.add_argument('--camera-error', type=float, nargs=2, metavar=('METRES', 'DEGREES'),
                    help="the overhead camera's calibrated pose off by this much (S3); the image is unchanged")
    ap.add_argument('--config', default='config/sim.yaml')
    ap.add_argument('--out', type=pathlib.Path)
    args = ap.parse_args()
    config = load_config(args.config)
    task = TASKS[args.task]
    out = args.out or ROOT / 'runs' / f'scripted_{time.strftime("%Y%m%d_%H%M%S")}'
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(12345)
    records = []
    with open(out / 'results.jsonl', 'w') as f:
        for seed in range(args.first_seed, args.first_seed + args.seeds):
            rec = run_seed(task, seed, config, args.segmenter, args.camera_error, rng)
            records.append(rec)
            f.write(json.dumps(rec) + '\n')
            first = rec['steps'][0] if rec['steps'] else {}
            print(f"seed {seed}: {'ok  ' if rec['success'] else 'FAIL'} find {first.get('find_error_mm')} "
                  f"refine {first.get('refine_mm')} place-estimate {first.get('place_estimate_error_mm')} mm, "
                  f"{json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in rec['metrics'].items()})}",
                  flush=True)
    n = sum(r['success'] for r in records)
    print(f'\n{task.name}: {n}/{len(records)} succeeded ({args.segmenter}'
          f'{", camera error %g m / %g deg" % tuple(args.camera_error) if args.camera_error else ""}); {out}')


if __name__ == '__main__':
    main()
