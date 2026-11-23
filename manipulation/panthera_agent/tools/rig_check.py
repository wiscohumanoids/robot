#!/usr/bin/env python3
"""Pre-flight check of the rig for panthera_agent, without moving anything.

    python tools/rig_check.py                 # every check; exits non-zero if one fails

Checks, in order: config/rig.yaml filled in (no "R0:" placeholders left where they matter), calibration files with
intrinsics and poses, each camera opening and giving a sensible frame (not black, not frozen), the arms' SDK
connection and joint feedback, the model's forward kinematics against the SDK's (RealRobot does this at start-up),
the current TCP inside the safety box, the home pose reachable, the gripper's reading inside its calibrated range,
the empty-table photo, DINOv3 loading, and a trained local policy if any. Run it before every session; it is the
"someone at the stop button" moment's checklist, not a substitute for one.
"""
from __future__ import annotations

import pathlib
import sys
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from core.build import load_config  # noqa: E402

results: list[tuple[str, bool, str]] = []


def check(name):
    def deco(f):
        try:
            ok, detail = f()
        except Exception as e:                                  # a failing check is a result, not a crash
            ok, detail = False, f'{type(e).__name__}: {e}'
            if '-v' in sys.argv:
                traceback.print_exc()
        results.append((name, bool(ok), detail))
        return f
    return deco


def main():
    cfg = load_config('config/rig.yaml')
    state = {}

    @check('config filled in')
    def _():
        text = (ROOT / 'config' / 'rig.yaml').read_text()
        left = [line.strip() for line in text.splitlines() if '"R0:' in line or "'R0:" in line]
        return not left, 'ok' if not left else f'{len(left)} R0 placeholders: ' + '; '.join(left[:3])

    @check('calibration files')
    def _():
        import json
        missing = []
        for name, c in cfg['cameras'].items():
            p = ROOT / c['calibration']
            if not p.exists():
                missing.append(f'{name}: no {c["calibration"]}')
                continue
            cal = json.loads(p.read_text())
            if 'T_base_cam' not in cal and 'T_flange_cam' not in cal:
                missing.append(f'{name}: intrinsics only')
        return not missing, 'ok' if not missing else '; '.join(missing)

    @check('robot connection, joint feedback, kinematics')
    def _():
        from robots.real_robot import RealRobot
        state['robot'] = RealRobot(cfg)
        return True, f'arms: {", ".join(state["robot"].arm_names())} (kinematics agree with the SDK)'

    @check('cameras give live frames')
    def _():
        r = state['robot']
        bad = []
        for name in r.camera_names():
            a = r.observe(name).rgb
            time.sleep(0.2)
            b = r.observe(name).rgb
            if a.mean() < 10:
                bad.append(f'{name}: black')
            elif np.abs(a.astype(int) - b.astype(int)).mean() == 0:
                bad.append(f'{name}: frozen')
        return not bad, 'ok' if not bad else '; '.join(bad)

    @check('TCP inside the safety box, home reachable')
    def _():
        from robots.real_robot import START_POSITION
        r = state['robot']
        out = []
        for arm in r.arm_names():
            r.select_arm(arm)
            p, R = r.ee_pose()
            why = r._safe(p)
            home, Rh = r._to_table(START_POSITION, r.start_rotation)
            out.append(f'{arm}: TCP {np.round(p, 3).tolist()} {"inside" if not why else why}, home '
                       f'{"reachable" if r.reachable(home, Rh) else "NOT reachable"}')
        return all('inside' in o and 'NOT' not in o for o in out), '; '.join(out)

    @check('gripper reading in range')
    def _():
        r = state['robot']
        w = r.gripper_width()
        return 0.0 <= w <= cfg['gripper']['open_width'] + 0.005, f'{w * 1000:.0f} mm'

    @check('empty-table photo')
    def _():
        from perception.memory import ObjectMemory
        p = ObjectMemory().folder / 'empty_overhead.png'
        return p.exists(), str(p) if p.exists() else 'missing: run tools/capture_empty_table.py'

    @check('DINOv3 loads')
    def _():
        from perception.dino import shared
        d = shared()
        return True, f'on {d.device}'

    @check('trained local policy (optional)')
    def _():
        from learning.policy import load
        p = load('local')
        return True, 'none yet (insert uses the scripted search)' if p is None else f'shared local policy: {p.record}'

    width = max(len(n) for n, _, _ in results)
    for name, ok, detail in results:
        print(f'{"PASS" if ok else "FAIL"}  {name:<{width}}  {detail}')
    sys.exit(0 if all(ok for _, ok, _ in results) else 1)


if __name__ == '__main__':
    main()
