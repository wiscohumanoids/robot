"""Ankle linkage analysis for training (uses the rod geometry in robot/linkage.json).

Why this exists: robot/linkage.py solves each crank angle with a secant iteration started
at 0, which diverges near the edge of the ankle range (HARDWARE_ISSUES H2). Here each crank
angle is found by bracketing the rod-length equation on a fine scan and refining with
brentq, then:

  1. reachability: which (pitch, roll) poses the rods can actually make
  2. the training ankle-pitch lower limit that keeps every roll angle reachable
  3. a grid of the 2x2 Jacobian d(motor angle)/d(joint angle) per ankle, so the training
     penalty can turn joint torques into motor torques on the GPU by interpolation:
         tau_motor = solve(J^T, tau_joint)

Run:  python linkage_tools.py      (writes ankle_linkage.json next to this file)
"""
import json
import sys

import numpy as np

from robot_params import LINKAGE_OUT, ROBOT_DIR, SOURCE_ANKLE_PITCH_LOWER, load_report

sys.path.insert(0, str(ROBOT_DIR))
from linkage import Mechanism  # noqa: E402  (robot/linkage.py, unchanged)
sys.dont_write_bytecode = True

def brentq(f, a, b, xtol=1e-12):
    """Bisection on a bracketing interval (f(a), f(b) of opposite sign)."""
    fa = f(a)
    while b - a > xtol:
        m = 0.5 * (a + b); fm = f(m)
        if np.sign(fm) == np.sign(fa):
            a, fa = m, fm
        else:
            b = m
    return 0.5 * (a + b)


SCAN = np.linspace(-2.5, 2.5, 2001)      # crank angles searched (rad)
MARGIN = 0.02                            # rad kept between training limit and the unreachable edge


def crank_angles(mech, q, ref=None):
    """Crank angle per motor at joint angles q=(a, b), or None where a rod can't reach.
    ref: angles to stay closest to (keeps the same assembly branch across a grid)."""
    out = []
    for i, c in enumerate(mech.m["cranks"]):
        e = np.array([mech._err(c, t, q) for t in SCAN])
        idx = np.where(np.sign(e[:-1]) != np.sign(e[1:]))[0]
        if len(idx) == 0:
            return None, float(np.abs(e).min())
        roots = [brentq(lambda t: mech._err(c, t, q), SCAN[k], SCAN[k + 1], xtol=1e-12) for k in idx]
        target = 0.0 if ref is None else ref[i]
        out.append(min(roots, key=lambda r: abs(r - target)))
    return np.array(out), 0.0


def jacobian(mech, q, ref, h=1e-5):
    J = np.zeros((2, 2))
    for k in range(2):
        d = np.zeros(2); d[k] = h
        a, _ = crank_angles(mech, np.asarray(q) + d, ref)
        b, _ = crank_angles(mech, np.asarray(q) - d, ref)
        if a is None or b is None:
            return None
        J[:, k] = (a - b) / (2 * h)
    return J


def main():
    rep = load_report()
    pitch_upper = rep["left_ankle_pitch"]["upper"]
    roll_lo, roll_hi = rep["left_ankle_roll"]["lower"], rep["left_ankle_roll"]["upper"]
    result = {"source": "robot/linkage.json", "sides": {}}

    # 1+2. lowest pitch where every roll in range is reachable, on both ankles
    rolls = np.linspace(roll_lo, roll_hi, 9)
    edge = {}
    for side in ("left", "right"):
        mech = Mechanism(f"{side}_ankle")
        p = SOURCE_ANKLE_PITCH_LOWER
        while p < 0:
            if all(crank_angles(mech, (p, r))[0] is not None for r in rolls):
                break
            p += 0.0025
        edge[side] = p
    reachable_lower = max(edge.values())
    train_lower = round(reachable_lower + MARGIN, 3)
    result["reachable_pitch_lower"] = edge
    result["training_ankle_pitch_lower"] = train_lower

    # 3. Jacobian grids over the training range
    pitches = np.linspace(train_lower, pitch_upper, 25)
    rolls_g = np.linspace(roll_lo, roll_hi, 13)
    for side in ("left", "right"):
        mech = Mechanism(f"{side}_ankle")
        ref0, _ = crank_angles(mech, (0.0, 0.0))
        grid = np.zeros((len(pitches), len(rolls_g), 2, 2))
        for i, p in enumerate(pitches):
            for k, r in enumerate(rolls_g):
                # predict this pose's crank angles from the zero-pose Jacobian to pick the branch
                J0 = jacobian(mech, (0.0, 0.0), ref0)
                ref = ref0 + J0 @ np.array([p, r])
                a, _ = crank_angles(mech, (p, r), ref)
                J = jacobian(mech, (p, r), a)
                if J is None:
                    raise RuntimeError(f"{side}: unreachable inside training range at {(p, r)}")
                grid[i, k] = J
        cond = np.array([[np.linalg.cond(grid[i, k]) for k in range(len(rolls_g))] for i in range(len(pitches))])
        result["sides"][side] = {
            "joints": [f"{side}_ankle_pitch", f"{side}_ankle_roll"],
            "crank_angle_at_zero_pose": ref0.round(4).tolist(),
            "jacobian_at_zero_pose": jacobian(mech, (0.0, 0.0), ref0).round(4).tolist(),
            "worst_condition_number": float(cond.max()),
            "grid": grid.round(6).tolist(),
        }
    result["grid_pitch"] = pitches.round(6).tolist()
    result["grid_roll"] = rolls_g.round(6).tolist()
    result["grid_layout"] = "grid[pitch_index][roll_index] = J (2x2), rows = motors (crank order in linkage.json), columns = (pitch, roll)"
    LINKAGE_OUT.write_text(json.dumps(result, indent=1))

    print(f"reachable ankle pitch lower limit: left {edge['left']:+.4f}, right {edge['right']:+.4f} rad "
          f"(source model {SOURCE_ANKLE_PITCH_LOWER:+.4f})")
    print(f"training ankle pitch lower limit:  {train_lower:+.3f} rad ({np.degrees(train_lower):+.1f} deg)")
    for side, s in result["sides"].items():
        print(f"{side}: crank angles at zero pose {s['crank_angle_at_zero_pose']}, J0 {s['jacobian_at_zero_pose']}, "
              f"worst condition number in range {s['worst_condition_number']:.1f}")
    print(f"wrote {LINKAGE_OUT}")


if __name__ == "__main__":
    main()
