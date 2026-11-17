"""Steadiness probe for a walking policy in plain MuJoCo (the physics the ROS sim runs).

Kitchen-like command script (stand, short walks, stops, turns, nudges); per phase it reports what "swaying,
shuffling, stumbling" look like in numbers:
  rock    torso roll/pitch rate RMS (deg/s)          sway / rocking
  tilt    max torso tilt from upright (deg)          stumbling
  steps   foot lift-offs per second (> 2 cm)         shuffling while standing
  slide   foot sliding while on the ground (cm/s)    shuffling without lifting
  drift   pelvis horizontal speed (cm/s) while told to stand

    .venv/bin/python tools/eval/steadiness.py models/locomotion/g1_velocity_arms_free [more policy folders...]
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from sim2sim_play import DEFAULT_SCENE, PREFIX, Policy, Robot  # noqa: E402

SCRIPT = [(4, (0, 0, 0), "stand"), (3, (0.4, 0, 0), "walk 0.4"), (4, (0, 0, 0), "stop"),
          (2, (0, 0, 0.9), "turn 0.9"), (4, (0, 0, 0), "stop"), (3, (0.5, 0, 0.3), "walk+turn"),
          (4, (0, 0, 0), "stop"), (0.8, (0.3, 0, 0), "nudge 0.3"), (3, (0, 0, 0), "stop"),
          (0.6, (0, 0, 0.7), "pulse 0.7"), (3, (0, 0, 0), "stop"), (3, (0, 0.3, 0), "side 0.3"), (5, (0, 0, 0), "stop")]


def run(folder, seed=0):
    """Run the command script with the policy in plain MuJoCo and report sway and drift per phase."""
    p = Policy.load(Path(folder))
    m = mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE))
    r, d = Robot(m, p), mujoco.MjData(m)
    r.reset(d)
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, PREFIX + "torso_link")
    feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, PREFIX + f"{s}_ankle_roll_link") for s in ("left", "right")]
    period = p.sim_dt * p.decimation
    out, totals = [], {}
    rng = np.random.default_rng(seed)
    for sec, cmd, label in SCRIPT:
        n = int(sec / period)
        rock, tilt, lifts, slide, drift = [], [], 0, 0.0, []
        ground = [d.xpos[f][2] for f in feet]
        prev_air = [False, False]
        prev_xy = [d.xpos[f][:2].copy() for f in feet]
        for i in range(n):
            r.control_step(d, np.array(cmd, float))
            if r.fallen(d):
                return out + [(label, "FELL")], {"fell": 1}
            R = d.xmat[torso].reshape(3, 3)
            w = R.T @ d.cvel[torso][:3]                      # body-frame angular velocity
            rock.append(np.hypot(w[0], w[1]))
            tilt.append(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
            for k, f in enumerate(feet):
                z = d.xpos[f][2]
                air = z > min(ground[k], 0.08) + 0.02
                if air and not prev_air[k]:
                    lifts += 1
                if not air and not prev_air[k]:
                    slide += np.linalg.norm(d.xpos[f][:2] - prev_xy[k])
                prev_air[k], prev_xy[k] = air, d.xpos[f][:2].copy()
            if not any(cmd):
                v = np.zeros(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, r.pelvis, v, 0)
                drift.append(np.hypot(v[3], v[4]))
        row = dict(rock=np.degrees(np.sqrt(np.mean(np.square(rock)))), tilt=max(tilt), steps=lifts / sec,
                   slide=100 * slide / sec, drift=100 * np.mean(drift) if drift else float("nan"))
        out.append((label, row))
        k = "standing" if not any(cmd) else "moving"
        totals.setdefault(k, []).append(row)
    summ = {k: {q: np.nanmean([x[q] for x in v]) for q in v[0]} for k, v in totals.items()}
    return out, summ


if __name__ == "__main__":
    for folder in sys.argv[1:]:
        rows, summ = run(folder)
        print(f"== {folder}")
        for label, row in rows:
            if row == "FELL":
                print(f"  {label:10s} FELL"); continue
            print(f"  {label:10s} rock {row['rock']:5.1f} deg/s  tilt {row['tilt']:4.1f} deg  steps {row['steps']:4.2f}/s"
                  f"  slide {row['slide']:4.1f} cm/s" + (f"  drift {row['drift']:4.1f} cm/s" if row['drift'] == row['drift'] else ""))
        for k, s in summ.items():
            print(f"  -> {k:8s}: rock {s['rock']:.1f} deg/s, max tilt {s['tilt']:.1f} deg, {s['steps']:.2f} steps/s, "
                  f"slide {s['slide']:.1f} cm/s" + (f", drift {s['drift']:.1f} cm/s" if k == 'standing' else ""))
