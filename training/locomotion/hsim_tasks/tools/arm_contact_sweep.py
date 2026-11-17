"""Sample random arm targets (G1, legs at the default pose) and count how often an arm touches the
body (legs, pelvis, waist, torso). Compares the old sampling (uniform within 60 % of each joint's soft
range around default) with the "safe" per-joint offsets used by mdp.random_arm_targets.
    python hsim_tasks/tools/arm_contact_sweep.py [--n 3000]"""
import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
from hsim_tasks.arm_ranges import G1_ARM_OFFSETS, offset_bounds  # noqa: E402

SCENE = HERE.parents[4] / "assets/robots/g1/mjlab_1.6.0/g1_velocity_flat.xml"
P = "robot/"
DEFAULT = {"hip_pitch_joint": -0.312, "knee_joint": 0.669, "ankle_pitch_joint": -0.363, "elbow_joint": 0.6,
           "left_shoulder_roll_joint": 0.2, "left_shoulder_pitch_joint": 0.2,
           "right_shoulder_roll_joint": -0.2, "right_shoulder_pitch_joint": 0.2}
ARM = ("shoulder", "elbow", "wrist")


def default_of(n):
    """The default angle of a joint."""
    if n in DEFAULT:
        return DEFAULT[n]
    for k, v in DEFAULT.items():
        if not k.startswith(("left", "right")) and n.endswith(k):
            return v
    return 0.0


def main():
    """Sample arm targets both ways and print how often an arm touches the body."""
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=3000); a = ap.parse_args()
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    m.body("robot/pelvis").pos[:] = [0, 0, 1.5]            # hold the pelvis up, out of the floor
    d = mujoco.MjData(m)
    names = [m.joint(j).name for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    arm = [n for n in names if any(k in n for k in ARM)]
    qadr = {n: m.jnt_qposadr[m.joint(n).id] for n in names}
    is_arm_body = np.array([any(k in m.body(b).name for k in ARM) for b in range(m.nbody)])
    rng = np.random.default_rng(0)

    def touches(sample):
        mujoco.mj_resetData(m, d)
        d.qpos[2] = 1.5; d.qpos[3] = 1.0
        for n in names:
            d.qpos[qadr[n]] = default_of(n.removeprefix(P))
        for n, q in sample.items():
            d.qpos[qadr[n]] = q
        mujoco.mj_forward(m, d)
        body_hit = arm_arm = False
        for c in d.contact[: d.ncon]:
            if c.dist > -0.001:
                continue
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if b1 == 0 or b2 == 0:
                continue
            a1, a2 = is_arm_body[b1], is_arm_body[b2]
            if a1 != a2:
                body_hit = True
            elif a1 and a2 and m.body(b1).name.split("_")[0] != m.body(b2).name.split("_")[0]:
                arm_arm = True
        return body_hit, arm_arm

    soft = {}
    for n in arm:
        lo, hi = m.jnt_range[m.joint(n).id]
        c, h = (lo + hi) / 2, (hi - lo) / 2 * 0.9
        soft[n] = (c - h, c + h)

    def old_sample():
        s = {}
        for n in arm:
            dft, (lo, hi) = default_of(n.removeprefix(P)), soft[n]
            s[n] = rng.uniform(dft - 0.6 * max(dft - lo, 0), dft + 0.6 * max(hi - dft, 0))
        return s

    def new_sample():
        s = {}
        for n in arm:
            dft, (lo, hi) = default_of(n.removeprefix(P)), soft[n]
            o_lo, o_hi = offset_bounds(n.removeprefix(P), G1_ARM_OFFSETS)
            s[n] = float(np.clip(dft + rng.uniform(o_lo, o_hi), lo, hi))
        return s

    for label, fn in (("old: 60 % of soft range", old_sample), ("new: safe offsets (L = 1)", new_sample)):
        res = np.array([touches(fn()) for _ in range(a.n)])
        print(f"{label:28s} arm touches body {100 * res[:, 0].mean():5.1f} %   arms touch each other {100 * res[:, 1].mean():4.1f} %   (n={a.n})")


if __name__ == "__main__":
    main()
