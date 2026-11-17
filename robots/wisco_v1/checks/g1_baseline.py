"""Reference numbers: the Unitree G1 (which trains well with mjlab) through the same stand and
step-response tests as our robot, with mjlab's own G1 gains and KNEES_BENT pose.

Needs the G1 model from the mjlab 1.6.0 package: pass the folder holding g1.xml.
  python g1_baseline.py <path to mjlab/asset_zoo/robots/unitree_g1/xmls>
Writes results/g1_baseline.json.
"""
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent / "results" / "g1_baseline"
W = 10 * 2 * math.pi
ZETA = 2.0


def two_stage(rot, gear):
    """Reflected inertia of a two-stage gear train."""
    return rot[0] * (gear[1] * gear[2]) ** 2 + rot[1] * gear[2] ** 2 + rot[2]


A5020 = two_stage((0.139e-4, 0.017e-4, 0.169e-4), (1, 1 + 46 / 18, 1 + 56 / 16))
A7520_14 = two_stage((0.489e-4, 0.098e-4, 0.533e-4), (1, 4.5, 1 + 48 / 22))
A7520_22 = two_stage((0.489e-4, 0.109e-4, 0.738e-4), (1, 4.5, 5))
A4010 = two_stage((0.068e-4, 0, 0), (1, 5, 5))
GROUPS = [  # (regex-ish suffixes, armature, effort)   from mjlab g1_constants.py
    (("elbow_joint", "shoulder_pitch_joint", "shoulder_roll_joint", "shoulder_yaw_joint", "wrist_roll_joint"), A5020, 25.0),
    (("hip_pitch_joint", "hip_yaw_joint", "waist_yaw_joint"), A7520_14, 88.0),
    (("hip_roll_joint", "knee_joint"), A7520_22, 139.0),
    (("wrist_pitch_joint", "wrist_yaw_joint"), A4010, 5.0),
    (("waist_pitch_joint", "waist_roll_joint", "ankle_pitch_joint", "ankle_roll_joint"), 2 * A5020, 50.0),
]
POSE = {"hip_pitch_joint": -0.312, "knee_joint": 0.669, "ankle_pitch_joint": -0.363, "elbow_joint": 0.6,
        "left_shoulder_roll_joint": 0.2, "left_shoulder_pitch_joint": 0.2,
        "right_shoulder_roll_joint": -0.2, "right_shoulder_pitch_joint": 0.2}


def group(name):
    """Actuator parameters for a G1 joint, by name suffix."""
    for suf, arm, eff in GROUPS:
        if any(name.endswith(s) for s in suf):
            return arm, eff
    raise KeyError(name)


def default(name):
    """The G1 default angle of a joint."""
    if name in POSE:
        return POSE[name]
    for k, v in POSE.items():
        if not k.startswith(("left", "right")) and name.endswith(k):
            return v
    return 0.0


def build(xml_dir, fixed=False):
    """G1 MuJoCo model with mjlab's actuator settings (optionally fixed base)."""
    spec = mujoco.MjSpec.from_file(str(Path(xml_dir) / "g1.xml"))
    spec.option.timestep = 0.005
    spec.option.iterations, spec.option.ls_iterations = 10, 20
    for g in spec.geoms:
        if g.name.endswith("_collision"):
            foot = "_foot" in g.name
            g.contype = g.conaffinity = 1
            g.condim, g.priority = (3, 1) if foot else (1, 0)
            if foot:
                g.friction = [0.6, g.friction[1], g.friction[2]]
    names = [j.name for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_HINGE]
    for n in names:
        arm, eff = group(n)
        a = spec.add_actuator(name=n, target=n)
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT; a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.gainprm[0] = arm * W ** 2; a.biasprm[1] = -arm * W ** 2; a.biasprm[2] = -2 * ZETA * arm * W
        a.inheritrange = 0.0; a.ctrllimited = False; a.forcelimited = True; a.forcerange = [-eff, eff]
        spec.joint(n).armature = arm
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], condim=3)
    if fixed:
        spec.delete(spec.joint("floating_base_joint")); spec.body("pelvis").pos = [0, 0, 1.2]
    m = spec.compile()
    return m, names


def main(xml_dir):
    """Run the stand and step-response tests on the G1 and write the reference numbers."""
    m, names = build(xml_dir)
    d = mujoco.MjData(m)
    qadr = [m.jnt_qposadr[m.joint(n).id] for n in names]
    q0 = np.array([default(n) for n in names])
    d.qpos[2] = 0.76; d.qpos[3] = 1; d.qpos[qadr] = q0; d.ctrl[:] = q0
    mujoco.mj_forward(m, d)
    # drop onto the floor exactly: lower until the lowest foot geom touches
    low = min(d.geom_xpos[g][2] - m.geom_size[g][0] for g in range(m.ngeom) if "_foot" in m.geom(g).name)
    d.qpos[2] -= low - 0.002; mujoco.mj_forward(m, d); z0 = d.qpos[2]
    tilts = []
    for _ in range(500):
        d.ctrl[:] = q0
        for _ in range(4):
            mujoco.mj_step(m, d)
        R = d.xmat[m.body("pelvis").id].reshape(3, 3); tilts.append(math.degrees(math.acos(min(1, R[2, 2]))))
    sag = np.abs(d.qpos[qadr] - q0)
    tau = np.abs(d.actuator_force)
    out = {"stand_10s": {"pelvis_height_change_m": round(abs(d.qpos[2] - z0), 4), "max_tilt_deg": round(max(tilts), 2),
                         "final_tilt_deg": round(tilts[-1], 2),
                         "horizontal_drift_m": round(float(np.hypot(d.qpos[0], d.qpos[1])), 4),
                         "max_joint_sag_rad": {names[i]: round(float(sag[i]), 4) for i in np.argsort(-sag)[:4]},
                         "knee_hold_nm": round(float(tau[names.index("left_knee_joint")]), 2)},
           "step_response": {}}
    for n in ("left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_elbow_joint"):
        m2, nm = build(xml_dir, fixed=True); d2 = mujoco.MjData(m2)
        qa = [m2.jnt_qposadr[m2.joint(x).id] for x in nm]; d2.qpos[qa] = q0; d2.ctrl[:] = q0
        for _ in range(100):
            mujoco.mj_step(m2, d2)
        i = nm.index(n); start = d2.qpos[qa[i]]; tgt = q0.copy(); tgt[i] += 0.2
        qs = []
        for _ in range(200):
            d2.ctrl[:] = tgt; mujoco.mj_step(m2, d2); qs.append(d2.qpos[qa[i]] - start)
        qs = np.array(qs); final = qs[-1]
        outside = np.where(np.abs(qs - final) > 0.05 * abs(final))[0]
        arm, eff = group(n)
        out["step_response"][n] = {"kp": round(arm * W ** 2, 1), "reached_frac": round(float(final / 0.2), 3),
                                   "overshoot_frac": round(float(max(0, (qs.max() - final) / final)), 3),
                                   "settle_s": round(float((outside[-1] + 1) * 0.005) if len(outside) else 0.0, 3)}
    (OUT / "g1_baseline.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
