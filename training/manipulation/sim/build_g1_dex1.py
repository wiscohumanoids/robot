"""Build the G1 + Dex1-1 manipulation robot MJCF from Unitree's official URDF.

Source (BSD-3, unitree_ros commit 725b59b, 2026-11-09):
  assets/robots/g1_dex1/source/g1_29dof_mode_15_with_dex1_1.urdf + source/meshes/
Output:
  assets/robots/g1_dex1/g1_dex1.xml   (meshes referenced from source/meshes)

What this adds on top of the URDF (standing, fixed-pelvis tabletop manipulation, v1-v3):
  - legs + waist welded at the zero pose (joints removed); pelvis is a fixed body the scene places
  - arms (2 x 7 joints): position actuators, gravity-compensated bodies, armature/damping for stable tracking
  - Dex1-1: the two prismatic fingers coupled by an equality constraint (real gripper: one motor,
    symmetric fingers), one position actuator per gripper (finger_joint_1), force-limited.
    Measured from the meshes: joint -0.020 = closed (5.8 mm pad gap), +0.0245 = open (94.8 mm gap).
  - finger pads: higher friction, condim 4, priority, so grasps hold in MuJoCo
  - sites: tcp_left / tcp_right (between the finger pads), ego camera at the real head D435 mount
    (d435_link: on torso, pitched 47.6 deg down), optional wrist cameras on the gripper bases
Run:  ../../../.venv/bin/python build_g1_dex1.py
"""
import os
import re
import tempfile
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[3]
ASSET = ROOT / "assets/robots/g1_dex1"
SRC_URDF = ASSET / "source/g1_29dof_mode_15_with_dex1_1.urdf"
OUT = ASSET / "g1_dex1.xml"

ARM_JOINTS = [f"{s}_{j}_joint" for s in ("left", "right") for j in
              ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
WAIST_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
GRIP_OPEN, GRIP_CLOSED = 0.0245, -0.020           # finger joint range (open / closed)
TCP_OFFSET = (0.115, 0.0, 0.0)                    # from dex1 base: middle of the finger pads (x 0.077-0.143)
# position-actuator gains (stiff tracking for data collection; bodies are gravity compensated)
ARM_KP = {"shoulder": 300.0, "elbow": 250.0, "wrist": 80.0}
ARM_KV = {"shoulder": 20.0, "elbow": 15.0, "wrist": 4.0}
ARM_FORCE = {"shoulder": 60.0, "elbow": 40.0, "wrist": 13.4}   # G1 motor_info.md (5010/5019/6010)
GRIP_KP, GRIP_FORCE = 3000.0, 40.0                 # N/m, N (Dex1-1: 5-120 N adjustable)
EGO_FOVY = 58.0                                    # D435 depth VFOV; rendered 4:3


def load_urdf_spec():
    text = SRC_URDF.read_text()
    meshdir = (ASSET / "source/meshes").resolve()
    text = re.sub(r"<mujoco>.*?</mujoco>",
                  f'<mujoco><compiler meshdir="{meshdir}" strippath="true" discardvisual="false" '
                  f'fusestatic="false" balanceinertia="true"/></mujoco>', text, flags=re.S)
    fd, tmp = tempfile.mkstemp(suffix=".urdf")
    os.write(fd, text.encode()); os.close(fd)
    try:
        spec = mujoco.MjSpec.from_file(tmp)
    finally:
        os.remove(tmp)
    return spec


def group(name):
    return "wrist" if "wrist" in name else "elbow" if "elbow" in name else "shoulder"


def build():
    spec = load_urdf_spec()
    spec.modelname = "g1_dex1"
    spec.compiler.degree = False

    # 1. weld legs and waist at the zero pose (pelvis stays fixed in the world: URDF import gives a fixed base)
    for j in list(spec.joints):
        if j.name not in ARM_JOINTS and "dex1" not in j.name:
            spec.delete(j)

    # 2. classes for visuals / collisions
    for g in spec.geoms:
        if g.contype == 0:                   # URDF visual copy
            g.group = 2
        else:                                # collision copy
            g.group = 3
            g.rgba = [0.2, 0.6, 0.9, 0.3]

    # 3. arm actuators + gravity compensation
    arm_bodies = set()
    for jn in ARM_JOINTS:
        j = spec.joint(jn)
        gname = group(jn)
        j.armature = 0.01
        j.damping = [0.5, 0, 0] if hasattr(j.damping, "__len__") else 0.5
        a = spec.add_actuator(name=jn.replace("_joint", ""), target=jn)
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.gainprm[0] = ARM_KP[gname]
        a.biasprm[1] = -ARM_KP[gname]
        a.biasprm[2] = -ARM_KV[gname]
        a.ctrllimited = True
        a.ctrlrange = j.range
        a.forcelimited = True
        a.forcerange = [-ARM_FORCE[gname], ARM_FORCE[gname]]
    for b in spec.bodies:
        if any(k in b.name for k in ("shoulder", "elbow", "wrist", "dex1")):
            b.gravcomp = 1.0
            arm_bodies.add(b.name)

    # 4. Dex1-1 grippers: couple fingers, one actuator, grasp-friendly pads, TCP + wrist camera
    for side in ("left", "right"):
        j1, j2 = f"{side}_dex1_finger_joint_1", f"{side}_dex1_finger_joint_2"
        for jn in (j1, j2):
            j = spec.joint(jn)
            j.armature = 0.001
            j.damping = [2.0, 0, 0] if hasattr(j.damping, "__len__") else 2.0
        eq = spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1=j2, name2=j1)
        eq.data[:5] = [0, 1, 0, 0, 0]       # q2 = q1 (symmetric fingers)
        eq.solref = [0.005, 1.0]
        a = spec.add_actuator(name=f"{side}_gripper", target=j1)
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.gainprm[0] = GRIP_KP
        a.biasprm[1] = -GRIP_KP
        a.biasprm[2] = -60.0
        a.ctrllimited = True
        a.ctrlrange = [GRIP_CLOSED, GRIP_OPEN]
        a.forcelimited = True
        a.forcerange = [-GRIP_FORCE, GRIP_FORCE]
        for fl in (f"{side}_dex1_finger_link_1", f"{side}_dex1_finger_link_2"):
            for g in spec.body(fl).geoms:
                if g.contype:
                    g.name = f"{fl}_pad"
                    g.condim = 6                    # sliding + torsional + rolling: a pinched cup can't pivot freely
                    g.priority = 1
                    g.friction = [1.5, 0.08, 0.004]
                    g.solimp = [0.95, 0.99, 0.001, 0.5, 2]
                    g.solref = [0.004, 1.0]
                    g.margin = 0.0
        base = spec.body(f"{side}_dex1_base_link")
        base.add_site(name=f"tcp_{side}", pos=list(TCP_OFFSET), size=[0.008, 0, 0], rgba=[1, 0.2, 0.2, 1], group=4)
        base.add_camera(name=f"wrist_{side}", pos=[0.02, 0.0, 0.06], xyaxes=[0, -1, 0, 0.6, 0, 0.8], fovy=75.0)

    # 4b. the URDF torso collision mesh is a convex hull fatter than the real chest: the upper arm (shoulder
    # links, elbow) hits it during normal reaching. Exclude those pairs; the expert planner instead keeps the elbow
    # >= 10 cm from the body centre line (real chest). Torso <-> forearm/wrist/gripper contacts stay on.
    for side in ("left", "right"):
        for link in ("shoulder_pitch_link", "shoulder_roll_link", "shoulder_yaw_link", "elbow_link"):
            spec.add_exclude(bodyname1="torso_link", bodyname2=f"{side}_{link}")

    # 5. ego camera at the real head D435: link x forward (pitched down), y left, z up
    spec.body("d435_link").add_camera(name="ego", pos=[0, 0, 0], xyaxes=[0, -1, 0, 0, 0, 1], fovy=EGO_FOVY)
    return spec


def main():
    spec = build()
    model = spec.compile()
    xml = spec.to_xml()
    meshdir = str((ASSET / "source/meshes").resolve())
    xml = xml.replace(f'meshdir="{meshdir}/"', 'meshdir="source/meshes/"').replace(f'meshdir="{meshdir}"', 'meshdir="source/meshes/"')
    assert 'meshdir="source/meshes/"' in xml, "meshdir rewrite failed"
    OUT.write_text(xml)
    m = mujoco.MjModel.from_xml_path(str(OUT))
    print(f"wrote {OUT.relative_to(ROOT)}: {m.njnt} joints, {m.nu} actuators ({[m.actuator(i).name for i in range(m.nu)]}), "
          f"mass {m.body_subtreemass[0]:.2f} kg, cameras {[m.cam(i).name for i in range(m.ncam)]}")


if __name__ == "__main__":
    main()
