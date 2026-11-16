"""Regenerate the team-facing robot files in robots/wisco_v1/ from the original CAD export plus
our measured / datasheet numbers, so the folder always matches what training uses.

Inputs (never edited):
  robots/wisco_v1/original_from_team/robot.urdf, robot.xml, model_report.json (as received from the team)
  robots/wisco_v1/model/robot_numbers.yaml, ankle_linkage.json (via robot_params.py)
Outputs (overwritten, do not hand-edit):
  robots/wisco_v1/robot.urdf, robots/wisco_v1/robot.xml, robots/wisco_v1/model_report.json

What changes vs the CAD export:
  - every revolute joint: effort (peak torque at the joint), velocity (no-load speed at the joint),
    friction (datasheet backdrive torque), damping (datasheet-bounded viscous) from MyActuator datasheets;
    ankle/waist values are linkage equivalents at the zero pose
  - MJCF joints: armature (reflected rotor inertia) too; motor ctrlrange = peak torque
  - ankle pitch lower limit -> reachable limit with margin (HARDWARE_ISSUES H2)
  - IMU frame orientation from the Legs V2 CAD (EVAL_HI05 board; HARDWARE_ISSUES H7)
  - ZED X camera: optical frame added (ROS REP-103 convention)
  - URDF: <transmission> per joint (gear ratio, motor, control mode)
  - MJCF: "home" keyframe = the knees-bent pose the walking policies start from
  - link masses from robot_numbers.yaml masses.overrides (none yet)
  - IMU pose and camera tilt from robot_numbers.yaml sensors; per-joint kp/kd recorded in model_report.json
Run:  python sync_robot_folder.py      (then build_training_model.py, phase_a checks)
"""
import json
import math
import sys
import xml.etree.ElementTree as ET

import robot_params as P

sys.dont_write_bytecode = True
SRC = P.ROBOT_DIR / "original_from_team"
SRC_URDF, SRC_XML, SRC_REPORT = SRC / "robot.urdf", SRC / "robot.xml", SRC / "model_report.json"

# IMU (EVAL_HI05 board on the "IMU Mount", back of pelvis) and camera: robot_numbers.yaml sensors
_S = P.sensors()
IMU_POS = tuple(_S["imu"]["pos_m"])
IMU_RPY = tuple(_S["imu"]["rpy"])
IMU_QUAT_WXYZ = tuple(_S["imu"]["quat_wxyz"])
CAMERA_TILT = float(_S["head_camera"].get("tilt_rad") or 0.0)
# ROS optical frame convention: z forward, x right, y down
OPTICAL_RPY = (-math.pi / 2, 0.0, -math.pi / 2)

# Planned motor-driver control mode per joint group (decided 2026-11-10, HARDWARE_ISSUES H11)
LEG_WAIST_MODE = "EtherCAT mode 5 (PVT/MIT impedance): position target + kp/kd per cycle, PD on the driver"
ARM_MODE = "EtherCAT mode 10 (CST): torque from the ROS controls stack (gravity compensation + PD)"
HEAD_MODE = "EtherCAT mode 5 (PVT/MIT impedance)"


def joint_values():
    """name -> dict(effort, velocity, friction, damping, armature, lower, motor, mode) for all 31 joints."""
    params = P.joint_params()
    head = P.motor("X4-10")
    hg = P.load_numbers()["control"]["head_gains"]
    out = {}
    for name, p in params.items():
        mode = ARM_MODE if any(k in name for k in ("shoulder", "elbow", "forearm", "wrist")) else LEG_WAIST_MODE
        out[name] = dict(effort=p["effort"], rated=p["rated"], velocity=p["velocity"], friction=p["frictionloss"],
                         damping=p["damping"], armature=p["armature"], lower=p["lower"], upper=p["upper"],
                         motor=p["motor"], mode=mode, kp=p["kp"], kd=p["kd"])
    for name in P.FIXED_JOINTS:
        out[name] = dict(effort=head["peak"], rated=head["rated"], velocity=head["w_noload"], friction=head["friction"],
                         damping=head["damping"], armature=head["armature"], lower=None, upper=None,
                         motor="X4-10", mode=HEAD_MODE, kp=float(hg["kp"]), kd=float(hg["kd"]))
    return out


def fmt(x):
    return f"{x:.6g}"


def sync_urdf(J):
    tree = ET.parse(SRC_URDF)
    root = tree.getroot()
    specs = P.load_motor_specs()["models"]
    for j in root.findall("joint"):
        name = j.get("name")
        if name in J:
            v = J[name]
            lim = j.find("limit")
            lim.set("effort", fmt(v["effort"]))
            lim.set("velocity", fmt(v["velocity"]))
            if v["lower"] is not None:
                lim.set("lower", fmt(v["lower"]))
            dyn = j.find("dynamics")
            if dyn is None:
                dyn = ET.SubElement(j, "dynamics")
            dyn.set("friction", fmt(v["friction"]))
            dyn.set("damping", fmt(v["damping"]))
        elif name == "imu_link_fixed":
            j.find("origin").set("rpy", " ".join(f"{a:.10f}" for a in IMU_RPY))
            j.find("origin").set("xyz", " ".join(fmt(a) for a in IMU_POS))
        elif name == "head_camera_fixed" and CAMERA_TILT:
            j.find("origin").set("rpy", f"0 {CAMERA_TILT:.10f} 0")
    # IMU / camera notes, camera optical frame
    head_cam_joint = root.find("joint[@name='head_camera_fixed']")
    idx = list(root).index(head_cam_joint) + 1
    root.insert(idx, ET.Comment(" ZED X stereo camera (HARDWARE_ISSUES H8). Mount tilt not yet confirmed: assumed level. "
                                "head_camera: x forward, y left, z up. head_camera_optical: ROS optical convention (z forward, x right, y down). "))
    opt_link = ET.Element("link", name="head_camera_optical")
    opt_joint = ET.Element("joint", name="head_camera_optical_fixed", type="fixed")
    ET.SubElement(opt_joint, "parent", link="head_camera")
    ET.SubElement(opt_joint, "child", link="head_camera_optical")
    ET.SubElement(opt_joint, "origin", xyz="0 0 0", rpy=" ".join(f"{a:.10f}" for a in OPTICAL_RPY))
    root.insert(idx + 1, opt_link)
    root.insert(idx + 2, opt_joint)
    imu_joint = root.find("joint[@name='imu_link_fixed']")
    root.insert(list(root).index(imu_joint), ET.Comment(
        " IMU: EVAL_HI05 board on the 'IMU Mount', back of the pelvis, chip facing backward (Legs V2 CAD). "
        "Frame axes = board axes: x -> robot right, y -> down, z -> forward. Still to check: the board's printed axes match the CAD part. "))
    # masses
    for link, o in P.load_mass_overrides().items():
        inert = root.find(f"link[@name='{link}']/inertial")
        m = inert.find("mass"); f = o["mass_kg"] / float(m.get("value"))
        m.set("value", fmt(o["mass_kg"]))
        I = inert.find("inertia")
        for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
            I.set(k, fmt(float(I.get(k)) * f))
        if "com" in o:
            inert.find("origin").set("xyz", " ".join(fmt(x) for x in o["com"]))
    # transmissions
    order = [j.get("name") for j in root.findall("joint") if j.get("name") in J]
    root.append(ET.Comment(" Transmissions: one MyActuator RMD-X motor per joint, except the ankles (2x X4-36) and waist roll/pitch "
                           "(2x X6-60), which are rod linkages: their joint effort/velocity above are linkage equivalents at the zero pose; "
                           "per-motor conversion: robots/wisco_v1/model/ankle_motor.py (ankles), robots/wisco_v1/linkage.py (both). "))
    for name in order:
        v = J[name]
        base = v["motor"].replace("2x", "")
        t = ET.SubElement(root, "transmission", name=f"{name}_transmission")
        ET.SubElement(t, "type").text = "transmission_interface/SimpleTransmission"
        jt = ET.SubElement(t, "joint", name=name)
        iface = "hardware_interface/EffortJointInterface"
        ET.SubElement(jt, "hardwareInterface").text = iface
        act = ET.SubElement(t, "actuator", name=f"{name}_motor")
        ET.SubElement(act, "mechanicalReduction").text = str(specs[base]["gear_ratio"]["value"])
        t.append(ET.Comment(f" MyActuator RMD-{v['motor']} | control: {v['mode']} "))
    root.insert(0, ET.Comment(
        " Wisco humanoid V1. Geometry, links, inertias: CAD export (Legs V2 Full Assembly.step + FullAssmb.step), "
        "kept unchanged in robots/wisco_v1/original_from_team/robot.urdf. Joint limits/dynamics, IMU orientation, camera optical frame "
        "and transmissions: generated by robots/wisco_v1/model/sync_robot_folder.py from MyActuator datasheets "
        "(robots/wisco_v1/model/robot_numbers.yaml) and our measurements. Do not hand-edit: change the sources and rerun. "
        "Units: m, kg, rad, N*m, rad/s. See robots/wisco_v1/CHANGES.md. "))
    ET.indent(tree, space="  ")
    tree.write(P.ROBOT_DIR / "robot.urdf", encoding="utf-8", xml_declaration=True)
    return len(order)


def sync_mjcf(J, home_pelvis_z):
    tree = ET.parse(SRC_XML)
    root = tree.getroot()
    n = 0
    for j in root.iter("joint"):
        name = j.get("name")
        if name in J:
            v = J[name]
            j.set("damping", fmt(v["damping"]))
            j.set("frictionloss", fmt(v["friction"]))
            j.set("armature", fmt(v["armature"]))
            if v["lower"] is not None:
                lo, hi = j.get("range").split()
                j.set("range", f"{fmt(v['lower'])} {hi}")
            n += 1
    for s in root.iter("site"):
        if s.get("name") == "imu":
            s.set("quat", " ".join(fmt(x) for x in IMU_QUAT_WXYZ))
            s.set("pos", " ".join(fmt(x) for x in IMU_POS))
    for m in root.iter("motor"):
        e = J[m.get("joint")]["effort"]
        m.set("ctrlrange", f"{fmt(-e)} {fmt(e)}")
    for b in root.iter("body"):
        o = P.load_mass_overrides().get(b.get("name"))
        if o:
            inert = b.find("inertial"); f = o["mass_kg"] / float(inert.get("mass"))
            inert.set("mass", fmt(o["mass_kg"]))
            inert.set("fullinertia", " ".join(fmt(float(x) * f) for x in inert.get("fullinertia").split()))
            if "com" in o:
                inert.set("pos", " ".join(fmt(x) for x in o["com"]))
    # home keyframe: knees-bent pose used by the walking policies
    kf = root.find("keyframe")
    joint_order = [j.get("name") for j in root.iter("joint") if j.get("name") in J]
    qpos = [0.0, 0.0, home_pelvis_z, 1.0, 0.0, 0.0, 0.0] + [P.DEFAULT_POSE.get(nm, 0.0) for nm in joint_order]
    home = ET.SubElement(kf, "key", name="home", qpos=" ".join(fmt(x) for x in qpos))
    root.insert(0, ET.Comment(
        " Wisco humanoid V1 MuJoCo scene. Generated by robots/wisco_v1/model/sync_robot_folder.py from "
        "robots/wisco_v1/original_from_team/robot.xml + MyActuator datasheets. Do not hand-edit. Keyframes: zero, stand, "
        "home (knees-bent walking start pose). The 'imu' site has the real board orientation (sensors read in board axes). "))
    ET.indent(tree, space="  ")
    tree.write(P.ROBOT_DIR / "robot.xml", encoding="utf-8", xml_declaration=False)
    return n


def sync_report(J):
    rep = json.loads(SRC_REPORT.read_text())
    for name, v in J.items():
        r = rep["joints"][name]
        r.update(effort=v["effort"], rated=v["rated"], vel=v["velocity"], fric=v["friction"],
                 damp=v["damping"], arm=v["armature"], control_mode=v["mode"], kp=v["kp"], kd=v["kd"])
        if v["lower"] is not None:
            r["lower"] = v["lower"]
    rep["_updated"] = ("Joint effort/rated/vel/fric/damp/arm from MyActuator datasheets (robots/wisco_v1/model/robot_numbers.yaml), kp/kd from its control section, "
                       "ankle pitch lower limit from the linkage reach analysis, control_mode per joint group. "
                       "Generated by robots/wisco_v1/model/sync_robot_folder.py; original in original_from_team/.")
    (P.ROBOT_DIR / "model_report.json").write_text(json.dumps(rep, indent=1) + "\n")


def main():
    J = joint_values()
    info = json.loads((P.HERE / "train_model_info.json").read_text())
    n_u = sync_urdf(J)
    n_x = sync_mjcf(J, info["default_pelvis_height"])
    sync_report(J)
    print(f"robots/wisco_v1/robot.urdf: {n_u} joints + IMU + camera optical frame + transmissions")
    print(f"robots/wisco_v1/robot.xml: {n_x} joints + IMU site + motor ranges + home keyframe")
    print("robots/wisco_v1/model_report.json updated")


if __name__ == "__main__":
    main()
