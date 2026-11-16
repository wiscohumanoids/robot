"""Update the Wisco V1 robot description in place from the numbers file, so the URDF, the MuJoCo scene and the
joint report always carry the values training uses.

Source and output are the same files (robots/wisco_v1/robot.urdf, robot.xml, model_report.json): geometry, links,
inertias and joint axes come from the CAD export and are never changed here; everything numeric that has a better
source is rewritten from robots/wisco_v1/model/robot_numbers.yaml (via robot_params.py). Running it twice gives the
same files.

What it sets:
  - every revolute joint: effort (peak torque at the joint), velocity (no-load speed at the joint), friction
    (datasheet backdrive torque) and damping (datasheet-bounded viscous loss); ankle and waist values are the
    two-motor linkage equivalents at the zero pose
  - MJCF joints: armature (rotor inertia reflected through the gearbox); motor ctrlrange = peak torque
  - ankle pitch lower limit: the reachable limit from the linkage analysis, with margin
  - IMU frame pose and orientation, head-camera tilt, and a ROS optical frame for the camera
  - URDF: one <transmission> per joint (gear ratio, motor model, planned driver control mode)
  - MJCF: the "home" keyframe, the knees-bent pose the walking policies start from
  - link masses from robot_numbers.yaml masses.overrides (inertia scaled with the mass)

Run from this folder:  python sync_robot_folder.py   (then build_training_model.py and the checks in ../checks/)
"""
import json
import math
import sys
import xml.etree.ElementTree as ET

import robot_params as P

sys.dont_write_bytecode = True
URDF_PATH, XML_PATH, REPORT_PATH = P.ROBOT_DIR / "robot.urdf", P.ROBOT_DIR / "robot.xml", P.ROBOT_DIR / "model_report.json"

# IMU (EVAL_HI05 board on the IMU mount at the back of the pelvis) and head camera poses, from robot_numbers.yaml
_S = P.sensors()
IMU_POS = tuple(_S["imu"]["pos_m"])
IMU_RPY = tuple(_S["imu"]["rpy"])
IMU_QUAT_WXYZ = tuple(_S["imu"]["quat_wxyz"])
CAMERA_TILT = float(_S["head_camera"].get("tilt_rad") or 0.0)
# ROS optical frame convention (REP 103): z forward, x right, y down
OPTICAL_RPY = (-math.pi / 2, 0.0, -math.pi / 2)

# Planned motor-driver control mode per joint group, recorded in the transmissions and the report
LEG_WAIST_MODE = "EtherCAT mode 5 (PVT/MIT impedance): position target + kp/kd per cycle, PD on the driver"
ARM_MODE = "EtherCAT CSP + torque offset or CST: arm controller (gravity compensation + PD)"
HEAD_MODE = "EtherCAT mode 5 (PVT/MIT impedance)"


def joint_values():
    """Per-joint values for all 31 joints: name -> dict(effort, rated, velocity, friction, damping, armature,
    lower, upper, motor, mode, kp, kd). The two head joints are welded in training and use the head gains."""
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
    """Numbers as written into the model files: 6 significant digits."""
    return f"{x:.6g}"


def _scale_mass(inertial_mass_get, inertial_mass_set, inertia_values, set_inertia, override):
    """Apply one mass override: set the mass and scale the inertia by the same ratio (idempotent: the ratio is
    1 once the mass already matches)."""
    f = override["mass_kg"] / float(inertial_mass_get())
    inertial_mass_set(fmt(override["mass_kg"]))
    set_inertia([fmt(float(x) * f) for x in inertia_values])


def sync_urdf(J):
    """Rewrite joint limits/dynamics, sensor frames, mass overrides and transmissions in robot.urdf.
    Returns the number of joints updated."""
    tree = ET.parse(URDF_PATH)   # comments are dropped on parse and re-added below, so reruns don't stack them
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

    # Camera optical frame: created once, after the head camera joint
    head_cam_joint = root.find("joint[@name='head_camera_fixed']")
    idx = list(root).index(head_cam_joint) + 1
    if root.find("link[@name='head_camera_optical']") is None:
        opt_link = ET.Element("link", name="head_camera_optical")
        opt_joint = ET.Element("joint", name="head_camera_optical_fixed", type="fixed")
        ET.SubElement(opt_joint, "parent", link="head_camera")
        ET.SubElement(opt_joint, "child", link="head_camera_optical")
        ET.SubElement(opt_joint, "origin", xyz="0 0 0", rpy=" ".join(f"{a:.10f}" for a in OPTICAL_RPY))
        root.insert(idx, opt_link)
        root.insert(idx + 1, opt_joint)
    root.insert(idx, ET.Comment(" ZED X stereo camera; mount tilt from robot_numbers.yaml sensors (assumed level until "
                                "measured). head_camera: x forward, y left, z up. head_camera_optical: ROS optical "
                                "convention (z forward, x right, y down). "))
    imu_joint = root.find("joint[@name='imu_link_fixed']")
    root.insert(list(root).index(imu_joint), ET.Comment(
        " IMU: EVAL_HI05 board on the IMU mount, back of the pelvis, chip facing backward. Frame axes = board axes: "
        "x -> robot right, y -> down, z -> forward. "))

    for link, o in P.load_mass_overrides().items():
        inert = root.find(f"link[@name='{link}']/inertial")
        m, inertia = inert.find("mass"), inert.find("inertia")
        keys = ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
        _scale_mass(lambda: m.get("value"), lambda s: m.set("value", s), [inertia.get(k) for k in keys],
                    lambda vals: [inertia.set(k, s) for k, s in zip(keys, vals)], o)
        if "com" in o:
            inert.find("origin").set("xyz", " ".join(fmt(x) for x in o["com"]))

    # Transmissions are regenerated from scratch at the end of the file
    for t in root.findall("transmission"):
        root.remove(t)
    order = [j.get("name") for j in root.findall("joint") if j.get("name") in J]
    root.append(ET.Comment(" Transmissions: one MyActuator RMD-X motor per joint, except the ankles (2x X4-36) and waist "
                           "roll/pitch (2x X6-60), which are rod linkages: their joint effort/velocity above are linkage "
                           "equivalents at the zero pose; per-motor conversion: robots/wisco_v1/model/ankle_motor.py "
                           "(ankles), robots/wisco_v1/linkage.py (both). "))
    for name in order:
        v = J[name]
        base = v["motor"].replace("2x", "")
        t = ET.SubElement(root, "transmission", name=f"{name}_transmission")
        ET.SubElement(t, "type").text = "transmission_interface/SimpleTransmission"
        jt = ET.SubElement(t, "joint", name=name)
        ET.SubElement(jt, "hardwareInterface").text = "hardware_interface/EffortJointInterface"
        act = ET.SubElement(t, "actuator", name=f"{name}_motor")
        ET.SubElement(act, "mechanicalReduction").text = str(specs[base]["gear_ratio"]["value"])
        t.append(ET.Comment(f" MyActuator RMD-{v['motor']} | control: {v['mode']} "))

    root.insert(0, ET.Comment(
        " Wisco V1. Geometry, links and inertias from the CAD export; joint limits and dynamics, sensor frames and "
        "transmissions written by robots/wisco_v1/model/sync_robot_folder.py from robots/wisco_v1/model/robot_numbers.yaml "
        "(MyActuator datasheets and measurements). Change the numbers file and rerun; don't hand-edit. "
        "Units: m, kg, rad, N*m, rad/s. "))
    ET.indent(tree, space="  ")
    tree.write(URDF_PATH, encoding="utf-8", xml_declaration=True)
    return len(order)


def sync_mjcf(J, home_pelvis_z):
    """Rewrite joint dynamics, the IMU site, motor ranges, mass overrides and the home keyframe in robot.xml.
    Returns the number of joints updated."""
    tree = ET.parse(XML_PATH)
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
            inert = b.find("inertial")
            _scale_mass(lambda: inert.get("mass"), lambda s: inert.set("mass", s), inert.get("fullinertia").split(),
                        lambda vals: inert.set("fullinertia", " ".join(vals)), o)
            if "com" in o:
                inert.set("pos", " ".join(fmt(x) for x in o["com"]))

    # Home keyframe: the knees-bent pose the walking policies start from (replaced on every run)
    kf = root.find("keyframe")
    for key in kf.findall("key[@name='home']"):
        kf.remove(key)
    joint_order = [j.get("name") for j in root.iter("joint") if j.get("name") in J]
    qpos = [0.0, 0.0, home_pelvis_z, 1.0, 0.0, 0.0, 0.0] + [P.DEFAULT_POSE.get(nm, 0.0) for nm in joint_order]
    ET.SubElement(kf, "key", name="home", qpos=" ".join(fmt(x) for x in qpos))
    root.insert(0, ET.Comment(
        " Wisco V1 MuJoCo scene. Joint dynamics, IMU site, motor ranges and the home keyframe written by "
        "robots/wisco_v1/model/sync_robot_folder.py from robots/wisco_v1/model/robot_numbers.yaml; don't hand-edit. "
        "Keyframes: zero, stand, home (knees-bent walking start pose). The 'imu' site has the real board orientation "
        "(sensors read in board axes). "))
    ET.indent(tree, space="  ")
    tree.write(XML_PATH, encoding="utf-8", xml_declaration=False)
    return n


def sync_report(J):
    """Rewrite the per-joint values in model_report.json (limits, dynamics, gains, control mode)."""
    rep = json.loads(REPORT_PATH.read_text())
    for name, v in J.items():
        r = rep["joints"][name]
        r.update(effort=v["effort"], rated=v["rated"], vel=v["velocity"], fric=v["friction"],
                 damp=v["damping"], arm=v["armature"], control_mode=v["mode"], kp=v["kp"], kd=v["kd"])
        if v["lower"] is not None:
            r["lower"] = v["lower"]
    rep["_updated"] = ("Joint effort/rated/vel/fric/damp/arm from MyActuator datasheets (robots/wisco_v1/model/robot_numbers.yaml), "
                       "kp/kd from its control section, ankle pitch lower limit from the linkage reach analysis, control_mode "
                       "per joint group. Written by robots/wisco_v1/model/sync_robot_folder.py.")
    REPORT_PATH.write_text(json.dumps(rep, indent=1) + "\n")


def main():
    """Update all three description files and print a one-line summary of each."""
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
