"""Build a clean MJCF for the Panthera-HT arm from the vendor URDF.

MuJoCo can import the URDF directly, but the result is not something you want to
maintain: it dedups meshes by basename, so the collision geoms end up pointing at
the high-poly *visual* STLs, and it has no actuators, no end-effector site and no
rest pose. This takes the imported kinematics/inertials verbatim -- those are the
part worth trusting, straight from the manufacturer's CAD -- and rewrites the
presentation around them.

Writes panthera/panthera.xml (robot) and panthera/scene.xml (robot + floor, light,
table and cubes to pick up; `--no-objects` leaves the scene bare),
the same split mujoco_menagerie uses, so the robot can be included in other scenes
without dragging a floor along.

Run sim/prep_meshes.py first.
"""
import argparse
import math
import pathlib
import xml.etree.ElementTree as ET

import mujoco

HERE = pathlib.Path(__file__).parent
ROBOT = HERE / "panthera"
URDF = ROBOT / "panthera_arm.urdf"

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

# Position-servo gains, scaled to each joint's share of the load.
#
# These are stiff on purpose. The arm's real torque ceiling is enforced by
# `forcerange` (the manufacturer's peak joint torque), not by soft gains, so
# a stiff loop plus a hard force limit models a high-ratio planetary reducer
# better than a soft loop does -- the reducer really is that stiff, and what
# actually stops the joint is running out of torque.
#
# Measured on a 0.5 Hz, 100 mm end-effector trajectory: peak demand reaches
# 20.9 of joint2's 36 Nm with no saturation on any joint, static droop is
# 0.05 deg, and residual velocity after settling is ~1e-11 rad/s (no ringing).
# Softer gains are not free: at a fifth of these the end-effector lagged its
# commanded target by 22 mm, which lands in recorded data as a robot that
# never went where the operator pointed.
GAINS = {
    "joint1": (3000.0, 54.0), "joint2": (5000.0, 90.0), "joint3": (4000.0, 68.0),
    "joint4": (1200.0, 23.0), "joint5": (800.0, 14.0),  "joint6": (800.0, 14.0),
}
GRIPPER_GAIN = (400.0, 20.0)
# Single actuator force shared by the coupled jaws: 3 N produces about 1.5 N
# per pad. Physics-rate interpolation prevents large acceleration bursts from
# stepped position targets, so physical grasps work without increasing force.
GRIPPER_FORCE = 3.0

# Elbow-up rest pose; joint2/joint3 have one-sided ranges so 0 is a hard stop.
HOME = [0.0, 1.2, 1.4, 0.0, 0.0, 0.0]

# Distance from link6's origin to the grasp point, along +x.
#
# This is the manufacturer's own TCP: the URDF puts `gripper_center_joint` here.
# Measured against the finger meshes, they span x in [0.065, 0.170] in link6's
# frame, so 0.165 sits just inside the fingertips -- where an object actually
# ends up when the jaws close. Do not be tempted to add the finger length on top
# of the mount offset; that was worth 60 mm of phantom reach, and since this site
# is both the IK target and the pose logged into every episode, an error here is
# a silent bias in the whole dataset.
GRIP_SITE_X = 0.165

# Cameras.
#
# Neither is a sensor: nothing reads pixels off them, and they add no bodies,
# no dofs and no contacts. They exist so the teleop view can be switched to a
# frame that answers a question the fixed over-the-shoulder view cannot --
# "are the jaws lined up with the cube", and "where is everything on the
# table" -- and they live in the model rather than in the teleop script so
# that every consumer of the scene gets the same two viewpoints.
#
# The wrist camera rides link6, above and behind the jaws. It is *not* aimed
# along the gripper's approach axis (+x in link6): mounted above that axis and
# pointed parallel to it, the grasp point lands low in the frame and whatever
# is being picked up slides off the bottom edge on the way in. The tilt is
# derived instead of chosen, so that the camera looks straight at the TCP --
# the same point the IK drives and the dataset logs -- which puts the target
# in the middle of the frame with the jaws framing it either side, and keeps
# the two agreeing if the mount ever moves. The fovy is wide, as a wrist
# camera's is: at 70 degrees the full span of the open jaws stays in frame
# from 5 cm away.
WRIST_CAM_POS = (0.055, 0.0, 0.055)
WRIST_CAM_TILT = math.atan2(WRIST_CAM_POS[2], GRIP_SITE_X - WRIST_CAM_POS[0])
WRIST_CAM_FOVY = 70.0

# Straight down over the middle of the table, with the robot's +x up the image
# so the view is oriented the way the operator is sitting rather than the way
# the world frame happens to be numbered. Height and fovy are picked together
# to contain the reachable strip of table rather than the table itself: the
# vertical field is along x, and 2 * (h - TABLE_TOP) * tan(fovy/2) = 0.79 m
# spans x in [0.01, 0.79], which is the teleop region's 0.6 m with room either
# side. The long side of the table runs across the image, where the 4:3 aspect
# gives it 1.05 m to sit in.
OVERHEAD_CAM_HEIGHT = 0.90
OVERHEAD_CAM_FOVY = 50.0

# Opacity of the arm's visual shell.
#
# During teleoperation the arm sits between the shoulder view and the table, and the
# links are exactly what the operator needs to see *through* to line the jaws up
# with a cube. Ghosting the structural links keeps the arm legible as a pose
# without letting it occlude the workspace. The gripper stays opaque: it is the
# part being aimed, and it is small enough not to hide anything.
LINK_ALPHA = 0.12
OPAQUE_MESHES = {"gripper_center", "L_finger", "R_finger"}

# Table and manipulanda, for grasping demonstrations.
#
# The table top sits 10 mm *below* the floor of the teleoperated workspace box
# (the useful low workspace is about z=[0.06, 0.30]), which is the reason
# these two numbers are chosen together. The operator cannot command the TCP
# below 0.06, and the fingertips reach about 5 mm past it along a 30-degree
# approach, so bottoming out leaves the jaws just clear of the surface: every
# low target the operator can ask for is one the arm can actually hold. Level with the
# box floor instead, and bottoming out -- which happens constantly, it is the
# bottom of the operator's range of motion -- grinds a fingertip into the table
# at 70 N, with the servo fighting the contact for 7 mm of tracking error.
# Move the table and you have to move `--region` with it.
TABLE_TOP = 0.05
TABLE_HALF = (0.22, 0.32)      # x, y half-extents
TABLE_CENTER_X = 0.40
TABLE_THICK = 0.02

# 45 mm cubes: the jaws span 80 mm fully open, so there is ~17 mm of clearance
# either side to be sloppy with, and a cube that small still fits between the
# fingertips when they close. Light enough (45 g) that the gripper's force is
# never the limiting factor, heavy enough not to be flicked away on contact.
CUBE_HALF = 0.0225
CUBE_DENSITY = 500.0
# Equal-priority geom contacts use the per-axis maximum. Explicit pad-cube
# pairs below override these coefficients for pinches; table contacts do not.
CUBE_FRICTION = "0.8 0.08 0.004"
CUBES = [
    ("cube_red",   (0.38, 0.12), (0.85, 0.20, 0.18, 1)),
    ("cube_green", (0.45, 0.00), (0.25, 0.70, 0.30, 1)),
    ("cube_blue",  (0.36, -0.13), (0.20, 0.40, 0.85, 1)),
]

# Parallel-jaw pads. The finger collision meshes are convex hulls of a concave
# jaw, so the inner face that should pinch a cube is a sloped fat surface
# instead -- at mid-finger the hull sits 23 mm outside the visual pad. A cube
# on that slope slides the moment the arm lifts. These boxes sit on the real
# inner face (y = 0 in each finger frame) and do the grasping.
#
# Size is a compromise against the table. The workspace floor is z = 0.06 and
# the table is at 0.05; on the 30-degree approach the fingertip is the lowest
# point, so a pad that reaches the tip *and* is taller than ~15 mm digs into
# the table, the servo fights that contact, and the jaws never actually close
# on the cube. End the pad at the TCP and grow it back into the jaw -- that
# direction is *up* on a 30-degree approach, so it stays clear and still
# catches a cube that sits deep in the fingers.
PAD_SIZE = (0.0225, 0.002, 0.010)      # 45 x 4 x 20 mm
PAD_POS_X = -0.0225                    # x in [-0.045, 0], ending at the TCP
PAD_POS_Y = 0.0015                     # 0.5 mm proud of the visual face
PAD_FRICTION = "2.0 0.2 0.01"
# Proximal outer bulk, well behind the pads. Must not reach the fingertip:
# that is what hit the table and held the jaws open.
BULK_SIZE = (0.035, 0.015, 0.015)
BULK_POS_X = -0.058                    # x in [-0.093, -0.023]


def indent(elem, level=0):
    pad = "\n" + "  " * level
    if len(elem):
        if not (elem.text or "").strip():
            elem.text = pad + "  "
        for child in elem:
            indent(child, level + 1)
        if not (child.tail or "").strip():
            child.tail = pad
    if level and not (elem.tail or "").strip():
        elem.tail = pad


def sub(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def add_objects(wb):
    """Table and loose cubes, appended to the scene's worldbody.

    These live in scene.xml rather than panthera.xml so the robot stays
    importable on its own -- same reason the floor does.

    The cubes are free bodies, so they add 7 qpos each *after* the arm's. That
    is safe for everything that indexes the arm, because those indices come
    from `jnt_qposadr` by name, not from a fixed slice. It also means the
    `home` keyframe -- defined in panthera.xml, where the cubes do not exist --
    is shorter than nq; MuJoCo pads the remainder with the identity pose, not
    these body poses. `PantheraSim.reset()` writes the free joints afterwards,
    scattering them on the table -- the positions here are only the fallback
    layout when randomisation is off.
    """
    tx, (hx, hy) = TABLE_CENTER_X, TABLE_HALF
    top, th = TABLE_TOP, TABLE_THICK
    t = sub(wb, "body", name="table", pos=f"{tx} 0 {top - th / 2:.4f}")
    g = sub(t, "geom", name="table_top", type="box")
    g.set("size", f"{hx} {hy} {th / 2:.4f}")
    g.set("material", "table")
    # condim 4 (torsional friction) so a cube set down on the table settles
    # instead of spinning on a frictionless point contact. Keep this below
    # the cube and pad: with elliptic cones and a high impratio the table
    # otherwise out-sticks the grasp and the jaws slide up the cube.
    g.set("condim", "4"); g.set("friction", "0.3 0.02 0.0005")
    leg_h = (top - th) / 2
    if leg_h > 0:
        for sx in (-1, 1):
            for sy in (-1, 1):
                g = sub(t, "geom", name=f"table_leg_{'p' if sx > 0 else 'm'}"
                                        f"{'p' if sy > 0 else 'm'}",
                        type="box")
                g.set("size", f"0.015 0.015 {leg_h:.4f}")
                g.set("pos", f"{sx * (hx - 0.03):.4f} {sy * (hy - 0.03):.4f} "
                             f"{-(th / 2 + leg_h):.4f}")
                g.set("material", "table")
                # Legs are scenery only. They sit under the top, well outside
                # anything the arm can reach, and making them collidable would
                # only give the solver contacts to worry about.
                g.set("contype", "0"); g.set("conaffinity", "0")

    h = CUBE_HALF
    for name, (x, y), rgba in CUBES:
        # Resting exactly on the table: half a side above the top surface.
        b = sub(wb, "body", name=name, pos=f"{x} {y} {top + h:.4f}")
        sub(b, "freejoint", name=name)
        g = sub(b, "geom", name=name, type="box")
        g.set("size", f"{h} {h} {h}")
        g.set("rgba", " ".join(str(v) for v in rgba))
        g.set("density", str(CUBE_DENSITY))
        # condim 6 gives sliding + rolling + torsion, so a cube pinched
        # between two flat faces cannot spin about the pinch axis or roll
        # out when the wrist pitches. Contact friction is the per-axis
        # maximum of equal-priority geoms, unless an explicit pair overrides it.
        g.set("condim", "6"); g.set("friction", CUBE_FRICTION)
        g.set("solimp", "0.98 0.99 0.001"); g.set("solref", "0.005 1")


def add_grasp_contacts(scene):
    """Give finger pads rubber-like friction without making the table sticky.

    Explicit pairs set the actual pad-cube friction. Inactive welds remain only
    for replaying weld-v1 recordings; contact-v2 never activates them.
    """
    eqs = sub(scene, "equality")
    for name, _, _ in CUBES:
        w = sub(eqs, "weld", name=f"pad_grasp_{name}")
        w.set("body1", "link6")
        w.set("body2", name)
        w.set("active", "false")
        # Compliance avoids an infinitely rigid wrist/object connection while
        # remaining stiff enough to represent a broad, deformed rubber patch.
        w.set("solref", "0.015 1")
        w.set("solimp", "0.90 0.98 0.002")

    ctc = sub(scene, "contact")
    for name, _, _ in CUBES:
        for pad in ("L_finger_pad", "R_finger_pad"):
            p = sub(ctc, "pair", geom1=pad, geom2=name)
            p.set("condim", "6")
            p.set("friction", "0.8 0.8 0.01 0.001 0.001")
            p.set("solref", "0.005 1")
            p.set("solimp", "0.98 0.99 0.001")


def build(objects: bool = True):
    # 1. Import the URDF and dump MuJoCo's own MJCF; this resolves the mesh
    #    inertias, axis conventions and the mimic->equality translation for us.
    model = mujoco.MjModel.from_xml_path(str(URDF))
    raw = ROBOT / ".panthera_raw.xml"
    mujoco.mj_saveLastXML(str(raw), model)
    tree = ET.parse(raw)
    root = tree.getroot()
    raw.unlink()

    root.set("model", "panthera_ht")
    compiler = root.find("compiler")
    compiler.set("meshdir", ".")
    compiler.set("autolimits", "true")

    opt = ET.Element("option")
    opt.set("timestep", "0.002")
    opt.set("integrator", "implicitfast")
    # Elliptic cones plus higher friction impedance suppress slow numerical
    # creep. Command interpolation addresses dynamic slip; no weld is needed.
    opt.set("cone", "elliptic")
    opt.set("impratio", "100")
    root.insert(1, opt)

    # 2. Default classes. Visual geoms never collide; collision geoms are never
    #    drawn (group 3 is hidden by default in the viewer).
    dflt = ET.Element("default")
    root.insert(2, dflt)
    jd = sub(dflt, "joint")
    jd.set("damping", "0.2")
    jd.set("armature", "0.05")
    jd.set("frictionloss", "0.1")
    gd = sub(dflt, "general")
    gd.set("biastype", "affine")

    vis = sub(dflt, "default", **{"class": "visual"})
    g = sub(vis, "geom")
    g.set("type", "mesh"); g.set("group", "2")
    g.set("contype", "0"); g.set("conaffinity", "0"); g.set("density", "0")

    col = sub(dflt, "default", **{"class": "collision"})
    g = sub(col, "geom")
    g.set("type", "mesh"); g.set("group", "3")
    g.set("contype", "1"); g.set("conaffinity", "1")
    g.set("condim", "4"); g.set("friction", "1.0 0.05 0.001")

    # 3. Register a second, convex-hull mesh asset per link for collision.
    asset = root.find("asset")
    names = [m.get("name") for m in asset.findall("mesh")]
    for name in names:
        m = sub(asset, "mesh", name=f"{name}_col")
        m.set("content_type", "model/stl")
        m.set("file", f"meshes/collision/{name}.STL")

    # 4. Retag every geom into one of the two classes. The importer marks visual
    #    geoms with density="0"; that is what distinguishes the pair.
    for body in [root.find("worldbody")] + root.find("worldbody").findall(".//body"):
        for geom in body.findall("geom"):
            mesh = geom.get("mesh")
            is_visual = geom.get("density") == "0"
            rgba = geom.get("rgba")
            pos, quat = geom.get("pos"), geom.get("quat")
            geom.clear()
            geom.set("class", "visual" if is_visual else "collision")
            geom.set("mesh", mesh if is_visual else f"{mesh}_col")
            if is_visual and rgba:
                if mesh not in OPAQUE_MESHES:
                    r, g_, b, _ = (float(v) for v in rgba.split())
                    rgba = f"{r:g} {g_:g} {b:g} {LINK_ALPHA:g}"
                geom.set("rgba", rgba)
            if pos:
                geom.set("pos", pos)
            if quat and quat != "1 0 0 0":
                geom.set("quat", quat)

    # 5. End-effector site, between the fingertips. This is the frame the
    #    retargeting drives and the frame logged in the dataset.
    link6 = root.find(".//body[@name='link6']")
    s = sub(link6, "site", name="grip_site")
    s.set("pos", f"{GRIP_SITE_X:.5f} 0 0")
    s.set("size", "0.008")
    s.set("rgba", "1 0.2 0.2 1")
    s.set("group", "1")

    # 5b. Wrist camera, on the same body as the site it looks past.
    #
    # MuJoCo cameras look down their own -z with +y up the image, so the axes
    # are built from the view direction rather than written out: tilt the
    # approach axis down about the camera's right, then read off the frame.
    a = WRIST_CAM_TILT
    right = (0.0, -1.0, 0.0)                     # link6 -y is image right
    up = (math.sin(a), 0.0, math.cos(a))         # -z x right, tilted with it
    c = sub(link6, "camera", name="wrist")
    c.set("pos", " ".join(f"{v:g}" for v in WRIST_CAM_POS))
    c.set("xyaxes", " ".join(f"{v:.5f}" for v in right + up))
    c.set("fovy", f"{WRIST_CAM_FOVY:g}")

    # Fingertip pads get their own sites for grasp checks.
    for finger, y_sign in (("L_finger", 1.0), ("R_finger", -1.0)):
        b = root.find(f".//body[@name='{finger}']")
        s = sub(b, "site", name=f"{finger}_tip")
        # +x tip of the jaw, on its inner face (the body slides in y).
        s.set("pos", "0.005 0 0")
        s.set("size", "0.005")
        s.set("group", "4")
        # Hull stays as a visual collision reference but must not contact:
        # its inner face is the slope that drops cubes. A flat box on the
        # real pad does the grasping; a second box covers the outer bulk.
        for geom in b.findall("geom"):
            if geom.get("class") == "collision":
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
        pad = sub(b, "geom", name=f"{finger}_pad", type="box")
        sx, sy, sz = PAD_SIZE
        pad.set("size", f"{sx} {sy} {sz}")
        pad.set("pos", f"{PAD_POS_X} {y_sign * PAD_POS_Y:.4f} 0")
        pad.set("group", "3")
        pad.set("condim", "6")
        pad.set("friction", PAD_FRICTION)
        pad.set("solref", "0.005 1")
        pad.set("solimp", "0.98 0.99 0.001")
        bx, by, bz = BULK_SIZE
        bulk = sub(b, "geom", name=f"{finger}_bulk", type="box")
        bulk.set("size", f"{bx} {by} {bz}")
        bulk.set("pos", f"{BULK_POS_X} {y_sign * 0.019:.4f} 0")
        bulk.set("group", "3")
        bulk.set("condim", "4")
        bulk.set("friction", "1.0 0.05 0.001")

    # Contact exclusions.
    #
    # MuJoCo already filters contacts between a body and its parent -- but NOT
    # when that parent is the world, since the world is usually the floor. The
    # URDF's base_link is fixed to the world, so it gets merged into it, and its
    # convex hull then overlaps link1's and grinds against it: a persistent
    # 4-point contact whose friction fights joint1 for several degrees. Exclude
    # it explicitly.
    ctc = ET.SubElement(root, "contact")
    sub(ctc, "exclude", body1="world", body2="link1")
    # Fingers must not collide with each other or with the palm they slide in.
    sub(ctc, "exclude", body1="L_finger", body2="R_finger")
    for finger in ("L_finger", "R_finger"):
        sub(ctc, "exclude", body1="link6", body2=finger)

    # 6. Actuators: one position servo per joint, force-limited to the
    #    manufacturer's peak joint torque.
    act = ET.SubElement(root, "actuator")
    for i, jname in enumerate(ARM_JOINTS):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        lo, hi = model.jnt_range[jid]
        frc = model.jnt_actfrcrange[jid][1]
        kp, kv = GAINS[jname]
        a = sub(act, "position", name=jname, joint=jname)
        a.set("kp", str(kp)); a.set("kv", str(kv))
        a.set("ctrlrange", f"{lo:.4f} {hi:.4f}")
        a.set("forcerange", f"{-frc:.1f} {frc:.1f}")

    kp, kv = GRIPPER_GAIN
    a = sub(act, "position", name="gripper", joint="L_finger_joint")
    a.set("kp", str(kp)); a.set("kv", str(kv))
    a.set("ctrlrange", "0 0.04")
    a.set("forcerange", f"{-GRIPPER_FORCE:g} {GRIPPER_FORCE:g}")

    # 7. Rest pose. nq includes the two finger slides; they start open.
    key = ET.SubElement(root, "keyframe")
    qpos = HOME + [0.04, -0.04]
    k = sub(key, "key", name="home")
    k.set("qpos", " ".join(f"{v:g}" for v in qpos))
    k.set("ctrl", " ".join(f"{v:g}" for v in HOME + [0.04]))

    indent(root)
    out = ROBOT / "panthera.xml"
    tree.write(out, encoding="utf-8", xml_declaration=False)
    print(f"wrote {out}")

    # 8. Scene wrapper.
    scene = ET.Element("mujoco", model="panthera_scene")
    sub(scene, "include", file="panthera.xml")
    st = sub(scene, "statistic")
    st.set("center", "0.3 0 0.3"); st.set("extent", "1.2")
    v = sub(scene, "visual")
    hz = sub(v, "headlight")
    hz.set("diffuse", "0.6 0.6 0.6"); hz.set("ambient", "0.3 0.3 0.3")
    hz.set("specular", "0 0 0")
    sub(v, "rgba").set("haze", "0.15 0.25 0.35 1")
    gl = sub(v, "global")
    gl.set("azimuth", "14"); gl.set("elevation", "-34")
    gl.set("offwidth", "1920"); gl.set("offheight", "1080")

    a = sub(scene, "asset")
    t = sub(a, "texture", type="skybox", builtin="gradient")
    t.set("rgb1", "0.3 0.5 0.7"); t.set("rgb2", "0 0 0")
    t.set("width", "512"); t.set("height", "3072")
    t = sub(a, "texture", type="2d", name="groundplane", builtin="checker")
    t.set("mark", "edge"); t.set("rgb1", "0.2 0.3 0.4")
    t.set("rgb2", "0.1 0.2 0.3"); t.set("markrgb", "0.8 0.8 0.8")
    t.set("width", "300"); t.set("height", "300")
    m = sub(a, "material", name="groundplane", texture="groundplane")
    m.set("texuniform", "true"); m.set("texrepeat", "5 5")
    m.set("reflectance", "0.1")
    if objects:
        m = sub(a, "material", name="table")
        m.set("rgba", "0.75 0.68 0.55 1"); m.set("reflectance", "0.05")

    wb = sub(scene, "worldbody")
    li = sub(wb, "light", pos="0 0 2")
    li.set("dir", "0 0 -1"); li.set("directional", "true")
    gp = sub(wb, "geom", name="floor", type="plane")
    gp.set("size", "0 0 0.05"); gp.set("material", "groundplane")
    # Robot +x up the image, so the view matches the operator's own sense of
    # forward; that makes image right the robot's -y.
    c = sub(wb, "camera", name="overhead")
    c.set("pos", f"{TABLE_CENTER_X:g} 0 {OVERHEAD_CAM_HEIGHT:g}")
    c.set("xyaxes", "0 -1 0 1 0 0")
    c.set("fovy", f"{OVERHEAD_CAM_FOVY:g}")
    if objects:
        add_objects(wb)
        add_grasp_contacts(scene)

    indent(scene)
    out = ROBOT / "scene.xml"
    ET.ElementTree(scene).write(out, encoding="utf-8", xml_declaration=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-objects", dest="objects", action="store_false",
                    help="bare scene: floor and arm only, no table or cubes")
    build(**vars(ap.parse_args()))
