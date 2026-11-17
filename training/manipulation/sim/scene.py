"""Compile a manipulation scene: RoboCasa kitchen + G1-Dex1 (fixed pelvis) + every object of a version.

Build on the Mac (needs the RoboCasa asset checkout), ship the .mjb to the VM:
    ../../../.venv/bin/python scene.py --scene kitchen_l1_s1 --version v1
    -> training/manipulation/sim/build/kitchen_l1_s1_v1.mjb + kitchen_l1_s1_v1.json (object/joint index)

Objects get free joints. Unused objects are parked on a hidden shelf far from the kitchen (out of view),
so one compiled model serves every episode; env.py moves the chosen object + target into the work area.
The robot is attached under a mocap-free "robot_base" body whose pose env.py sets per episode
(pelvis jitter), via a weld-free trick: robot_base is a plain body with its pose written to model.body_pos
before mj_forward (allowed: it has no joint).
"""
import argparse
import json
import math
import os
from pathlib import Path

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
CATALOG = HERE / "catalog.yaml"
ROBOT_XML = ROOT / "assets/robots/g1_dex1/g1_dex1.xml"
OBJ_LIB = Path(os.path.expanduser("~/hsim_third_party/robocasa/robocasa/models/assets/objects"))
BUILD = HERE / "build"
PARK = np.array([30.0, 30.0, 0.0])     # hidden shelf (top at z=0.1)


def obj_path(entry):
    """'aigen_objs/cup_0' -> <lib>/aigen_objs/cup/cup_0/model.xml"""
    coll, inst = entry.split("/")
    return OBJ_LIB / coll / inst.rsplit("_", 1)[0] / inst / "model.xml"


def resolve_objects(entries):
    """Object model paths in the RoboCasa library (raises if missing)."""
    out = []
    for e in entries:
        p = obj_path(e)
        if not p.exists():
            raise FileNotFoundError(p)
        out.append(e)
    return out


def version_objects(cat, version):
    """Objects and targets of a dataset version."""
    v = cat["versions"][version]
    objs, tgts = [], []
    for c, lst in v["objects"].items():
        if lst == "same_as_v1":
            lst = cat["versions"]["v1"]["objects"][c]
        if isinstance(lst, list) and lst and "/" in lst[0]:
            objs += [(c, e) for e in resolve_objects(lst)]
    for c, lst in v.get("targets", cat["versions"]["v1"]["targets"]).items():
        tgts += [(c, e) for e in resolve_objects(lst)]
    return objs, tgts


ROBOSUITE_ASSETS = Path(os.path.expanduser("~/hsim_third_party/robosuite/robosuite/models/assets"))


def fix_foreign_paths(child):
    """Some RoboCasa objects reference textures by absolute paths from their author's machine
    (/home/<user>/.../robosuite/models/assets/textures/x.png). Point them at our robosuite checkout."""
    for t in child.textures:
        f = t.file
        if f and f.startswith("/") and not os.path.exists(f) and "robosuite/models/assets/" in f:
            t.file = str(ROBOSUITE_ASSETS / f.split("robosuite/models/assets/", 1)[1])


def robot_pelvis_height():
    """Pelvis z so the lowest foot vertex touches the floor (legs straight, zero pose)."""
    m = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    low = np.inf
    for g in range(m.ngeom):
        if "ankle_roll" in m.body(m.geom_bodyid[g]).name and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            low = min(low, (v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min())
    return -low


def build(scene_name, version):
    """Compile a scene: kitchen, robot and every object of a version, parked off the counter."""
    cat = yaml.safe_load(CATALOG.read_text())
    sc = cat["scenes"][scene_name]
    spec = mujoco.MjSpec.from_file(str(ROOT / sc["room"]))
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.option.noslip_iterations = 3
    # free-floating RoboCasa task objects from the export (sugar cube etc.): leave them; they are away from the work area

    # robot
    z = robot_pelvis_height()
    yaw = math.radians(sc["robot_yaw_deg"])
    base = spec.worldbody.add_body(name="robot_base", pos=[*sc["robot_xy"], z],
                                   quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    robot = mujoco.MjSpec.from_file(str(ROBOT_XML))
    spec.attach(robot, prefix="g1/", frame=base.add_frame())

    # hidden parking shelf
    shelf = spec.worldbody.add_body(name="park_shelf", pos=list(PARK))
    shelf.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[6, 6, 0.1], rgba=[0.3, 0.3, 0.3, 1])

    objs, tgts = version_objects(cat, version)
    index = {"scene": scene_name, "version": version, "pelvis_z": z, "robot_xy": sc["robot_xy"],
             "robot_yaw_deg": sc["robot_yaw_deg"], "counter_top_z": sc["counter_top_z"], "work_area": sc["work_area"], "reach_area_robot": sc.get("reach_area_robot"),
             "objects": [], "targets": [], "instruction_templates": cat["versions"][version].get("instruction_templates", [])}
    for k, (role, lst) in enumerate((("objects", objs), ("targets", tgts))):
        for i, (category, entry) in enumerate(lst):
            name = f"{role[:3]}{i:02d}"
            slot = len(index["objects"]) + len(index["targets"])
            park = PARK + np.array([-5 + (slot % 10) * 1.0, -5 + (slot // 10) * 1.0, 0.35])
            b = spec.worldbody.add_body(name=name, pos=park.tolist())
            b.add_freejoint(name=f"{name}_free")
            child = mujoco.MjSpec.from_file(str(obj_path(entry)))
            fix_foreign_paths(child)
            for g in child.geoms:                       # grasp-friendly contacts on the collision copies
                if g.contype != 0 or g.conaffinity != 0:
                    if g.classname.name == "collision" or g.group == 0:
                        g.friction = [1.0, 0.01, 0.001]
                        g.condim = 4
            spec.attach(child, prefix=f"{name}/", frame=b.add_frame())
            index[role].append({"name": name, "category": category, "asset": entry,
                                "label": category.replace("_", " ")})
    model = spec.compile()
    BUILD.mkdir(exist_ok=True)
    out = BUILD / f"{scene_name}_{version}.mjb"
    mujoco.mj_saveModel(model, str(out), None)
    # bounding boxes (object frame) from the reg_bbox geoms, for placement and grasp heights
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    for role in ("objects", "targets"):
        for o in index[role]:
            gid = model.geom(f"{o['name']}/reg_bbox").id if f"{o['name']}/reg_bbox" in [model.geom(g).name for g in range(model.ngeom)] else None
            if gid is not None:
                o["half_extents"] = model.geom_size[gid].tolist()
                o["bbox_offset"] = model.geom_pos[gid].tolist()
            o["mass"] = float(model.body_subtreemass[model.body(o["name"]).id])
    (BUILD / f"{scene_name}_{version}.json").write_text(json.dumps(index, indent=1))
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.0f} MB): {len(objs)} objects, {len(tgts)} targets, "
          f"pelvis z {z:.3f}, nq {model.nq}, nu {model.nu}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="kitchen_l1_s1")
    ap.add_argument("--version", default="v1")
    a = ap.parse_args()
    build(a.scene, a.version)
