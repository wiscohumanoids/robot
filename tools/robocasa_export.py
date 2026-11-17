"""Export a RoboCasa kitchen into our environment format (assets/envs/robocasa/<name>/).

Runs in RoboCasa's own Python environment (~/hsim_third_party/.venv-robocasa), from any
folder except ~/hsim_third_party (that folder's robocasa/ checkout shadows the package):

    ~/hsim_third_party/.venv-robocasa/bin/python tools/robocasa_export.py --layout 1 --style 1

How: RoboCasa only builds kitchens inside a task with a robot, so this builds a real task
scene with its default robot, then removes the robot and writes:
  room.xml   the kitchen (fixtures + task objects at their placed poses); asset paths point into the
             RoboCasa checkout through the $HSIM_THIRD_PARTY token, which env_build.py expands
  env.yaml   our environment definition: spawn, map bounds, named objects
Then build it like any environment:
    .venv/bin/python tools/env_build.py assets/envs/robocasa/<name>
"""

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import robocasa  # registers the kitchen tasks; also locates the asset checkout
import robosuite
import yaml

ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_TOKEN = "$HSIM_THIRD_PARTY"     # expanded by tools/env_build.py

# What an object detector would call each RoboCasa fixture class (None = structure, not an object)
FIXTURE_LABELS = {
    "counter": "counter", "drawer": "drawer", "hingecabinet": "cabinet", "singlecabinet": "cabinet",
    "panelcabinet": "cabinet", "housingcabinet": "cabinet", "opencabinet": "shelf", "sink": "sink",
    "microwave": "microwave", "dishwasher": "dishwasher", "toaster": "toaster", "coffeemachine": "coffee machine",
    "toasteroven": "toaster oven", "stove": "stove", "stovetop": "stovetop", "oven": "oven", "hood": "range hood",
    "fridgefrenchdoor": "fridge", "fridgebottomfreezer": "fridge", "fridgesidebyside": "fridge", "stool": "stool",
    "wall": None, "floor": None, "box": None, "window": None, "wallaccessory": None, "accessory": None,
}


def strip_robot(root: ET.Element, prefixes: tuple[str, ...]):
    """Remove every element that belongs to the robot (named or referencing a robot prefix)."""
    def is_robot(el):
        return any(str(v).startswith(prefixes) for v in el.attrib.values())

    removed = 0
    for parent in root.iter():
        for child in list(parent):
            if child.tag in ("body", "geom", "site", "camera", "light", "joint", "mesh", "texture",
                             "material", "motor", "position", "velocity", "general", "intvelocity",
                             "jointpos", "jointvel", "force", "torque", "framepos", "framequat",
                             "accelerometer", "gyro", "touch", "exclude", "pair", "weld", "connect",
                             "joint", "fixed", "spatial", "rangefinder", "velocimeter", "actuatorpos",
                             "actuatorvel", "actuatorfrc", "subtreecom", "framelinvel", "frameangvel",
                             "default") and is_robot(child):
                parent.remove(child)
                removed += 1
    for section in ("keyframe",):          # robot-sized keyframes; our builder makes its own
        for el in root.findall(section):
            root.remove(el)
    # drop now-empty sections MuJoCo would complain about
    for section in ("actuator", "sensor", "contact", "equality", "tendon"):
        for el in root.findall(section):
            if len(el) == 0:
                root.remove(el)
    return removed


def main():
    """Export one RoboCasa kitchen (layout, style, task) as an environment."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", type=int, required=True)
    ap.add_argument("--style", type=int, required=True)
    ap.add_argument("--task", default="PickPlaceCounterToCabinet", help="RoboCasa task that places the objects")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--obj-registries", nargs="+", default=["objaverse", "lightwheel"])
    ap.add_argument("--name", help="default: kitchen_l<layout>_s<style>")
    args = ap.parse_args()

    name = args.name or f"kitchen_l{args.layout}_s{args.style}"
    out = ROOT / "assets/envs/robocasa" / name
    out.mkdir(parents=True, exist_ok=True)

    env = robosuite.make(args.task, robots="PandaOmron", layout_ids=args.layout, style_ids=args.style,
                         obj_registries=tuple(args.obj_registries), has_renderer=False,
                         has_offscreen_renderer=False, use_camera_obs=False, seed=args.seed)
    env.reset()
    sim, model = env.sim, env.sim.model

    # Robot spawn spot RoboCasa chose (clear of fixtures) -> G1 spawn
    robot = env.robots[0]
    base_body = robot.robot_model.base.root_body if hasattr(robot.robot_model, "base") and robot.robot_model.base else robot.robot_model.root_body
    bid = model.body_name2id(base_body)
    bpos = sim.data.body_xpos[bid]
    bmat = sim.data.body_xmat[bid].reshape(3, 3)
    # RoboCasa faces its arm-on-a-cart robot toward the counter; a walking G1 starts facing
    # the open room instead (turn around to see the counter).
    yaw = (math.degrees(math.atan2(bmat[1, 0], bmat[0, 0])) + 180.0 + 180.0) % 360.0 - 180.0

    root = ET.fromstring(env.model.get_xml())

    # Task objects: write their placed poses (set in the sim at reset) into the XML
    objects = []
    wb = root.find("worldbody")
    category = {cfg["name"]: str(cfg.get("info", {}).get("cat") or "") or None for cfg in env.object_cfgs}
    for key, obj in env.objects.items():
        body = obj.root_body
        jname = obj.joints[0] if obj.joints else None
        el = next((b for b in wb.iter("body") if b.get("name") == body), None)
        if el is None:
            continue
        if jname:
            q = sim.data.get_joint_qpos(jname)
            el.set("pos", " ".join(f"{v:.5f}" for v in q[:3]))
            el.set("quat", " ".join(f"{v:.6f}" for v in q[3:7]))
        cat = category.get(key) or key
        objects.append({"name": key, "class": cat, "label": cat.replace("_", " "), "detectable": True, "body": body})

    prefixes = tuple({robot.robot_model.naming_prefix, "mobilebase0_", "gripper0_", "robot0_"})
    removed = strip_robot(root, prefixes)

    # Our G1 world already has the floor plane: make RoboCasa's floor visual-only.
    for g in root.iter("geom"):
        if "floor" in (g.get("name") or "").lower():
            g.set("contype", "0")
            g.set("conaffinity", "0")

    # Asset paths are absolute into the local checkout; store them machine-independent.
    third_party = str(Path(robocasa.__file__).resolve().parents[2])
    xml = ET.tostring(root, encoding="unicode").replace(third_party, THIRD_PARTY_TOKEN)
    (out / "room.xml").write_text(xml)

    # Fixtures: name them by their RoboCasa key, sized/placed later by env_build
    for fname, fx in env.fixtures.items():
        fbody = getattr(fx, "root_body", None)
        if fbody and any(b.get("name") == fbody for b in wb.iter("body")):
            cls = type(fx).__name__.lower()
            label = FIXTURE_LABELS.get(cls, cls)
            objects.append({"name": fname, "class": cls, "label": label or cls, "detectable": label is not None,
                            "body": fbody})

    env_yaml = {
        "name": f"robocasa/{name}",
        "description": f"RoboCasa kitchen layout {args.layout}, style {args.style}; objects placed by task "
                       f"{args.task} (seed {args.seed})",
        "robot_world": "assets/robots/g1/mjlab_1.6.0/g1_velocity_flat.xml",
        "room": "room.xml",
        "hide_collision_geoms": True,   # RoboCasa's group-0 collision shapes are debug-colored
        "spawn": {"x": round(float(bpos[0]), 3), "y": round(float(bpos[1]), 3), "yaw_deg": round(yaw, 1)},
        "map": {"resolution": 0.05,
                "bounds": "auto",   # env_build computes them from solid footprints
                "min_obstacle_height": 0.05, "max_obstacle_height": 1.5},
        "objects": objects,
    }
    (out / "env.yaml").write_text(yaml.safe_dump(env_yaml, sort_keys=False))
    env.close()
    print(f"exported {name}: removed {removed} robot elements, {len(env.fixtures)} fixtures, "
          f"{len(env.objects)} task objects, spawn ({bpos[0]:.2f}, {bpos[1]:.2f}, yaw {yaw:.0f}°) -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
