"""Build an environment: G1 training world + a room -> files the stack loads.

    .venv/bin/python tools/env_build.py assets/envs/test/table_room

Reads <env>/env.yaml and writes <env>/build/:
  world.mjb     G1 training world with the room attached (prefix "room/"), compiled to
                MuJoCo's self-contained binary format; keyframe "init_state" = G1 standing at
                the spawn pose + every loose object where the room file puts it
  map.pgm/.yaml 2D occupancy map in Nav2 map_server format: footprints of every solid
                shape that overlaps the robot's height band; bounds may be "auto"
  objects.yaml  named objects: position, size, top height
  preview.png   top-down render (plus preview_persp.png) for a visual check

Physics options come from the G1 training world, so the robot behaves as in training.
"""

from __future__ import annotations

import argparse
import math
import struct
import zlib
from pathlib import Path

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
ROOM_PREFIX = "room/"
KEY = "init_state"
FREE, OCCUPIED = 254, 0          # PGM values Nav2 reads as free / occupied (negate: 0)


def compose(env: dict, env_dir: Path) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(ROOT / env["robot_world"]))
    room = mujoco.MjSpec.from_file(str(env_dir / env["room"]))

    # Remember the G1 standing keyframe, then drop it: its qpos only covers the robot.
    key = spec.keys[0]
    robot_qpos, robot_ctrl = np.array(key.qpos), np.array(key.ctrl)
    spec.delete(key)

    # If the room brings its own (visual) floor, keep the training floor for physics but hide
    # it, so the room's floor shows instead of the training checkerboard.
    if any("floor" in (g.name or "").lower() for g in room.geoms):
        terrain = spec.geom("terrain")
        terrain.group = 3        # not drawn by default; still collides
    # Sources like RoboCasa model every fixture twice: textured visual meshes (group 1) and
    # simple debug-colored collision shapes (group 0). Viewers draw both, which looks garish
    # and flickers where they overlap. Keep the collision shapes solid, just don't draw them.
    if env.get("hide_collision_geoms", False):
        for g in room.geoms:
            if g.group == 0:
                g.rgba = [0.0, 0.0, 0.0, 0.0]
                g.material = ""
    frame = spec.worldbody.add_frame()
    spec.attach(room, prefix=ROOM_PREFIX, frame=frame)

    # qpos0 places loose objects where the room file puts them; robot part from the keyframe.
    m0 = spec.compile()
    qpos = m0.qpos0.copy()
    n_robot = len(robot_qpos)
    first_room_jnt = min(j for j in range(m0.njnt) if m0.joint(j).name.startswith(ROOM_PREFIX)) \
        if any(m0.joint(j).name.startswith(ROOM_PREFIX) for j in range(m0.njnt)) else m0.njnt
    if first_room_jnt < m0.njnt and m0.jnt_qposadr[first_room_jnt] != n_robot:
        raise RuntimeError("expected the robot's joints to come first in qpos")
    qpos[:n_robot] = robot_qpos
    sp = env["spawn"]
    yaw = math.radians(sp.get("yaw_deg", 0.0))
    qpos[0], qpos[1] = sp["x"], sp["y"]
    qpos[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    spec.add_key(name=KEY, qpos=qpos.tolist(), ctrl=robot_ctrl.tolist())
    return spec.compile()


def solid_footprints(m: mujoco.MjModel, d: mujoco.MjData, hmin: float, hmax: float):
    """XY footprints (convex polygons) of every collision geom that overlaps the robot's
    height band [hmin, hmax]. Skips the floor, visual-only geoms, and the robot itself."""
    robot_root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/pelvis")
    robot = {robot_root}
    for b in range(robot_root + 1, m.nbody):
        if m.body_parentid[b] in robot:
            robot.add(b)
    polys = []
    for g in range(m.ngeom):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE or m.geom_bodyid[g] in robot:
            continue
        if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
            continue          # visual only: doesn't block anything
        c, half = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
        R, p0 = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
        corners = np.array([p0 + R @ (c + half * np.array([sx, sy, sz]))
                            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        if corners[:, 2].max() <= hmin or corners[:, 2].min() >= hmax:
            continue          # entirely below the ankle or above the head
        polys.append(convex_hull(corners[:, :2]))
    return polys


def convex_hull(pts: np.ndarray) -> np.ndarray:
    pts = sorted(map(tuple, np.round(pts, 6)))
    def half(points):
        h = []
        for q in points:
            while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (q[1] - h[-2][1])
                                   - (h[-1][1] - h[-2][1]) * (q[0] - h[-2][0])) <= 0:
                h.pop()
            h.append(q)
        return h
    lower, upper = half(pts), half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def auto_bounds(polys, margin=0.3):
    allp = np.vstack(polys)
    lo, hi = allp.min(0) - margin, allp.max(0) + margin
    return {"xmin": round(float(lo[0]), 2), "xmax": round(float(hi[0]), 2),
            "ymin": round(float(lo[1]), 2), "ymax": round(float(hi[1]), 2)}


def occupancy(m: mujoco.MjModel, cfg: dict):
    """Rasterize solid footprints into a Nav2 occupancy grid. Returns (image, bounds)."""
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, KEY))
    mujoco.mj_forward(m, d)
    hmin, hmax = cfg["min_obstacle_height"], cfg.get("max_obstacle_height", 1.5)
    polys = solid_footprints(m, d, hmin, hmax)
    b = cfg["bounds"] if cfg.get("bounds", "auto") != "auto" else auto_bounds(polys)
    res = cfg["resolution"]
    xs = np.arange(b["xmin"] + res / 2, b["xmax"], res)
    ys = np.arange(b["ymin"] + res / 2, b["ymax"], res)
    grid = np.full((len(ys), len(xs)), FREE, dtype=np.uint8)
    X, Y = np.meshgrid(xs, ys)
    for poly in polys:
        lo, hi = poly.min(0), poly.max(0)
        ix = (xs >= lo[0] - res / 2) & (xs <= hi[0] + res / 2)
        iy = (ys >= lo[1] - res / 2) & (ys <= hi[1] + res / 2)
        if not ix.any() or not iy.any():
            continue
        sub_x, sub_y = X[np.ix_(iy, ix)], Y[np.ix_(iy, ix)]
        inside = np.ones(sub_x.shape, bool)
        n = len(poly)
        for k in range(n):        # inside a counter-clockwise convex polygon = left of every edge
            a, bb = poly[k], poly[(k + 1) % n]
            inside &= (bb[0] - a[0]) * (sub_y - a[1]) - (bb[1] - a[1]) * (sub_x - a[0]) >= -1e-9
        # also mark cells the polygon merely touches (thin walls narrower than a cell)
        if not inside.any():
            cx = np.clip(np.searchsorted(xs, (lo[0] + hi[0]) / 2), 0, len(xs) - 1)
            cy = np.clip(np.searchsorted(ys, (lo[1] + hi[1]) / 2), 0, len(ys) - 1)
            grid[cy, cx] = OCCUPIED
        block = grid[np.ix_(iy, ix)]
        block[inside] = OCCUPIED
        grid[np.ix_(iy, ix)] = block
    return np.flipud(grid), b


def safe_spawn(grid: np.ndarray, bounds: dict, res: float, x: float, y: float, clearance: float):
    """Nearest map cell to (x, y) with at least `clearance` m to every obstacle.
    Returns (x, y, moved_by_m)."""
    img = np.flipud(grid)                       # back to row 0 = ymin
    occ = np.argwhere(img == OCCUPIED)
    free = np.argwhere(img == FREE)
    if len(occ) == 0:
        return x, y, 0.0
    free_xy = np.c_[bounds["xmin"] + (free[:, 1] + 0.5) * res, bounds["ymin"] + (free[:, 0] + 0.5) * res]
    occ_xy = np.c_[bounds["xmin"] + (occ[:, 1] + 0.5) * res, bounds["ymin"] + (occ[:, 0] + 0.5) * res]
    if np.sqrt(((occ_xy - [x, y]) ** 2).sum(1)).min() >= clearance:
        return x, y, 0.0                        # already clear: keep it exactly where it was
    clear = np.empty(len(free_xy))
    for i in range(0, len(free_xy), 2048):     # distance from each free cell to the nearest obstacle
        chunk = free_xy[i:i + 2048]
        clear[i:i + 2048] = np.sqrt(((chunk[:, None, :] - occ_xy[None, :, :]) ** 2).sum(-1)).min(1)
    ok = free_xy[clear >= clearance]
    if len(ok) == 0:
        raise RuntimeError(f"no spot in the map has {clearance} m clearance")
    best = ok[np.argmin(((ok - [x, y]) ** 2).sum(1))]
    return float(best[0]), float(best[1]), float(np.hypot(best[0] - x, best[1] - y))


def write_pgm(path: Path, img: np.ndarray):
    h, w = img.shape
    path.write_bytes(f"P5\n{w} {h}\n255\n".encode() + img.tobytes())


def write_png(path: Path, rgb: np.ndarray):
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[r].tobytes() for r in range(h))
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def objects(m: mujoco.MjModel, env: dict):
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, KEY))
    mujoco.mj_forward(m, d)
    out = []
    for o in env["objects"]:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ROOM_PREFIX + o["body"])
        if bid < 0:
            raise ValueError(f"object {o['name']}: body {o['body']} not in the room")
        subtree = {bid}
        for b in range(bid + 1, m.nbody):          # children always come after their parent
            if m.body_parentid[b] in subtree:
                subtree.add(b)
        corners = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] not in subtree:
                continue
            c, half = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
            R, p = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        corners.append(p + R @ (c + half * np.array([sx, sy, sz])))
        corners = np.array(corners)
        lo, hi = corners.min(0), corners.max(0)
        out.append({
            "name": o["name"], "class": o["class"], "body": ROOM_PREFIX + o["body"],
            "label": o.get("label", o["name"].replace("_", " ")),     # what a detector would call it
            "detectable": o.get("detectable", True),                  # structure (walls, floor) is not
            "position": [round(float(v), 3) for v in (lo + hi) / 2],
            "size": [round(float(v), 3) for v in hi - lo],
            "top_z": round(float(hi[2]), 3),
            "movable": bool(m.body_jntnum[bid] > 0 and m.jnt_type[m.body_jntadr[bid]] == mujoco.mjtJoint.mjJNT_FREE),
        })
    return out


def render(m: mujoco.MjModel, cfg: dict, out_dir: Path):
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, KEY))
    mujoco.mj_forward(m, d)
    b = cfg["bounds"]
    center = [(b["xmin"] + b["xmax"]) / 2, (b["ymin"] + b["ymax"]) / 2, 0.0]
    span = max(b["xmax"] - b["xmin"], b["ymax"] - b["ymin"])
    r = mujoco.Renderer(m, height=600, width=800)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = center
    for name, (dist, elev, azim) in {"preview": (span * 1.25, -90.0, 90.0),
                                     "preview_persp": (span * 0.95, -35.0, 135.0)}.items():
        cam.distance, cam.elevation, cam.azimuth = dist, elev, azim
        r.update_scene(d, camera=cam)
        write_png(out_dir / f"{name}.png", r.render())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("env_dir", type=Path)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--robot", default="g1", help="robot profile (ros2_ws/src/hsim_bringup/config/robots/<robot>.yaml): "
                    "its training world replaces env.yaml's robot_world, output goes to the profile's env_build_dir "
                    "(g1: build/, wisco: build_wisco/)")
    args = ap.parse_args()
    env_dir = args.env_dir.resolve()
    env = yaml.safe_load((env_dir / "env.yaml").read_text())
    prof = yaml.safe_load((ROOT / "ros2_ws/src/hsim_bringup/config/robots" / f"{args.robot}.yaml").read_text())
    if args.robot != "g1":           # g1 keeps env.yaml's robot_world, so existing builds are reproduced exactly
        env["robot_world"] = prof["world"].removeprefix("/hsim/")
    out = env_dir / prof["env_build_dir"]
    out.mkdir(exist_ok=True)

    m = compose(env, env_dir)
    grid, b = occupancy(m, env["map"])

    # The robot must start with room around it; nudge the spawn if it's too close to anything.
    sp = env["spawn"]
    need = sp.get("min_clearance", 0.5)
    x, y, moved = safe_spawn(grid, b, env["map"]["resolution"], sp["x"], sp["y"], need)
    if moved > 1e-6:
        print(f"spawn ({sp['x']:.2f}, {sp['y']:.2f}) has < {need} m clearance -> moved {moved:.2f} m to ({x:.2f}, {y:.2f})")
        env["spawn"] = {**sp, "x": round(x, 3), "y": round(y, 3)}
        m = compose(env, env_dir)
    mujoco.mj_saveModel(m, str(out / "world.mjb"), None)
    write_pgm(out / "map.pgm", grid)
    (out / "map.yaml").write_text(yaml.safe_dump({
        "image": "map.pgm", "mode": "trinary", "resolution": env["map"]["resolution"],
        "origin": [b["xmin"], b["ymin"], 0.0], "negate": 0,
        "occupied_thresh": 0.65, "free_thresh": 0.25,
    }, sort_keys=False))

    objs = objects(m, env)
    (out / "objects.yaml").write_text(yaml.safe_dump(
        {"env": env["name"], "frame": "map", "spawn": env["spawn"], "objects": objs}, sort_keys=False))

    if not args.no_render:
        render(m, {**env["map"], "bounds": b}, out)

    occ = float((grid == OCCUPIED).mean())
    print(f"built {env['name']} for {args.robot}: nq={m.nq} (robot 36 + room {m.nq - 36}), nbody={m.nbody}, "
          f"map {grid.shape[1]}x{grid.shape[0]} @ {env['map']['resolution']} m ({occ:.0%} occupied), "
          f"{len(objs)} objects -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
