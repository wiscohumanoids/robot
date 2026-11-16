"""Render benchmark RGB-D frames from our MuJoCo kitchens, exactly like the ZED X sim node
(ros2_ws/src/hsim_sim/hsim_sim/stereo_camera.py): 640x320, 110 deg HFOV, camera 0.45 m above the torso
(~1.28 m above the floor for a standing G1), pitched 25 deg down, stereo-like depth noise and holes.

Ground truth comes from a segmentation render: which objects are visible (>= MIN_PIXELS pixels) and their
true 3D boxes (union of their geoms' AABBs, like the line-of-sight stub). Viewpoints: free map cells
1.2-2.6 m from a randomly chosen object, looking at it (plus the robot spawn, looking around).

Run on the Mac (repo .venv: mujoco 3.11 + numpy + yaml):
  .venv/bin/python services/perception_server/bench/render_frames.py --out /tmp/perc_bench --kitchens kitchen_l1_s1 kitchen_l4_s4 kitchen_l9_s9 --views 12
Writes <out>/<kitchen>_<i>.npz (rgb, depth, K, T_map_cam, gt json) + <out>/index.json.
"""
import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[3]
ENVS = REPO / "assets" / "envs" / "robocasa"
W, H, HFOV = 640, 320, 110.0
CAM_Z, PITCH = 1.28, math.radians(25.0)
MIN_PIXELS = 150
SKIP = {"wall", "floor", "wallaccessory", "accessory", "box"}          # structural placeholders, not prompts


def box_std(gray, k=7):
    pad = k // 2
    g = np.pad(gray, pad, mode="edge").astype(np.float64)

    def box(a):
        c = np.cumsum(np.cumsum(a, 0), 1)
        c = np.pad(c, ((1, 0), (1, 0)))
        return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)
    m = box(g)
    return np.sqrt(np.maximum(box(g * g) - m * m, 0.0))


def stereo_noise(rgb, z, fx, rng, baseline=0.12, sig_d=0.15, zmin=0.3, zmax=12.0, tex=2.5, drop=0.01):
    """Same model as stereo_camera.py."""
    gray = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    textured = box_std(gray) > tex
    sigma = z * z * sig_d / (fx * baseline)
    z = z + rng.normal(0.0, 1.0, z.shape).astype(np.float32) * sigma
    valid = textured & (z > zmin) & (z < zmax) & (rng.random(z.shape) > drop)
    return np.where(valid, z, np.nan).astype(np.float32)


def load_map(kdir):
    meta = yaml.safe_load(open(kdir / "build" / "map.yaml"))
    raw = open(kdir / "build" / meta["image"], "rb").read()
    parts, i = [], 0
    while len(parts) < 4:                                       # P5 header: magic, w, h, maxval
        while raw[i:i + 1].isspace():
            i += 1
        if raw[i:i + 1] == b"#":
            while raw[i:i + 1] != b"\n":
                i += 1
            continue
        j = i
        while not raw[j:j + 1].isspace():
            j += 1
        parts.append(raw[i:j]); i = j
    w, h = int(parts[1]), int(parts[2])
    img = np.frombuffer(raw[i + 1:i + 1 + w * h], np.uint8).reshape(h, w)
    free = img > 250                                            # trinary map: 254 free, 0 occupied, 205 unknown
    return free, meta["resolution"], meta["origin"]


def is_free(free, res, origin, x, y, margin=0.35):
    h, w = free.shape
    r = int(math.ceil(margin / res))
    c = int((x - origin[0]) / res)
    row = h - 1 - int((y - origin[1]) / res)
    if not (r <= c < w - r and r <= row < h - r):
        return False
    return bool(free[row - r:row + r + 1, c - r:c + r + 1].all())


def optical_pose(pos, yaw):
    fwd = np.array([math.cos(yaw) * math.cos(PITCH), math.sin(yaw) * math.cos(PITCH), -math.sin(PITCH)])
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = right, down, fwd, pos
    return T, fwd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--kitchens", nargs="+", default=["kitchen_l1_s1", "kitchen_l4_s4", "kitchen_l9_s9"])
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    index = []
    fx = (W / 2) / math.tan(math.radians(HFOV) / 2)
    K = {"fx": fx, "fy": fx, "cx": (W - 1) / 2, "cy": (H - 1) / 2, "width": W, "height": H}
    for kname in args.kitchens:
        kdir = ENVS / kname
        m = mujoco.MjModel.from_binary_path(str(kdir / "build" / "world.mjb"))
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        m.vis.global_.fovy = math.degrees(2 * math.atan((H / 2) / fx))
        for g in range(m.ngeom):                                       # hide the robot, like the camera node
            if m.body(m.geom_bodyid[g]).name.startswith("robot/"):
                m.geom_group[g] = 4 if m.geom_group[g] <= 2 else 5
        objs_yaml = yaml.safe_load(open(kdir / "build" / "objects.yaml"))
        objs, all_objs = [], []                       # all_objs: every object incl. walls/floor/"box" (for fair scoring)
        for o in objs_yaml["objects"]:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o["body"])
            if bid < 0:
                continue
            sub = {bid}
            for b in range(bid + 1, m.nbody):
                if m.body_parentid[b] in sub:
                    sub.add(b)
            geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in sub]
            corners = []
            for g in geoms:
                c, half = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
                R, p = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
                for s in np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]):
                    corners.append(p + R @ (c + half * s))
            corners = np.array(corners)
            rec = {"label": o["label"], "name": o["name"], "class": o.get("class", ""), "geoms": set(geoms),
                   "center": ((corners.min(0) + corners.max(0)) / 2).tolist(),
                   "size": (corners.max(0) - corners.min(0)).tolist(), "movable": o.get("movable", False)}
            all_objs.append(rec)
            if o["label"] not in SKIP:
                objs.append(rec)
        free, res, origin = load_map(kdir)
        rgb_r = mujoco.Renderer(m, height=H, width=W)
        dep_r = mujoco.Renderer(m, height=H, width=W); dep_r.enable_depth_rendering()
        seg_r = mujoco.Renderer(m, height=H, width=W); seg_r.enable_segmentation_rendering()
        opt = mujoco.MjvOption()
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        sp = objs_yaml["spawn"]
        views = [(sp["x"], sp["y"], math.radians(sp["yaw_deg"]) + a) for a in (0.0, 1.57, 3.14, -1.57)]
        movable = [o for o in objs if o["movable"]] or objs
        tries = 0
        while len(views) < args.views and tries < 5000:
            tries += 1
            o = movable[rng.integers(len(movable))] if rng.random() < 0.6 else objs[rng.integers(len(objs))]
            dist, ang = rng.uniform(1.2, 2.6), rng.uniform(-math.pi, math.pi)
            x, y = o["center"][0] + dist * math.cos(ang), o["center"][1] + dist * math.sin(ang)
            if is_free(free, res, origin, x, y):
                views.append((x, y, math.atan2(o["center"][1] - y, o["center"][0] - x)))
        for i, (x, y, yaw) in enumerate(views):
            T, fwd = optical_pose(np.array([x, y, CAM_Z]), yaw)
            cam.lookat[:] = T[:3, 3] + fwd; cam.distance = 1.0
            cam.azimuth = math.degrees(yaw); cam.elevation = -math.degrees(PITCH)
            for r in (rgb_r, dep_r, seg_r):
                r.update_scene(d, camera=cam, scene_option=opt)
            rgb = rgb_r.render()
            z = dep_r.render().astype(np.float32)
            seg = seg_r.render()
            depth = stereo_noise(rgb, z, fx, rng)
            geom_ids = np.where(seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, seg[..., 0], -1)
            # per-pixel object index into all_objs (-1 = nothing labelled), for mask-based scoring
            g2o = np.full(m.ngeom + 1, -1, np.int16)
            for oi, o in enumerate(all_objs):
                for g in o["geoms"]:
                    g2o[g] = oi
            objmap = g2o[np.where(geom_ids >= 0, geom_ids, m.ngeom)]
            gt = []
            for o in objs:
                mask = np.isin(geom_ids, list(o["geoms"]))
                n = int(mask.sum())
                if n < MIN_PIXELS:
                    continue
                ys, xs = np.nonzero(mask)
                # check the projection convention: true centre projected vs mask centroid
                pc = np.linalg.inv(T) @ np.array([*o["center"], 1.0])
                proj = [K["fx"] * pc[0] / pc[2] + K["cx"], K["fy"] * pc[1] / pc[2] + K["cy"]] if pc[2] > 0 else None
                gt.append({"label": o["label"], "name": o["name"], "pixels": n, "center_map": o["center"],
                           "size": o["size"], "box_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                           "mask_centroid_px": [float(xs.mean()), float(ys.mean())], "proj_center_px": proj})
            fname = f"{kname}_{i:02d}.npz"
            np.savez_compressed(out / fname, rgb=rgb, depth=depth, T_map_cam=T, objmap=objmap)
            index.append({"file": fname, "kitchen": kname, "pose": [x, y, yaw], "gt": gt,
                          "objects_all": [{"label": o["label"], "class": o["class"], "name": o["name"], "center_map": o["center"],
                                           "size": o["size"]} for o in all_objs],
                          "vocab": sorted({o["label"] for o in objs}),
                          "movable": sorted({o["label"] for o in objs if o["movable"]})})
        print(f"{kname}: {len(views)} views, {len(objs)} labelled objects")
    json.dump({"K": K, "frames": index}, open(out / "index.json", "w"), indent=1)
    n_gt = sum(len(f["gt"]) for f in index)
    print(f"wrote {len(index)} frames, {n_gt} visible ground-truth objects -> {out}")


if __name__ == "__main__":
    main()
