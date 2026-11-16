"""The viewer's perception panel: the world as the ROBOT knows it, drawn on black.

Fed by the `viewer_bridge` ROS node over UDP (see ros2_ws/src/hsim_teleop/hsim_teleop/viewer_bridge.py).
Everything is in the map frame and comes from the robot's own estimates, not the simulator:
  - the robot itself, highlighted, at its OWN estimated pose
  - its map (known map or SLAM), filling in as it explores; newly mapped cells flash
  - lidar returns with a short fading trail, and a radar pulse around the robot
  - stereo point cloud (sensors:=stereo), colored by height
  - object memory: boxes with labels and confidence
  - Nav2's planned path
"""

import colorsys
import json
import math
import time
import zlib
from collections import deque

import mujoco
import numpy as np

ROBOT_PREFIX = "robot/"
HIDDEN_GROUP = 5
EYE = np.eye(3).flatten()

# Everything is drawn OPAQUE (fading = blending toward the background color): see-through geometry
# has to be re-sorted every frame, and thousands of see-through dots/boxes pop and flicker.
C_BG = np.array([0.0, 0.0, 0.0])
C_FREE = (0.05, 0.10, 0.18, 1.0)
C_OCC = np.array([0.10, 0.75, 1.00])
C_FRESH = np.array([0.90, 1.00, 1.00])
C_SCAN = np.array([1.00, 0.40, 0.12])
C_PLAN = (0.20, 1.00, 0.45, 1.0)
C_PULSE = np.array([0.20, 1.00, 0.70])
C_ROBOT = (0.75, 0.95, 1.00, 1.0)
TRAIL_S = 0.6            # how long lidar hits linger
FRESH_S = 1.5            # how long newly mapped walls glow


def fade(color, a, bg=C_BG):
    """`color` faded toward the background by a (1 = full color), as an opaque rgba."""
    c = bg + (np.asarray(color[:3]) - bg) * float(np.clip(a, 0.0, 1.0))
    return (c[0], c[1], c[2], 1.0)


def merge_rects(cells, kind):
    """Cells of one kind -> few rectangles (row runs, merged down identical runs). [(i0, i1, j0, j1)]"""
    rects, open_ = [], {}
    h, w = cells.shape
    for j in range(h + 1):
        runs = set()
        if j < h:
            row = (cells[j] == kind).astype(np.int8)
            edges = np.flatnonzero(np.diff(np.r_[0, row, 0]))
            runs = set(zip(edges[0::2], edges[1::2]))
        for key in list(open_):
            if key not in runs:
                rects.append((key[0], key[1], open_.pop(key), j))
        for key in runs:
            open_.setdefault(key, j)
    return rects


def label_color(label, alpha):
    h = (zlib.crc32(label.encode()) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.65, 1.0)
    return (r, g, b, alpha)


def height_color(z):
    """Blue (floor) -> green -> yellow -> red (head height)."""
    t = float(np.clip(z / 1.8, 0.0, 1.0))
    r, g, b = colorsys.hsv_to_rgb(0.66 * (1 - t), 0.9, 1.0)
    return (r, g, b, 1.0)


def yaw_quat(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


class PerceptionView:
    def __init__(self, model: mujoco.MjModel, max_width=1920, max_height=1200):
        m = self.m = model                      # a private copy: we recolor and hide geoms in it
        m.vis.global_.offwidth, m.vis.global_.offheight = max_width, max_height
        for g in range(m.ngeom):
            if m.body(m.geom_bodyid[g]).name.startswith(ROBOT_PREFIX):
                if m.geom_group[g] <= 2:         # visual + collision shapes of the robot: draw the visual ones
                    m.geom_matid[g] = -1
                    m.geom_rgba[g] = C_ROBOT
                    m.geom_group[g] = 0 if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0 else HIDDEN_GROUP
            else:
                m.geom_group[g] = HIDDEN_GROUP    # the real room is NOT shown: only what the robot knows
        if not any(m.geom_group[g] == 0 for g in range(m.ngeom)):
            for g in range(m.ngeom):              # robot model without visual-only meshes: show its collision shapes
                if m.body(m.geom_bodyid[g]).name.startswith(ROBOT_PREFIX):
                    m.geom_group[g] = 0
        m.vis.headlight.ambient[:] = 0.45
        m.vis.headlight.diffuse[:] = 0.6
        self.d = mujoco.MjData(m)
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 0
        self.opt.geomgroup[0] = 1
        self.opt.sitegroup[:] = 0
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.renderer = None
        self.size = (0, 0)

        self.live = {}
        self.live_t = 0.0
        self.trail = deque(maxlen=6)             # recent scans: (time, points)
        self.offset = None                       # smoothed (estimate - truth): dx, dy, dyaw
        self.map_rects = []                      # (kind, cx, cy, hx, hy)
        self.fresh = []                          # newly mapped cells: (time, cx, cy, hx, hy)
        self.cells = None
        self.map_origin = None
        self.t0 = time.monotonic()

    # ------------------------------------------------------------------ data in
    def on_packet(self, data: bytes):
        try:
            if data[:4] == b"HPER":
                self.live = json.loads(zlib.decompress(data[4:]))
                self.live_t = time.monotonic()
                if self.live.get("scan"):
                    self.trail.append((self.live_t, np.asarray(self.live["scan"], np.float32)))
            elif data[:4] == b"HMAP":
                raw = zlib.decompress(data[4:])
                head, grid = raw.split(b"\n", 1)
                self.on_map(json.loads(head), np.frombuffer(grid, np.uint8))
        except (ValueError, zlib.error, KeyError):
            pass

    def on_map(self, h, grid):
        w, hh, res, ox, oy = h["w"], h["h"], h["res"], h["ox"], h["oy"]
        cells = grid.reshape(hh, w)
        if self.cells is not None and np.array_equal(cells, self.cells) and self.map_origin == (ox, oy, res):
            return                                # periodic resend, nothing new
        rects = []
        for kind in (0, 1):                       # free floor, then walls/furniture; unknown stays black
            for i0, i1, j0, j1 in merge_rects(cells, kind):
                rects.append((kind, ox + (i0 + i1) / 2 * res, oy + (j0 + j1) / 2 * res,
                              (i1 - i0) / 2 * res, (j1 - j0) / 2 * res))
        now = time.monotonic()
        fresh = []
        if self.cells is not None and self.cells.shape == cells.shape and self.map_origin == (ox, oy, res):
            new = (cells == 1) & (self.cells != 1)        # walls/furniture seen for the first time
            fresh = [(now, ox + (i0 + i1) / 2 * res, oy + (j0 + j1) / 2 * res, (i1 - i0) / 2 * res, (j1 - j0) / 2 * res)
                     for i0, i1, j0, j1 in merge_rects(new.astype(np.uint8), 1)]
        self.fresh = [f for f in self.fresh if now - f[0] < FRESH_S] + fresh
        self.cells, self.map_rects, self.map_origin = cells, rects, (ox, oy, res)

    @property
    def connected(self):
        return time.monotonic() - self.live_t < 1.0

    # ------------------------------------------------------------------ drawing helpers
    def _geom(self, gtype, size, pos, rgba, mat=EYE, label=""):
        scn = self.renderer.scene
        if scn.ngeom >= scn.maxgeom:
            return None
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, gtype, np.asarray(size, np.float64), np.asarray(pos, np.float64), mat,
                            np.asarray(rgba, np.float32))
        if label:
            g.label = label
        scn.ngeom += 1
        return g

    def _line(self, a, b, rgba, width=3.0):
        scn = self.renderer.scene
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), EYE, np.asarray(rgba, np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, width, np.asarray(a, np.float64), np.asarray(b, np.float64))
        scn.ngeom += 1

    # ------------------------------------------------------------------ render
    def render(self, qpos, main_cam: mujoco.MjvCamera, width, height):
        """Returns an RGB image, top row first (the viewer's set_images draws it as is)."""
        width, height = min(width, self.m.vis.global_.offwidth), min(height, self.m.vis.global_.offheight)
        if self.renderer is None or self.size != (width, height):
            if self.renderer is not None:
                self.renderer.close()
            self.renderer = mujoco.Renderer(self.m, height, width, max_geom=20000)
            for f in (mujoco.mjtRndFlag.mjRND_SKYBOX, mujoco.mjtRndFlag.mjRND_SHADOW, mujoco.mjtRndFlag.mjRND_REFLECTION,
                      mujoco.mjtRndFlag.mjRND_FOG, mujoco.mjtRndFlag.mjRND_HAZE):
                self.renderer.scene.flags[f] = 0
            self.size = (width, height)

        # the robot at its OWN estimated pose (joint angles from the sim, like its encoders)
        d = self.d
        d.qpos[:] = qpos
        # The estimate arrives 10x a second; the joints 50x. Draw the robot from the smooth joint stream
        # shifted by a smoothed (estimate - truth) offset, so it glides instead of hopping.
        pose = self.live.get("pose") if self.connected else None
        true_xy = np.array(qpos[:2])
        q = qpos[3:7]
        true_yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        if pose:
            target = np.array([pose[0] - qpos[0], pose[1] - qpos[1],
                               math.atan2(math.sin(pose[2] - true_yaw), math.cos(pose[2] - true_yaw))])
            if self.offset is None or np.linalg.norm(target[:2] - self.offset[:2]) > 1.0:
                self.offset = target              # first estimate, or a big jump (relocalized): snap
            else:
                dt = min(0.1, time.monotonic() - getattr(self, "_last_render", 0.0))
                self.offset += (target - self.offset) * (1.0 - math.exp(-dt / 0.4))
        if self.offset is not None and pose:
            d.qpos[0], d.qpos[1] = qpos[0] + self.offset[0], qpos[1] + self.offset[1]
            d.qpos[3:7] = quat_mul(yaw_quat(self.offset[2]), q)
        self._last_render = time.monotonic()
        mujoco.mj_forward(self.m, d)             # (kinematics alone leaves parts of the robot undrawn)
        est_xy = np.array(d.qpos[:2])

        # same angle as the main camera, centered on where the robot thinks it is
        c = self.cam
        c.lookat[:] = np.array(main_cam.lookat) + np.r_[est_xy - true_xy, 0.0]    # follows the smoothed estimate
        c.azimuth, c.elevation, c.distance = main_cam.azimuth, main_cam.elevation, main_cam.distance
        self.renderer.update_scene(d, camera=c, scene_option=self.opt)
        scn = self.renderer.scene
        for i in range(scn.ngeom):               # the robot glows: lit evenly from every side
            scn.geoms[i].emission = 0.55

        now = time.monotonic()
        G = mujoco.mjtGeom
        occ = (*C_OCC, 1.0)
        # map: dark floor where it's free, glowing walls/furniture; new discoveries glow white, then settle
        for k, cx, cy, hx, hy in list(self.map_rects):
            if k == 0:
                self._geom(G.mjGEOM_BOX, (hx, hy, 0.002), (cx, cy, -0.002), C_FREE)
            else:
                self._geom(G.mjGEOM_BOX, (hx, hy, 0.12), (cx, cy, 0.12), occ)
        for t, cx, cy, hx, hy in list(self.fresh):
            a = 1.0 - (now - t) / FRESH_S
            if a > 0:
                self._geom(G.mjGEOM_BOX, (hx + 0.004, hy + 0.004, 0.125), (cx, cy, 0.125),
                           fade(C_FRESH, a, bg=C_OCC))

        if self.connected:
            live = self.live
            # radar pulse: an expanding ring every 2 s
            phase = ((now - self.t0) % 2.0) / 2.0
            r, n = 0.3 + 5.5 * phase, 96
            ring = [(est_xy[0] + r * math.cos(2 * math.pi * i / n), est_xy[1] + r * math.sin(2 * math.pi * i / n), 0.01)
                    for i in range(n + 1)]
            col = fade(C_PULSE, 0.9 * (1 - phase))
            for a, b in zip(ring[:-1], ring[1:]):
                self._line(a, b, col, 2.0)
            # lidar: latest scan bright, the previous few dimming
            for t, pts in list(self.trail):        # (packets arrive on another thread)
                a = 1.0 - (now - t) / TRAIL_S
                if a <= 0:
                    continue
                col = fade(C_SCAN, a)
                for px, py in pts:
                    self._geom(G.mjGEOM_BOX, (0.022, 0.022, 0.022), (px, py, 0.35), col)
            # stereo point cloud
            for px, py, pz in live.get("cloud", []):
                self._geom(G.mjGEOM_BOX, (0.018, 0.018, 0.018), (px, py, pz), height_color(pz))
            # Nav2's plan
            plan = live.get("plan", [])
            for a, b in zip(plan[:-1], plan[1:]):
                self._line((a[0], a[1], 0.03), (b[0], b[1], 0.03), C_PLAN, 5.0)
            if plan:
                self._geom(G.mjGEOM_CYLINDER, (0.12, 0.004, 0), (plan[-1][0], plan[-1][1], 0.01), C_PLAN)
            # object memory: outlined boxes; labels on the nearest confident ones
            objs = live.get("objects", [])
            near = sorted((o for o in objs if o["c"] >= 0.5),
                          key=lambda o: math.hypot(o["p"][0] - est_xy[0], o["p"][1] - est_xy[1]))
            chosen = []                           # nearest first, at most 6, none within 0.6 m of another
            for o in near:
                if len(chosen) < 6 and all(math.hypot(o["p"][0] - c["p"][0], o["p"][1] - c["p"][1]) > 0.6 for c in chosen):
                    chosen.append(o)
            labeled = {id(o) for o in chosen}
            for o in objs:
                (px, py, pz), (sx, sy, sz) = o["p"], o["s"]
                col = label_color(o["label"], 1.0)
                corners = [(px + dx * sx / 2, py + dy * sy / 2, pz + dz * sz / 2)
                           for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)]
                for i, j in ((0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)):
                    self._line(corners[i], corners[j], col, 2.0)
                if id(o) in labeled:
                    self._geom(G.mjGEOM_SPHERE, (0.001, 0, 0), (px, py, pz + sz / 2 + 0.12), (0, 0, 0, 1),
                               label=f"{o['label']} {int(round(100 * o['c']))}%")

        return self.renderer.render()
