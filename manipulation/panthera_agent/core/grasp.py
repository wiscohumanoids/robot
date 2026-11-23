"""Where to grasp an object of any shape: candidate grasps from its outline and parts, best first.

A cube is grasped at its centre across two faces (core/skills.py did only that). A learned object has a measured
top-view outline (core/learn.py), and possibly named parts (a mug's handle, a tool's handle). For a parallel gripper
coming down at the configured pitch, a grasp is a point on the object, the jaws' closing direction, and a height:

- the jaws close across the object; the width they meet there is the outline's extent along the closing direction,
  over the strip the finger pads cover;
- a usable width is inside (min_width + margin, open_width - margin);
- among those: narrower is firmer, nearer the object's centre (or the named part) wastes less torque, and closer to
  the radial direction from the arm's base is easier to reach (the reach and clearance checks come later, in
  Skills._choose, which tries these in order).

Flat and deformable objects (a towel) are pinched at the table instead: a corner or edge, jaws closing across it.
Tall objects are gripped at mid-height (up to `grasp_depth` below the top): gripped near the top, a can swings about
the jaws when the arm moves. The part "top" grips just below the top instead, for inserting the object.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GraspOption:
    """One way to grasp, in the base frame."""
    center_xy: np.ndarray       # where the jaws close (between the pads)
    z: float                    # height of the grasp point (the material between the pads)
    yaw: float                  # gripper yaw (approach direction); the jaws close across yaw + 90 deg
    width: float                # expected jaw width at the grasp, m
    score: float
    why: str
    pitch: float | None = None  # radians; None: the configured approach pitch


def _rot(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def width_across(outline_local: np.ndarray, point: np.ndarray, closing: np.ndarray, along: np.ndarray,
                 pad_half: float) -> tuple[float, float]:
    """The outline's extent along `closing` (unit, object frame) through `point`, over the strip of half-width
    `pad_half` along `along`: (width, offset of the strip's middle from `point` along `closing`)."""
    # Densify so a strip narrower than the outline's vertex spacing still catches its edges.
    dense = []
    n = len(outline_local)
    for i in range(n):
        a, b = outline_local[i], outline_local[(i + 1) % n]
        k = max(2, int(np.linalg.norm(b - a) / 0.002))
        dense.append(np.linspace(a, b, k, endpoint=False))
    dense = np.concatenate(dense) - point
    t = dense @ along
    sel = dense[np.abs(t) <= pad_half]
    if len(sel) < 2:
        return np.inf, 0.0
    s = sel @ closing
    return float(s.max() - s.min()), float((s.max() + s.min()) / 2)


def lowest_tcp(cfg: dict, margin: float = 0.004) -> float:
    """How far above the table the TCP must stay so no finger reaches into it, at the configured pitch: the lowest
    corner of the finger boxes (`grasp.gripper_boxes`, the first two) below the TCP, plus `margin`."""
    from core.geometry import gripper_rotation
    R = gripper_rotation(0.0, np.deg2rad(cfg['grasp']['pitch']))
    low = 0.0
    for x0, x1, y0, y1, z0, z1 in cfg['grasp']['gripper_boxes'][:2]:
        c = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
        low = min(low, float((c @ R.T)[:, 2].min()))
    return -low + margin


def fingers_clear(cfg: dict, center_xy, z: float, yaw: float, outline_world: np.ndarray, top_z: float,
                  margin: float = 0.002) -> bool:
    """Whether the open fingers (the first two `grasp.gripper_boxes`), lowered to this grasp, stay clear of the
    object's own outline: the jaws must straddle it, not land on a handle or a corner."""
    from core.geometry import convex_hull, gripper_rotation, polygon_gap
    R = gripper_rotation(yaw, np.deg2rad(cfg['grasp']['pitch']))
    tcp = np.array([center_xy[0], center_xy[1], z]) + float(cfg['grasp']['tcp_offset']) * R[:, 0]
    hull = convex_hull(outline_world)
    for x0, x1, y0, y1, z0, z1 in cfg['grasp']['gripper_boxes'][:2]:
        c = np.array([[x, y, zz] for x in (x0, x1) for y in (y0, y1) for zz in (z0, z1)]) @ R.T + tcp
        if c[:, 2].min() >= top_z:
            continue
        if polygon_gap(convex_hull(c[:, :2]), hull) < margin:
            return False
    return True


def grasp_options(obj, cfg: dict, base_xy=(0.0, 0.0), part: str | None = None, table_z: float = 0.05,
                  step_deg: float = 15.0) -> list[GraspOption]:
    """Candidate grasps for a found object (core/world.py Found with a learned type), best first."""
    t = obj.type
    g = cfg['grasp']
    open_w = float(g.get('open_width', 0.08))
    min_w = float(g['min_width'])
    margin = float(g.get('width_margin', 0.010))
    pad_half = float(g.get('pad_half_length', 0.0225)) * np.cos(np.deg2rad(g['pitch']))
    depth = float(g.get('grasp_depth', 0.04))       # at most this far below the top (the fingers' reach)
    center = obj.top[:2]
    R = _rot(obj.yaw)
    outline = np.asarray(t.outline, float) if t.outline else None
    bearing = np.arctan2(center[1] - base_xy[1], center[0] - base_xy[0])

    # Points to try (object frame): the named part, else the centre and points along the long axis. "top" is built
    # in: the whole object, gripped just below its top (leaving its body free to go into a hole).
    if part == 'top':
        depth = float(g.get('top_grasp_depth', 0.014))
        part = None
    if part:
        if not t.parts or part not in t.parts:
            raise KeyError(part)
        points = [np.asarray(t.parts[part][:2], float)]
    else:
        points = [np.zeros(2)]
        if outline is not None and t.footprint[0] > 1.6 * t.footprint[1]:
            span = t.footprint[0] / 2 * 0.7
            points += [np.array([s, 0.0]) for s in np.linspace(-span, span, 7) if abs(s) > 1e-6]
    flat = t.deformable or t.kind == 'flat'
    # Flat or soft things are pinched straight down, fingertips at the table (at the usual tilt the fingertips meet
    # the table first).
    flat_pitch = np.deg2rad(float(g.get('flat_pitch', 88)))
    z = table_z + float(g.get('pinch_height', 0.003)) if flat else max(obj.top[2] - min(t.height / 2, depth),
                                                                         table_z + lowest_tcp(cfg))
    outline_world = None if outline is None else outline @ R.T + center
    out = []
    for p in points:
        for k in range(int(round(360 / step_deg))):
            yaw_local = np.deg2rad(k * step_deg)
            along = np.array([np.cos(yaw_local), np.sin(yaw_local)])          # approach, object frame
            closing = np.array([-along[1], along[0]])
            if (outline is None or t.kind in ('box', 'long')) and abs(np.sin(2 * yaw_local)) > 0.05:
                continue                    # flat faces: grip across faces, never across corners (they slip)
            if outline is None:
                w = t.footprint[1] if abs(np.sin(yaw_local)) < 0.5 else t.footprint[0]
                off = 0.0
            elif flat:
                w, off = 0.004, 0.0
            else:
                w, off = width_across(outline, p, closing, along, pad_half)
            if not (min_w + margin <= w <= open_w - margin) and not flat:
                continue
            c_local = p + off * closing
            c = center + R @ c_local
            yaw = obj.yaw + yaw_local
            radial = abs(np.angle(np.exp(1j * (yaw - bearing))))
            if radial > np.deg2rad(100):
                continue
            if outline is not None and not flat and not fingers_clear(cfg, c, z, yaw, outline_world, obj.top[2]):
                continue
            off_centre = float(np.linalg.norm(c_local)) if not part else 0.0
            score = -(w / open_w) - 0.6 * radial / np.pi - 4.0 * off_centre
            out.append(GraspOption(c, float(z), float(yaw), float(w), float(score),
                                   f'{"part " + part if part else "body"}, {w * 1000:.0f} mm across, '
                                   f'{np.rad2deg(radial):.0f} deg off radial' + (', pinched from above' if flat else ''),
                                   flat_pitch if flat else None))
    out.sort(key=lambda o: -o.score)
    return out


def draw_options(outline_local, obj, options: list[GraspOption], size_px: int = 260) -> np.ndarray:
    """A top-view sketch: the outline, and the first few grasps as jaw pairs (the chosen one thicker)."""
    import cv2
    img = np.full((size_px, size_px, 3), 245, np.uint8)
    R = _rot(obj.yaw)
    pts = (np.asarray(outline_local) @ R.T + obj.top[:2]) if outline_local else None
    allp = [o.center_xy for o in options[:6]] + ([*pts] if pts is not None else [obj.top[:2]])
    c = np.mean(allp, axis=0)
    scale = size_px * 0.35 / max(0.04, max(np.linalg.norm(p - c) for p in allp))

    def px(p):
        d = (np.asarray(p) - c) * scale                       # x up, y left (as the overhead image)
        return int(size_px / 2 - d[1]), int(size_px / 2 - d[0])
    if pts is not None:
        cv2.polylines(img, [np.array([px(p) for p in pts], np.int32)], True, (60, 60, 60), 2, cv2.LINE_AA)
    for i, o in enumerate(options[:6][::-1]):
        closing = np.array([-np.sin(o.yaw), np.cos(o.yaw)])
        a, b = o.center_xy + closing * o.width / 2, o.center_xy - closing * o.width / 2
        col = (220, 60, 40) if o is options[0] else (120, 160, 220)
        cv2.line(img, px(a), px(b), col, 3 if o is options[0] else 1, cv2.LINE_AA)
        cv2.circle(img, px(o.center_xy), 3, col, -1)
    return img
