"""Learning a new object from a few chosen points: its appearance examples, then its height, footprint and shape.

The pure parts of the `learn` skill (core/skills.py drives the arm and the cameras):

- `examples`: the LLM's chosen candidate points (perception/candidates.py) -> the object's DINO examples (the
  patches of the chosen regions) and background examples (the other regions and bare table).
- `measure_height`: no depth camera, so the height comes from two views. The overhead camera and the wrist camera
  (looking from the side, at a known pose) each see the object; the footprint each implies depends on the height
  assumed for the top face, and only the true height makes the two agree.
- `shape_of`: the footprint's outline -> a shape class (box, round, long, flat, irregular) and the yaw symmetry the
  grasp planner uses.
"""
from __future__ import annotations

import cv2
import numpy as np

from core.geometry import footprint, pixels_to_plane


def object_pixels(rgb: np.ndarray, empty_rgb: np.ndarray | None, region_patches: np.ndarray,
                  change: float = 14.0) -> np.ndarray:
    """(H, W) bool: the object's own pixels: the connected blobs of pixels that differ from the
    empty table (Lab distance above `change`, lightness half-weighted), except shadows (darker, same chromaticity).
    Without an empty-table image, the whole region."""
    from perception.candidates import colour_distance
    from perception.dino import PATCH
    from perception.dino_local import lab_image
    H, W = rgb.shape[:2]
    gh, gw = region_patches.shape
    grown = cv2.dilate(region_patches.astype(np.uint8), np.ones((3, 3), np.uint8))
    px = np.zeros((H, W), bool)
    px[:gh * PATCH, :gw * PATCH] = cv2.resize(grown, (gw * PATCH, gh * PATCH), interpolation=cv2.INTER_NEAREST) > 0
    if empty_rgb is None:
        return px & (cv2.resize(region_patches.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0)
    changed = colour_distance(lab_image(rgb), lab_image(empty_rgb)) > change
    # A cast shadow changes the table's brightness, not its chromaticity (r, g over r + g + b): not the object.
    a, b = rgb.astype(np.float32), empty_rgb.astype(np.float32)
    sa, sb = a.sum(-1) + 1e-3, b.sum(-1) + 1e-3
    same_hue = np.linalg.norm(a[..., :2] / sa[..., None] - b[..., :2] / sb[..., None], axis=-1) < 0.02
    changed &= ~(same_hue & (sa < 0.92 * sb))
    # Whole blobs of changed pixels that the chosen region touches: one point on an object brings all of it. The
    # opening breaks thin bridges (a shadow edge, a reflection) between neighbouring objects.
    m = cv2.morphologyEx(changed.astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, comp = cv2.connectedComponents(m, connectivity=8)
    if n <= 1:
        return px
    touched = np.unique(comp[px & (m > 0)])
    sizes = np.bincount(comp.ravel())
    keep = np.isin(comp, [c for c in touched if c > 0 and sizes[c] >= 20])
    # A blob much larger than the chosen region is the object touching a neighbour (a peg against its block): then
    # only the changed pixels of the chosen region itself.
    chosen = np.zeros((H, W), bool)
    chosen[:gh * PATCH, :gw * PATCH] = cv2.resize(region_patches.astype(np.uint8), (gw * PATCH, gh * PATCH),
                                                  interpolation=cv2.INTER_NEAREST) > 0
    if keep.sum() > 3 * max(chosen.sum(), 1):
        keep = px & changed
    return keep if keep.any() else px


def examples(features: np.ndarray, proposals, chosen: list[int], pixels: np.ndarray | None = None,
             max_fg: int = 48, max_bg: int = 96, rng: np.random.Generator | None = None):
    """(fg, bg, region): feature examples of the object, examples of the bare table around it, and the object's patches
    (gh, gw) bool. The object is the chosen points' regions, or, given its exact `pixels` (`object_pixels`), the
    patches at least 40 % covered by them (an edge patch that is mostly table would teach the wrong thing)."""
    from perception.dino import PATCH
    rng = rng or np.random.default_rng(0)
    gh, gw, C = features.shape
    region = np.zeros((gh, gw), bool)
    for n in chosen:
        c = proposals.by_number(n)
        if c is None:
            raise ValueError(f'no point {n}')
        region |= c.patches
    if pixels is not None:
        cover = cv2.resize(pixels[:gh * PATCH, :gw * PATCH].astype(np.float32), (gw, gh),
                           interpolation=cv2.INTER_AREA)
        if (cover >= 0.4).sum() >= 1:
            region = cover >= 0.4
    others = np.zeros((gh, gw), bool)
    for c in proposals.candidates:
        if c.number not in chosen and c.kind == 'region' and not (c.patches & region).any():
            others |= c.patches
    touched = region | (cv2.resize(pixels[:gh * PATCH, :gw * PATCH].astype(np.float32), (gw, gh),
                                   interpolation=cv2.INTER_AREA) > 0.02 if pixels is not None else region)
    grown = cv2.dilate(touched.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    table = proposals.background & ~grown

    def sample(mask, k):
        idx = np.argwhere(mask)
        if len(idx) > k:
            idx = idx[rng.choice(len(idx), k, replace=False)]
        return features[tuple(idx.T)]

    fg = sample(region, max_fg)
    bg = sample(table, max_bg)
    return fg, bg, region


def measure_height(mask_a: np.ndarray, cam_a, mask_b: np.ndarray, cam_b, z_table: float,
                   heights=np.arange(0.004, 0.201, 0.002)):
    """The height (m) at which the two views' footprints agree best, and the disagreement (m) there.

    Each view's mask, projected to a top face at table + h and a base at the table, gives a footprint (the two-plane
    overlap). Seen from two directions, those footprints coincide only at the true h: a wrong h shifts each one
    along its own line of sight. Returns (h, centre gap in m, per-height gaps)."""
    gaps = []
    for h in heights:
        fa = footprint(mask_a, cam_a, z_table + h, z_table)
        fb = footprint(mask_b, cam_b, z_table + h, z_table)
        gaps.append(np.inf if fa is None or fb is None else float(np.linalg.norm(fa.center[:2] - fb.center[:2])))
    gaps = np.array(gaps)
    if not np.isfinite(gaps).any():
        return None, None, gaps
    i = int(np.argmin(gaps))
    # Sub-step refinement: a parabola through the minimum and its neighbours.
    h = float(heights[i])
    if 0 < i < len(gaps) - 1 and np.isfinite(gaps[i - 1:i + 2]).all():
        a, b, c = gaps[i - 1], gaps[i], gaps[i + 1]
        den = a - 2 * b + c
        if den > 1e-12:
            h += 0.5 * (a - c) / den * float(heights[1] - heights[0])
    return h, float(gaps[i]), gaps


def top_outline(mask: np.ndarray, camera, z_top: float, z_table: float) -> np.ndarray | None:
    """The footprint's outline (N, 2) in metres, base frame: the two-plane overlap of the mask's outline."""
    from core.geometry import _overlap
    m = np.asarray(mask, np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    outline = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
    top = pixels_to_plane(camera, outline, z_top)
    base = pixels_to_plane(camera, outline, z_table)
    if not (np.isfinite(top).all() and np.isfinite(base).all()):
        return None
    mm = _overlap(top[:, :2] * 1000, base[:, :2] * 1000)
    return None if mm is None else mm / 1000.0


def shape_of(outline_xy: np.ndarray, height: float, center: np.ndarray, yaw: float) -> dict:
    """Shape class, symmetry and the outline in the object's frame (origin at the centre, x along `yaw`).

    round: circular (any yaw is the same; grasped from any side). box: square-ish and rectangular (90 deg).
    long: a rectangle clearly longer than wide (180 deg). flat: much wider than tall (a towel, a plate).
    irregular: anything a rectangle fits badly (a mug with its handle, an L-shaped tool): no symmetry assumed."""
    c, s = np.cos(-yaw), np.sin(-yaw)
    local = (outline_xy - center[:2]) @ np.array([[c, -s], [s, c]]).T
    pts = local.astype(np.float32)
    area = abs(cv2.contourArea(pts))
    perim = cv2.arcLength(pts, True)
    (_, _), (w, h), _ = cv2.minAreaRect(pts)
    long_, short = max(w, h), min(w, h)
    rect_fill = area / max(long_ * short, 1e-12)
    circ = 4 * np.pi * area / max(perim ** 2, 1e-12)
    aspect = long_ / max(short, 1e-9)
    if aspect < 1.15 and (circ > 0.84 or 0.7 < rect_fill < 0.86):     # a disc fills pi/4 of its square
        kind, sym = 'round', 30.0
    elif rect_fill > 0.86 and aspect < 1.2:
        kind, sym = 'box', 90.0
    elif rect_fill > 0.86:
        kind, sym = 'long', 180.0
    else:
        kind, sym = 'irregular', 360.0
    if height < 0.25 * short and height < 0.012:
        kind = 'flat'
    step = max(1, len(local) // 48)
    return dict(kind=kind, symmetry_deg=sym, footprint=(float(long_), float(short)),
                outline=[[round(float(x), 4), round(float(y), 4)] for x, y in local[::step]],
                fill=float(rect_fill), circularity=float(circ))


def obstacles(rgb: np.ndarray, empty_rgb: np.ndarray | None, exclude: np.ndarray, camera, z_table: float,
              change: float = 14.0, min_pixels: int = 150) -> list[np.ndarray]:
    """Everything else on the table, as conservative footprints: the blobs of pixels that differ from the empty
    table (shadows excluded), other than `exclude`, each as the convex hull (N, 2) of its outline on the table plane.
    From above, a taller thing's outline on the table plane is larger than its footprint, so this errs on the safe
    side. Used to keep the arm clear while it looks at a new object."""
    if empty_rgb is None:
        return []
    from perception.candidates import colour_distance
    from perception.dino_local import lab_image
    changed = colour_distance(lab_image(rgb), lab_image(empty_rgb)) > change
    a, b = rgb.astype(np.float32), empty_rgb.astype(np.float32)
    sa, sb = a.sum(-1) + 1e-3, b.sum(-1) + 1e-3
    same_hue = np.linalg.norm(a[..., :2] / sa[..., None] - b[..., :2] / sb[..., None], axis=-1) < 0.02
    changed &= ~(same_hue & (sa < 0.92 * sb))
    changed &= ~(cv2.dilate(exclude.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)
    m = cv2.morphologyEx(changed.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        if cv2.contourArea(c) < min_pixels:
            continue
        pts = pixels_to_plane(camera, c.reshape(-1, 2).astype(float), z_table)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) >= 3:
            out.append(cv2.convexHull(pts[:, :2].astype(np.float32)).reshape(-1, 2).astype(float))
    return out
