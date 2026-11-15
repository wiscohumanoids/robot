"""Mask + depth -> 3D object estimate (camera frame and, given the camera pose, map frame).

Conventions: camera = ROS optical frame (x right, y down, z forward), depth = metres along z,
intrinsics given for the RGB image; the depth image may be a different (e.g. half) resolution.
T_map_cam: 4x4, maps camera-frame points into the map frame.

Robustness: the mask is eroded (boundary pixels often sample the background), invalid depth
(NaN / <= 0) is dropped, and depth outliers beyond a median-absolute-deviation band are removed
(a mug's mask that leaks onto the wall behind it). Extents use the 5th-95th percentile range.

A camera only sees the front surface, so the visible-point box is too thin along the view ray and
its centre sits slightly in front of the object. For compact objects (< `complete_max` m) the box is
completed by assuming the object is about as deep as it is wide (shift the centre away from the
camera, set the depth extent); fixtures (counters, fridges) are left as seen.
"""
from __future__ import annotations

import numpy as np


def erode(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask
    m = mask.copy()
    for _ in range(r):
        m[1:, :] &= mask[:-1, :]
        m[:-1, :] &= mask[1:, :]
        m[:, 1:] &= mask[:, :-1]
        m[:, :-1] &= mask[:, 1:]
        mask = m.copy()
    return m


def nearest_cluster(z: np.ndarray, min_frac: float = 0.25) -> np.ndarray:
    """Split depths at gaps > max(5 cm, 5 % of depth); return the nearest cluster holding >= min_frac of the
    points (a mask that leaks onto the wall behind an object adds a FARTHER cluster), else the largest."""
    zs = np.sort(z)
    gaps = np.diff(zs) > np.maximum(0.05, 0.05 * zs[:-1])
    starts = np.concatenate(([0], np.flatnonzero(gaps) + 1))
    ends = np.concatenate((starts[1:], [zs.size]))
    sizes = ends - starts
    for s, e, n in zip(starts, ends, sizes):
        if n >= min_frac * zs.size:
            return zs[s:e]
    i = int(np.argmax(sizes))
    return zs[starts[i]:ends[i]]


def resize_nearest(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == tuple(shape):
        return mask
    H, W = mask.shape
    h, w = shape
    rows = (np.arange(h) * H / h).astype(int)
    cols = (np.arange(w) * W / w).astype(int)
    return mask[rows[:, None], cols[None, :]]


def lift(mask: np.ndarray, depth: np.ndarray, K: dict, T_map_cam: np.ndarray | None = None,
         erode_px: int = 2, mad_k: float = 3.0, min_points: int = 20, complete_max: float = 0.4) -> dict | None:
    """Returns None if too few valid depth points under the mask."""
    h, w = depth.shape
    sx, sy = w / K["width"], h / K["height"]
    fx, fy = K["fx"] * sx, K["fy"] * sy
    cx, cy = (K["cx"] + 0.5) * sx - 0.5, (K["cy"] + 0.5) * sy - 0.5
    m = resize_nearest(mask, (h, w))
    valid = np.isfinite(depth) & (depth > 0)
    # 1. the object's depth, from the mask core (erosion drops boundary pixels that sample the background)
    r = max(1, int(round(erode_px * w / K["width"])))
    core = erode(m, r) & valid
    if core.sum() < min_points:                    # small objects: erosion ate everything
        core = m & valid
    zc = depth[core]
    if zc.size < min_points:
        return None
    zc = nearest_cluster(zc)
    med = np.median(zc)
    mad = np.median(np.abs(zc - med)) * 1.4826 + 1e-3
    # 2. every mask pixel at that depth (full mask, so extents are not shrunk by the erosion)
    sel = m & valid & (np.abs(depth - med) < mad_k * mad + 0.02)
    v, u = np.nonzero(sel)
    z = depth[v, u]
    if z.size < min_points:
        return None
    pts = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], axis=1)
    lo, hi = np.percentile(pts, 2, axis=0), np.percentile(pts, 98, axis=0)
    out = {
        "n_points": int(z.size),
        "centroid_cam": ((lo + hi) / 2).tolist(),
        "extent_cam": (hi - lo).tolist(),
        "depth_median": float(np.median(z)),
    }
    if T_map_cam is not None:
        R, t = T_map_cam[:3, :3], T_map_cam[:3, 3]
        pm = pts @ R.T + t
        lo_m, hi_m = np.percentile(pm, 5, axis=0), np.percentile(pm, 95, axis=0)
        c_m, e_m = (lo_m + hi_m) / 2, hi_m - lo_m
        completed = False
        horiz = max(e_m[0], e_m[1])
        if horiz < complete_max:
            # view direction in the map plane, from the camera to the object
            d = c_m[:2] - t[:2]
            n = np.linalg.norm(d)
            if n > 1e-6:
                d = d / n
                axis = 0 if abs(d[0]) >= abs(d[1]) else 1               # map axis closest to the view ray
                along, across = e_m[axis], e_m[1 - axis]                 # observed depth vs width
                if across > along:
                    c_m[:2] += d * (across - along) / 2
                    e_m[axis] = across
                    completed = True
        c_cam = R.T @ (c_m - t)
        out.update({
            "centroid_map": c_m.tolist(),
            "extent_map": e_m.tolist(),
            "centroid_cam_completed": c_cam.tolist(),
            "completed": completed,
        })
    return out
