"""Pixels to positions and grasp poses, without depth.

An object resting on the table has its top face on a known horizontal plane (table top + the object's height), so a
pixel on that face is the intersection of its ray with the plane: exact geometry, given the camera's calibration.
A mask's outline projected onto that plane gives the top face's footprint, whose minimum-area rectangle gives the
centre, the size and the yaw. Side faces seen at an angle land outside the true footprint; `footprint` reports the
measured size so callers can compare it with the expected one and distrust an occluded or merged mask.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.camera import Camera


def intersect_plane(origin: np.ndarray, directions: np.ndarray, z: float) -> np.ndarray:
    """Points where rays from `origin` along `directions` (N, 3) meet the plane z = `z` (NaN if parallel/behind)."""
    d = np.asarray(directions, float).reshape(-1, 3)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (z - origin[2]) / d[:, 2]
    t = np.where(t > 0, t, np.nan)
    return origin + t[:, None] * d


def pixels_to_plane(camera: Camera, uv: np.ndarray, z: float) -> np.ndarray:
    """(N, 2) pixels -> (N, 3) points on the plane z = `z`, through `camera`."""
    return intersect_plane(camera.position, camera.rays(uv), z)


@dataclass
class Footprint:
    """A mask's top face on its plane: centre (x, y, z), size (long, short) in metres, yaw of the long side (rad)."""
    center: np.ndarray
    size: tuple[float, float]
    yaw: float
    pixels: int


def footprint(mask: np.ndarray, camera: Camera, z: float, z_base: float | None = None) -> Footprint | None:
    """The top face of a mask's object, whose top face lies in the plane z = `z` (and its base on z = `z_base`).

    The outline of the largest connected region is projected onto the plane and fitted with a minimum-area
    rectangle. Seen at an angle, the side faces facing the camera make that projection too big toward the camera.
    With `z_base` (an upright object with vertical sides resting on that plane), the outline is also projected onto
    the base plane, where the side faces push it away from the camera instead; the overlap of the two is the
    footprint itself. None when the mask is empty or its outline does not reach the plane.
    """
    m = np.asarray(mask, np.uint8)
    if m.sum() == 0:
        return None
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    outline = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
    top = pixels_to_plane(camera, outline, z)
    if not np.isfinite(top).all() or len(top) < 3:
        return None
    mm = top[:, :2] * 1000.0                                 # work in millimetres (minAreaRect takes float32)
    if z_base is not None:
        base = pixels_to_plane(camera, outline, z_base)
        if np.isfinite(base).all():
            mm = _overlap(mm, base[:, :2] * 1000.0)
            if mm is None:
                return None
    (cx, cy), (w, h), angle = cv2.minAreaRect(mm.astype(np.float32))
    yaw = np.deg2rad(angle)
    if w < h:                                               # report the long side's direction
        w, h = h, w
        yaw += np.pi / 2
    yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2
    return Footprint(np.array([cx / 1000.0, cy / 1000.0, z]), (w / 1000.0, h / 1000.0), float(yaw), int(m.sum()))


def _overlap(a_mm: np.ndarray, b_mm: np.ndarray, resolution: float = 0.25) -> np.ndarray | None:
    """Outline of the overlap of two polygons (millimetres), rasterised at `resolution` mm."""
    lo = np.minimum(a_mm.min(0), b_mm.min(0)) - 2
    hi = np.maximum(a_mm.max(0), b_mm.max(0)) + 2
    size = np.ceil((hi - lo) / resolution).astype(int) + 1
    if size.max() > 4000:                                     # over a metre across: not an object on the table
        return None

    def raster(poly):
        img = np.zeros((size[1], size[0]), np.uint8)
        cv2.fillPoly(img, [np.round((poly - lo) / resolution).astype(np.int32)], 1)
        return img

    both = raster(a_mm) & raster(b_mm)
    contours, _ = cv2.findContours(both, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
    return c * resolution + lo


def fold_yaw(yaw: float, symmetry: float, toward: float = 0.0) -> float:
    """The equivalent yaw (yaw + k * symmetry) closest to `toward`."""
    k = np.round((toward - yaw) / symmetry)
    return float(yaw + k * symmetry)


def rot_z(yaw: float) -> np.ndarray:
    """Rotation by `yaw` about +z."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rot_y(angle: float) -> np.ndarray:
    """Rotation by `angle` about +y."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def gripper_rotation(yaw: float, pitch: float) -> np.ndarray:
    """The gripper frame for a pitched approach: approach axis (+x) tipped `pitch` below horizontal toward +x after
    turning by `yaw`; the jaws close along +y, so they close across the direction `yaw` + 90 deg.

    The same construction as panthera_sim's scripted collector (`grasp_rotation`): yaw about z, then pitch about y.
    """
    return rot_z(yaw) @ rot_y(pitch)



def rectangle(center_xy, size, yaw) -> np.ndarray:
    """(4, 2) corners of a rectangle (long side `size[0]` along `yaw`)."""
    u = np.array([np.cos(yaw), np.sin(yaw)]) * size[0] / 2
    v = np.array([-np.sin(yaw), np.cos(yaw)]) * size[1] / 2
    c = np.asarray(center_xy, float)
    return np.array([c + u + v, c - u + v, c - u - v, c + u - v])


def convex_hull(points: np.ndarray) -> np.ndarray:
    """(N, 2) points -> their convex hull's vertices, in order."""
    return cv2.convexHull(np.asarray(points, np.float32)).reshape(-1, 2).astype(float)


def _segment_distance(p, a, b) -> float:
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-18), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def polygon_gap(a: np.ndarray, b: np.ndarray) -> float:
    """Distance between two convex polygons (vertices in order); 0 when they overlap."""
    for poly in (a, b):                                  # separating axis test on both polygons' edge normals
        for i in range(len(poly)):
            e = poly[(i + 1) % len(poly)] - poly[i]
            n = np.array([-e[1], e[0]])
            pa, pb = a @ n, b @ n
            if pa.max() < pb.min() or pb.max() < pa.min():
                break
        else:
            continue
        break
    else:
        return 0.0
    gaps = [_segment_distance(p, q[i], q[(i + 1) % len(q)]) for p, q in ((p, b) for p in a) for i in range(len(q))]
    gaps += [_segment_distance(p, q[i], q[(i + 1) % len(q)]) for p, q in ((p, a) for p in b) for i in range(len(q))]
    return float(min(gaps))
