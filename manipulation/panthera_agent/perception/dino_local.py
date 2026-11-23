"""`dino_local`: find learned objects by DINO feature similarity, on this machine (no perception server).

For a query ("purple mug", "mug"), the seen-objects list (`perception/memory.py`) gives the matching entries. For each
one, every patch of the image is scored

    score = max cosine to the object's examples  -  max cosine to the table examples

and a patch also needs some of its pixels in the object's colour (`colour_share`). The two do different jobs. A
tabletop object seen from 85 cm is a few patches across, and at that size DINO's features say "an object, not
table" very reliably but tell one small object from another poorly (a red cube and a teal puck: cosine about 0.9);
colour tells them apart. The share of object-coloured pixels is used rather than a patch's average colour, because
an edge patch is half table. Patches passing both form connected regions; each region is an instance. Two learned
objects of the same colour and size stay ambiguous: `find` reports both and the LLM tells them apart by position. Its mask is then sharpened at the pixel level by colour
(patches are 16 pixels; an object's edge is not), when the object is evenly coloured enough for that to work.

The masks then go through the same geometry as every other segmenter's (core/geometry.py): footprint, height, yaw.
"""
from __future__ import annotations

import cv2
import numpy as np

from perception.base import Instance
from perception.candidates import colour_distance
from perception.dino import PATCH, shared, upsample
from perception.memory import Entry, ObjectMemory
from robots.base import Observation


def _examples(by_camera: dict[str, np.ndarray], camera: str) -> np.ndarray | None:
    """An entry's examples for a camera: that camera's own first, then the others' (DINO features carry over between
    views of the same object reasonably well)."""
    if not by_camera:
        return None
    own = by_camera.get(camera)
    rest = [a for c, a in by_camera.items() if c != camera]
    parts = ([own] if own is not None else []) + rest
    return np.concatenate(parts).astype(np.float32)


class DinoLocalSegmenter:
    """Learned objects, found by feature similarity."""

    name = 'dino_local'

    def __init__(self, memory: ObjectMemory | None = None, margin: float = 0.0, colour: float = 22.0,
                 colour_share: float = 0.15, min_pixels: int = 60, features=None):
        self.memory = memory or ObjectMemory()
        self.margin, self.colour, self.colour_share, self.min_pixels = margin, colour, colour_share, min_pixels
        self._features = features
        self.last_timing = None
        self.last_heat: np.ndarray | None = None          # an RGB picture of the last score map, for the viewer

    @property
    def features(self):
        if self._features is None:
            self._features = shared()
        return self._features

    def score_map(self, f: np.ndarray, pix_lab: np.ndarray, entry: Entry, camera: str,
                  rgb: np.ndarray | None = None) -> np.ndarray:
        """(gh, gw) score of each patch for `entry`."""
        gh, gw, C = f.shape
        flat = f.reshape(-1, C).astype(np.float32)
        fg = _examples(entry.fg, camera)
        bg = _examples(entry.bg, camera)
        pos = (flat @ fg.T).max(axis=1)
        negs = (flat @ bg.T).max(axis=1) if bg is not None and len(bg) else np.zeros(len(flat))
        s = (pos - negs).reshape(gh, gw)
        near = (palette_distance(pix_lab, entry, rgb) < self.colour).astype(np.float32)
        share = cv2.resize(near[:gh * PATCH, :gw * PATCH], (gw, gh), interpolation=cv2.INTER_AREA)
        return np.where(share < self.colour_share, np.minimum(s, -0.05), s)

    def segment(self, obs: Observation, query: str) -> list[Instance]:
        import time
        t0 = time.perf_counter()
        entries = self.memory.lookup(query)
        if not entries:
            self.last_timing = None
            return []
        f = self.features(obs.rgb)
        pix_lab = lab_image(obs.rgb)
        out = []
        heat = np.full(f.shape[:2], -1.0, np.float32)
        for e in entries:
            s = self.score_map(f, pix_lab, e, obs.camera.name, obs.rgb)
            heat = np.maximum(heat, s)
            up = upsample(s, obs.rgb.shape)
            n, comp = cv2.connectedComponents((up > self.margin).astype(np.uint8), connectivity=8)
            for c in range(1, n):
                blob = comp == c
                if blob.sum() < self.min_pixels:
                    continue
                mask, how = self._sharpen(blob, pix_lab, e, obs.rgb)
                out.append(Instance(mask, e.name, float(np.clip(up[blob].mean() * 2, 0, 1)),
                                    extra=dict(similarity=round(float(up[blob].max()), 3), mask=how,
                                               colour_distance=round(float(np.median(
                                                   palette_distance(pix_lab[mask][None], e))), 1))))
        self.last_heat = _heat_picture(obs.rgb, heat)
        self.last_timing = dict(total=(time.perf_counter() - t0) * 1000)
        return sorted(out, key=lambda i: -int(i.mask.sum()))

    def _sharpen(self, blob: np.ndarray, pix_lab: np.ndarray, e: Entry, rgb=None) -> tuple[np.ndarray, str]:
        """The blob's mask at pixel resolution: the largest connected set of the object's colours within two patches
        of the blob. Kept when the blob confirms most of it and it is a plausible size (an object of a few clear
        colours); otherwise the blob itself (patch resolution)."""
        near = cv2.dilate(blob.astype(np.uint8), np.ones((2 * PATCH + 1, 2 * PATCH + 1), np.uint8)) > 0
        close = palette_distance(pix_lab, e, rgb) < self.colour
        m = near & close
        n, comp = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
        if n <= 1:
            return blob, 'patches'
        sizes = np.bincount(comp.ravel())[1:]
        m = comp == 1 + int(np.argmax(sizes))
        inside = (m & blob).sum() / max(m.sum(), 1)            # how much of the colour mask the blob confirms
        ratio = m.sum() / max(blob.sum(), 1)
        if inside > 0.6 and 0.25 < ratio < 1.8:
            m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
            return m, 'colour'
        return blob, 'patches'


def _lab_to_chroma(lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(chromaticity (..., 2): r and g over r + g + b; brightness (...)) of Lab colours (this module's scaling)."""
    x = np.asarray(lab, np.float32).reshape(-1, 1, 3).copy()
    x[..., 0] *= 255.0 / 100.0
    x[..., 1:] += 128.0
    rgb = cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32).reshape(
        *np.shape(lab)[:-1], 3)
    total = rgb.sum(-1) + 1e-3
    return rgb[..., :2] / total[..., None], total


def palette_distance(lab: np.ndarray, entry: Entry, rgb: np.ndarray | None = None) -> np.ndarray:
    """Each pixel's colour distance to the nearest of the object's colours (Lab, lightness half-weighted).

    Shadows: with the image's `rgb`, a pixel darker than one of the object's colours but of the same chromaticity
    (r, g over r + g + b, which shading leaves alone) counts as that colour in shadow, at its chromaticity distance
    (scaled to Lab-like units). The arm's own shadow falls across objects in the wrist view."""
    pal = entry.palette()
    d = np.min(np.stack([colour_distance(lab, c) for c in pal]), axis=0)
    if rgb is None:
        return d
    rgbf = rgb.astype(np.float32)
    total = rgbf.sum(-1) + 1e-3
    chroma = rgbf[..., :2] / total[..., None]
    pc, pt = _lab_to_chroma(pal)
    for c, t in zip(pc, pt):
        sat = float(np.abs(c - 1 / 3).max())
        if sat < 0.04:                            # grey-ish colours: chromaticity says nothing about them
            continue
        darker = (total < t * 0.95) & (total > 60)
        dc = np.linalg.norm(chroma - c, axis=-1) * 400.0
        d = np.where(darker, np.minimum(d, dc), d)
    return d


def palette_of(lab_pixels: np.ndarray, k: int = 3, min_share: float = 0.12) -> list[list[float]]:
    """The main colours of an object's pixels: k-means in Lab, clusters with at least `min_share` of the pixels."""
    x = lab_pixels.reshape(-1, 3).astype(np.float32)
    if len(x) > 4000:
        x = x[np.linspace(0, len(x) - 1, 4000).astype(int)]
    if len(x) < k * 5:
        return [x.mean(0).tolist()]
    cv2.setRNGSeed(0)
    _, lab, c = cv2.kmeans(x, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5), 3,
                           cv2.KMEANS_PP_CENTERS)
    share = np.bincount(lab.ravel(), minlength=k) / len(x)
    return [c[i].tolist() for i in np.argsort(-share) if share[i] >= min_share]


def lab_image(rgb: np.ndarray) -> np.ndarray:
    """(H, W, 3) CIE Lab, L 0-100, a and b centred on 0."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    return lab


def _heat_picture(rgb: np.ndarray, heat: np.ndarray) -> np.ndarray:
    """The image dimmed, with the score map in colour where it is positive."""
    up = upsample(heat, rgb.shape)
    v = np.clip(up / 0.4, 0, 1)
    colour = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]
    out = (rgb * 0.35).astype(np.uint8)
    pos = up > 0
    out[pos] = (0.4 * rgb[pos] + 0.6 * colour[pos]).astype(np.uint8)
    return out
