"""Numbered candidate points for the LLM to choose from, proposed by the image's own DINO features.

The LLM never types pixel coordinates (pointing benchmarks put frontier models at 7-16 % for that). Instead the code
proposes points and draws them numbered; the LLM answers with numbers ("the mug is 4, 5, 6; its handle is 6").

How the points are made, for one image:
1. Only the table is searched: its outline (from the config) projected into the image is the region of interest.
   Background is what still looks like the empty table: the fixed overhead camera's image of the bare table is taken
   once (a calibration step, `empty_table` in tools), and a patch resembling the empty table's patch at the same
   place is table. Anything put on the table, a towel or a peg, changes its patches; uneven light does not, because
   the empty image has the same light. Without that image: what resembles the table's edge.
2. The other patches, grouped into connected regions, are proposals: things that differ from the background. Each
   proposal gets a point at its most interior patch.
3. A proposal big enough to have parts is clustered again on its own features, and each part gets a point too
   (a mug's body and its handle, a tool's handle and its hook).

Each candidate remembers its patches, so a chosen point brings its whole region along as the object's examples.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from perception.dino import PATCH


@dataclass
class Candidate:
    """One numbered point: where it is drawn, and the patches it stands for."""
    number: int
    uv: tuple[float, float]                # pixel
    patch: tuple[int, int]                 # (row, col)
    patches: np.ndarray                    # (gh, gw) bool: the region this point represents
    proposal: int                          # which proposal (region) it is on
    kind: str                              # 'region' (a whole proposal) or 'part' (a cluster inside one)


@dataclass
class Proposals:
    """The candidates of one image, and its background."""
    candidates: list[Candidate]
    background: np.ndarray                 # (gh, gw) bool
    grid_shape: tuple[int, int]

    def by_number(self, n: int) -> Candidate | None:
        return next((c for c in self.candidates if c.number == n), None)


def _kmeans(x: np.ndarray, k: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    cv2.setRNGSeed(seed)
    k = max(1, min(k, len(x)))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
    _, labels, centres = cv2.kmeans(x.astype(np.float32), k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    return labels.ravel(), centres


def _interior(region: np.ndarray) -> tuple[int, int]:
    """The patch of a region farthest from its border."""
    d = cv2.distanceTransform(np.pad(region.astype(np.uint8), 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
    i, j = np.unravel_index(np.argmax(d * region), region.shape)
    return int(i), int(j)


def commonness(features: np.ndarray, similar: float = 0.5) -> np.ndarray:
    """(gh, gw): the share of the image's patches each patch resembles (cosine above `similar`, after removing the
    image's mean feature: DINO features share a large common component that would make everything look alike)."""
    gh, gw, C = features.shape
    flat = features.reshape(-1, C).astype(np.float32)
    flat = flat - flat.mean(axis=0)
    flat /= np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8
    return ((flat @ flat.T) > similar).mean(axis=1).reshape(gh, gw)


def roi_patches(roi_pixels: np.ndarray, grid_shape) -> np.ndarray:
    """A pixel mask of the region of interest -> the patches wholly inside it."""
    gh, gw = grid_shape
    crop = roi_pixels[:gh * PATCH, :gw * PATCH].astype(np.float32)
    return cv2.resize(crop, (gw, gh), interpolation=cv2.INTER_AREA) > 0.99


def table_background(features: np.ndarray, roi: np.ndarray, table_like: float = 0.82, ring: int = 2) -> np.ndarray:
    """(gh, gw) bool, the fallback without an empty-table image: patches that resemble (cosine above `table_like`) a
    patch on the ROI's inside edge (a ring `ring` patches deep, the image's edges included), or lie outside the ROI.
    Uneven light can make the table's middle look unlike its edge; the empty-table image (`empty_background`) does
    not have that problem."""
    gh, gw, C = features.shape
    padded = np.pad(roi.astype(np.uint8), 1)
    inner = cv2.erode(padded, np.ones((2 * ring + 1, 2 * ring + 1), np.uint8))[1:-1, 1:-1] > 0
    border = roi & ~inner
    if border.sum() < 4:
        return ~roi
    B = features[border]
    if len(B) > 400:
        B = B[np.linspace(0, len(B) - 1, 400).astype(int)]
    like = (features.reshape(-1, C) @ B.T).max(axis=1).reshape(gh, gw)
    return (like > table_like) | ~roi


def patch_lab(rgb: np.ndarray, grid_shape) -> np.ndarray:
    """(gh, gw, 3) mean CIE Lab colour of each patch (L 0-100, a and b about -128..127)."""
    gh, gw = grid_shape
    lab = cv2.cvtColor(rgb[:gh * PATCH, :gw * PATCH], cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    return cv2.resize(lab, (gw, gh), interpolation=cv2.INTER_AREA)


def colour_distance(a: np.ndarray, b: np.ndarray, light_weight: float = 0.5) -> np.ndarray:
    """Lab distance with lightness down-weighted (shading changes lightness, paint changes hue)."""
    d = a - b
    return np.sqrt((light_weight * d[..., 0]) ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)


def empty_background(features: np.ndarray, empty: dict, lab: np.ndarray | None = None, same: float = 0.75,
                     colour: float = 12.0, reach: int = 1) -> np.ndarray:
    """(gh, gw) bool: patches that still look like the empty table at (about) the same place: DINO cosine above
    `same` and Lab colour within `colour` of the empty image's patch within `reach` patches. Colour matters because
    DINO describes texture and shape: a plain orange towel on a plain cream table can look alike to it. (A fixed
    camera, so the table looks the same each time; a moving camera has no such image.)"""
    gh, gw, C = features.shape
    ef, el = empty['features'], empty.get('lab')
    keep = np.zeros((gh, gw), bool)
    for di in range(-reach, reach + 1):
        for dj in range(-reach, reach + 1):
            shift = lambda x: np.roll(np.roll(x, di, axis=0), dj, axis=1)
            ok = (features * shift(ef)).sum(-1) > same
            if lab is not None and el is not None:
                ok &= colour_distance(lab, shift(el)) < colour
            keep |= ok
    return keep


def propose(features: np.ndarray, roi: np.ndarray | None = None, empty: dict | None = None,
            lab: np.ndarray | None = None,
            table_like: float = 0.82, common: float = 0.045, similar: float = 0.5, min_patches: int = 2,
            part_min: int = 6, max_parts: int = 6, exclude: np.ndarray | None = None) -> Proposals:
    """Candidates for an image's (gh, gw, C) features. `roi` (gh, gw) bool: the table's patches (`table_roi`).
    `empty`: {features, lab} of the same camera's empty-table image and `lab` this image's patch colours
    (`patch_lab`): the best background test (`empty_background`); without
    it, the table's edge (`table_background`); without a table, a patch resembling more than `common` of the image.
    `exclude` (gh, gw) bool: patches never proposed."""
    gh, gw, C = features.shape
    if empty is not None and empty['features'].shape == features.shape:
        background = empty_background(features, empty, lab) | (~roi if roi is not None else False)
    elif roi is not None and roi.any():
        background = table_background(features, roi, table_like)
    else:
        background = commonness(features, similar) > common
    if exclude is not None:
        background |= exclude
    fg = (~background).astype(np.uint8)
    n, comp = cv2.connectedComponents(fg, connectivity=8)
    cands: list[Candidate] = []
    regions = []
    for r in range(1, n):
        region = comp == r
        if region.sum() < min_patches:
            background |= region
            continue
        regions.append(region)
    # Biggest first, so the numbering reads roughly by size.
    regions.sort(key=lambda m: -int(m.sum()))
    for p, region in enumerate(regions):
        i, j = _interior(region)
        cands.append(Candidate(0, ((j + 0.5) * PATCH, (i + 0.5) * PATCH), (i, j), region, p, 'region'))
        size = int(region.sum())
        if size >= part_min:
            k = min(max_parts, size // part_min + 1)
            idx = np.argwhere(region)
            # Features plus position, so parts are compact and spread over the region (a long tool's two ends are
            # different parts even where they look alike).
            spread = float(np.ptp(idx, axis=0).max()) + 1.0
            x = np.concatenate([features[region], 0.8 * (idx - idx.mean(0)) / spread], axis=1)
            lab, _ = _kmeans(x, k, seed=p + 1)
            for c in range(k):
                part = np.zeros_like(region)
                part[tuple(idx[lab == c].T)] = True
                if part.sum() < 2:
                    continue
                # A part's point: its most interior patch, skipped if it is the region's own point.
                n2, comp2 = cv2.connectedComponents(part.astype(np.uint8), connectivity=8)
                biggest = max(range(1, n2), key=lambda q: (comp2 == q).sum())
                pi, pj = _interior(comp2 == biggest)
                if (pi, pj) == (i, j):
                    continue
                cands.append(Candidate(0, ((pj + 0.5) * PATCH, (pi + 0.5) * PATCH), (pi, pj), part, p, 'part'))
    for k, c in enumerate(cands, 1):
        c.number = k
    return Proposals(cands, background, (gh, gw))


def detail(proposals: Proposals, letter: str, max_points: int = 30) -> Proposals:
    """Dense points over one region (by its letter in `describe`): one per patch, thinned to `max_points`, spread
    evenly. Each stands for the whole region when chosen as the object, and for its own spot as a part."""
    p = ord(letter.strip().upper()[:1]) - 65
    region = next((c.patches for c in proposals.candidates if c.proposal == p and c.kind == 'region'), None)
    if region is None:
        raise KeyError(letter)
    idx = np.argwhere(region)
    if len(idx) > max_points:                          # farthest-point thinning
        keep = [0]
        d = np.linalg.norm(idx - idx[0], axis=1)
        for _ in range(max_points - 1):
            keep.append(int(np.argmax(d)))
            d = np.minimum(d, np.linalg.norm(idx - idx[keep[-1]], axis=1))
        idx = idx[keep]
    cands = [Candidate(k, ((j + 0.5) * PATCH, (i + 0.5) * PATCH), (int(i), int(j)), region, p, 'part')
             for k, (i, j) in enumerate(idx, 1)]
    return Proposals(cands, proposals.background, proposals.grid_shape)


def draw(rgb: np.ndarray, proposals: Proposals, highlight: dict[int, tuple[int, int, int]] | None = None) -> np.ndarray:
    """The image with each candidate as a numbered dot (region points larger than part points). `highlight`:
    number -> colour for chosen points."""
    img = rgb.copy()
    palette = [(255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 255, 0), (0, 160, 255),
               (255, 90, 90), (200, 120, 255)]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for c in proposals.candidates:
        col = (highlight or {}).get(c.number, palette[c.proposal % len(palette)])
        u, v = int(c.uv[0]), int(c.uv[1])
        r = 6 if c.kind == 'region' else 4
        cv2.circle(img, (u, v), r + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, (u, v), r, col, -1, cv2.LINE_AA)
        text = str(c.number)
        (w, h), _ = cv2.getTextSize(text, font, 0.45, 1)
        x, y = u + r + 2, v - r - 1
        cv2.rectangle(img, (x - 1, y - h - 2), (x + w + 1, y + 2), (0, 0, 0), -1)
        cv2.putText(img, text, (x, y), font, 0.45, col, 1, cv2.LINE_AA)
    return img


def describe(proposals: Proposals) -> str:
    """The points grouped by region, for the LLM."""
    groups: dict[int, list[int]] = {}
    for c in proposals.candidates:
        groups.setdefault(c.proposal, []).append(c.number)
    lines = [f'region {chr(65 + p) if p < 26 else p}: points {", ".join(map(str, ns))}' for p, ns in groups.items()]
    return '\n'.join(lines)


def table_roi(camera, table: dict, image_shape, grid_shape, margin: float = 0.0) -> np.ndarray:
    """The table's patches in a camera's image (its rectangle from the config, at the table's height)."""
    (x0, x1), (y0, y1), z = table['x'], table['y'], float(table['z'])
    pts = np.array([[x0 + margin, y0 + margin, z], [x1 - margin, y0 + margin, z], [x1 - margin, y1 - margin, z],
                    [x0 + margin, y1 - margin, z]])
    # Densify the outline so lens distortion and near-camera clipping are followed.
    dense = np.concatenate([np.linspace(pts[i], pts[(i + 1) % 4], 20) for i in range(4)])
    uv = camera.project(dense)
    uv = uv[np.isfinite(uv).all(axis=1)]
    m = np.zeros(image_shape[:2], np.uint8)
    if len(uv) >= 3:
        cv2.fillPoly(m, [np.round(uv).astype(np.int32)], 1)
    return roi_patches(m, grid_shape)
