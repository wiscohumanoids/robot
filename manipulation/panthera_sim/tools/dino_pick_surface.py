"""Locate raised support surfaces with calibrated multiview DINO descriptors."""

from dataclasses import dataclass

import cv2
import numpy as np


_MIN_BASELINE_M = .020
_MAX_VIEWS = 8
# Broad calibrated workspace envelope. Each support height is estimated
# from multiview DINO descriptor agreement within this interval.
_MIN_HEIGHT_M = -.050
_MAX_HEIGHT_M = .230
_COARSE_HEIGHTS = np.arange(_MIN_HEIGHT_M, _MAX_HEIGHT_M, .005)


@dataclass(frozen=True)
class PlatformObservation:
    polygon: np.ndarray
    center: np.ndarray

    def contains(self, pixel, margin=0.):
        """Test whether a pixel lies at least margin pixels inside the support."""
        return cv2.pointPolygonTest(self.polygon, tuple(np.asarray(pixel, dtype=float)), True) >= margin


@dataclass(frozen=True)
class SurfaceEstimate:
    point: np.ndarray
    normal: np.ndarray
    feature_count: int
    view_count: int
    reprojection_rmse_px: float
    plane_rmse_m: float


def _stone(rgb):
    red, green, blue = np.moveaxis(rgb.astype(np.float32) / 255., -1, 0)
    light = (red + green + blue) / 3
    spread = np.max(rgb, axis=2).astype(float) - np.min(rgb, axis=2)
    mask = (light > .21) & (blue >= .91 * red) & (spread < 67.)
    return light, mask


def _polygons(mask):
    binary = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < binary.size * .006:
            continue
        hull = cv2.convexHull(contour)
        if area < .52 * cv2.contourArea(hull):
            continue
        outline = cv2.approxPolyDP(hull, .019 * cv2.arcLength(hull, True), True)
        if len(outline) >= 3:
            polygons.append((area, outline.astype(np.float32)))
    return polygons


def _split_touching_supports(mask):
    # A rear support can touch the silhouette of the front support. Look
    # for a step in the outer edges where the two nearly horizontal top
    # boundaries meet. A single perspective trapezoid has continuous edges.
    expanded = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                                np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(expanded)
    if count < 2:
        return []
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = mask & (labels == largest)
    row_counts = mask.sum(axis=1)
    rows = np.flatnonzero(row_counts >= 12)
    if len(rows) < 45:
        return []
    first, last = rows[0], rows[-1]
    left = np.argmax(mask, axis=1)
    right = mask.shape[1] - np.argmax(mask[:, ::-1], axis=1) - 1
    width = np.max(right[rows]) - np.min(left[rows])
    edges = np.column_stack((left, right))
    cuts = []
    for y in range(first + 20, last - 19):
        if not np.all(row_counts[y - 4:y + 4] >= 12):
            continue
        jump = np.median(edges[y:y + 4], axis=0) - np.median(edges[y - 4:y], axis=0)
        strength = np.max(np.abs(jump))
        if strength < .19 * width:
            continue
        top, bottom = mask.copy(), mask.copy()
        top[y - 4:] = False
        bottom[:y + 4] = False
        parts = [max(_polygons(band), default=None, key=lambda item: item[0])
                 for band in (top, bottom)]
        if all(part is not None for part in parts):
            cuts.append((strength, parts))
    return max(cuts, default=(0., []), key=lambda cut: cut[0])[1]


def find_platforms(rgb):
    """Segment support outlines and assign temporary IDs within this image."""
    light, stone = _stone(rgb)
    smooth = cv2.GaussianBlur(light, (0, 0), 1.2)
    split = sorted(_polygons(stone & (smooth < .37)) + _polygons(stone & (smooth >= .37)),
                   key=lambda item: -item[0])
    polygons = []
    for _, outline in split + sorted(_polygons(stone), key=lambda item: -item[0]):
        moments = cv2.moments(outline)
        center = np.array([moments["m10"], moments["m01"]]) / moments["m00"]
        if any(board.contains(center) or cv2.pointPolygonTest(
                outline, tuple(board.center), False) >= 0 for board in polygons):
            continue
        polygons.append(PlatformObservation(outline, center))
        if len(polygons) >= 2:
            break
    if len(polygons) == 1:
        split = _split_touching_supports(stone)
        if split:
            polygons = []
            for _, outline in split:
                moments = cv2.moments(outline)
                center = np.array([moments["m10"], moments["m01"]]) / moments["m00"]
                polygons.append(PlatformObservation(outline, center))
    polygons.sort(key=lambda observation: tuple(observation.center[::-1]))
    return dict(enumerate(polygons))


def _rays(camera, pixels):
    normalized = (pixels - camera.center) / camera.focal
    return (camera.forward + normalized[..., 0, None] * camera.right
            - normalized[..., 1, None] * camera.up)


def _projections(camera, xyz):
    offset = xyz - camera.position
    depth = offset @ camera.forward
    safe_depth = np.maximum(depth, 1e-6)
    x = camera.center[0] + camera.focal * (offset @ camera.right) / safe_depth
    y = camera.center[1] - camera.focal * (offset @ camera.up) / safe_depth
    return x, y, depth


_CROP_PIXELS = 96
_TOKEN_PIXELS = 16
_TOKENS = _CROP_PIXELS // _TOKEN_PIXELS


def _coverage(mask):
    return mask.reshape(-1, _TOKENS, _TOKEN_PIXELS, _TOKENS, _TOKEN_PIXELS).mean(axis=(2, 4))


class SurfaceTracker:
    """Match DINO patch descriptors across depth hypotheses for an overhead click."""

    def __init__(self, overhead_rgb, overhead_camera, click_pixel, *, describe, platform_id=None):
        self.camera = overhead_camera
        self.pixel = np.asarray(click_pixel, dtype=float)
        self._describe = describe
        platforms = find_platforms(overhead_rgb)
        hits = [index for index, observation in platforms.items()
                if (platform_id is None or platform_id == index) and observation.contains(self.pixel)]
        self.platform_id = hits[0] if len(hits) == 1 else None
        self._views = []
        self._coarse = []
        self._dirty = True
        self._estimate = None
        if self.platform_id is None:
            return

        observation = platforms[self.platform_id]
        outline = observation.polygon.astype(np.int32)
        x, y, width, height = cv2.boundingRect(outline)
        x -= .11 * width
        y -= .11 * height
        width *= 1.22
        height *= 1.22
        columns = x - .5 + (np.arange(_CROP_PIXELS, dtype=np.float32) + .5) * width / _CROP_PIXELS
        rows = y - .5 + (np.arange(_CROP_PIXELS, dtype=np.float32) + .5) * height / _CROP_PIXELS
        xx, yy = (grid.astype(np.float32) for grid in np.meshgrid(columns, rows))
        pixels = np.stack((xx, yy), axis=-1)
        mask = np.zeros(overhead_rgb.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, outline, 1)
        mask = np.ascontiguousarray(mask & _stone(overhead_rgb)[1], dtype=np.uint8)
        selected = cv2.remap(mask, xx, yy, cv2.INTER_NEAREST) > 0
        coverage = _coverage(selected)[0]
        self._tokens = coverage >= .06
        self._edges = (coverage >= .06) & (coverage <= .94)
        crop = cv2.remap(overhead_rgb, xx, yy, cv2.INTER_LINEAR)
        crop[~selected] = 104
        self._reference = self._describe(crop[None])[0]
        self._directions = _rays(self.camera, pixels)
        self._patches = [(slice(row, row + 2), slice(col, col + 2))
                         for row in range(0, _TOKENS, 2) for col in range(0, _TOKENS, 2)]

    @property
    def view_count(self):
        """Count the reference frame and distinct wrist camera positions."""
        return len(self._views) + 1 if self.platform_id is not None else 0

    def add_frame(self, rgb, camera):
        """Keep a camera position when it adds at least 20 mm of stereo baseline."""
        if self.platform_id is None or len(self._views) >= _MAX_VIEWS:
            return 0
        positions = [self.camera.position] + [view[0].position for view in self._views]
        if min(np.linalg.norm(camera.position - position) for position in positions) < _MIN_BASELINE_M:
            return 0
        stone = _stone(rgb)[1].astype(np.uint8)
        stone = cv2.morphologyEx(stone, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        self._views.append((camera, rgb, stone))
        self._dirty = True
        return 1

    def _xyz(self, heights, directions):
        depth = (np.asarray(heights)[..., None, None] - self.camera.position[2]) / directions[..., 2]
        return self.camera.position + depth[..., None] * directions

    def _describe_view(self, camera, rgb, stone, heights):
        xyz = self._xyz(heights, self._directions)
        x, y, depth = _projections(camera, xyz)
        shape = (-1, _CROP_PIXELS)
        map_x, map_y = x.reshape(shape).astype(np.float32), y.reshape(shape).astype(np.float32)
        mask = cv2.remap(stone, map_x, map_y, cv2.INTER_NEAREST).reshape(depth.shape) > 0
        mask &= depth > .015
        in_frame = (depth > .015) & (x >= 0) & (y >= 0) & (x < rgb.shape[1] - 1) & (y < rgb.shape[0] - 1)
        coverage = _coverage(mask)
        valid = (_coverage(in_frame) >= .98) & self._tokens
        crops = cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR).reshape((*depth.shape, 3))
        crops[~mask] = 104
        scores = np.zeros(valid.shape, dtype=np.float32)
        candidates = np.flatnonzero(((coverage > .30) & self._tokens).sum(axis=(1, 2)) >= 4)
        for batch in np.array_split(candidates, max(1, (len(candidates) + 23) // 24)):
            if len(batch):
                descriptors = self._describe(crops[batch])
                scores[batch] = np.sum(descriptors * self._reference, axis=-1)
        return scores, valid

    def _alignment(self, observed):
        scores, valid = (np.stack(result, axis=1) for result in zip(*observed))
        # A zero-coverage view cannot win merely by matching a single token.
        weights = np.where(self._edges, 1., .20) * valid
        totals = (scores * weights).sum(axis=(2, 3))
        count = weights.sum(axis=(2, 3))
        merit = totals / np.maximum(count, 1)
        merit = np.where(count >= 3., merit, 0.)
        ordered = np.sort(merit, axis=1)
        combined = .60 * ordered[:, -1] + .40 * ordered[:, -2]
        return combined, merit, scores, valid

    def _solve(self):
        if self.platform_id is None or len(self._views) < 2 or self._tokens.sum() < 6:
            return None
        self._coarse.extend(self._describe_view(*view, _COARSE_HEIGHTS)
                            for view in self._views[len(self._coarse):])
        merit, _, _, _ = self._alignment(self._coarse)
        winner = int(np.argmax(merit))
        height = _COARSE_HEIGHTS[winner]
        distant = np.abs(_COARSE_HEIGHTS - height) > .015
        if (winner in (0, len(_COARSE_HEIGHTS) - 1) or merit[winner] < .65
                or merit[winner] - np.max(merit[distant]) < .005):
            return None
        fine = height + np.arange(-.011, .0112, .001)
        fine = fine[(fine >= _MIN_HEIGHT_M) & (fine <= _MAX_HEIGHT_M)]
        merit, individual, scores, valid = self._alignment(
            [self._describe_view(*view, fine) for view in self._views])
        best = int(np.argmax(merit))
        trusted = individual[best] >= .60
        if trusted.sum() < 2:
            return None

        patch_points = []
        token_directions = self._directions[8::_TOKEN_PIXELS, 8::_TOKEN_PIXELS]
        for rows, columns in self._patches:
            if not np.any(self._edges[rows, columns]):
                continue
            patch = scores[:, trusted, rows, columns]
            observed = valid[:, trusted, rows, columns]
            count = observed.sum(axis=(1, 2, 3))
            local = (patch * observed).sum(axis=(1, 2, 3)) / np.maximum(count, 1)
            enough = (observed.sum(axis=(2, 3)) > 0).sum(axis=1) >= 2
            winner = int(np.argmax(np.where(enough & (count >= 3), local, 0.)))
            if winner in (0, len(fine) - 1) or local[winner] < .58:
                continue
            directions = token_directions[rows, columns]
            patch_points.append(self._xyz(fine[winner], directions).mean(axis=(0, 1)))
        if len(patch_points) < 3:
            return None
        xyz = np.asarray(patch_points)
        center = xyz.mean(axis=0)
        _, spread, axes = np.linalg.svd(xyz - center, full_matrices=False)
        normal = axes[-1] * np.sign(axes[-1, 2])
        residual = spread[-1] / np.sqrt(len(xyz))
        if spread[1] < .013 or normal[2] < .96 or residual > .005:
            return None
        direction = _rays(self.camera, self.pixel)
        distance = normal @ (center - self.camera.position) / (normal @ direction)
        if distance <= 0:
            return None
        point = self.camera.position + distance * direction

        patch_x, patch_y, _ = _projections(self.camera, xyz)
        directions = _rays(self.camera, np.column_stack([patch_x, patch_y]))
        distances = ((center - self.camera.position) @ normal) / (directions @ normal)
        planar = self.camera.position + distances[:, None] * directions
        errors = []
        for (camera, _, _), keep in zip(self._views, trusted):
            if keep:
                x, y, _ = _projections(camera, xyz)
                plane_x, plane_y, _ = _projections(camera, planar)
                errors.extend(np.square(x - plane_x) + np.square(y - plane_y))
        return SurfaceEstimate(point, normal, int(self._tokens.sum()), int(trusted.sum() + 1),
                               float(np.sqrt(np.mean(errors))), float(residual))

    def estimate(self):
        """Return the current click-plane intersection when the DINO views agree."""
        if self._dirty:
            self._estimate = self._solve()
            self._dirty = False
        return self._estimate
