"""Select image detections and orient a grasp from their stud pattern."""

import numpy as np
from PIL import Image, ImageDraw


class Miss(Exception):
    """A trial stopped at a defined perception or motion gate."""


def jaw_axes(studs):
    """Candidate jaw headings from reconstructed horizontal stud positions in meters."""
    studs = np.asarray(studs)[:, :2] if len(studs) else np.empty((0, 2))
    if len(studs) >= 2:
        _, singular, axes = np.linalg.svd(studs - studs.mean(axis=0), full_matrices=False)
        # The two side axes of a 2x2 stud grid are equivalent grasps.
        # Average adjacent-stud angles modulo a quarter turn to avoid
        # aiming the jaws across the block's wider diagonal.
        offsets = [a - b for i, a in enumerate(studs) for b in studs[:i]]
        sides = [v for v in offsets if .023 <= np.linalg.norm(v) <= .038]
        if singular[1] > .008 and len(sides) >= 2:
            phase = np.mean([np.exp(4j * np.arctan2(v[1], v[0])) for v in sides])
            if abs(phase) > .7:
                axis = np.angle(phase) / 4
                return [axis, axis + np.pi / 2]
        if singular[0] > 1.4 * singular[1]:
            return [np.arctan2(axes[0, 1], axes[0, 0]) + np.pi / 2]
    return [0., np.pi / 4, np.pi / 2, -np.pi / 4]


def mark(path, rgb, blocks=(), source=None, destination=None):
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    for block in blocks:
        draw.rectangle(block.bbox_xyxy, outline="white")
    for point, color in ((source, "lime"), (destination, "magenta")):
        if point is not None:
            x, y = point
            draw.ellipse((x-6, y-6, x+6, y+6), outline=color, width=2)
    image.save(path)


def clicked_block(blocks, pixel):
    x, y = pixel
    hits = [b for b in blocks if b.bbox_xyxy[0] <= x < b.bbox_xyxy[2]
            and b.bbox_xyxy[1] <= y < b.bbox_xyxy[3]]
    if not hits:
        raise Miss("source_outside_detection")
    if len(hits) > 1:
        raise Miss("source_overlaps_detections")
    if hits[0].color is None:
        raise Miss("source_color_unknown")
    return hits[0]
