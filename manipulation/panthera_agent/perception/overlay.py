"""Drawing for the LLM: numbered outlines on found objects, and a metric grid on the table.

The LLM answers "which one" by number and "where on the table" by grid coordinates, never by reading pixel
positions off the image.
"""
from __future__ import annotations

import cv2
import numpy as np

from core.camera import Camera

_PALETTE = [(255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 255, 0), (0, 128, 255),
            (255, 255, 255), (255, 0, 128)]


def _label(img, text, xy, colour):
    """Text on a dark box at `xy` (its centre)."""
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    (w, h), base = cv2.getTextSize(text, font, scale, thick)
    x, y = int(xy[0] - w / 2), int(xy[1] + h / 2)
    cv2.rectangle(img, (x - 3, y - h - 4), (x + w + 3, y + base), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, scale, colour, thick, cv2.LINE_AA)


def draw_instances(rgb: np.ndarray, items: list[tuple[int, np.ndarray]]) -> np.ndarray:
    """Copy of `rgb` with each (number, mask) outlined and its number at the mask's centre."""
    img = rgb.copy()
    for k, (number, mask) in enumerate(items):
        colour = _PALETTE[k % len(_PALETTE)]
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, colour, 2, cv2.LINE_AA)
        ys, xs = np.nonzero(mask)
        _label(img, f'#{number}', (xs.mean(), ys.mean()), colour)
    return img


def draw_table_grid(rgb: np.ndarray, camera: Camera, z: float, x_range, y_range, step: float = 0.1) -> np.ndarray:
    """Copy of `rgb` with grid lines every `step` metres on the plane z = `z` (x, y in the robot's base frame), each
    labelled in centimetres at one end ("x=40" along the +y edge, "y=-10" along the near edge), so a place on the
    table can be named by coordinates."""
    img = rgb.copy()
    xs = np.arange(np.ceil(x_range[0] / step - 1e-9) * step, x_range[1] + 1e-9, step)
    ys = np.arange(np.ceil(y_range[0] / step - 1e-9) * step, y_range[1] + 1e-9, step)

    def line(a, b):
        uv = camera.project(np.linspace(a, b, 20))
        uv = uv[np.isfinite(uv).all(axis=1)].astype(np.int32)
        if len(uv) > 1:
            cv2.polylines(img, [uv], False, (70, 70, 70), 1, cv2.LINE_AA)

    def tag(text, point):
        uv = camera.project(np.array([point]))[0]
        if np.isfinite(uv).all():
            font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.45
            (w, h), _ = cv2.getTextSize(text, font, scale, 1)
            x, y = int(uv[0] - w / 2), int(uv[1] + h / 2)
            cv2.rectangle(img, (x - 2, y - h - 3), (x + w + 2, y + 3), (255, 255, 255), -1)
            cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), 1, cv2.LINE_AA)

    for x in xs:
        line([x, y_range[0], z], [x, y_range[1], z])
        tag(f'x={x * 100:.0f}', [x, y_range[1], z])
    for y in ys:
        line([x_range[0], y, z], [x_range[1], y, z])
        tag(f'y={y * 100:.0f}', [x_range[0], y, z])
    return img
