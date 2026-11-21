"""Camera appearance changes that preserve pixel coordinates and instance labels."""

import cv2
import numpy as np


CONDITIONS = ("clean", "lighting", "blur", "sensor", "mixed")


def _lighting(rgb, rng):
    height, width = rgb.shape[:2]
    # Approximate exposure and white balance in linear light. A broad
    # brightness gradient represents uneven bench illumination.
    gain = 2 ** rng.uniform(-1.35, .85)
    red, blue = 2 ** rng.uniform(-.48, .48, size=2)
    x = np.linspace(-1, 1, width, dtype=np.float32)[None, :]
    y = np.linspace(-1, 1, height, dtype=np.float32)[:, None]
    angle = rng.uniform(0, 2 * np.pi)
    gradient = np.exp2(rng.uniform(.15, .65) * (np.cos(angle) * x + np.sin(angle) * y))
    linear = (rgb.astype(np.float32) / 255) ** 2.2
    linear *= (gain * gradient[:, :, None]) * np.array([red, 1, blue], dtype=np.float32)
    return np.rint(255 * np.clip(linear, 0, 1) ** (1 / 2.2)).astype(np.uint8)


def _blur(rgb, rng):
    scale = min(rgb.shape[:2]) / 512
    if rng.random() < .5:
        sigma = max(.15, rng.uniform(.85, 2.15) * scale)
        return cv2.GaussianBlur(rgb, (0, 0), sigma)
    radius = max(1, round(rng.uniform(2, 5) * scale))
    angle = rng.uniform(0, np.pi)
    dx, dy = np.rint(radius * np.array([np.cos(angle), np.sin(angle)])).astype(int)
    kernel = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    cv2.line(kernel, (radius - dx, radius - dy), (radius + dx, radius + dy), 255,
             lineType=cv2.LINE_AA)
    return cv2.filter2D(rgb, -1, kernel.astype(np.float32) / kernel.sum())


def _sensor(rgb, rng):
    # Noise is in eight-bit camera levels. JPEG also introduces chroma
    # subsampling and blocking around the small stud boundaries.
    noise = rng.standard_normal(rgb.shape, dtype=np.float32) * rng.uniform(2, 7)
    noisy = np.rint(np.clip(rgb.astype(np.float32) + noise, 0, 255)).astype(np.uint8)
    quality = int(rng.integers(40, 86))
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(noisy, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Could not encode the augmented camera frame")
    return cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def augment_rgb(rgb, condition, rng):
    """Return an RGB uint8 exposure of the same shape using the supplied random generator."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected a uint8 RGB image")
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown camera condition {condition!r}")
    if condition in ("lighting", "mixed"):
        rgb = _lighting(rgb, rng)
    if condition in ("blur", "mixed"):
        rgb = _blur(rgb, rng)
    if condition in ("sensor", "mixed"):
        rgb = _sensor(rgb, rng)
    return rgb
