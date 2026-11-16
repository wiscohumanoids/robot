"""COCO-style uncompressed run-length encoding of binary masks (column-major, counts start with zeros).

Pure numpy so both the server and the ROS client (no pycocotools there) can use it.
"""
import numpy as np


def encode(mask: np.ndarray) -> dict:
    """bool (H, W) -> {"size": [H, W], "counts": [n0, n1, ...]} (first count = leading zeros)."""
    flat = np.asarray(mask, dtype=bool).ravel(order="F")
    if flat.size == 0:
        return {"size": list(mask.shape), "counts": []}
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate(([0], change, [flat.size]))
    runs = np.diff(bounds).tolist()
    if flat[0]:
        runs = [0] + runs
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": [int(r) for r in runs]}


def decode(rle: dict) -> np.ndarray:
    h, w = rle["size"]
    flat = np.zeros(h * w, dtype=bool)
    pos, val = 0, False
    for n in rle["counts"]:
        if val:
            flat[pos:pos + n] = True
        pos += n
        val = not val
    return flat.reshape((h, w), order="F")


def area(rle: dict) -> int:
    return int(sum(rle["counts"][1::2]))
