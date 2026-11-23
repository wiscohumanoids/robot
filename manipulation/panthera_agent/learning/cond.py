"""What the shared local policy is told besides what it sees: which skill, and which object.

    condition(skill, object_feature) -> (CONDITION_DIM,) float32
                                      = one-hot over SKILLS (8 slots) + the object's DINO feature (384, unit length)

The object feature is the mean DINOv3 feature of the object's pixels in the overhead image, the same thing the
seen-objects list keeps as its examples (perception/memory.py), so the sim's training and the agent's runs describe
an object the same way. One network then covers every local skill and every object, and each new task the
curriculum adds is more data for it rather than a new model.
"""
from __future__ import annotations

import numpy as np

SKILLS = ['insert', 'seat', 'pull', 'pinch', 'pour', 'place', 'grasp', 'other']
FEATURE_DIM = 384
CONDITION_DIM = len(SKILLS) + FEATURE_DIM


def condition(skill: str, feature: np.ndarray | None) -> np.ndarray:
    c = np.zeros(CONDITION_DIM, np.float32)
    c[SKILLS.index(skill) if skill in SKILLS else SKILLS.index('other')] = 1.0
    if feature is not None:
        f = np.asarray(feature, np.float32).ravel()[:FEATURE_DIM]
        n = float(np.linalg.norm(f))
        c[len(SKILLS):len(SKILLS) + len(f)] = f / n if n > 0 else f
    return c


def object_feature(rgb: np.ndarray, mask: np.ndarray, features=None) -> np.ndarray | None:
    """The mean DINO feature of the patches a mask covers at least 40 % of (overhead image)."""
    import cv2
    from perception.dino import PATCH, shared
    f = (features or shared())(rgb)
    gh, gw = f.shape[:2]
    cover = cv2.resize(mask[:gh * PATCH, :gw * PATCH].astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)
    sel = cover >= 0.4
    if not sel.any():
        sel = cover > 0
    if not sel.any():
        return None
    v = f[sel].mean(axis=0)
    return v / (np.linalg.norm(v) + 1e-8)


def memory_feature(entry) -> np.ndarray | None:
    """The same for a learned object: the mean of its overhead examples."""
    ex = entry.fg.get('overhead') if entry is not None else None
    if ex is None or not len(ex):
        return None
    v = np.asarray(ex, np.float32).mean(axis=0)
    return v / (np.linalg.norm(v) + 1e-8)
