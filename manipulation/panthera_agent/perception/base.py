"""Segmenters: an image and a text query in, one mask per matching object out."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from robots.base import Observation

COLOURS = {'red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'white', 'black', 'grey', 'gray', 'brown'}


@dataclass
class Instance:
    """One object found in an image."""
    mask: np.ndarray            # (H, W) bool
    label: str                  # what the segmenter calls it
    score: float                # its confidence (1.0 for ground truth)
    extra: dict = field(default_factory=dict)      # segmenter-specific (dino_local: similarity, colour distance)


class Segmenter(Protocol):
    """Text-prompted instance segmentation."""

    name: str

    def segment(self, obs: Observation, query: str) -> list[Instance]:
        """Every object in `obs` matching `query` ("red cube", "block"), largest first."""


def words(text: str) -> list[str]:
    """Lower-case words of a query or a name ('cube_red' -> ['cube', 'red'])."""
    return [w for w in text.lower().replace('_', ' ').replace('-', ' ').split() if w]
