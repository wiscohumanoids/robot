"""Ground-truth segmentation from the sim's segmentation render (S1): perfect masks, so every remaining error is
the geometry's or the motion's.

A query matches a scene body when its object word names one of the body's object types (config `objects:` words:
"block" finds the cubes), or is itself one of the body's name words ("mug" finds mug_purple, for learned objects),
and any colour it names is in the body's name (cube_red is red).
"""
from __future__ import annotations

from perception.base import COLOURS, Instance, words
from robots.base import Observation


class SimTruthSegmenter:
    """Masks straight from MuJoCo: which body is under each pixel."""

    name = 'sim_truth'

    def __init__(self, objects: dict):
        # body-name word that marks an object type -> the words a query may use for it
        self.types = {t: set(spec['words']) | {t} for t, spec in objects.items()}

    def _matches(self, query: list[str], body: list[str]) -> bool:
        colours = [w for w in query if w in COLOURS]
        if any(c not in body for c in colours):
            return False
        for t, vocab in self.types.items():
            if t in body and any(w in vocab or w.rstrip('s') in vocab for w in query):
                return True
        # Any other object (a learned one: "purple mug" -> body mug_purple): a query noun is the body's own word.
        nouns = [w.rstrip('s') if len(w) > 3 else w for w in query if w not in COLOURS]
        return any(n in body for n in nouns)

    def segment(self, obs: Observation, query: str) -> list[Instance]:
        """One mask per matching body, largest first."""
        seg, names = obs.extra['segmentation'], obs.extra['body_names']
        q = words(query)
        out = []
        for bid, name in enumerate(names):
            if name and self._matches(q, words(name)):
                mask = seg == bid
                if mask.any():
                    out.append(Instance(mask, name, 1.0))
        return sorted(out, key=lambda i: -int(i.mask.sum()))
