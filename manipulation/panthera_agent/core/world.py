"""What the agent has found: numbered objects with their estimated top-face pose and size.

Numbers are never reused within a run, so an old number cannot silently point at a new object. An estimate is a
snapshot (`step` says when); after anything moves, find again.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ObjectType:
    """What `find` needs to know about a kind of object (config `objects:`)."""
    name: str
    words: set[str]
    height: float
    footprint: tuple[float, float]       # long, short
    symmetry: float                      # radians


@dataclass
class Found:
    """One object seen by `find`."""
    number: int
    query: str
    label: str
    type: ObjectType
    top: np.ndarray                      # centre of the top face, base frame
    yaw: float                           # long side's direction, radians
    size: tuple[float, float]            # measured long, short
    score: float
    camera: str
    step: int
    trusted: bool                        # measured size close to the expected one (False: occluded or merged)
    level: int = 1                       # 1 on the table, 2 on top of one object of its height, ...
    raw_top: np.ndarray | None = None    # the camera's own measurement, before the overhead correction
    refined: bool = False                # corrected by the wrist camera since it was found

    @property
    def center(self) -> np.ndarray:
        """The object's centre."""
        return self.top - np.array([0.0, 0.0, self.type.height / 2])

    def describe(self) -> str:
        """One line for the LLM."""
        flag = '' if self.trusted else ' (partly hidden or merged with something: size does not match)'
        return (f'#{self.number} {self.label}: top centre x={self.top[0] * 100:.1f} y={self.top[1] * 100:.1f} '
                f'z={self.top[2] * 100:.1f} cm, {self.size[0] * 1000:.0f} x {self.size[1] * 1000:.0f} mm, '
                f'yaw {np.rad2deg(self.yaw):.0f} deg, score {self.score:.2f}{flag}')


@dataclass
class World:
    """The found objects, the held one and the step counter."""
    types: dict[str, ObjectType]
    objects: dict[int, Found] = field(default_factory=dict)
    held: Found | None = None
    step: int = 0
    _next: int = 1

    @staticmethod
    def from_config(objects: dict) -> 'World':
        """Object types from config `objects:` (symmetry in degrees there)."""
        types = {name: ObjectType(name, set(o['words']) | {name}, float(o['height']),
                                  (float(o['footprint'][0]), float(o['footprint'][1])), np.deg2rad(o['symmetry']))
                 for name, o in objects.items()}
        return World(types)

    def type_for(self, query: str) -> ObjectType | None:
        """The object type a query names ("red block" -> cube), or None."""
        q = query.lower().replace('_', ' ').split()
        for t in self.types.values():
            if any(w in t.words or w.rstrip('s') in t.words for w in q):
                return t
        return None

    def add(self, **kw) -> Found:
        """Record a new found object under the next number."""
        f = Found(number=self._next, step=self.step, **kw)
        self.objects[f.number] = f
        self._next += 1
        return f
