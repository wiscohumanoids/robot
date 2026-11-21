"""What the agent has found: numbered objects with their estimated top-face pose and size.

Numbers are never reused within a run, so an old number cannot silently point at a new object. An estimate is a
snapshot (`step` says when); after anything moves, find again.

The world also remembers the stacks the agent built itself (`stacks`): how many levels high each one is, by
position. Without depth, `find` can only guess a level from how big the mask looks, and a segmenter's mask of a
tower covers the side faces of every block below the top one, which looks like a taller tower. A remembered level
is used in place of that guess.
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
    level_known: bool = False            # the level comes from a stack the agent built, not from the mask's size

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


@dataclass(eq=False)                     # compared by identity (it holds an array)
class Stack:
    """A stack the agent built: its top face's (x, y) where the last block went, and its height in levels."""
    xy: np.ndarray
    levels: int


@dataclass
class World:
    """The found objects, the held one and the step counter."""
    types: dict[str, ObjectType]
    objects: dict[int, Found] = field(default_factory=dict)
    held: Found | None = None
    stacks: list[Stack] = field(default_factory=list)
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

    def stack_at(self, xy, radius: float = 0.025) -> Stack | None:
        """The built stack whose top is within `radius` (m) of (x, y), or None."""
        near = [s for s in self.stacks if np.linalg.norm(s.xy - np.asarray(xy, float)[:2]) < radius]
        return min(near, key=lambda s: np.linalg.norm(s.xy - np.asarray(xy, float)[:2]), default=None)

    def set_stack(self, xy, levels: int) -> None:
        """Remember that the stack at (x, y) is `levels` high (one level: a block on the table, nothing to keep)."""
        old = self.stack_at(xy)
        self.stacks = [s for s in self.stacks if s is not old]
        if levels > 1:
            self.stacks.append(Stack(np.asarray(xy, float)[:2].copy(), int(levels)))
