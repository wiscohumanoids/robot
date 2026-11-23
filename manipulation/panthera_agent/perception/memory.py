"""The seen-objects list: what the agent has learned about each object it was shown, kept on disk between runs.

An entry is made the first time an object is seen (the `learn` tool: the LLM chooses numbered points on it), and
grows as the object is used:

- appearance: DINO feature examples of the object and of what surrounds it, per camera (the overhead camera's at
  first; the wrist camera's are added the first time the wrist sees it confidently), and its mean colour;
- geometry, measured, not given: height (from the overhead and wrist views together), footprint, outline, a shape
  class and the yaw symmetry that follows from it;
- parts: named points the LLM chose on it (a handle, a rim, a hook, a corner), each with where it sits on the object
  and its own feature example, so later views can find it again;
- experience: grasps tried on it and how they went.

Files: `memory/<slug>/entry.json` (everything but arrays), `features.npz`, `thumb.jpg`. The folder is per robot (the
features depend on its cameras), so it is not kept in the repository; set PANTHERA_AGENT_MEMORY to use another.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import time
from dataclasses import asdict, dataclass, field

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
STOP = {'the', 'a', 'an', 'of', 'with', 'and', 'on', 'small', 'big', 'large', 'little'}


def slug(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_') or 'object'


def name_words(text: str) -> list[str]:
    return [w.rstrip('s') if len(w) > 3 and not w.endswith('ss') else w
            for w in re.findall(r'[a-z0-9]+', text.lower()) if w not in STOP]


@dataclass
class Part:
    """A named point on an object, in the object's frame (origin at its top face's centre, x along its yaw)."""
    name: str
    offset: list[float]                      # (x, y, z) metres
    feature: list[float] | None = None       # its DINO example (overhead)


@dataclass
class Shape:
    """What the geometry measured."""
    height: float                            # m, table to top
    footprint: tuple[float, float]           # long, short (m)
    kind: str                                # box | round | long | irregular | flat
    symmetry_deg: float                      # the yaw repeats every this many degrees
    outline: list[list[float]] = field(default_factory=list)     # top-view outline, object frame, metres
    deformable: bool = False


@dataclass
class Entry:
    """One seen object."""
    name: str
    words: list[str]
    colour_lab: list[float]                                     # mean colour (CIE Lab)
    shape: Shape | None = None
    parts: dict[str, Part] = field(default_factory=dict)
    grasps: list[dict] = field(default_factory=list)            # {yaw_rel_deg, part, ok, width, time}
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    notes: str = ''
    colours: list[list[float]] = field(default_factory=list)    # its main colours (Lab), a mug: outside and inside
    # arrays (features.npz): fg/bg examples per camera
    fg: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    bg: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def palette(self) -> np.ndarray:
        """(K, 3) its colours, or the mean colour alone."""
        return np.asarray(self.colours or [self.colour_lab], np.float32)

    @property
    def slug(self) -> str:
        return slug(self.name)

    def matches(self, query: str) -> float:
        """How well a query names this object: 0 (not at all) to 1. Every content word of the name must be in the
        query, or the query must name one of its words (and no conflicting colour)."""
        from perception.base import COLOURS
        q = set(name_words(query))
        mine = set(name_words(self.name)) | set(self.words)
        colours_q = q & COLOURS
        colours_me = set(name_words(self.name)) & COLOURS
        if colours_q and colours_me and not colours_q & colours_me:
            return 0.0
        nouns_q = q - COLOURS
        if not nouns_q & mine:
            return 0.0
        return len(q & mine) / max(len(q), 1)

    def summary(self) -> str:
        """One line for the LLM."""
        if self.shape is None:
            return f'{self.name}: appearance only (shape not measured yet)'
        s = self.shape
        parts = f', parts: {", ".join(self.parts)}' if self.parts else ''
        ok = sum(g['ok'] for g in self.grasps)
        grasps = f', grasped {ok}/{len(self.grasps)}' if self.grasps else ''
        return (f'{self.name}: {s.kind}, {s.footprint[0] * 1000:.0f} x {s.footprint[1] * 1000:.0f} mm, '
                f'{s.height * 1000:.0f} mm tall{" (deformable)" if s.deformable else ""}{parts}{grasps}')


class ObjectMemory:
    """The seen-objects list on disk."""

    def __init__(self, folder: str | pathlib.Path | None = None):
        self.folder = pathlib.Path(folder or os.environ.get('PANTHERA_AGENT_MEMORY') or ROOT / 'memory')
        self.entries: dict[str, Entry] = {}
        self.load()

    # ---------- disk ----------
    def load(self) -> None:
        self.entries = {}
        if not self.folder.exists():
            return
        for d in sorted(self.folder.iterdir()):
            f = d / 'entry.json'
            if not f.exists():
                continue
            j = json.loads(f.read_text())
            shape = Shape(**j['shape']) if j.get('shape') else None
            parts = {k: Part(**v) for k, v in (j.get('parts') or {}).items()}
            e = Entry(j['name'], j['words'], j['colour_lab'], shape, parts, j.get('grasps', []), j.get('created', 0),
                      j.get('updated', 0), j.get('notes', ''), j.get('colours', []))
            npz = d / 'features.npz'
            if npz.exists():
                z = np.load(npz)
                for k in z.files:
                    kind, cam = k.split(':', 1)
                    (e.fg if kind == 'fg' else e.bg)[cam] = z[k]
            self.entries[e.slug] = e

    def save(self, e: Entry, thumb: np.ndarray | None = None) -> None:
        e.updated = time.time()
        d = self.folder / e.slug
        d.mkdir(parents=True, exist_ok=True)
        j = {k: v for k, v in asdict(e).items() if k not in ('fg', 'bg')}
        (d / 'entry.json').write_text(json.dumps(j, indent=1))
        arrays = {f'fg:{c}': a.astype(np.float16) for c, a in e.fg.items()}
        arrays |= {f'bg:{c}': a.astype(np.float16) for c, a in e.bg.items()}
        np.savez_compressed(d / 'features.npz', **arrays)
        if thumb is not None:
            import cv2
            cv2.imwrite(str(d / 'thumb.jpg'), cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR))
        self.entries[e.slug] = e

    def forget(self, name: str) -> bool:
        import shutil
        s = slug(name)
        if s not in self.entries:
            return False
        shutil.rmtree(self.folder / s, ignore_errors=True)
        del self.entries[s]
        return True

    # ---------- lookup ----------
    def lookup(self, query: str) -> list[Entry]:
        """Entries a query names, best first ("mug" -> purple mug; "purple mug" -> only that one)."""
        scored = [(e.matches(query), e) for e in self.entries.values()]
        scored = [(s, e) for s, e in scored if s > 0]
        if not scored:
            return []
        best = max(s for s, _ in scored)
        # A query that names one object fully ("purple mug") keeps only the best; a generic one keeps all matches.
        return [e for s, e in sorted(scored, key=lambda x: -x[0]) if s >= best - 1e-9 or best < 0.99]

    def thumb_path(self, name: str) -> pathlib.Path:
        return self.folder / slug(name) / 'thumb.jpg'

    def listing(self) -> str:
        if not self.entries:
            return 'No objects learned yet.'
        return '\n'.join('- ' + e.summary() for e in self.entries.values())
