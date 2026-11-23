"""The agent's sim scenes beyond the three cubes: odd shapes, a peg and its hole, a hook tool, a cup of beads, a towel.

Each scene is an MJCF file next to panthera_sim's own (`../panthera_sim/sim/panthera/agent_<name>.xml`, written by
`python -m scenes.build`), because the arm's model is included by a relative path. The objects are plain MuJoCo
geoms: what the agent knows about them comes from perception (sim_truth or dino_local) and from what it measures or
learns, never from this file. This file is only the simulator's side, plus how to lay a scene out on reset.

Body names are "<kind>_<colour>" (can_yellow), so the ground-truth segmenter's colour + word matching keeps working.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TABLE_Z = 0.05
PAD_PAIR = ('condim="6" friction="0.9 0.9 0.04 0.002 0.002" solref="0.005 1" solimp="0.98 0.99 0.001"')
COLOURS = {'red': '0.85 0.2 0.18 1', 'green': '0.25 0.7 0.3 1', 'blue': '0.2 0.4 0.85 1',
           'yellow': '0.92 0.78 0.15 1', 'orange': '0.95 0.5 0.12 1', 'purple': '0.55 0.3 0.75 1',
           'white': '0.92 0.92 0.9 1', 'grey': '0.32 0.33 0.36 1', 'brown': '0.5 0.33 0.2 1', 'teal': '0.1 0.6 0.6 1'}


@dataclass
class Geom:
    """One MuJoCo geom of an object, in the body's frame (z = 0 is the body's base on the table)."""
    type: str                                  # box | cylinder | sphere | capsule
    size: tuple                                # MuJoCo half-sizes (box), (radius, half-height) for cylinders
    pos: tuple = (0.0, 0.0, 0.0)
    euler: tuple | None = None                 # degrees
    name: str | None = None


@dataclass
class Body:
    """An object in a scene: free (it can be picked and pushed) or fixed (a fixture)."""
    name: str
    colour: str
    geoms: list[Geom]
    free: bool = True
    pos: tuple | None = None                   # base position on the table; None: placed at random on reset
    yaw: float | None = None                   # degrees; None: random on reset
    density: float = 500.0
    graspable: bool = True                     # gets the finger pads' high-friction contact pairs
    radius: float = 0.03                       # keep-apart radius for the random layout
    condim: int = 6                            # 3 for containers: rolling friction would hold beads in

    @property
    def height(self) -> float:
        return max(g.pos[2] + (g.size[2] if g.type == 'box' else g.size[1] if g.type in ('cylinder', 'capsule')
                               else g.size[0]) for g in self.geoms)


@dataclass
class Scene:
    """A scene: objects, an optional towel, beads in a cup, the table's extent, and where random objects go."""
    name: str
    bodies: list[Body]
    description: str
    table_x: tuple = (0.18, 0.62)
    table_y: tuple = (-0.32, 0.32)
    spawn_x: tuple = (0.28, 0.52)
    spawn_y: tuple = (-0.22, 0.22)
    beads: dict | None = None                  # {'count', 'radius', 'colour', 'in': cup body name}
    towel: dict | None = None                  # {'count', 'spacing', 'colour', 'pos'}
    extra_xml: str = ''


# ---------- object shapes ----------
def cube(colour, edge=0.045):
    h = edge / 2
    return Body(f'cube_{colour}', colour, [Geom('box', (h, h, h), (0, 0, h))], radius=0.035)


def cylinder(kind, colour, radius, height):
    return Body(f'{kind}_{colour}', colour, [Geom('cylinder', (radius, height / 2), (0, 0, height / 2))],
                radius=radius + 0.01)


def box(kind, colour, lx, ly, lz):
    return Body(f'{kind}_{colour}', colour, [Geom('box', (lx / 2, ly / 2, lz / 2), (0, 0, lz / 2))],
                radius=max(lx, ly) / 2 + 0.01)


def hollow(kind, colour, radius, height, wall=0.004, segments=14, bottom=0.004, handle=False):
    """A cup / mug / bowl: a ring of wall boxes on a disc. With `handle`, a loop on the +x side."""
    geoms = [Geom('cylinder', (radius, bottom / 2), (0, 0, bottom / 2), name=f'{kind}_{colour}')]
    seg_w = 2 * np.pi * radius / segments * 0.55
    for i in range(segments):
        a = 2 * np.pi * i / segments
        r = radius - wall / 2
        geoms.append(Geom('box', (wall / 2, seg_w, height / 2), (r * np.cos(a), r * np.sin(a), height / 2),
                          euler=(0, 0, float(np.rad2deg(a)))))
    if handle:
        hx = radius + 0.012
        geoms += [Geom('box', (0.004, 0.006, height * 0.32), (hx, 0, height / 2)),
                  Geom('box', (0.007, 0.006, 0.004), (radius + 0.005, 0, height * 0.8)),
                  Geom('box', (0.007, 0.006, 0.004), (radius + 0.005, 0, height * 0.2))]
    return Body(f'{kind}_{colour}', colour, geoms, radius=radius + (0.03 if handle else 0.012), density=400.0, condim=3)


def hook_tool(colour='brown'):
    """An L-shaped stick: a 20 cm handle along x and a 6 cm hook across its far end."""
    return Body(f'hook_{colour}', colour, [
        Geom('box', (0.10, 0.009, 0.009), (0.0, 0.0, 0.009), name=f'hook_{colour}'),
        Geom('box', (0.009, 0.035, 0.009), (0.091, 0.026, 0.009))], radius=0.11, density=300.0)


def peg_fixture(colour='orange', hole=0.030, outer=0.075, height=0.035, place=(0.42, -0.08)):
    """A fixed block with a square through-hole `hole` wide (a 24 mm peg has 3 mm each side)."""
    wall = (outer - hole) / 2
    off = hole / 2 + wall / 2
    hz = height / 2
    return Body(f'fixture_{colour}', colour, [
        Geom('box', (outer / 2, wall / 2, hz), (0, off, hz), name=f'fixture_{colour}'),
        Geom('box', (outer / 2, wall / 2, hz), (0, -off, hz)),
        Geom('box', (wall / 2, hole / 2, hz), (off, 0, hz)),
        Geom('box', (wall / 2, hole / 2, hz), (-off, 0, hz))],
        free=False, pos=None if place is None else (place[0], place[1], 0.0), yaw=None if place is None else 0.0,
        graspable=False, radius=0.06)


# ---------- scenes ----------
SCENES = {s.name: s for s in [
    Scene('objects', [cube('red'), cylinder('can', 'yellow', 0.022, 0.075), box('box', 'blue', 0.075, 0.034, 0.03),
                      hollow('mug', 'purple', 0.03, 0.065, handle=True), cylinder('puck', 'teal', 0.028, 0.025)],
          'five different shapes: a cube, a can, a long box, a mug with a handle, a flat puck'),
    Scene('peg', [box('peg', 'red', 0.024, 0.024, 0.06), peg_fixture(), cube('green')],
          'a 24 mm square peg and a fixed block with a 30 mm square hole (3 mm clearance a side)',
          spawn_x=(0.30, 0.50), spawn_y=(0.02, 0.22)),
    Scene('peg_tight', [box('peg', 'red', 0.024, 0.024, 0.06), peg_fixture(hole=0.026, place=None)],
          'a 24 mm square peg and a fixed block with a 26 mm hole (1 mm clearance a side): for training the '
          'insertion policy', spawn_x=(0.30, 0.48), spawn_y=(0.04, 0.20)),
    Scene('hook', [hook_tool(), Body('puck_green', 'green', [Geom('cylinder', (0.025, 0.0125), (0, 0, 0.0125))],
                                     pos=(0.70, 0.02, 0.0), yaw=0.0, radius=0.035)],
          'a green puck beyond the arm\'s reach and an L-shaped hook tool', table_x=(0.18, 0.80),
          spawn_x=(0.30, 0.42), spawn_y=(-0.20, -0.05)),
    Scene('pour', [hollow('cup', 'teal', 0.026, 0.07), hollow('bowl', 'blue', 0.06, 0.035, wall=0.005,
                                                                segments=20)],
          'a teal cup holding twelve red beads, and a wide blue bowl', beads=dict(count=12, radius=0.0065,
                                                                                    colour='red', into='cup_teal')),
    Scene('towel', [cube('blue')], 'a 20 x 20 cm cloth lying flat, and a blue cube',
          towel=dict(count=11, spacing=0.02, colour='orange', pos=(0.42, 0.06))),
]}


# ---------- MJCF ----------
def _geom_xml(body: Body, g: Geom, i: int) -> str:
    name = g.name or f'{body.name}_g{i}'
    size = ' '.join(f'{v:.5f}' for v in g.size)
    pos = ' '.join(f'{v:.5f}' for v in g.pos)
    euler = f' euler="{" ".join(f"{np.deg2rad(e):.5f}" for e in g.euler)}"' if g.euler else ''
    return (f'      <geom name="{name}" type="{g.type}" size="{size}" pos="{pos}"{euler} '
            f'rgba="{COLOURS[body.colour]}" density="{body.density}" condim="{body.condim}" friction="0.8 0.08 0.004" '
            f'solimp="0.98 0.99 0.001" solref="0.005 1" />')


def geom_names(body: Body) -> list[str]:
    return [g.name or f'{body.name}_g{i}' for i, g in enumerate(body.geoms)]


def scene_xml(scene: Scene) -> str:
    """The scene's MJCF (includes the arm's panthera.xml from the same folder)."""
    tx, ty = scene.table_x, scene.table_y
    cx, cy, hx, hy = (tx[0] + tx[1]) / 2, (ty[0] + ty[1]) / 2, (tx[1] - tx[0]) / 2, (ty[1] - ty[0]) / 2
    out = [f'<!-- Generated by panthera_agent/scenes/build.py ({scene.name}): {scene.description}. Edit the catalog, '
           'not this file. -->',
           f'<mujoco model="agent_{scene.name}">', '  <include file="panthera.xml" />',
           '  <statistic center="0.3 0 0.3" extent="1.2" />',
           '  <default>\n    <default class="cloth"><joint damping="0" armature="0" frictionloss="0" /></default>'
           '\n  </default>',
           '  <visual>\n    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0" />',
           '    <rgba haze="0.15 0.25 0.35 1" />',
           '    <global azimuth="14" elevation="-34" offwidth="1920" offheight="1080" />\n  </visual>',
           '  <asset>',
           '    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072" />',
           '    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" '
           'rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300" />',
           '    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" '
           'reflectance="0.1" />',
           '    <material name="table" rgba="0.75 0.68 0.55 1" reflectance="0.05" />', '  </asset>',
           '  <worldbody>', '    <light pos="0 0 2" dir="0 0 -1" directional="true" />',
           '    <geom name="floor" type="plane" size="0 0 0.05" material="groundplane" />',
           '    <camera name="overhead" pos="0.4 0 0.9" xyaxes="0 -1 0 1 0 0" fovy="50" />',
           f'    <body name="table" pos="{cx:.4f} {cy:.4f} 0.0400">',
           f'      <geom name="table_top" type="box" size="{hx:.4f} {hy:.4f} 0.0100" material="table" condim="4" '
           'friction="0.3 0.02 0.0005" />', '    </body>']
    for b in scene.bodies:
        x, y = (b.pos or (0.4, 0.0, 0.0))[:2]
        yaw = np.deg2rad(b.yaw or 0.0)
        out.append(f'    <body name="{b.name}" pos="{x:.4f} {y:.4f} {TABLE_Z + 0.0005:.4f}" '
                   f'euler="0 0 {yaw:.4f}">')
        if b.free:
            out.append(f'      <freejoint name="{b.name}" />')
        out += [_geom_xml(b, g, i) for i, g in enumerate(b.geoms)]
        out.append('    </body>')
    if scene.beads:
        bd = scene.beads
        for i in range(bd['count']):
            out.append(f'    <body name="bead{i}_{bd["colour"]}" pos="0.4 0 0.2"><freejoint />'
                       f'<geom name="bead{i}_{bd["colour"]}" type="sphere" size="{bd["radius"]}" '
                       f'rgba="{COLOURS[bd["colour"]]}" density="1500" condim="3" friction="0.3 0.01 0.001" '
                       'solref="0.004 1" /></body>')
    if scene.towel:
        t = scene.towel
        n, s = t['count'], t['spacing']
        x, y = t['pos']
        # The arm's default joint damping would apply to the cloth's vertex joints, which MuJoCo forbids.
        out.append('    <body name="towel_frame" childclass="cloth">')
        out.append(f'    <flexcomp name="towel_{t["colour"]}" type="grid" dim="2" count="{n} {n} 1" '
                   f'spacing="{s} {s} {s}" pos="{x} {y} {TABLE_Z + 0.004}" radius="0.0025" '
                   f'rgba="{COLOURS[t["colour"]]}" mass="0.05">'
                   '<contact condim="3" solref="0.003 1" solimp=".95 .99 .0001" selfcollide="none" '
                   'friction="1.0 0.01 0.001"/>'
                   '<edge equality="true" damping="0.002"/></flexcomp>')
        out.append('    </body>')
    out.append('  </worldbody>')
    pads = []
    for b in scene.bodies:
        if b.graspable and b.free:
            for g in geom_names(b):
                pads += [f'    <pair geom1="L_finger_pad" geom2="{g}" {PAD_PAIR} />',
                         f'    <pair geom1="R_finger_pad" geom2="{g}" {PAD_PAIR} />']
    if pads:
        out += ['  <contact>'] + pads + ['  </contact>']
    out += [scene.extra_xml, '</mujoco>', '']
    return '\n'.join(line for line in out if line != '')


# ---------- layout on reset ----------
def layout(scene: Scene, rng: np.random.Generator) -> dict[str, tuple[np.ndarray, float]]:
    """{body: (base xy, yaw)} for a seeded reset: fixed positions where the catalog gives them, else random in
    the spawn box, kept apart by the bodies' radii (and clear of the fixtures and the towel). A fixture with no
    catalog position (`pos` None) is placed at random too, before the free objects."""
    placed: list[tuple[np.ndarray, float]] = []
    for b in scene.bodies:
        if b.pos is not None:
            placed.append((np.array(b.pos[:2], float), b.radius))
    if scene.towel:
        t = scene.towel
        half = (t['count'] - 1) * t['spacing'] / 2
        placed.append((np.array(t['pos'], float) + half, half * 1.45))
    out = {}
    for b in sorted(scene.bodies, key=lambda b: b.free):          # fixtures first: they need the most room
        if not b.free and b.pos is not None:
            continue
        if b.pos is not None:
            out[b.name] = (np.array(b.pos[:2], float), np.deg2rad(b.yaw or 0.0))
            continue
        xy = None
        for _ in range(200):
            c = rng.uniform([scene.spawn_x[0], scene.spawn_y[0]], [scene.spawn_x[1], scene.spawn_y[1]])
            if all(np.linalg.norm(c - p) >= r + b.radius for p, r in placed):
                xy = c
                break
        if xy is None:
            xy = rng.uniform([scene.spawn_x[0], scene.spawn_y[0]], [scene.spawn_x[1], scene.spawn_y[1]])
        placed.append((xy, b.radius))
        out[b.name] = (xy, float(rng.uniform(-np.pi, np.pi)) if b.yaw is None else np.deg2rad(b.yaw))
    return out
