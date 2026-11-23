"""Sim test tasks: the instruction the agent gets, an optional scene set-up, and success judged from the
simulator's true object poses (never from what the agent believes).

Success thresholds are panthera_sim's (`sim/stack_task.py`), so a stack here is a stack there.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from robots.sim_robot import SIM_ROOT  # noqa: F401  (puts panthera_sim on the path)
from sim.stack_task import CUBE_EDGE, TABLE_CUBE_Z, ordered_two_stack_metrics, stack_metrics


@dataclass
class Task:
    """One sim task."""
    name: str
    instruction: str
    check: Callable[[dict], dict]                 # truth {name: (pos, R)} -> metrics with 'success'
    setup: Callable | None = None                 # (SimRobot) -> None, after the seeded reset
    scene: str = 'scene.xml'


def _pos(truth, name):
    return truth[name][0]


def _on_table(p):
    return abs(p[2] - TABLE_CUBE_Z) < 0.01


def check_red_on_green(truth: dict) -> dict:
    """Red released on green, green on the table, within 12 mm."""
    m = ordered_two_stack_metrics(np.array([_pos(truth, 'cube_red'), _pos(truth, 'cube_green')]))
    return dict(success=m['two_stack'], alignment_mm=m['horizontal_alignment_m'] * 1000,
                gap_mm=m['vertical_gap_m'] * 1000)


def check_three_stack(truth: dict) -> dict:
    """All three in one tower, any order."""
    m = stack_metrics(np.array([_pos(truth, n) for n in ('cube_red', 'cube_green', 'cube_blue')]))
    return dict(success=m['three_stack'], spread_mm=m['horizontal_spread_m'] * 1000)


def check_all_on_table(truth: dict) -> dict:
    """Nothing stacked: every cube on the table."""
    on = {n: _on_table(p) for n, (p, _) in truth.items()}
    return dict(success=all(on.values()), on_table=on)


def check_left_half(truth: dict) -> dict:
    """Every cube on the table with y > 3 cm (the robot's left half)."""
    ys = {n: float(p[1]) for n, (p, _) in truth.items()}
    return dict(success=all(y > 0.03 for y in ys.values()) and all(_on_table(p) for p, _ in truth.values()), y=ys)


def setup_red_on_green(robot) -> None:
    """Start with red already stacked on green."""
    pos, quat = robot.sim.object_poses()
    names = robot.sim.object_names
    r, g = names.index('cube_red'), names.index('cube_green')
    pos[r] = pos[g] + np.array([0.0, 0.0, CUBE_EDGE])
    quat[r] = quat[g]
    robot.sim.set_object_poses(pos, quat)
    robot.settle()


# ---------- the agent's scenes (scenes/catalog.py): body origins sit at each object's base ----------
TABLE_TOP = 0.05


def check_puck_on_cube(truth: dict) -> dict:
    """The teal puck resting on the red cube's top (within 15 mm of its centre), the cube on the table."""
    puck, cube = _pos(truth, 'puck_teal'), _pos(truth, 'cube_red')
    off = float(np.linalg.norm(puck[:2] - cube[:2]))
    on = abs(puck[2] - (cube[2] + 0.045)) < 0.006 and off < 0.015 and abs(cube[2] - TABLE_TOP) < 0.005
    return dict(success=bool(on), offset_mm=off * 1000, puck_z=float(puck[2]))


def check_mug_left(truth: dict) -> dict:
    p = _pos(truth, 'mug_purple')
    return dict(success=bool(p[1] > 0.05 and abs(p[2] - TABLE_TOP) < 0.005), y_cm=float(p[1] * 100))


def check_peg_in(truth: dict) -> dict:
    """The peg inside the fixture's hole: within 8 mm of its centre (the fixture is fixed, at its catalog place) and
    its base 15 mm or more below the fixture's top (35 mm)."""
    from scenes.catalog import SCENES
    fx = next(b for b in SCENES['peg'].bodies if b.name.startswith('fixture'))
    peg = _pos(truth, 'peg_red')
    off = float(np.linalg.norm(peg[:2] - np.array(fx.pos[:2])))
    return dict(success=bool(off < 0.008 and peg[2] < TABLE_TOP + 0.035 - 0.015 + 1e-3), offset_mm=off * 1000,
                peg_base_z=float(peg[2]))


def check_puck_in_reach(truth: dict) -> dict:
    p = _pos(truth, 'puck_green')
    return dict(success=bool(np.linalg.norm(p[:2]) < 0.55), distance_cm=float(np.linalg.norm(p[:2]) * 100))


def check_beads_in_bowl(truth: dict) -> dict:
    bowl = _pos(truth, 'bowl_blue')
    beads = [p for n, (p, _) in truth.items() if n.startswith('bead')]
    inside = sum(1 for p in beads if np.linalg.norm(p[:2] - bowl[:2]) < 0.055 and p[2] < bowl[2] + 0.05)
    return dict(success=bool(inside >= 0.75 * len(beads)), in_bowl=inside, beads=len(beads))


TASKS = {t.name: t for t in [
    Task('stack_red_on_green', 'Stack the red cube on top of the green cube.', check_red_on_green,
         scene='scene_two_blocks.xml'),
    Task('stack_three', 'Stack all three cubes into a single tower.', check_three_stack),
    Task('unstack', 'Make sure no cube is on top of another: every cube should rest on the table.',
         check_all_on_table, setup=setup_red_on_green),
    Task('gather_left', 'Move every cube to the left half of the table (y greater than 5 cm).', check_left_half),
    Task('learn_and_stack', 'Put the teal puck on top of the red cube. You have not seen these objects before.',
         check_puck_on_cube, scene='agent_objects.xml'),
    Task('mug_left', 'Move the purple mug to the left half of the table (y greater than 5 cm).', check_mug_left,
         scene='agent_objects.xml'),
    Task('peg_insert', 'Put the red peg into the hole in the orange block.', check_peg_in, scene='agent_peg.xml'),
    Task('hook_puck', 'The green puck is out of the arm\'s reach. Use the brown hook to bring it within reach.',
         check_puck_in_reach, scene='agent_hook.xml'),
    Task('pour_beads', 'Pour the red beads from the teal cup into the blue bowl.', check_beads_in_bowl,
         scene='agent_pour.xml'),
]}
