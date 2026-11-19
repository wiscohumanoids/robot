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


TASKS = {t.name: t for t in [
    Task('stack_red_on_green', 'Stack the red cube on top of the green cube.', check_red_on_green,
         scene='scene_two_blocks.xml'),
    Task('stack_three', 'Stack all three cubes into a single tower.', check_three_stack),
    Task('unstack', 'Make sure no cube is on top of another: every cube should rest on the table.',
         check_all_on_table, setup=setup_red_on_green),
    Task('gather_left', 'Move every cube to the left half of the table (y greater than 5 cm).', check_left_half),
]}
