"""Fixed-start deployment settings recovered from demonstration provenance."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sim.stack_task import TABLE_CUBE_Z, ordered_two_stack_metrics, stack_metrics
from sim.dynamics import recorded_dynamics


def fixed_environment(provenance: dict) -> dict | None:
    metadata = [json.loads((Path(source['source']) / 'meta.json').read_text())
                for source in provenance['sources'].values()]
    for meta in metadata:
        meta['simulation_dynamics'] = recorded_dynamics(meta)
    if len({meta['simulation_dynamics'] for meta in metadata}) > 1:
        raise ValueError('Dataset has inconsistent simulation dynamics')
    if not any(meta.get('arm_start') == 'fixed' for meta in metadata):
        return None
    keys = ('scene', 'task', 'objects', 'arm_start', 'initial_arm_q', 'gripper_open_m', 'simulation_dynamics')
    environment = {key: metadata[0][key] for key in keys}
    if any({key: meta.get(key) for key in keys} != environment for meta in metadata):
        raise ValueError('Fixed-start dataset has inconsistent scene/reset settings')
    environment['success_hold_seconds'] = 1.0
    return environment


def reset_fixed_arm(sim, environment: dict) -> None:
    import mujoco
    q = np.asarray(environment['initial_arm_q'], dtype=float)
    opening = float(environment['gripper_open_m'])
    sim.data.qpos[sim.arm_qadr] = q
    sim.data.qvel[:] = 0
    sim.data.qpos[sim.finger_qadr] = [opening, -opening]
    sim.set_arm_ctrl(q, immediate=True)
    sim.set_gripper(opening / .04)
    mujoco.mj_forward(sim.model, sim.data)


def rollout_metrics(positions: np.ndarray) -> dict:
    if len(positions) != 2:
        metrics = stack_metrics(positions)
        return {**metrics, 'task_stack': metrics['three_stack']}
    ordered = ordered_two_stack_metrics(positions)
    return dict(two_stack=ordered['two_stack'], three_stack=False,
                task_stack=ordered['two_stack'],
                max_height_m=float(positions[:, 2].max()),
                lifted_cubes=int(np.sum(positions[:, 2] > TABLE_CUBE_Z + .025)),
                horizontal_spread_m=ordered['horizontal_alignment_m'],
                vertical_gaps_m=[ordered['vertical_gap_m']])
