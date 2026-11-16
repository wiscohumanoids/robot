#!/usr/bin/env python3
"""Check recorded ACT targets under the deployment clock, without a policy."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.panthera_env import PantheraSim
from sim.dynamics import recorded_dynamics
from sim.stack_task import stack_metrics
from teleop.dataset_contract import DEFAULT_RENDERED, PhysicsClock


def replay(path: Path, max_joint_step: float | None = None) -> dict:
    data = np.load(path / 'trajectory.npz')
    source = json.loads((path / 'source.json').read_text())
    raw = np.load(Path(source['source']) / 'data.npz')
    meta = json.loads((Path(source['source']) / 'meta.json').read_text())
    sim = PantheraSim(ROOT / meta.get('scene', 'sim/panthera/scene.xml'), dynamics=recorded_dynamics(meta))
    sim.reset(randomize=False)
    sim.data.qpos[sim.arm_qadr] = data['q'][0]
    sim.data.qvel[sim.arm_dofadr] = raw['dq'][0]
    sim.data.qpos[sim.finger_qadr] = data['finger_q'][0]
    sim.data.ctrl[:] = data['ctrl'][0]
    sim.sync_control_state()
    sim.set_object_poses(data['obj_pos'][0], data['obj_quat'][0])
    mujoco.mj_forward(sim.model, sim.data)
    clock = PhysicsClock(float(data['sample_hz']), sim.dt)
    errors, clipped = [], 0
    grasped = lifted = False
    for index, action in enumerate(data['ctrl'][1:], start=1):
        target = np.clip(action[:6], sim.arm_range[:, 0], sim.arm_range[:, 1])
        if max_joint_step is not None:
            limited = np.clip(target, sim.q - max_joint_step, sim.q + max_joint_step)
            clipped += int(not np.allclose(limited, target))
            target = limited
        sim.set_arm_ctrl(target)
        sim.set_gripper(float(action[6]) / .04)
        sim.step(clock.next_steps())
        errors.append(np.abs(sim.q - data['q'][index]).max())
        grasped |= sim.grasped
        lifted |= bool(stack_metrics(sim.object_poses()[0])['lifted_cubes'])
    metrics = stack_metrics(sim.object_poses()[0])
    return {'episode': path.name, 'max_joint_step': max_joint_step,
            'grasped': grasped, 'lifted': lifted,
            'final_two_stack': metrics['two_stack'], 'final_three_stack': metrics['three_stack'],
            'clipped_steps': clipped, 'steps': len(errors),
            'joint_error_p50_rad': float(np.median(errors)),
            'joint_error_p95_rad': float(np.percentile(errors, 95))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rendered', type=Path, default=DEFAULT_RENDERED)
    parser.add_argument('--episodes', type=int, default=8)
    parser.add_argument('--report', type=Path, default=ROOT / 'outputs/act/replay_30hz.json')
    args = parser.parse_args()
    paths = sorted(args.rendered.glob('episode_*/trajectory.npz'))
    chosen = np.linspace(0, len(paths)-1, min(args.episodes, len(paths)), dtype=int)
    results = []
    for index in chosen:
        for cap in (None, .15):
            result = replay(paths[index].parent, cap)
            results.append(result)
            print(json.dumps(result), flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
