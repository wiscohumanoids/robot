#!/usr/bin/env python3
"""Replay native demonstrations under ACT's next-sample 30 Hz control contract."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.panthera_env import PantheraSim
from sim.dynamics import recorded_dynamics
from sim.stack_task import stack_metrics, ordered_two_stack_metrics
from teleop.render_vla_dataset import recording_times, interpolate
from teleop.dataset_contract import PhysicsClock


def replay(path: Path) -> dict:
    meta = json.loads((path / 'meta.json').read_text())
    sim = PantheraSim(ROOT / meta.get('scene', 'sim/panthera/scene.xml'),
                      dynamics=recorded_dynamics(meta))
    two_blocks = len(sim.object_names) == 2
    sim.reset(randomize=False)
    with np.load(path / 'data.npz') as data:
        times, _ = recording_times(data, sim.dt)
        grid = np.arange(0, times[-1] + 1e-9, 1 / 30)
        controls = interpolate(times, data['ctrl'], grid)
        sim.data.qpos[sim.arm_qadr] = data['q'][0]
        sim.data.qvel[sim.arm_dofadr] = data['dq'][0]
        sim.data.qpos[sim.finger_qadr] = data['finger_q'][0]
        sim.data.qvel[sim.finger_dofadr] = data['finger_dq'][0]
        sim.set_object_poses(data['obj_pos'][0], data['obj_quat'][0])
        sim.data.ctrl[:] = controls[0]
        sim.sync_control_state()
        mujoco.mj_forward(sim.model, sim.data)
    clock = PhysicsClock(30, sim.dt)
    stable = 0
    grasped = set()
    for action in controls[1:]:
        sim.set_arm_ctrl(action[:6])
        sim.set_gripper(action[6] / .04)
        sim.step(clock.next_steps())
        grasped.update(np.flatnonzero(sim.grasp_flags()).tolist())
        released = not sim.grasped
        positions = sim.object_poses()[0]
        valid_stack = (ordered_two_stack_metrics(positions)['two_stack'] if two_blocks
                       else stack_metrics(positions)['three_stack'])
        success = valid_stack and released
        stable = stable + 1 if success else 0
    return dict(episode=path.name, simulation_dynamics=sim.dynamics, success=stable >= 30, stable_final_steps=stable,
                grasped_blocks=sorted(grasped), final_positions=sim.object_poses()[0].tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/scripted_stack')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--episodes', type=int, default=0,
                        help='0 replays every episode; otherwise sample evenly')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.workers < 1 or args.episodes < 0:
        parser.error('workers must be positive and episodes nonnegative')
    paths = sorted(p.parent for p in args.input.glob('episode_*/data.npz'))
    if not paths:
        parser.error('No native episodes found')
    if args.episodes:
        chosen = np.linspace(0, len(paths)-1, min(args.episodes, len(paths)), dtype=int)
        paths = [paths[i] for i in chosen]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(replay, paths):
            results.append(result)
            if not result['success']:
                print('FAIL', result, flush=True)
            if len(results) % 100 == 0:
                print(f"{len(results)} replayed, {sum(r['success'] for r in results)} successful", flush=True)
    report = dict(episodes=len(results), successes=sum(r['success'] for r in results),
                  control_hz=30, action_alignment='next_uniform_sample', results=results)
    destination = args.report or args.input / 'replay_audit.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(f"{report['successes']}/{len(results)} successful replays -> {destination}")
    if report['successes'] != len(results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
