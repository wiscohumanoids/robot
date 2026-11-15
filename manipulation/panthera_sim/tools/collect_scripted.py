#!/usr/bin/env python3
"""Collect physics-validated scripted stacks in the native teleop episode format."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from sim.panthera_env import PantheraSim, mat_to_quat
from sim.dynamics import CONTACT_DYNAMICS
from sim.stack_task import stack_metrics, ordered_two_stack_metrics, CUBE_EDGE
from teleop.dataset_contract import PhysicsClock, file_hash
from teleop.episode import Episode


class DemoFailure(RuntimeError):
    pass


def grasp_rotation(sim, index):
    """Face-aligned jaws, with a downward approach that stays in the IK workspace."""
    rotation = sim.data.xmat[sim.object_bodies[index]].reshape(3, 3)
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    # Equivalent cube faces; choose the yaw closest to the robot's forward axis.
    yaw = (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
    pitch = np.deg2rad(55)
    c, s = np.cos(yaw), np.sin(yaw)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
    c, s = np.cos(pitch), np.sin(pitch)
    return rz @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


class Planner:
    def __init__(self, seed: int, blocks: int = 3, arm_start: str = "random"):
        self.blocks = blocks
        self.arm_start = arm_start
        self.scene = ROOT / "sim/panthera" / ("scene_two_blocks.xml" if blocks == 2 else "scene.xml")
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.sim = PantheraSim(self.scene)
        self.sim.reset(rng=self.rng)
        self.clock = PhysicsClock(30, self.sim.dt)
        self.episode = Episode()
        self.stages = []
        self.speed = float(self.rng.uniform(.10, .16))
        self.clearance = float(self.rng.uniform(.06, .08))
        self.target, self.quat = self.sim.ee_pose()
        self.grip = 1.
        self.qctrl = self.sim.q

    def tick(self, target, quat, grip):
        sim = self.sim
        q, pe, re = sim.ik(target, quat, q_init=self.qctrl, iters=35,
                           max_joint_step=.08, posture_gain=0.)
        self.qctrl = q
        sim.set_arm_ctrl(q)
        sim.set_gripper(grip)
        ticks = self.clock.next_steps()
        sim.step(ticks)
        if not np.isfinite(sim.data.qpos).all():
            raise DemoFailure('nonfinite simulator state')
        pos, rot = sim.object_poses()
        self.episode.add(dict(t=sim.data.time, sim_time=sim.data.time,
            physics_steps=ticks, finger_q=sim.data.qpos[sim.finger_qadr].copy(),
            finger_dq=sim.data.qvel[sim.finger_dofadr].copy(),
            q=sim.q, dq=sim.dq, ctrl=sim.data.ctrl.copy(),
            ee_pos=sim.ee_pos(), ee_quat=sim.ee_quat(), obj_pos=pos, obj_quat=rot,
            target_pos=np.array(target), target_quat=np.array(quat), gripper=grip,
            ik_pos_err=pe, ik_rot_err=re))
        self.target, self.quat, self.grip = np.array(target), np.array(quat), grip

    def move(self, name, target, quat=None, grip=None, duration=None):
        self.stages.append(dict(name=name, start=len(self.episode)))
        target = np.array(target)
        quat = self.quat.copy() if quat is None else np.array(quat)
        grip = self.grip if grip is None else grip
        start, qstart, gstart = self.target.copy(), self.quat.copy(), self.grip
        if np.dot(quat, qstart) < 0:
            quat = -quat
        angle = 2 * np.arccos(np.clip(np.dot(quat, qstart), -1, 1))
        duration = duration or max(.4, 1.5 * np.linalg.norm(target-start)/self.speed, angle/.7)
        for i in range(1, math.ceil(duration*30)+1):
            u = i / math.ceil(duration*30)
            u = u*u*(3-2*u)
            q = qstart*(1-u)+quat*u
            q /= np.linalg.norm(q)
            self.tick(start*(1-u)+target*u, q, gstart*(1-u)+grip*u)
        # Let the position servos finish tracking before a contact transition.
        for _ in range(9):
            self.tick(target, quat, grip)
        if np.linalg.norm(self.sim.ee_pos()-target) > .012:
            raise DemoFailure(f'{name}: tracking error {np.linalg.norm(self.sim.ee_pos()-target):.4f}')

    def hold(self, name, seconds, grip=None):
        self.move(name, self.target, grip=grip, duration=seconds)

    def run(self):
        sim = self.sim
        # Fixed starts do not depend on object yaw or the episode RNG.
        for _ in range(1 if self.arm_start == 'fixed' else 100):
            if self.arm_start == 'fixed':
                start = np.array([.38, 0., .24])
                pitch = np.deg2rad(55)
                c, s = np.cos(pitch), np.sin(pitch)
                r = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            else:
                start = self.rng.uniform([.30,-.16,.18], [.46,.16,.28])
                r = grasp_rotation(sim, 0)
            quat = mat_to_quat(r)
            q, pe, re = sim.ik(start, quat, max_joint_step=None, iters=150)
            if pe < .001 and re < .01:
                break
        else:
            raise DemoFailure('initial IK')
        self.initial_arm_q = q.tolist()
        self.initial_target_pos = start.tolist()
        self.initial_target_quat = quat.tolist()
        sim.data.qpos[sim.arm_qadr] = q
        sim.data.qvel[:] = 0
        sim.data.qpos[sim.finger_qadr] = [.04,-.04]
        sim.set_arm_ctrl(q, immediate=True); sim.set_gripper(1)
        mujoco.mj_forward(sim.model, sim.data)
        self.target, self.quat, self.qctrl = sim.ee_pos(), sim.ee_quat(), q
        self.hold('settle', .3)
        # Consistent color order avoids an ambiguous multimodal imitation target.
        # Red on green, then blue on red.
        pairs = ((0, 1),) if self.blocks == 2 else ((0, 1), (2, 0))
        for level, (block, support) in enumerate(pairs, start=1):
            r = grasp_rotation(sim, block)
            quat = mat_to_quat(r)
            offset = .018*r[:,0]
            obj = sim.object_poses()[0][block]
            grasp = obj + offset
            above = grasp.copy(); above[2] += self.clearance
            # Lift before lateral travel so the open fingers clear other cubes.
            safe = self.target.copy(); safe[2] = max(safe[2], above[2]+.015)
            self.move(f'{level}_clear', safe)
            self.move(f'{level}_approach', above, quat, 1.)
            self.move(f'{level}_descend', grasp)
            self.hold(f'{level}_close', .5, 0.)
            if not sim.grasp_flags()[block]:
                raise DemoFailure(f'{level}: no bilateral grasp')
            self.move(f'{level}_lift', above)
            if sim.object_poses()[0][block,2] < obj[2]+.04:
                raise DemoFailure(f'{level}: failed lift')
            # Measured held offset accounts for contact seating and servo lag.
            held_offset = sim.ee_pos()-sim.object_poses()[0][block]
            dest = sim.object_poses()[0][support] + [0,0,CUBE_EDGE+.014]
            place = dest + held_offset
            transit = place.copy(); transit[2] += self.clearance
            lift = self.target.copy(); lift[2] = transit[2]
            self.move(f'{level}_raise', lift)
            self.move(f'{level}_transfer', transit)
            self.move(f'{level}_place', place)
            self.hold(f'{level}_release', .5, 1.)
            self.move(f'{level}_retreat', transit)
        self.stages.append(dict(name='validate', start=len(self.episode)))
        for _ in range(30):
            self.tick(self.target, self.quat, 1.)
            metrics = (ordered_two_stack_metrics(sim.object_poses()[0]) if self.blocks == 2
                       else stack_metrics(sim.object_poses()[0]))
            if not metrics['two_stack' if self.blocks == 2 else 'three_stack'] or sim.grasped:
                raise DemoFailure('released stack did not remain stable for one second')
            p = sim.object_poses()[0][[1,0] if self.blocks == 2 else [1,0,2]]
            if np.max(np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1)) > .012:
                raise DemoFailure('stack alignment exceeds 12 mm')
        return metrics


def attempt(seed, blocks=3, arm_start="random"):
    planner = Planner(seed, blocks=blocks, arm_start=arm_start)
    try:
        metrics = planner.run()
        return planner.episode, dict(seed=seed, stages=planner.stages, metrics=metrics,
            speed_m_s=planner.speed, approach_clearance_m=planner.clearance,
            initial_arm_q=planner.initial_arm_q, initial_target_pos=planner.initial_target_pos,
            initial_target_quat=planner.initial_target_quat, arm_start=arm_start, blocks=blocks)
    except DemoFailure as error:
        return None, dict(seed=seed, failure=str(error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'data/scripted_stack')
    parser.add_argument('--blocks', type=int, choices=(2, 3), default=3)
    parser.add_argument('--arm-start', choices=('fixed', 'random'), default='random')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=230923)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-attempts', type=int, default=10000)
    parser.add_argument('--min-free-gb', type=float, default=5.)
    args = parser.parse_args()
    if min(args.episodes,args.workers,args.max_attempts) < 1:
        parser.error('episodes, workers and max-attempts must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    signature = file_hash(Path(__file__))
    contract = dict(simulation_dynamics=CONTACT_DYNAMICS, generator_sha256=signature, seed=args.seed, control_hz=30,
        blocks=args.blocks, arm_start=args.arm_start,
        task_metrics_sha256=file_hash(ROOT/"sim/stack_task.py"),
        scene_sha256={p.name:file_hash(p) for p in sorted((ROOT/'sim/panthera').glob('*.xml'))},
        simulator_sha256=file_hash(ROOT/'sim/panthera_env.py'))
    contract_path = args.output/'collection.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise SystemExit('Collection code/settings changed; use a new output directory.')
    if not contract_path.exists() and list(args.output.glob('episode_*')):
        raise SystemExit('Output contains episodes without a collection contract; use a new directory.')
    contract_path.write_text(json.dumps(contract, indent=2)+'\n')
    existing = sorted(args.output.glob('episode_*/meta.json'))
    for i, path in enumerate(existing):
        if path.parent.name != f'episode_{i:04d}' or not (path.parent/'data.npz').exists():
            raise SystemExit('Incomplete or noncontiguous collection; inspect before resuming.')
    count = len(existing)
    log = args.output/'attempts.jsonl'
    prior = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    seed = max([args.seed-1]+[item['seed'] for item in prior]+
               [json.loads(p.read_text())['seed'] for p in existing])+1
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while count < args.episodes and seed-args.seed < args.max_attempts:
            if shutil.disk_usage(args.output).free < args.min_free_gb*1024**3:
                raise RuntimeError('Low disk space; collector stopped safely. Free space and resume.')
            seeds = range(seed, seed+min(args.workers, args.episodes-count, args.max_attempts-(seed-args.seed)))
            for episode, result in pool.map(partial(attempt, blocks=args.blocks, arm_start=args.arm_start), seeds):
                if episode is not None:
                    name = f'episode_{count:04d}'
                    meta = dict(robot='Panthera-HT (HighTorque) 6-DoF + parallel gripper',
                        scene='sim/panthera/scene_two_blocks.xml' if args.blocks == 2 else 'sim/panthera/scene.xml',
                        control_mode='scripted_ik',
                        task='stack the red cube on the green cube' if args.blocks == 2 else 'stack the three colored cubes',
                        objects=['cube_red','cube_green'] if args.blocks == 2 else ['cube_red','cube_green','cube_blue'],
                        arm_joints=[f'joint{i}' for i in range(1,7)], gripper_open_m=.04,
                        frames={'ee_*/target_*':'robot base frame','quat':'(w, x, y, z)'},
                        recording={'control_hz':30,'clock':'simulation','row_alignment':'post_action'},
                        generator_sha256=signature, **result)
                    temporary = args.output/f'.{name}.tmp'
                    episode.save(temporary, meta, 30, False)
                    temporary.rename(args.output/name)
                    result['episode'] = name
                    count += 1
                with log.open('a') as stream:
                    stream.write(json.dumps(result)+'\n')
                print(json.dumps(dict(accepted=count, requested=args.episodes, **result)), flush=True)
            seed += len(seeds)
            status = dict(accepted=count, requested=args.episodes, attempts=seed-args.seed,
                next_seed=seed, elapsed_s=time.time()-started,
                free_gb=shutil.disk_usage(args.output).free/1024**3, complete=count>=args.episodes)
            temp = args.output/'status.tmp'; temp.write_text(json.dumps(status,indent=2)+'\n')
            temp.replace(args.output/'status.json')
    if count < args.episodes:
        raise SystemExit(f'Only {count}/{args.episodes} accepted before max-attempts')


if __name__ == '__main__':
    main()
