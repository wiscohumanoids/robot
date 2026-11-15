"""Policy deploy contract: reads a manifest (v1 or v2) and builds the observation / joint
targets exactly like mjlab did in training. Pure numpy, no ROS, so tools/sim2sim_play.py and
the tests use the same code as policy_node.

Manifest v1: the policy drives every joint in `joints` (each has action_scale).
Manifest v2: the policy drives `action.joints` (e.g. legs + waist); the rest are `held_joints`
             (role: held), commanded by another controller on the robot (arm targets on
             held_joints.source_topic, default_pos until the first message).

Observation (mjlab velocity task, 'actor' group, in this order):
  base_lin_vel(3) base_ang_vel(3) projected_gravity(3)
  joint_pos - default (every joint in `joints`, manifest order = training joint order)
  joint_vel           (every joint in `joints`)
  actions             (last raw policy output, one per driven joint)
  command(3)          (vx, vy, wz)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

OBS_LAYOUT = ['base_lin_vel', 'base_ang_vel', 'projected_gravity', 'joint_pos', 'joint_vel', 'actions', 'command']


@dataclass
class Contract:
    policy_id: str
    version: int
    obs_joint_names: list[str]          # all joints, observation order
    default_pos: np.ndarray             # per obs joint
    kp: np.ndarray
    kd: np.ndarray
    act_joint_names: list[str]          # driven joints, action order
    act_default: np.ndarray
    act_scale: np.ndarray
    held_joint_names: list[str] = field(default_factory=list)
    held_default: np.ndarray = field(default_factory=lambda: np.zeros(0))
    held_topic: str | None = None
    sim_dt: float = 0.005
    decimation: int = 4

    @property
    def policy_hz(self) -> float:
        return 1.0 / (self.sim_dt * self.decimation)

    @property
    def obs_dim(self) -> int:
        n = len(self.obs_joint_names)
        return 9 + 2 * n + len(self.act_joint_names) + 3

    def observation(self, lin_vel_b, ang_vel_b, gravity_b, q, dq, last_action, command) -> np.ndarray:
        """q, dq: positions/velocities of obs_joint_names (same order)."""
        return np.concatenate([
            np.asarray(lin_vel_b, float), np.asarray(ang_vel_b, float), np.asarray(gravity_b, float),
            np.asarray(q, float) - self.default_pos, np.asarray(dq, float),
            np.asarray(last_action, float), np.asarray(command, float),
        ]).astype(np.float32)

    def action_targets(self, action) -> np.ndarray:
        """Joint position targets for act_joint_names."""
        return self.act_default + self.act_scale * np.asarray(action, float)


def load_contract(folder: str | Path) -> Contract:
    folder = Path(folder)
    man = yaml.safe_load((folder / 'manifest.yaml').read_text())
    if man['observation']['terms'] != OBS_LAYOUT:
        raise ValueError(f'unsupported observation layout {man["observation"]["terms"]}')
    version = int(man.get('manifest_version', 1))
    joints = man['joints']
    by_name = {j['name']: j for j in joints}
    obs_names = [j['name'] for j in joints]
    default = np.array([j['default_pos'] for j in joints], float)
    kp = np.array([j.get('kp', 0.0) for j in joints], float)
    kd = np.array([j.get('kd', 0.0) for j in joints], float)
    if version >= 2:
        act_names = list(man['action']['joints'])
        missing = [n for n in act_names if n not in by_name or 'action_scale' not in by_name[n]]
        if missing:
            raise ValueError(f'action joints missing from manifest joints / action_scale: {missing}')
        held = man.get('held_joints', {})
        held_names = list(held.get('names', [n for n in obs_names if n not in act_names]))
        held_topic = held.get('source_topic')
    elif version == 1:
        act_names = obs_names
        held_names, held_topic = [], None
    else:
        raise ValueError(f'unsupported manifest_version {version}')
    return Contract(
        policy_id=man['policy_id'], version=version,
        obs_joint_names=obs_names, default_pos=default, kp=kp, kd=kd,
        act_joint_names=act_names,
        act_default=np.array([by_name[n]['default_pos'] for n in act_names], float),
        act_scale=np.array([by_name[n]['action_scale'] for n in act_names], float),
        held_joint_names=held_names,
        held_default=np.array([by_name[n]['default_pos'] for n in held_names], float),
        held_topic=held_topic,
        sim_dt=float(man['timing']['sim_dt']), decimation=int(man['timing']['decimation']),
    )


class HeldTargets:
    """Targets for the held (non-policy) joints: default pose until a command arrives, then the
    latest commanded target per joint (joints not in a message keep their previous target)."""

    def __init__(self, contract: Contract):
        self.names = contract.held_joint_names
        self.index = {n: i for i, n in enumerate(self.names)}
        self.targets = contract.held_default.copy()
        self.received = False

    def update(self, names, positions) -> list[str]:
        """Apply a command (e.g. a JointState). Returns names that are not held joints (ignored)."""
        unknown = []
        for n, p in zip(names, positions):
            i = self.index.get(n)
            if i is None:
                unknown.append(n)
                continue
            self.targets[i] = float(p)
            self.received = True
        return unknown
