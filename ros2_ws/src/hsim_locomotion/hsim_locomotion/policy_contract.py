"""Policy deploy contract: reads a manifest (v1, v2 or v3) and builds the observation / joint
targets exactly like mjlab did in training. Pure numpy, no ROS, so tools/sim/sim2sim_play.py and
the tests use the same code as policy_node.

Manifest v1: velocity policy that drives every joint in `joints` (each has action_scale).
Manifest v2: velocity policy that drives `action.joints` (e.g. legs + waist); the rest are `held_joints`
             (role: held), commanded by another controller on the robot (arm targets on
             held_joints.source_topic, default_pos until the first message).
Manifest v3: motion-tracking policy (kind: tracking) that drives every joint and plays one motion clip,
             which the ONNX file carries (it takes `time_step` and returns the reference for that step).

Velocity observation (mjlab velocity task, 'actor' group, in this order):
  base_lin_vel(3) base_ang_vel(3) projected_gravity(3)
  joint_pos - default (every joint in `joints`, manifest order = training joint order)
  joint_vel           (every joint in `joints`)
  actions             (last raw policy output, one per driven joint)
  command(3)          (vx, vy, wz)

Tracking observation (mjlab tracking task, 'actor' group, in this order; the two marked * are dropped when the
policy was trained without state estimation):
  command             reference joint_pos and joint_vel of the clip at this step (2 per joint)
  motion_anchor_pos_b* (3) the clip's anchor body position in the robot's anchor body frame
  motion_anchor_ori_b (6) the clip's anchor orientation in that frame, first two rotation-matrix columns
  base_lin_vel*(3) base_ang_vel(3) joint_pos - default, joint_vel, actions
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

OBS_LAYOUT = ['base_lin_vel', 'base_ang_vel', 'projected_gravity', 'joint_pos', 'joint_vel', 'actions', 'command']
TRACKING_OBS_LAYOUT = ['command', 'motion_anchor_pos_b', 'motion_anchor_ori_b', 'base_lin_vel', 'base_ang_vel',
                       'joint_pos', 'joint_vel', 'actions']
TRACKING_OBS_LAYOUT_NO_ESTIMATION = [t for t in TRACKING_OBS_LAYOUT if t not in ('motion_anchor_pos_b', 'base_lin_vel')]


def quat_to_mat(q) -> np.ndarray:
    """Rotation matrix of a (w, x, y, z) quaternion."""
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def yaw_of(q) -> float:
    """Yaw of a (w, x, y, z) quaternion."""
    w, x, y, z = np.asarray(q, float)
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class MotionAlignment:
    """Maps the clip's world frame onto the robot's: a yaw rotation and an xy shift chosen at the start so the
    clip's anchor body starts where the robot's anchor body is, facing the same way (height is the clip's own)."""

    def __init__(self, motion_anchor_pos, motion_anchor_quat, robot_anchor_pos, robot_anchor_quat):
        """Fix the alignment from the clip's first anchor pose and the robot's anchor pose at that moment."""
        dyaw = yaw_of(robot_anchor_quat) - yaw_of(motion_anchor_quat)
        c, s = np.cos(dyaw), np.sin(dyaw)
        self.R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        p_m, p_r = np.asarray(motion_anchor_pos, float), np.asarray(robot_anchor_pos, float)
        self.t = np.array([*(p_r[:2] - (self.R @ p_m)[:2]), 0.0])

    def apply(self, pos, quat):
        """Clip pose (position, (w, x, y, z)) in the robot's world frame, as (position, rotation matrix)."""
        return self.R @ np.asarray(pos, float) + self.t, self.R @ quat_to_mat(quat)


@dataclass
class Contract:
    """Everything the node needs from a policy folder: joint order and roles, defaults, gains, action scales,
    observation layout, timing.
    """
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
    kind: str = 'velocity'              # 'velocity' or 'tracking'
    obs_terms: list[str] = field(default_factory=lambda: list(OBS_LAYOUT))
    anchor_body: str | None = None      # tracking: body the clip is anchored to (ROS frame torso_link)
    anchor_index: int = 0               # tracking: its index in the ONNX body_pos_w / body_quat_w outputs
    motion_steps: int | None = None     # tracking: clip length in policy steps

    @property
    def policy_hz(self) -> float:
        """Policy rate."""
        return 1.0 / (self.sim_dt * self.decimation)

    @property
    def obs_dim(self) -> int:
        """Observation size."""
        n = len(self.obs_joint_names)
        if self.kind == 'tracking':
            sizes = {'command': 2 * n, 'motion_anchor_pos_b': 3, 'motion_anchor_ori_b': 6, 'base_lin_vel': 3,
                     'base_ang_vel': 3, 'joint_pos': n, 'joint_vel': n, 'actions': len(self.act_joint_names)}
            return sum(sizes[t] for t in self.obs_terms)
        return 9 + 2 * n + len(self.act_joint_names) + 3

    def observation(self, lin_vel_b, ang_vel_b, gravity_b, q, dq, last_action, command) -> np.ndarray:
        """q, dq: positions/velocities of obs_joint_names (same order)."""
        return np.concatenate([
            np.asarray(lin_vel_b, float), np.asarray(ang_vel_b, float), np.asarray(gravity_b, float),
            np.asarray(q, float) - self.default_pos, np.asarray(dq, float),
            np.asarray(last_action, float), np.asarray(command, float),
        ]).astype(np.float32)

    def tracking_observation(self, ref_q, ref_dq, motion_anchor_pos, motion_anchor_R, robot_anchor_pos,
                             robot_anchor_R, lin_vel_b, ang_vel_b, q, dq, last_action) -> np.ndarray:
        """Tracking observation (see the module docstring). Anchor poses are in the same world frame (the clip's
        already mapped through MotionAlignment); rotations are 3x3 matrices; q, dq follow obs_joint_names."""
        R_r = np.asarray(robot_anchor_R, float)
        pos_b = R_r.T @ (np.asarray(motion_anchor_pos, float) - np.asarray(robot_anchor_pos, float))
        ori_b = (R_r.T @ np.asarray(motion_anchor_R, float))[:, :2].reshape(-1)   # mjlab: mat[..., :2], row-major
        parts = {
            'command': np.concatenate([np.asarray(ref_q, float), np.asarray(ref_dq, float)]),
            'motion_anchor_pos_b': pos_b, 'motion_anchor_ori_b': ori_b,
            'base_lin_vel': np.asarray(lin_vel_b, float), 'base_ang_vel': np.asarray(ang_vel_b, float),
            'joint_pos': np.asarray(q, float) - self.default_pos, 'joint_vel': np.asarray(dq, float),
            'actions': np.asarray(last_action, float),
        }
        return np.concatenate([parts[t] for t in self.obs_terms]).astype(np.float32)

    def action_targets(self, action) -> np.ndarray:
        """Joint position targets for act_joint_names."""
        return self.act_default + self.act_scale * np.asarray(action, float)


def load_contract(folder: str | Path) -> Contract:
    """Read and check `manifest.yaml` in `folder` (v1 all-joints, v2 arms-free or v3 tracking); raises ValueError
    on an unsupported layout.
    """
    folder = Path(folder)
    man = yaml.safe_load((folder / 'manifest.yaml').read_text())
    version = int(man.get('manifest_version', 1))
    kind = man.get('kind', 'tracking' if version == 3 else 'velocity')
    terms = list(man['observation']['terms'])
    allowed = (TRACKING_OBS_LAYOUT, TRACKING_OBS_LAYOUT_NO_ESTIMATION) if kind == 'tracking' else (OBS_LAYOUT,)
    if terms not in allowed:
        raise ValueError(f'unsupported observation layout {terms} for a {kind} policy')
    joints = man['joints']
    by_name = {j['name']: j for j in joints}
    obs_names = [j['name'] for j in joints]
    default = np.array([j['default_pos'] for j in joints], float)
    kp = np.array([j.get('kp', 0.0) for j in joints], float)
    kd = np.array([j.get('kd', 0.0) for j in joints], float)
    if version == 2:
        act_names = list(man['action']['joints'])
        missing = [n for n in act_names if n not in by_name or 'action_scale' not in by_name[n]]
        if missing:
            raise ValueError(f'action joints missing from manifest joints / action_scale: {missing}')
        held = man.get('held_joints', {})
        held_names = list(held.get('names', [n for n in obs_names if n not in act_names]))
        held_topic = held.get('source_topic')
    elif version in (1, 3):
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
        kind=kind, obs_terms=terms,
        anchor_body=(man.get('tracking') or {}).get('anchor_body'),
        anchor_index=int((man.get('tracking') or {}).get('anchor_index', 0)),
        motion_steps=(man.get('tracking') or {}).get('motion_steps'),
    )


class HeldTargets:
    """Targets for the held (non-policy) joints: default pose until a command arrives, then the
    latest commanded target per joint (joints not in a message keep their previous target)."""

    def __init__(self, contract: Contract):
        """Start every held joint at its manifest default until the first target message arrives."""
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
