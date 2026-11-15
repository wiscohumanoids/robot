"""Custom terms for the v2 locomotion recipe (docs/runs/2026-11-10_overnight_plan.md) that mjlab 1.6.0 lacks.

1. YawInPlaceVelocityCommand: like mjlab's UniformVelocityCommand, plus a fraction of samples that are
   pure turns in place (zero linear velocity, |yaw| in a set range). Those envs are taken out of
   heading / world-frame / forward modes so nothing overwrites the yaw. (decision 0004: v1 ignored small
   turns and undershot turn-in-place.)
2. Arm targets: the arms are not policy actions (the ROS controls stack owns them on the robot). In sim
   they are held by their PD actuators while their position targets move toward a goal that is resampled
   every 1-3 s: 30 % of goals are the default pose, 70 % are uniform within (range_frac x level) of each arm
   joint's soft range around default; each goal is approached at a random speed. Runs every env step.
3. Difficulty curriculum (coordinator fix 2026-11-10, same design as the G1 v2 task): a performance-gated level
   L in [0, 1] scales the disturbances (arm target range 0 -> 60 %, hand payload max 0 -> 1 kg, push 0.5 -> 0.8 m/s);
   once L = 1 a second level L2 tightens velocity-tracking std from the G1 defaults (0.5 / 0.707) to 0.25 / 0.35.
   Gate, evaluated on episode resets over recent finished episodes: +0.05 when mean episode length > 0.8 x max AND the
   linear-tracking reward rate > 50 % of its weight AND the angular-tracking rate > 40 % of its weight, at most once per
   50 training iterations (24 x 50 env steps); -0.1 when mean episode length < 0.4 x max (fix 2, 2026-11-10).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


# ---------------------------------------------------------------------------------------- commands
class YawInPlaceVelocityCommand(UniformVelocityCommand):
  cfg: "YawInPlaceVelocityCommandCfg"

  def __init__(self, cfg: "YawInPlaceVelocityCommandCfg", env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.is_yaw_only_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    r = torch.rand(len(env_ids), device=self.device)
    yaw_only = (r < self.cfg.rel_yaw_only_envs) & ~self.is_standing_env[env_ids]
    self.is_yaw_only_env[env_ids] = yaw_only
    ids = env_ids[yaw_only]
    if len(ids) == 0:
      return
    lo, hi = self.cfg.yaw_only_range
    mag = torch.empty(len(ids), device=self.device).uniform_(lo, hi)
    sign = torch.where(torch.rand(len(ids), device=self.device) < 0.5, -1.0, 1.0)
    self.vel_command_b[ids, 0] = 0.0
    self.vel_command_b[ids, 1] = 0.0
    self.vel_command_b[ids, 2] = sign * mag
    self.vel_command_w[ids] = self.vel_command_b[ids]
    self.is_heading_env[ids] = False
    self.is_world_env[ids] = False
    self.is_forward_env[ids] = False


@dataclass(kw_only=True)
class YawInPlaceVelocityCommandCfg(UniformVelocityCommandCfg):
  rel_yaw_only_envs: float = 0.15
  """Fraction of command samples that are pure turns in place (zero linear velocity)."""
  yaw_only_range: tuple[float, float] = (0.3, 1.5)
  """|yaw rate| range (rad/s) for the turn-in-place samples."""

  def build(self, env: ManagerBasedRlEnv) -> YawInPlaceVelocityCommand:
    return YawInPlaceVelocityCommand(self, env)


# ---------------------------------------------------------------------------------------- arm targets
class _ArmTargetState:
  def __init__(self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, fracs: dict):
    asset = env.scene[asset_cfg.name]
    self.asset = asset
    self.joint_ids, names = asset.find_joints(asset_cfg.joint_names, preserve_order=True)
    lo_f, hi_f = [], []
    for n in names:
      key = n.replace("left_", "").replace("right_", "")
      lo_f.append(fracs[key][0]); hi_f.append(fracs[key][1])
    self.frac_lo = torch.tensor(lo_f, device=env.device)
    self.frac_hi = torch.tensor(hi_f, device=env.device)
    ids = torch.tensor(self.joint_ids, device=env.device)
    self.ids = ids
    n, j = env.num_envs, len(self.joint_ids)
    default = asset.data.default_joint_pos[:, ids]                       # (N, J)
    soft = asset.data.soft_joint_pos_limits[:, ids]                      # (N, J, 2)
    self.default = default.clone()
    self.soft_lo, self.soft_hi = soft[..., 0].clone(), soft[..., 1].clone()
    self.target = default.clone()
    self.goal = default.clone()
    self.speed = torch.ones(n, 1, device=env.device)
    self.time_left = torch.zeros(n, device=env.device)
    self.n, self.j, self.device = n, j, env.device


def _state(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, fracs: dict) -> _ArmTargetState:
  st = getattr(env, "_wisco_arm_state", None)
  if st is None:
    st = _ArmTargetState(env, asset_cfg, fracs)
    env._wisco_arm_state = st
  return st


def arm_targets_reset(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, asset_cfg: SceneEntityCfg,
                      fracs: dict) -> None:
  """On episode reset: arms back to the default pose, new goal timer."""
  st = _state(env, asset_cfg, fracs)
  if env_ids is None:
    env_ids = torch.arange(st.n, device=st.device)
  st.target[env_ids] = st.default[env_ids]
  st.goal[env_ids] = st.default[env_ids]
  st.time_left[env_ids] = 0.0
  st.asset.set_joint_position_target(st.target[env_ids], joint_ids=st.ids, env_ids=env_ids)


def arm_targets_step(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, asset_cfg: SceneEntityCfg,
                     fracs: dict, hold_prob: float = 0.3, period_s: tuple[float, float] = (1.0, 3.0),
                     speed_range: tuple[float, float] = (0.3, 2.5)) -> None:
  """Every env step: resample goals whose timer ran out, move targets toward goals at their speed (rad/s).
  Goal range per joint: default + level x (frac_lo toward the lower soft limit .. frac_hi toward the upper)."""
  del env_ids
  st = _state(env, asset_cfg, fracs)
  dt = env.step_dt
  st.time_left -= dt
  due = (st.time_left <= 0.0).nonzero(as_tuple=False).flatten()
  if len(due) > 0:
    k = len(due)
    lv = level(env)                                                      # curriculum: 0 -> full per-joint range
    lo = st.default[due] + lv * st.frac_lo * (st.soft_lo[due] - st.default[due])
    hi = st.default[due] + lv * st.frac_hi * (st.soft_hi[due] - st.default[due])
    u = torch.rand(k, st.j, device=st.device)
    new_goal = lo + u * (hi - lo)
    hold = torch.rand(k, 1, device=st.device) < hold_prob
    st.goal[due] = torch.where(hold, st.default[due], new_goal)
    st.speed[due] = torch.empty(k, 1, device=st.device).uniform_(*speed_range)
    st.time_left[due] = torch.empty(k, device=st.device).uniform_(*period_s)
  step = st.speed * dt
  st.target = st.target + torch.clamp(st.goal - st.target, -step, step)
  st.asset.set_joint_position_target(st.target, joint_ids=st.ids)


# ---------------------------------------------------------------------------------------- curriculum
class _Curriculum:
  def __init__(self):
    import os
    # Resumed runs keep their progress: WISCO_START_LEVEL / WISCO_START_LEVEL2 (default 0) set the initial levels; the
    # first raise from there still waits the dwell time (fix 3).
    self.L = float(os.environ.get("WISCO_START_LEVEL", "0"))
    self.L2 = float(os.environ.get("WISCO_START_LEVEL2", "0"))
    self.n = 0
    self.sum_len = 0.0
    self.sum_rate = 0.0
    self.sum_ang = 0.0
    self.last_eval_step = 0
    self.last_raise_step = 0
    self.last_len_frac = 0.0
    self.last_rate_frac = 0.0
    self.last_ang_frac = 0.0


def _curr(env) -> _Curriculum:
  c = getattr(env, "_wisco_curr", None)
  if c is None:
    c = _Curriculum()
    env._wisco_curr = c
  return c


def level(env) -> float:
  return _curr(env).L


def _lerp(a, b, t):
  return a + (b - a) * t


def _apply(env, c: _Curriculum, cfg: dict) -> None:
  """Push the current levels into the live term configs (managers read params every call)."""
  try:
    push = env.event_manager.get_term_cfg("push_robot").params["velocity_range"]
  except (KeyError, ValueError):
    push = None                                   # play mode: no pushes
  if push is not None:
    v = _lerp(cfg["push_min"], cfg["push_max"], c.L)
    push["x"], push["y"] = (-v, v), (-v, v)
  rm = env.reward_manager
  rm.get_term_cfg(cfg["lin_reward"]).params["std"] = _lerp(cfg["lin_std"][0], cfg["lin_std"][1], c.L2)
  rm.get_term_cfg(cfg["ang_reward"]).params["std"] = _lerp(cfg["ang_std"][0], cfg["ang_std"][1], c.L2)
  # arm target range reads level(env) directly; hand payload reads it in hand_payload()


DEFAULT_CURRICULUM = dict(
  lin_reward="track_linear_velocity", ang_reward="track_angular_velocity",
  lin_gate="gate_track_lin", ang_gate="gate_track_ang",
  up_len=0.8, down_len=0.4, rate_frac=0.5, ang_rate_frac=0.4, step_up=0.05, step_down=0.1,
  eval_interval_steps=24 * 10, min_episodes=64, min_steps_between_raises=24 * 50,
  push_min=0.5, push_max=0.8, lin_std=(0.5, 0.25), ang_std=(0.5 ** 0.5, 0.35),
)


def difficulty_level(env: ManagerBasedRlEnv, env_ids, **params) -> torch.Tensor:
  """Curriculum term (logged as Curriculum/level). Accumulates finished-episode stats, moves L / L2."""
  cfg = {**DEFAULT_CURRICULUM, **params}
  c = _curr(env)
  if isinstance(env_ids, slice) or env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  lens = env.episode_length_buf[env_ids].float()
  done = lens > 0
  if bool(done.any()):
    secs = lens[done] * env.step_dt
    # J3f: gate on fixed-std copies of the tracking rewards (L2 tightening must not lower its own gate score)
    rate = env.reward_manager._episode_sums[cfg["lin_gate"]][env_ids][done] / secs
    ang = env.reward_manager._episode_sums[cfg["ang_gate"]][env_ids][done] / secs
    c.n += int(done.sum())
    c.sum_len += float(lens[done].sum())
    c.sum_rate += float(rate.sum())
    c.sum_ang += float(ang.sum())
  step_now = int(env.common_step_counter)
  if step_now - c.last_eval_step >= cfg["eval_interval_steps"] and c.n >= cfg["min_episodes"]:
    mean_len_frac = c.sum_len / c.n / env.max_episode_length
    rm = env.reward_manager
    mean_rate_frac = (c.sum_rate / c.n) / rm.get_term_cfg(cfg["lin_gate"]).weight
    mean_ang_frac = (c.sum_ang / c.n) / rm.get_term_cfg(cfg["ang_gate"]).weight
    c.last_len_frac, c.last_rate_frac, c.last_ang_frac = mean_len_frac, mean_rate_frac, mean_ang_frac
    can_raise = step_now - c.last_raise_step >= cfg["min_steps_between_raises"]
    if (can_raise and mean_len_frac > cfg["up_len"] and mean_rate_frac > cfg["rate_frac"]
        and mean_ang_frac > cfg["ang_rate_frac"]):
      if c.L < 1.0 - 1e-6:
        c.L = min(1.0, round(c.L + cfg["step_up"], 6))
      else:
        c.L2 = min(1.0, round(c.L2 + cfg["step_up"], 6))
      c.last_raise_step = step_now
    elif mean_len_frac < cfg["down_len"]:
      if c.L2 > 1e-6:
        c.L2 = max(0.0, round(c.L2 - cfg["step_down"], 6))
      else:
        c.L = max(0.0, round(c.L - cfg["step_down"], 6))
    c.n, c.sum_len, c.sum_rate, c.sum_ang, c.last_eval_step = 0, 0.0, 0.0, 0.0, step_now
  _apply(env, c, cfg)
  return torch.tensor(c.L)


def level_info(env: ManagerBasedRlEnv, env_ids, **params) -> dict:
  """Read-only curriculum term: logs Curriculum/level_info/{level2, lin_std, ang_std, ep_len_frac, lin_track_frac}."""
  del env_ids
  cfg = {**DEFAULT_CURRICULUM, **params}
  c = _curr(env)
  rm = env.reward_manager
  return {
    "level2": torch.tensor(c.L2),
    "lin_std": torch.tensor(float(rm.get_term_cfg(cfg["lin_reward"]).params["std"])),
    "ang_std": torch.tensor(float(rm.get_term_cfg(cfg["ang_reward"]).params["std"])),
    "ep_len_frac": torch.tensor(c.last_len_frac),
    "lin_track_frac": torch.tensor(c.last_rate_frac),
    "ang_track_frac": torch.tensor(c.last_ang_frac),
  }


def hand_payload(env: ManagerBasedRlEnv, env_ids, asset_cfg: SceneEntityCfg, max_kg: float = 1.0) -> None:
  """Interval event: each hand's mass = default + U(0, max_kg x level)."""
  from mjlab.envs.mdp import dr
  asset = env.scene[asset_cfg.name]
  body_ids, _ = asset.find_bodies(asset_cfg.body_names, preserve_order=True)
  gid = asset.indexing.body_ids[torch.tensor(body_ids, device=env.device)]
  m0 = float(env.sim.get_default_field("body_mass")[gid].max())
  hi = m0 + max_kg * level(env)
  dr.body_mass(env, env_ids, ranges=(m0, max(hi, m0 + 1e-6)), asset_cfg=asset_cfg, operation="abs")


def fixed_level(env: ManagerBasedRlEnv, env_ids, L: float = 1.0, L2: float = 1.0, **params) -> torch.Tensor:
  """Play / evaluation: hold the curriculum at fixed levels (default: full difficulty) and apply them."""
  del env_ids
  cfg = {**DEFAULT_CURRICULUM, **params}
  c = _curr(env)
  c.L, c.L2 = L, L2
  _apply(env, c, cfg)
  return torch.tensor(c.L)


def self_collision_cost_multi(env: ManagerBasedRlEnv, sensor_names: tuple[str, ...], force_threshold: float = 10.0) -> torch.Tensor:
  """mjlab's self_collision_cost summed over several contact sensors (non-arm body groups, fix 2)."""
  from mjlab.tasks.velocity import mdp
  return sum(mdp.self_collision_cost(env, sensor_name=n, force_threshold=force_threshold) for n in sensor_names)


def policy_body_angular_momentum_sq(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """|L|^2 of the policy-controlled bodies only (fix 3): pelvis, legs, torso links and the welded head, NOT arms/hands.

  mjlab's angular_momentum_penalty uses the whole-body subtree angular momentum, which is dominated by the scripted
  arms + hand payload as the curriculum scales them up (J3c: -0.02 -> -1.10 per second at weight -0.005).
  L = sum_i m_i (p_i - c) x (v_i - v_c) + R_i diag(I_i) R_i^T w_i, about the selected bodies' common COM c.
  """
  from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
  asset = env.scene[asset_cfg.name]
  ids = asset_cfg.body_ids
  gid = asset.indexing.body_ids[ids]
  pose = asset.data.body_com_pose_w[:, ids]                     # (N, B, 7) COM pos + inertial-frame quat
  vel = asset.data.body_com_vel_w[:, ids]                       # (N, B, 6) lin + ang (world)
  m = env.sim.model.body_mass[..., gid]
  inertia = env.sim.model.body_inertia[..., gid, :]
  if m.dim() == 1:
    m = m.unsqueeze(0).expand(pose.shape[0], -1)
  if inertia.dim() == 2:
    inertia = inertia.unsqueeze(0).expand(pose.shape[0], -1, -1)
  p, q, v, w = pose[..., :3], pose[..., 3:], vel[..., :3], vel[..., 3:]
  M = m.sum(dim=1, keepdim=True)
  c = (m.unsqueeze(-1) * p).sum(dim=1, keepdim=True) / M.unsqueeze(-1)
  vc = (m.unsqueeze(-1) * v).sum(dim=1, keepdim=True) / M.unsqueeze(-1)
  L_orb = (m.unsqueeze(-1) * torch.cross(p - c, v - vc, dim=-1)).sum(dim=1)
  w_b = quat_apply_inverse(q, w)
  L_rot = quat_apply(q, inertia * w_b).sum(dim=1)
  L = L_orb + L_rot
  env.extras.setdefault("log", {})["Metrics/policy_angmom_mean"] = torch.mean(torch.linalg.norm(L, dim=-1))
  return torch.sum(L * L, dim=-1)


def wholebody_angular_momentum_sq(env: ManagerBasedRlEnv, sensor_name: str = "robot/root_angmom") -> torch.Tensor:
  """Metric: mjlab's whole-body (incl. arms) |L|^2, kept for comparison with the policy-body term."""
  return torch.sum(torch.square(env.scene[sensor_name].data), dim=-1)


def track_yaw_rate(env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
  """Yaw-rate-only tracking reward exp(-(cmd_wz - wz)^2 / std^2) (legged_gym style).

  mjlab's track_angular_velocity also counts roll/pitch rate as error; once the std is tightened (0.35) the rocking from
  stepping costs more than a slow turn earns, so the policy stands still for |wz| < ~0.8 rad/s. Roll/pitch rate stays
  penalised by the separate body_ang_vel term."""
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  return torch.exp(-torch.square(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2]) / std**2)
