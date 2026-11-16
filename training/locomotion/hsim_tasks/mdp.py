"""Custom MDP terms for the arms-free locomotion recipe (docs/runs/2026-11-10_overnight_plan.md).

- TurnInPlaceVelocityCommandCfg: mjlab's UniformVelocityCommand plus a fraction of pure-yaw
  commands (zero linear velocity, |yaw| in a range), so turning in place gets practiced
  (decision 0004: the all-joints policy undershot / ignored small turns).
- random_arm_targets: the arms are not policy actions. This event term stands in for the
  ROS controls stack: it writes arm joint position targets every env step, moving smoothly
  to a new random goal every 1-3 s (30 % of goals = the default pose, 70 % = uniform within
  60 % of each joint's soft range around default), at a random move duration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


class TurnInPlaceVelocityCommand(UniformVelocityCommand):
  cfg: TurnInPlaceVelocityCommandCfg

  def __init__(self, cfg: TurnInPlaceVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.is_turn_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    r = torch.empty(len(env_ids), device=self.device)
    turn = r.uniform_(0.0, 1.0) < self.cfg.rel_turn_envs
    if getattr(self.cfg, "keep_standing", False):   # J1k: standing resamples stay standing (turns no longer take ~30 % of them)
      turn &= ~self.is_standing_env[env_ids]
    self.is_turn_env[env_ids] = turn
    ids = env_ids[turn]
    if len(ids) == 0:
      self._small_commands(env_ids)
      return
    lo, hi = self.cfg.turn_yaw_range
    mag = torch.empty(len(ids), device=self.device).uniform_(lo, hi)
    sign = torch.where(torch.rand(len(ids), device=self.device) < 0.5, -1.0, 1.0)
    self.vel_command_b[ids, 0] = 0.0
    self.vel_command_b[ids, 1] = 0.0
    self.vel_command_b[ids, 2] = sign * mag
    self.vel_command_w[ids] = self.vel_command_b[ids]
    # a pure-yaw env is not a heading, standing, world-frame or forward-only env
    self.is_heading_env[ids] = False
    self.is_standing_env[ids] = False
    self.is_world_env[ids] = False
    self.is_forward_env[ids] = False
    self._small_commands(env_ids[~turn])

  def _small_commands(self, env_ids: torch.Tensor) -> None:
    """J1i: a fraction of resamples get one small command (slow walk, slow sidestep or slow in-place turn) so the
    policy practises the slow end it otherwise ignores (deadband)."""
    frac = getattr(self.cfg, "rel_small_envs", 0.0)
    if frac <= 0.0 or len(env_ids) == 0:
      return
    env_ids = env_ids[~self.is_standing_env[env_ids]]
    pick = torch.rand(len(env_ids), device=self.device) < frac
    ids = env_ids[pick]
    n = len(ids)
    if n == 0:
      return
    mode = torch.randint(0, 3, (n,), device=self.device)
    sign = torch.where(torch.rand(n, device=self.device) < 0.5, -1.0, 1.0)
    u = torch.rand(n, device=self.device)
    cmd = torch.zeros(n, 3, device=self.device)
    lx, ly, lw = self.cfg.small_lin_x_range, self.cfg.small_lin_y_range, self.cfg.small_yaw_range
    cmd[:, 0] = torch.where(mode == 0, sign * (lx[0] + u * (lx[1] - lx[0])), 0.0)
    cmd[:, 1] = torch.where(mode == 1, sign * (ly[0] + u * (ly[1] - ly[0])), 0.0)
    cmd[:, 2] = torch.where(mode == 2, sign * (lw[0] + u * (lw[1] - lw[0])), 0.0)
    self.vel_command_b[ids] = cmd
    self.vel_command_w[ids] = cmd
    self.is_heading_env[ids] = False
    self.is_world_env[ids] = False
    self.is_forward_env[ids] = False


@dataclass(kw_only=True)
class TurnInPlaceVelocityCommandCfg(UniformVelocityCommandCfg):
  rel_turn_envs: float = 0.2
  """Fraction of resamples that are pure yaw (zero linear velocity)."""
  turn_yaw_range: tuple[float, float] = (0.3, 1.5)
  """|yaw rate| range for pure-yaw commands, rad/s."""
  rel_small_envs: float = 0.0
  """J1i: fraction of the other (non-turn, non-standing) resamples that get one small command."""
  small_lin_x_range: tuple[float, float] = (0.1, 0.35)
  small_lin_y_range: tuple[float, float] = (0.1, 0.3)
  small_yaw_range: tuple[float, float] = (0.15, 0.5)
  keep_standing: bool = False
  """J1k: pure-yaw resamples never replace a standing one."""

  def build(self, env: ManagerBasedRlEnv) -> TurnInPlaceVelocityCommand:
    return TurnInPlaceVelocityCommand(self, env)


class random_arm_targets:
  """Event term (mode="step"): scripted arm motion standing in for the ROS arm controller.

  params:
    asset_cfg: SceneEntityCfg("robot", joint_names=<arm joints>)
    resample_s: (min, max) seconds between new goals
    hold_default_prob: probability a new goal is the default pose
    range_frac: scale f of the goal range (set by the curriculum).
    offsets: optional {joint suffix: (lo, hi)} safe offsets from default (arm_ranges.G1_ARM_OFFSETS,
             right arm mirrored). With offsets: goal = default + f * U(lo, hi), clamped to soft limits.
             Without: goal uniform in [default - f*(default-lo), default + f*(hi-default)] (soft limits).
    move_s: (min, max) seconds to move from the current target to the new goal
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    p = cfg.params
    asset_cfg: SceneEntityCfg = p["asset_cfg"]
    self._asset: Entity = env.scene[asset_cfg.name]
    ids, names = self._asset.find_joints(asset_cfg.joint_names, preserve_order=True)
    self._ids = torch.tensor(ids, device=env.device, dtype=torch.long)
    self._dt = env.step_dt
    self._dev = env.device
    n, k = env.num_envs, len(ids)
    self._resample = p.get("resample_s", (1.0, 3.0))
    self._hold_p = p.get("hold_default_prob", 0.3)
    self._frac = p.get("range_frac", 0.6)
    self._move = p.get("move_s", (0.3, 1.5))
    self._off = None
    if p.get("offsets"):
      from .arm_ranges import offset_bounds
      b = [offset_bounds(n, p["offsets"]) for n in names]
      self._off = torch.tensor(b, device=env.device, dtype=torch.float32)      # (k, 2)
    self._start = torch.zeros(n, k, device=env.device)
    self._goal = torch.zeros(n, k, device=env.device)
    self._t = torch.zeros(n, device=env.device)          # time since goal was set
    self._T = torch.ones(n, device=env.device)           # move duration
    self._next = torch.zeros(n, device=env.device)       # time until next goal

  def _default(self, env_ids):
    return self._asset.data.default_joint_pos[env_ids][:, self._ids]

  def _current(self):
    a = torch.clamp(self._t / self._T, 0.0, 1.0)[:, None]
    a = a * a * (3.0 - 2.0 * a)                          # smoothstep
    return self._start + a * (self._goal - self._start)

  def _new_goals(self, env_ids: torch.Tensor):
    m = len(env_ids)
    if m == 0:
      return
    cur = self._current()[env_ids]
    default = self._default(env_ids)
    lim = self._asset.data.soft_joint_pos_limits[env_ids][:, self._ids]
    if self._off is not None:
      lo = default + self._frac * self._off[:, 0]
      hi = default + self._frac * self._off[:, 1]
      goal = torch.minimum(torch.maximum(lo + torch.rand_like(lo) * (hi - lo), lim[..., 0]), lim[..., 1])
    else:
      lo = default - self._frac * (default - lim[..., 0]).clamp(min=0.0)
      hi = default + self._frac * (lim[..., 1] - default).clamp(min=0.0)
      goal = lo + torch.rand_like(lo) * (hi - lo)
    hold = torch.rand(m, device=self._dev) < self._hold_p
    goal[hold] = default[hold]
    self._start[env_ids] = cur
    self._goal[env_ids] = goal
    self._t[env_ids] = 0.0
    self._T[env_ids] = torch.empty(m, device=self._dev).uniform_(*self._move)
    self._next[env_ids] = torch.empty(m, device=self._dev).uniform_(*self._resample)

  def _write(self):
    self._asset.set_joint_position_target(self._current(), joint_ids=self._ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    # Called after the scene reset cleared all joint targets: hold the default pose.
    if env_ids is None or isinstance(env_ids, slice):
      env_ids = torch.arange(self._start.shape[0], device=self._dev)
    d = self._default(env_ids)
    self._start[env_ids] = d
    self._goal[env_ids] = d
    self._t[env_ids] = 0.0
    self._T[env_ids] = 1.0
    self._next[env_ids] = torch.empty(len(env_ids), device=self._dev).uniform_(*self._resample)
    self._write()

  def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, **kwargs) -> None:
    del env, env_ids, kwargs
    self._t += self._dt
    self._next -= self._dt
    due = (self._next <= 0.0).nonzero(as_tuple=False).flatten()
    self._new_goals(due)
    self._write()


class difficulty_curriculum:
  """Performance-gated curriculum (coordinator spec, 2026-11-10; same design as helper A's Wisco one).

  Two levels in [0, 1], updated on episode resets from a window of finished episodes:
    up   (+step) if mean episode length > up_len_frac * max  AND  linear tracking reward per
                 second > up_track_frac * its weight
    down (-step) if mean episode length < down_len_frac * max
  L  scales the disturbances: arm random-target range 0 -> arm_range_max (of soft range),
     hand payload max 0 -> payload_max_kg, push velocity push_min -> push_max (x and y).
  L2 only rises once L == 1; it tightens the tracking stds linearly from the mjlab G1 defaults
     (std_lin_start, std_ang_start) to (std_lin_end, std_ang_end). Going down lowers L2 first.
  Logged every reset as Curriculum/difficulty/<key> (level, level2, std_lin, std_ang, ...).

  J1c: up also needs angular tracking per second > up_ang_frac * its weight, and at least
  dwell_iters training iterations (x steps_per_iter env steps) since the last raise (L or L2).
  params: up_len_frac, up_track_frac, up_ang_frac, down_len_frac, step, window_episodes,
          dwell_iters, steps_per_iter,
          arm_range_max, payload_max_kg, push_min, push_max,
          std_lin_start, std_lin_end, std_ang_start, std_ang_end,
          arm_term, payload_term, push_term, lin_reward_term, ang_reward_term
  """

  def __init__(self, cfg, env):
    self._env = env
    self.p = dict(cfg.params)
    start = float(self.p.get("start_level", 0.0) or 0.0)    # resumed runs: continue from this level
    self.level = min(max(start, 0.0), 1.0)
    self.level2 = min(max(start - 1.0, 0.0), 1.0)
    self._n = 0
    self._len_sum = 0.0
    self._track_sum = 0.0
    self._ang_sum = 0.0
    self._last_raise_step = -10**12
    self._last = {"mean_len_frac": 0.0, "lin_track_frac": 0.0, "ang_track_frac": 0.0}
    self._applied = False

  # -- parameters driven by the levels -------------------------------------------------------
  def stds(self):
    p, a = self.p, self.level2
    return (p["std_lin_start"] + a * (p["std_lin_end"] - p["std_lin_start"]),
            p["std_ang_start"] + a * (p["std_ang_end"] - p["std_ang_start"]))

  def apply(self):
    env, p, L = self._env, self.p, self.level
    em, rm = env.event_manager, env.reward_manager
    names = set(em.active_terms) if isinstance(em.active_terms, list) else {
      n for mode_terms in em.active_terms.values() for n in mode_terms}
    if p["arm_term"] in names:
      em.get_term_cfg(p["arm_term"]).func._frac = p["arm_range_max"] * L
    if p["payload_term"] in names:
      em.get_term_cfg(p["payload_term"]).params["ranges"] = (0.0, p["payload_max_kg"] * L)
    if p["push_term"] in names:                       # absent in play / eval configs
      v = p["push_min"] + L * (p["push_max"] - p["push_min"])
      vr = em.get_term_cfg(p["push_term"]).params["velocity_range"]
      vr["x"] = (-v, v)
      vr["y"] = (-v, v)
    s_lin, s_ang = self.stds()
    rm.get_term_cfg(p["lin_reward_term"]).params["std"] = s_lin
    rm.get_term_cfg(p["ang_reward_term"]).params["std"] = s_ang
    self._applied = True

  def set_levels(self, level: float, level2: float = 0.0):
    """For tests / resuming: force the levels and apply them."""
    self.level = float(min(max(level, 0.0), 1.0))
    self.level2 = float(min(max(level2, 0.0), 1.0)) if self.level >= 1.0 else 0.0
    self.apply()

  def state(self):
    p, L = self.p, self.level
    s_lin, s_ang = self.stds()
    return {
      "level": torch.tensor(L), "level2": torch.tensor(self.level2),
      "arm_range_frac": torch.tensor(p["arm_range_max"] * L),
      "payload_max_kg": torch.tensor(p["payload_max_kg"] * L),
      "push_mps": torch.tensor(p["push_min"] + L * (p["push_max"] - p["push_min"])),
      "std_lin": torch.tensor(s_lin), "std_ang": torch.tensor(s_ang),
      "mean_len_frac": torch.tensor(self._last["mean_len_frac"]),
      "lin_track_frac": torch.tensor(self._last["lin_track_frac"]),
      "ang_track_frac": torch.tensor(self._last["ang_track_frac"]),
    }

  # -- mjlab hooks ----------------------------------------------------------------------------
  def reset(self, env_ids=None):
    pass

  def __call__(self, env, env_ids, **kwargs):
    del kwargs
    if not self._applied:
      self.apply()
    if isinstance(env_ids, slice) or env_ids is None:
      env_ids = torch.arange(env.num_envs, device=env.device)
    lens = env.episode_length_buf[env_ids].float()
    done = lens > 0                       # skip the very first reset (no episode yet)
    if done.any():
      ids = env_ids[done]
      lens = lens[done]
      p = self.p
      # J1f: gate on fixed-std copies of the tracking rewards when given, so tightening the training stds (L2)
      # doesn't lower the gate score by itself
      lin_t, ang_t = p.get("lin_gate_term", p["lin_reward_term"]), p.get("ang_gate_term", p["ang_reward_term"])
      lin_sum = env.reward_manager._episode_sums[lin_t][ids]
      per_s = lin_sum / (lens * env.step_dt)
      ang_s = env.reward_manager._episode_sums[ang_t][ids] / (lens * env.step_dt)
      self._n += int(done.sum())
      self._len_sum += float(lens.sum())
      self._track_sum += float(per_s.sum())
      self._ang_sum += float(ang_s.sum())
    if self._n >= self.p["window_episodes"]:
      p = self.p
      mean_len_frac = self._len_sum / self._n / env.max_episode_length
      w = env.reward_manager.get_term_cfg(p.get("lin_gate_term", p["lin_reward_term"])).weight
      lin_track_frac = (self._track_sum / self._n) / w if w else 0.0
      wa = env.reward_manager.get_term_cfg(p.get("ang_gate_term", p["ang_reward_term"])).weight
      ang_track_frac = (self._ang_sum / self._n) / wa if wa else 0.0
      self._last = {"mean_len_frac": mean_len_frac, "lin_track_frac": lin_track_frac, "ang_track_frac": ang_track_frac}
      self._n, self._len_sum, self._track_sum, self._ang_sum = 0, 0.0, 0.0, 0.0
      changed = False
      dwell_ok = (env.common_step_counter - self._last_raise_step) >= p.get("dwell_iters", 0) * p.get("steps_per_iter", 24)
      if (mean_len_frac > p["up_len_frac"] and lin_track_frac > p["up_track_frac"]
          and ang_track_frac > (p.get("up_ang_frac_l2", p.get("up_ang_frac", 0.0)) if self.level >= 1.0 - 1e-6
                                else p.get("up_ang_frac", 0.0))
          and dwell_ok):   # L2 (tracking tightening) can use a lower turn gate than L
        if self.level < 1.0 - 1e-6:
          self.level = min(1.0, round(self.level + p["step"], 6)); changed = True
        elif self.level2 < 1.0 - 1e-6:
          self.level2 = min(1.0, round(self.level2 + p["step"], 6)); changed = True
        if changed:
          self._last_raise_step = env.common_step_counter
      elif mean_len_frac < p["down_len_frac"]:
        if self.level2 > 1e-6:
          self.level2 = max(0.0, round(self.level2 - p["step"], 6)); changed = True
        elif self.level > 1e-6:
          self.level = max(0.0, round(self.level - p["step"], 6)); changed = True
      if changed:
        self.apply()
    return self.state()


def body_self_collision_cost(env, sensor_names: tuple[str, ...], force_threshold: float = 10.0) -> torch.Tensor:
  """Like mjlab's self_collision_cost, but over several contact sensors (OR across sensors and
  primaries per physics substep, count substeps). J1c: the sensors only pair leg links with the
  torso, the other leg and the pelvis, so contacts involving the scripted arms are never penalized."""
  hit = None
  for name in sensor_names:
    data = env.scene[name].data
    if data.force_history is not None:
      h = (torch.norm(data.force_history, dim=-1) > force_threshold).any(dim=1)   # [B, H]
    else:
      h = (data.found > 0).any(dim=-1, keepdim=True)
    hit = h if hit is None else (hit | h)
  return hit.sum(dim=-1).float()


def _quat_to_mat(q: torch.Tensor) -> torch.Tensor:
  """(..., 4) wxyz -> (..., 3, 3)."""
  w, x, y, z = q.unbind(-1)
  return torch.stack([
    1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
  ], dim=-1).reshape(q.shape[:-1] + (3, 3))


def bodies_angular_momentum(env, asset_cfg) -> torch.Tensor:
  """World-frame angular momentum (N, 3) of the bodies in asset_cfg about their combined COM:
  sum_i m_i (p_i - c) x (v_i - v_c) + R_i diag(I_i) R_i^T w_i  (same definition as MuJoCo's
  subtreeangmom, restricted to the chosen bodies). Uses the current (possibly randomized) masses."""
  asset = env.scene[asset_cfg.name]
  ids = asset_cfg.body_ids
  if isinstance(ids, slice):
    ids = list(range(asset.num_bodies))[ids]
  ids_t = torch.as_tensor(ids, device=env.device, dtype=torch.long)
  gid = torch.as_tensor(asset.indexing.body_ids, device=env.device)[ids_t]
  mass = torch.as_tensor(env.sim.model.body_mass, device=env.device)
  inert = torch.as_tensor(env.sim.model.body_inertia, device=env.device)
  n = env.num_envs
  m = (mass[:, gid] if mass.dim() == 2 else mass[gid].expand(n, -1)).float()          # (N, K)
  I = (inert[:, gid] if inert.dim() == 3 else inert[gid].expand(n, -1, -1)).float()   # (N, K, 3)
  d = asset.data
  p = d.body_com_pos_w[:, ids_t]
  v = d.body_com_lin_vel_w[:, ids_t]
  w = d.body_com_ang_vel_w[:, ids_t]
  R = _quat_to_mat(d.body_com_quat_w[:, ids_t])                                       # (N, K, 3, 3)
  M = m.sum(dim=1, keepdim=True)
  c = (m[..., None] * p).sum(dim=1, keepdim=True) / M[..., None]
  vc = (m[..., None] * v).sum(dim=1, keepdim=True) / M[..., None]
  orbital = (m[..., None] * torch.cross(p - c, v - vc, dim=-1)).sum(dim=1)
  w_local = torch.einsum("nkji,nkj->nki", R, w)                                        # R^T w
  spin = torch.einsum("nkij,nkj->nki", R, I * w_local).sum(dim=1)
  return orbital + spin


def policy_bodies_angular_momentum_penalty(env, asset_cfg) -> torch.Tensor:
  """J1d: like mjlab's angular_momentum_penalty (|L|^2) but only over the policy-controlled bodies
  (pelvis, legs, waist, torso link), so the scripted arms and hand payload don't count."""
  L = bodies_angular_momentum(env, asset_cfg)
  sq = torch.sum(L * L, dim=-1)
  env.extras.setdefault("log", {})["Metrics/policy_angular_momentum_mean"] = torch.mean(torch.sqrt(sq))
  return sq


def track_yaw_rate(env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
  """Yaw-rate-only tracking reward exp(-(cmd_wz - wz)^2 / std^2) (legged_gym style).

  mjlab's track_angular_velocity also counts roll/pitch rate as error; once the std is tightened (0.35) the rocking from
  stepping costs more than a slow turn earns, so the policy stands still for |wz| < ~0.8 rad/s. Roll/pitch rate stays
  penalised by the separate body_ang_vel term."""
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  return torch.exp(-torch.square(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2]) / std**2)


def stand_still(env, command_name: str, sensor_name: str, command_threshold: float = 0.05, foot_weight: float = 0.5,
                asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
  """J1k: penalty while told to stand (|v| + |w| command < threshold): body drift |v_xy| + 0.5 |w_z| plus foot_weight
  per foot off the ground. Tracking's exp kernel (std 0.25) barely notices 5 cm/s of drift or a shuffle step; in the
  kitchen the robot then wobbles and steps around in place while waiting."""
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  standing = (torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])) < command_threshold
  drift = torch.norm(asset.data.root_link_lin_vel_b[:, :2], dim=1) + 0.5 * torch.abs(asset.data.root_link_ang_vel_b[:, 2])
  in_air = (env.scene[sensor_name].data.current_air_time > 0).float().sum(dim=1)
  cost = (drift + foot_weight * in_air) * standing.float()
  log = env.extras.setdefault("log", {})
  n = torch.clamp(standing.float().sum(), min=1)
  log["Metrics/stand_drift"] = (drift * standing.float()).sum() / n
  log["Metrics/stand_feet_in_air"] = (in_air * standing.float()).sum() / n
  return cost
