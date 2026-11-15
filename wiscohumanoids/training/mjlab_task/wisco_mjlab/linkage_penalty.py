"""Ankle linkage: per-motor torque from joint torques, as a reward penalty and a metric (torch).

Each ankle's pitch and roll are driven together by two X4-36 motors through rods
(HARDWARE_ISSUES H3). The simulation treats pitch and roll as separate joints with their own
torque caps, which would let the policy use pitch + roll torque combinations the real ankle
can't produce. This converts joint torques to motor torques,
    tau_motor = solve(J(q)^T, tau_joint),  J = d(crank angle)/d(joint angle)
with J bilinearly interpolated from the grid in model/ankle_linkage.json (made by
model/linkage_tools.py from robot/linkage.json). Same maths as model/ankle_motor.py (numpy).

Penalty per step: sum over the 4 ankle motors of relu(|tau_motor| / limit - 1)^2
(zero while every motor is within its limit).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

LINKAGE_JSON = Path(__file__).resolve().parents[2] / "model" / "ankle_linkage.json"
SIDES = ("left", "right")


class JacobianGrid:
  """Ankle linkage Jacobian J(pitch, roll) (rows = motors, cols = joints), bilinear on the grid. Device-only."""

  def __init__(self, device):
    info = json.loads(LINKAGE_JSON.read_text())
    self.pitch = torch.tensor(info["grid_pitch"], device=device, dtype=torch.float32)
    self.roll = torch.tensor(info["grid_roll"], device=device, dtype=torch.float32)
    self.grid = {s: torch.tensor(info["sides"][s]["grid"], device=device, dtype=torch.float32) for s in SIDES}

  def jacobian(self, s: str, p: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    P, R, g = self.pitch, self.roll, self.grid[s]
    p = p.clamp(P[0], P[-1]); r = r.clamp(R[0], R[-1])
    i = (torch.searchsorted(P, p.contiguous()) - 1).clamp(0, len(P) - 2)
    k = (torch.searchsorted(R, r.contiguous()) - 1).clamp(0, len(R) - 2)
    tp = ((p - P[i]) / (P[i + 1] - P[i]))[:, None, None]
    tr = ((r - R[k]) / (R[k + 1] - R[k]))[:, None, None]
    return ((1 - tp) * (1 - tr) * g[i, k] + tp * (1 - tr) * g[i + 1, k]
            + (1 - tp) * tr * g[i, k + 1] + tp * tr * g[i + 1, k + 1])


def joint_to_motor(J: torch.Tensor, tau_j: torch.Tensor) -> torch.Tensor:
  """tau_m = solve(J^T, tau_j) for (N, 2, 2) J and (N, 2) joint torques (closed-form 2x2)."""
  a, b, c, d = J[:, 0, 0], J[:, 1, 0], J[:, 0, 1], J[:, 1, 1]    # J^T = [[a, b], [c, d]]
  det = a * d - b * c
  return torch.stack(((d * tau_j[:, 0] - b * tau_j[:, 1]) / det, (-c * tau_j[:, 0] + a * tau_j[:, 1]) / det), dim=1)


def motor_to_joint(J: torch.Tensor, tau_m: torch.Tensor) -> torch.Tensor:
  """tau_j = J^T tau_m."""
  return torch.einsum("nij,ni->nj", J, tau_m)


class AnkleMotorModel:
  """Holds the Jacobian grids on the GPU and the joint/actuator indices. One per env."""

  def __init__(self, env: ManagerBasedRlEnv, asset_name: str = "robot"):
    g = JacobianGrid(env.device)
    self.pitch, self.roll, self.grid, self._g = g.pitch, g.roll, g.grid, g
    asset = env.scene[asset_name]
    self.asset = asset
    self._ankle_act = next((a for a in getattr(asset, "actuators", ()) if type(a).__name__ == "AnkleLinkageActuator"), None)
    self.joint_ids, self.act_ids = {}, {}
    for s in SIDES:
      names = [f"{s}_ankle_pitch", f"{s}_ankle_roll"]
      self.joint_ids[s] = asset.find_joints(names, preserve_order=True)[0]
      self.act_ids[s] = asset.find_actuators(names, preserve_order=True)[0]

  def _jacobian(self, s: str, p: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    return self._g.jacobian(s, p, r)

  def motor_torques(self) -> torch.Tensor:
    """(num_envs, 4): [left motor A, left motor B, right motor A, right motor B] in N*m.

    With the linkage ankle actuator (fix 4) this is the actuator's own clamped motor torque from the last physics
    step (exact). Otherwise the applied joint torques are mapped back through J at the current joint angles (after
    the step, so it can read a few N*m off when the ankle moves fast)."""
    if self._ankle_act is not None:
      return self._ankle_act.last_motor_torque
    q = self.asset.data.joint_pos
    tau = self.asset.data.actuator_force
    out = []
    for s in SIDES:
      jp, jr = self.joint_ids[s]
      J = self._jacobian(s, q[:, jp], q[:, jr])                      # (N, 2, 2)
      t = tau[:, self.act_ids[s]]                                    # (N, 2) joint torques
      x = joint_to_motor(J, t)
      out += [x[:, 0], x[:, 1]]
    return torch.stack(out, dim=1)


def _model(env: ManagerBasedRlEnv) -> AnkleMotorModel:
  m = getattr(env, "_wisco_ankle_model", None)
  if m is None:
    m = AnkleMotorModel(env)
    env._wisco_ankle_model = m
  return m


def ankle_motor_overload(env: ManagerBasedRlEnv, motor_limit: float) -> torch.Tensor:
  """Reward term: sum of squared relative overload of the 4 ankle motors (0 when within limit)."""
  tau = _model(env).motor_torques()
  return torch.sum(torch.relu(tau.abs() / motor_limit - 1.0) ** 2, dim=1)


def ankle_motor_peak(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Metric: largest ankle motor torque this step (N*m)."""
  return _model(env).motor_torques().abs().max(dim=1).values


class ankle_motor_thermal:
  """Reward term (class): sustained ankle motor torque above rated (fix 4, HARDWARE_ISSUES H3/H13).

  Per motor, exponential moving average of tau^2 with time constant tau_s (default 2 s); penalty
  sum over the 4 motors of relu(sqrt(ema) / rated - 1)^2. Resets per episode. Also logs metrics.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self.ema = torch.zeros(env.num_envs, 4, device=env.device)
    self.env = env

  def reset(self, env_ids=None):
    if env_ids is None:
      env_ids = slice(None)
    self.ema[env_ids] = 0.0

  def __call__(self, env: ManagerBasedRlEnv, rated_nm: float = 10.5, tau_s: float = 2.0) -> torch.Tensor:
    tau = _model(env).motor_torques()
    alpha = min(1.0, env.step_dt / tau_s)
    self.ema += alpha * (tau * tau - self.ema)
    over = torch.relu(torch.sqrt(self.ema) / rated_nm - 1.0)
    log = env.extras.setdefault("log", {})
    log["Metrics/ankle_motor_rms_nm"] = torch.sqrt(torch.mean(tau * tau))
    log["Metrics/ankle_motor_frac_above_rated"] = torch.mean((tau.abs() > rated_nm).float())
    log["Metrics/ankle_motor_ema_rms_nm"] = torch.mean(torch.sqrt(self.ema))
    return torch.sum(over * over, dim=1)


def ankle_motor_frac_above_rated(env: ManagerBasedRlEnv, rated_nm: float = 10.5) -> torch.Tensor:
  """Metric: fraction of the 4 ankle motors above rated torque this step (episode mean = % of time)."""
  return (_model(env).motor_torques().abs() > rated_nm).float().mean(dim=1)


def ankle_motor_rms(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Metric: RMS over the 4 ankle motors this step (N*m)."""
  t = _model(env).motor_torques()
  return torch.sqrt(torch.mean(t * t, dim=1))
