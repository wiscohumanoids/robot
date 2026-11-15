"""Ankle actuator with per-MOTOR limits through the rod linkage (fix 4, HARDWARE_ISSUES H3).

Each ankle (pitch + roll) is driven by two X4-36 motors through rods. Joint-level torque caps (pitch 86.6 /
roll 35.2 N*m) let the policy combine pitch and roll torque that needs one motor above its 34 N*m peak (J3d:
2.7 % of robot-steps above peak, max 76.8 N*m). This actuator does what the real drivers do, every physics step:

  1. joint-space PD (our kp/kd per joint, with mjlab's gain randomization):  tau_j = kp (q* - q) - kd qdot
  2. motor torques through the linkage:  tau_m = solve(J^T, tau_j),  motor speeds  w_m = J qdot
  3. clamp each motor to its DC-motor envelope: +/- peak, and the back-EMF torque-speed line
     (mjlab dc_motor_clip with the X4-36 values: saturation = slope * w_noload, velocity_limit = w_noload)
  4. back to the joints:  tau_j = J^T tau_m
Commands inside the motor envelope pass through unchanged. J = d(motor angle)/d(joint angle) from the
ankle_linkage.json grid (same as the overload penalty).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from mjlab.actuator.actuator import ActuatorCmd
from mjlab.actuator.dc_actuator import DcMotorActuator, DcMotorActuatorCfg, dc_motor_clip
from mjlab.actuator.pd_actuator import pd_torque

from .linkage_penalty import JacobianGrid, joint_to_motor, motor_to_joint

SIDES = ("left", "right")


def clamp_ankle_pair(J: torch.Tensor, tau_j: torch.Tensor, qd: torch.Tensor, peak: float, sat: float,
                     vlim: float) -> tuple[torch.Tensor, torch.Tensor]:
  """Clamp one ankle's (pitch, roll) joint torques to the two motors' envelopes. Returns (tau_j, tau_m)."""
  tau_m = joint_to_motor(J, tau_j)
  w_m = torch.einsum("nij,nj->ni", J, qd)
  pk = torch.full_like(tau_m, peak)
  tau_m = dc_motor_clip(tau_m, torch.full_like(tau_m, sat), torch.full_like(tau_m, vlim), pk, w_m)
  return motor_to_joint(J, tau_m), tau_m


@dataclass(kw_only=True)
class AnkleLinkageActuatorCfg(DcMotorActuatorCfg):
  """One actuator for the 4 ankle joints. stiffness/damping/saturation_effort/velocity_limit of the base class
  are placeholders; per-joint gains come from joint_kp / joint_kd, motor limits from motor_*."""
  joint_kp: dict[str, float] = field(default_factory=dict)
  joint_kd: dict[str, float] = field(default_factory=dict)
  motor_peak: float = 34.0
  motor_saturation: float = 117.0
  motor_velocity_limit: float = 11.62

  def build(self, entity, target_ids, target_names) -> "AnkleLinkageActuator":
    return AnkleLinkageActuator(self, entity, target_ids, target_names)


class AnkleLinkageActuator(DcMotorActuator):
  cfg: AnkleLinkageActuatorCfg

  def initialize(self, mj_model, model, data, device) -> None:
    super().initialize(mj_model, model, data, device)
    names = list(self._target_names)
    for j, n in enumerate(names):
      self.stiffness[:, j] = float(self.cfg.joint_kp[n])
      self.damping[:, j] = float(self.cfg.joint_kd[n])
    self.default_stiffness = self.stiffness.clone()
    self.default_damping = self.damping.clone()
    self._grid = JacobianGrid(device)
    self._pairs = {s: (names.index(f"{s}_ankle_pitch"), names.index(f"{s}_ankle_roll")) for s in SIDES}
    self.last_motor_torque = torch.zeros(data.nworld, 4, device=device)

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    tau = pd_torque(self.stiffness, self.damping, cmd)
    out = tau.clone()
    for k, s in enumerate(SIDES):
      ip, ir = self._pairs[s]
      J = self._grid.jacobian(s, cmd.pos[:, ip], cmd.pos[:, ir])
      tj, tm = clamp_ankle_pair(J, torch.stack((tau[:, ip], tau[:, ir]), 1), torch.stack((cmd.vel[:, ip], cmd.vel[:, ir]), 1),
                                self.cfg.motor_peak, self.cfg.motor_saturation, self.cfg.motor_velocity_limit)
      out[:, ip], out[:, ir] = tj[:, 0], tj[:, 1]
      self.last_motor_torque[:, 2 * k:2 * k + 2] = tm
    return out
