"""PACE replay backends: simulate logged commands with candidate actuator params.

The fit (spec stage 9) is over armature, friction (Coulomb + viscous),
encoder bias and delay per joint, with link inertials held at their stage 5
values. Drive CSP gains are fitted too when the joint config does not give
them (spec risk: CSP gains may not be readable over SDO).

Two backends:

  JointReplayBackend  built in. Each joint is replayed on its own (others held
                      during the chirp), with the link inertia about the joint
                      axis and gravity along the sweep taken from the
                      Pinocchio model. Vectorized over every candidate of every
                      joint, so one CMA-ES generation is one batched replay.
  IsaacLabBackend     replays the same arrays in Isaac Lab (full arm, one env
                      per candidate), isaac_backend.py. Needs isaaclab;
                      selected with backend='isaac'.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

PARAMS = ('armature', 'coulomb', 'viscous', 'encoder_bias', 'delay_s', 'drive_kp', 'drive_kd')


@dataclass
class JointLog:
    """One joint's replay data: command and measured position at fixed rate."""
    joint: int
    dt: float
    q_cmd: np.ndarray          # (T,)
    qd_cmd: np.ndarray
    q_meas: np.ndarray
    q_hold: np.ndarray         # full-arm pose the other joints held
    sweep: tuple = (0.0, 0.0)  # (center, amplitude) for the PACE recorder
    freq: tuple = (0.0, 0.0)


class JointReplayBackend:
    """Per-joint replay. With `hold_gains` (the drives' CSP kp, kd for every joint)
    the other joints are simulated too, held by their drive loops, through the arm's
    mass matrix at the hold pose: the chirped joint drags them (and they it), which
    a one-joint model cannot show and which otherwise ends up in a biased armature
    and viscous fit. The chirp is small, so inertia is constant and gravity is the
    chirped joint's exact curve plus a linear term in the others' deflection.

    `stiffness` (Nm/rad per joint, from plate / mocap-calib; inf = rigid) adds the
    transmission's compliance as the plant has it: the encoder, and so the drive's
    PD and the log, read the motor angle, link angle + tau / k (tau from the last
    substep). Without it a compliant shoulder's softer loop ends up in the fitted
    armature and viscous.

    `torque_limit` (Nm per joint) is the drive's 0x6072 clamp on its total torque.
    A chirp near the servo resonance can hit it, and an unclipped replay then
    overshoots exactly where the log is most informative."""

    def __init__(self, model, logs: List[JointLog], gravity_ff: bool = True, substeps: int = 1,
                 hold_gains=None, hold_armature=None, stiffness=None, torque_limit=None):
        self.model = model
        self.logs = logs
        self.gravity_ff = gravity_ff
        self.substeps = substeps
        self.coupled = hold_gains is not None
        n = len(logs[0].q_hold) if logs else 0
        self.kp_hold = None if hold_gains is None else np.asarray(hold_gains[0], float)
        self.kd_hold = None if hold_gains is None else np.asarray(hold_gains[1], float)
        arm0 = np.zeros(n) if hold_armature is None else np.asarray(hold_armature, float)
        k = np.full(n, np.inf) if stiffness is None else np.asarray(stiffness, float)
        self.compliance = np.where(np.isfinite(k) & (k > 0), 1.0 / np.where(k > 0, k, 1.0), 0.0)
        self.tau_max = np.full(n, np.inf) if torque_limit is None else np.asarray(torque_limit, float)
        self.inertia = []
        self.g_grid = []
        self.M0 = []          # mass matrix at the hold pose, others' armature on the diagonal
        self.G = []           # d gravity / d q at the hold pose
        self.sub_k = []       # substeps per log (coupled)
        self.g_vec_grid = []  # full gravity vector along the chirped joint's sweep
        for log in logs:
            j = log.joint
            M = model.mass_matrix(log.q_hold)
            self.inertia.append(float(M[j, j]))
            lo = min(log.q_meas.min(), log.q_cmd.min()) - 0.2
            hi = max(log.q_meas.max(), log.q_cmd.max()) + 0.2
            grid = np.linspace(lo, hi, 121)
            gv = []
            for v in grid:
                q = log.q_hold.copy()
                q[j] = v
                gv.append(model.gravity(q))
            gv = np.array(gv)
            self.g_grid.append((grid, gv[:, j]))
            self.g_vec_grid.append(gv)
            if self.coupled:
                Ma = M + np.diag(np.where(np.arange(n) == j, 0.0, arm0))
                self.M0.append(Ma)
                h = 1e-5
                G = np.zeros((n, n))
                for k in range(n):
                    qp, qm = log.q_hold.copy(), log.q_hold.copy()
                    qp[k] += h
                    qm[k] -= h
                    G[:, k] = (model.gravity(qp) - model.gravity(qm)) / (2 * h)
                self.G.append(G)
                # explicit stepping: keep the stiffest and the most damped mode well
                # inside stability (the chirped joint at zero candidate armature is
                # the worst case)
                w = float(np.sqrt(np.max(np.abs(np.linalg.eigvals(np.linalg.solve(Ma, np.diag(self.kp_hold)))))))
                c = float(np.max(np.abs(np.linalg.eigvals(np.linalg.solve(Ma, np.diag(self.kd_hold))))))
                self.sub_k.append(max(substeps, int(np.ceil(max(w, c) * log.dt / 0.5))))
        self.friction_ff = [(model.coulomb[l.joint], model.viscous[l.joint]) for l in logs]
        self.fric_all = (np.asarray(model.coulomb, float), np.asarray(model.viscous, float))

    def simulate(self, k: int, P: np.ndarray) -> np.ndarray:
        """Replay log k for a population P (pop x 7, PARAMS order); returns q (pop x T)."""
        if self.coupled:
            return self._simulate_coupled(k, P)
        log = self.logs[k]
        arm, fc, fv, bias, delay, kp, kd = [P[:, i] for i in range(7)]
        pop = len(P)
        T = len(log.q_cmd)
        dt = log.dt
        grid, g = self.g_grid[k]
        I = self.inertia[k] + arm
        fc0, fv0 = self.friction_ff[k]
        # fractional delay: interpolate the command history
        tgrid = np.arange(T) * dt
        c = self.compliance[log.joint] if len(self.compliance) else 0.0
        # start in static equilibrium: the drive holds gravity, the link sits tau / k
        # below the motor angle the log starts at
        tau = np.full(pop, float(np.interp(log.q_meas[0], grid, g)))
        q = log.q_meas[0] - c * tau              # true link angle
        qd = np.zeros(pop)
        out = np.empty((pop, T))
        h = dt / self.substeps
        for t in range(T):
            ts = tgrid[t] - delay
            qc = np.interp(ts, tgrid, log.q_cmd)
            qdc = np.interp(ts, tgrid, log.qd_cmd)
            g_true = np.interp(q, grid, g)
            # the host computed the feedforward from what it measured when it sent the
            # frame, and the frame arrives `delay` later
            q_sent = np.interp(ts, tgrid, log.q_meas)
            ff = np.interp(q_sent, grid, g) if self.gravity_ff else 0.0
            ff = ff + fc0 * np.tanh(qdc / 0.02) + fv0 * qdc
            for _ in range(self.substeps):
                tau = kp * (qc - (q + c * tau + bias)) - kd * qd + ff
                if len(self.tau_max):
                    tau = np.clip(tau, -self.tau_max[log.joint], self.tau_max[log.joint])
                fr = fc * np.tanh(qd / 0.01) + fv * qd
                qdd = (tau - g_true - fr) / I
                qd = qd + h * qdd
                q = q + h * qd
            out[:, t] = q + c * tau + bias
        return out


    def _simulate_coupled(self, k: int, P: np.ndarray) -> np.ndarray:
        log = self.logs[k]
        j = log.joint
        arm, fc, fv, bias, delay, kp, kd = [P[:, i] for i in range(7)]
        pop = len(P)
        T = len(log.q_cmd)
        dt = log.dt
        grid, gj = self.g_grid[k]
        gv = self.g_vec_grid[k]
        n = gv.shape[1]
        hold = log.q_hold
        others = np.arange(n) != j
        M0 = self.M0[k]
        Minv0 = np.linalg.inv(M0)
        G = self.G[k]
        # (M0 + a e_j e_j^T)^-1 per candidate, Sherman-Morrison
        u = Minv0[:, j]
        denom = 1.0 + arm * Minv0[j, j]
        Minv = Minv0[None] - (arm / denom)[:, None, None] * np.einsum('i,k->ik', u, u)[None]
        fc0, fv0 = self.friction_ff[k]
        Fc, Fv = self.fric_all
        kp_h, kd_h = self.kp_hold, self.kd_hold
        tgrid = np.arange(T) * dt
        q = np.repeat(hold[None], pop, axis=0)
        q[:, j] = log.q_meas[0]
        qd = np.zeros((pop, n))
        cmp = self.compliance
        tau = np.repeat(np.array([np.interp(log.q_meas[0], grid, gv[:, i]) for i in range(n)])[None], pop, axis=0)
        q = q - cmp[None] * tau                  # link angles; hold and the log are motor angles
        out = np.empty((pop, T))
        sub = self.sub_k[k]
        h = dt / sub
        gcols = [gv[:, c] for c in range(n)]
        for t in range(T):
            ts = tgrid[t] - delay
            qc = np.interp(ts, tgrid, log.q_cmd)
            qdc = np.interp(ts, tgrid, log.qd_cmd)
            q_sent = np.interp(ts, tgrid, log.q_meas)
            # feedforward the host sent: gravity at the measured pose (others ~ at hold)
            if self.gravity_ff:
                ff = np.stack([np.interp(q_sent, grid, gcols[c]) for c in range(n)], axis=1)
            else:
                ff = np.zeros((pop, n))
            ff[:, j] += fc0 * np.tanh(qdc / 0.02) + fv0 * qdc
            for _ in range(sub):
                g_true = np.stack([np.interp(q[:, j], grid, gcols[c]) for c in range(n)], axis=1)
                dqo = np.where(others[None], q - hold[None], 0.0)
                g_true = g_true + dqo @ G.T
                enc = q + cmp[None] * tau          # motor angle (last substep's torque)
                tau = ff.copy()
                tau[:, j] += kp * (qc - (enc[:, j] + bias)) - kd * qd[:, j]
                tau[:, others] += kp_h[others] * (hold[others] - enc[:, others]) - kd_h[others] * qd[:, others]
                tau = np.clip(tau, -self.tau_max[None], self.tau_max[None])
                fr = np.empty_like(qd)
                fr[:, others] = Fc[others] * np.tanh(qd[:, others] / 0.01) + Fv[others] * qd[:, others]
                fr[:, j] = fc * np.tanh(qd[:, j] / 0.01) + fv * qd[:, j]
                qdd = np.einsum('pik,pk->pi', Minv, tau - g_true - fr)
                qd = qd + h * qdd
                q = q + h * qd
            out[:, t] = q[:, j] + cmp[j] * tau[:, j] + bias
        return out


class IsaacLabBackend:
    """The Isaac Lab replay (full arm, same parameters): arm_stack/pace/isaac_backend.py.

    Imported lazily so the stack runs without Isaac Lab; constructing it without
    an importable `isaaclab` raises an ImportError saying how to get it.
    """

    def __new__(cls, *args, **kwargs):
        from .isaac_backend import IsaacLabBackend as _Impl
        return _Impl(*args, **kwargs)


def default_bounds(spec, drive_known: Optional[Dict] = None):
    """Search box per joint, PARAMS order."""
    peak = spec.peak_torque_nm
    kp0 = (drive_known or {}).get('kp')
    kd0 = (drive_known or {}).get('kd')
    lo = [0.0, 0.0, 0.0, -0.02, 0.0, kp0 or 5.0, kd0 or 0.05]
    hi = [0.2, 0.08 * peak, 0.05 * peak, 0.02, 0.012, kp0 or 2000.0, kd0 or 40.0]
    return np.array(lo), np.array(hi)
