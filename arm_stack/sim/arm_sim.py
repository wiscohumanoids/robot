"""Sim backend behind the ArmClient API (spec: Phase 0).

The plant is the URDF with known, injectable errors, so every procedure can be
checked for recovering them:

  zero offsets (encoder bias)       direction (wiring) flips
  link mass/COM errors              kinematic errors (joint origin/axis)
  Kt error per joint                Coulomb, viscous and Stribeck stiction
  command delay (cycles)            backlash and joint compliance
  hard stops outside the URDF limits    thermal zero drift with time on

The RT side (what the controller believes) is the nominal URDF plus the
current calibration file, run through the same ControlLaw as the C++
controller. The drive's own CSP loop is a PD on the motor-side angle.
Everything is synchronous and deterministic for a given seed.
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pinocchio as pin

from ..config import ArmConfig
from ..control.law import ControlLaw, WATCHDOG_S
from ..hw.client import ArmClient, Command, JointState, Payload
from ..model.robot import ArmModel
from ..units import Units


@dataclass
class SimErrors:
    """Truth the procedures must recover. All per joint in config order."""
    zero_offsets_rad: Optional[np.ndarray] = None
    direction_flip: Optional[np.ndarray] = None       # bool
    kt_scale: Optional[np.ndarray] = None             # true Kt / nominal Kt
    com_offsets_m: Optional[np.ndarray] = None        # (n, 3) added to URDF COM
    mass_scale: Optional[np.ndarray] = None           # true mass / URDF mass
    coulomb_nm: Optional[np.ndarray] = None
    viscous_nm_s: Optional[np.ndarray] = None
    stiction_extra_nm: Optional[np.ndarray] = None    # breakaway above Coulomb
    delay_cycles: int = 0
    backlash_rad: Optional[np.ndarray] = None
    stiffness_nm_per_rad: Optional[np.ndarray] = None
    kinematic: Optional[Dict[str, Dict]] = None       # overlay applied to the truth
    hard_stop_margin_rad: Optional[np.ndarray] = None  # true stop beyond URDF limit
    thermal_drift_rad: Optional[np.ndarray] = None    # zero drift after warm-up
    thermal_tau_s: float = 600.0
    armature: Optional[np.ndarray] = None
    jig_reseat_sigma_rad: float = math.radians(0.01)
    current_noise_permille: float = 2.0                # 0x6077 noise, 1 sigma
    mocap_sigma_m: float = 3e-5
    # true counts per rad / configured (a wrong counts_per_rev, e.g. motor-side
    # encoder read as output-side); `arm indicator` must catch it
    counts_scale: Optional[np.ndarray] = None

    @staticmethod
    def random(n: int, seed: int = 0, scale: float = 1.0) -> 'SimErrors':
        r = np.random.default_rng(seed)
        return SimErrors(
            zero_offsets_rad=r.uniform(-1, 1, n) * math.radians(5) * scale,
            kt_scale=1 + r.uniform(-0.1, 0.1, n) * scale,
            com_offsets_m=r.uniform(-1, 1, (n, 3)) * 0.008 * scale,
            mass_scale=np.ones(n),
            coulomb_nm=r.uniform(0.05, 0.25, n) * np.array([4, 4, 2, 2, 1.5, 0.8, 0.8])[:n],
            viscous_nm_s=r.uniform(0.02, 0.1, n) * np.array([3, 3, 2, 2, 1, 0.5, 0.5])[:n],
            stiction_extra_nm=r.uniform(0.0, 0.05, n),
            delay_cycles=2,
            backlash_rad=np.full(n, math.radians(0.05)) * scale,
            stiffness_nm_per_rad=np.array([4000, 4000, 1500, 1500, 1500, 600, 600][:n], float),
            kinematic={},
            hard_stop_margin_rad=np.full(n, math.radians(2.0)),
            thermal_drift_rad=r.uniform(-1, 1, n) * math.radians(0.01) * scale,
        )


DRIVE_KP = {'x6_60': 300.0, 'x4_36': 150.0, 'x4_10': 60.0}
DRIVE_KD = {'x6_60': 4.0, 'x4_36': 1.5, 'x4_10': 0.5}
ARMATURE = {'x6_60': 0.03, 'x4_36': 0.012, 'x4_10': 0.004}  # kg m^2 at the output


class SimArm(ArmClient):

    def __init__(self, arm: ArmConfig, calibration: Dict, errors: Optional[SimErrors] = None,
                 dt: float = 0.001, substeps: int = 2, q0: Optional[np.ndarray] = None,
                 seed: int = 0, mocap_enabled: bool = True):
        self.arm = arm
        self.names = arm.names
        self.n = arm.n
        self.dt = dt
        self.cmd_dt = dt
        self.substeps = substeps
        self.rng = np.random.default_rng(seed)
        e = errors or SimErrors()
        self.err = e
        n = self.n
        z = np.zeros(n)
        self.true_zero = np.array(e.zero_offsets_rad if e.zero_offsets_rad is not None else z, float)
        self.dir_cfg = np.array([j.direction for j in arm.joints], float)
        flips = np.array(e.direction_flip if e.direction_flip is not None else [False] * n)
        self.dir_true = np.where(flips, -self.dir_cfg, self.dir_cfg)
        self.kt_scale = np.array(e.kt_scale if e.kt_scale is not None else np.ones(n), float)
        self.counts_scale = np.array(e.counts_scale if e.counts_scale is not None else np.ones(n), float)
        self.coulomb = np.array(e.coulomb_nm if e.coulomb_nm is not None else z, float)
        self.viscous = np.array(e.viscous_nm_s if e.viscous_nm_s is not None else z, float)
        self.stiction = np.array(e.stiction_extra_nm if e.stiction_extra_nm is not None else z, float)
        self.backlash = np.array(e.backlash_rad if e.backlash_rad is not None else z, float)
        self.stiffness = np.array(e.stiffness_nm_per_rad if e.stiffness_nm_per_rad is not None
                                  else np.full(n, np.inf), float)
        self.thermal = np.array(e.thermal_drift_rad if e.thermal_drift_rad is not None else z, float)
        self.armature = np.array(e.armature if e.armature is not None
                                 else [ARMATURE.get(j.model, 0.01) for j in arm.joints], float)
        self.drive_kp = np.array([DRIVE_KP.get(j.model, 100.0) for j in arm.joints])
        self.drive_kd = np.array([DRIVE_KD.get(j.model, 1.0) for j in arm.joints])
        self.drive_kp0 = self.drive_kp.copy()
        self.drive_kd0 = self.drive_kd.copy()

        # truth plant
        self.truth = ArmModel(arm.urdf, arm.names, home=arm.home)
        if e.kinematic:
            self.truth.apply_overlay(e.kinematic)
        self._urdf_inertias = [self.truth.model.inertias[j].copy() for j in self.truth.jid]
        ms = np.array(e.mass_scale if e.mass_scale is not None else np.ones(n), float)
        co = np.array(e.com_offsets_m if e.com_offsets_m is not None else np.zeros((n, 3)), float)
        for k, j in enumerate(self.truth.jid):
            I = self._urdf_inertias[k]
            self.truth.model.inertias[j] = pin.Inertia(I.mass * ms[k], I.lever + co[k], I.inertia * ms[k])
        self._truth_inertias = [self.truth.model.inertias[j].copy() for j in self.truth.jid]
        self.truth.data = self.truth.model.createData()
        margin = np.array(e.hard_stop_margin_rad if e.hard_stop_margin_rad is not None
                          else np.full(n, math.radians(2)), float)
        self.stop_lower = self.truth.lower - margin
        self.stop_upper = self.truth.upper + margin
        self.payload: Optional[Payload] = None

        # plant state (link side)
        # power-on pose: the arm's home (hanging), or q = 0 for a config without one
        start = self.truth.home if self.truth.home is not None else np.zeros(n)
        self.q = np.array(q0 if q0 is not None else np.clip(start, self.truth.lower + math.radians(3),
                                                             self.truth.upper - math.radians(3)), float)
        self.qd = np.zeros(n)
        self.tau_drive = np.zeros(n)     # motor torque actually produced
        self.tau_cycle = np.zeros(n)     # its mean over the last cycle: what 0x6077 reports
        self.deflection = np.zeros(n)    # motor angle - link angle
        self.tau_ext = np.zeros(n)       # operator / environment torque on the links
        self.t = 0.0
        self.power_on_t = 0.0

        # RT side
        self.cmd: Optional[Command] = None
        self.cmd_t = -1.0
        self.queue = collections.deque([None] * max(e.delay_cycles, 0))
        self.mode = 'P'
        self.events: List[Dict] = []
        self.mocap_enabled = mocap_enabled
        self.reload_calibration(calibration)
        self._last_out = None
        self._q_target_prev = None

    # ------------------------------------------------------------------ RT side
    def reload_calibration(self, calibration: Dict):
        self.calibration = calibration
        self.units = Units(self.arm, calibration)
        self.model = ArmModel(self.arm.urdf, self.arm.names, home=self.arm.home)
        self.model.apply_calibration(calibration, self.arm.link_masses_kg)
        soft = calibration.get('soft_limits_rad') or {}
        lower = self.model.lower.copy()
        upper = self.model.upper.copy()
        for i, nme in enumerate(self.names):
            lo, hi = (soft.get(nme) or [None, None])
            if lo is not None:
                lower[i] = lo
            if hi is not None:
                upper[i] = hi
        self.law = ControlLaw(self.model, [j.max_torque_nm for j in self.arm.joints],
                              [j.max_vel_rad_s for j in self.arm.joints], lower, upper,
                              kd_fallback=[max(j.kd, 0.2) for j in self.arm.joints])

    # ------------------------------------------------------------------ sensors
    def _encoder_counts(self) -> np.ndarray:
        drift = self.thermal * (1 - math.exp(-(self.t - self.power_on_t) / self.err.thermal_tau_s))
        motor = self.q + self.deflection
        return np.round(self.dir_true * (motor + self.true_zero + drift) * self.units.counts_per_rad
                        * self.counts_scale)

    def _measured(self):
        counts = self._encoder_counts()
        # what the host computes with its configured direction and zero offsets
        q = self.dir_cfg * counts / self.units.counts_per_rad - self.units.zero
        qd = self.dir_cfg * self.dir_true * self.qd
        # the drive's current reading is filtered over its PWM period, so it reports the
        # cycle mean, not the last substep (substep-to-substep chatter would bias it)
        current = self.tau_cycle / (self.kt_scale * self.units.kt)    # A (true Kt)
        noise = self.rng.normal(0, self.err.current_noise_permille, self.n) if self.err.current_noise_permille else 0.0
        permille = np.round(self.dir_true * current / self.units.rated_a * 1000.0 + noise)
        tau = self.dir_cfg * permille * self.units.nm_per_permille()
        return counts, q, qd, permille, tau

    def state(self) -> JointState:
        counts, q, qd, permille, tau = self._measured()
        return JointState(t=self.t, q=q, qd=qd, tau=tau, raw_counts=counts, raw_permille=permille,
                          status=np.full(self.n, 0x1237), mode=self.mode,
                          fallback=bool(self._last_out and self._last_out.fallback),
                          tau_cmd=None if self._last_out is None else self._last_out.tau.copy())

    # ------------------------------------------------------------------ commands
    def send(self, cmd: Command) -> None:
        if cmd.mode != self.mode:
            self.events.append({'t': self.t, 'event': 'mode_switch', 'from': self.mode, 'to': cmd.mode})
        self.cmd = cmd
        self.cmd_t = self.t

    def step(self, seconds: float = 0.0) -> JointState:
        cycles = max(1, int(round(seconds / self.dt)))
        for _ in range(cycles):
            self._cycle()
        return self.state()

    def _cycle(self):
        _, q_meas, qd_meas, _, _ = self._measured()
        age = self.t - self.cmd_t if self.cmd is not None else 1e9
        out = self.law.compute(self.cmd, age, q_meas, qd_meas, self.mode)
        if out.fallback and not (self._last_out and self._last_out.fallback):
            self.events.append({'t': self.t, 'event': 'watchdog_fallback'})
        if out.limit_hit:
            if not (self._last_out and self._last_out.limit_hit):
                self.events.append({'t': self.t, 'event': 'limit_hit'})
        # bumpless CSP, as in ArmController::update: the target moves at most
        # max_velocity per cycle and restarts from the actual position on entry
        if out.mode == 'P':
            if self.mode != 'P' or self._q_target_prev is None:
                self._q_target_prev = q_meas.copy()
            step = self.law.max_vel * self.dt
            self._q_target_prev = self._q_target_prev + np.clip(out.q_target - self._q_target_prev,
                                                                -step, step)
            out.q_target = self._q_target_prev.copy()
        else:
            self._q_target_prev = None
        self.mode = out.mode
        self._last_out = out
        # drive-side values (sent over the bus, then delayed)
        target_counts = self.units.rad_to_counts(out.q_target)
        tau_permille = self.units.nm_to_permille(out.tau)
        # 0x6072 follows the command's clamp; the fallback uses the configured max
        clamp = self.law.max_torque
        if self.cmd is not None and self.cmd.torque_clamp is not None and not out.fallback:
            clamp = np.minimum(clamp, self.cmd.torque_clamp)
        self.queue.append((out.mode, target_counts, tau_permille, clamp))
        applied = self.queue.popleft()
        if applied is None:
            applied = ('P', self._encoder_counts(), np.zeros(self.n), self.law.max_torque)
        mode, counts_t, permille_t, clamp_t = applied
        # what the drive does with raw values (true direction, true Kt)
        tau_cmd = self.dir_true * permille_t * self.units.rated_a / 1000.0 * self.kt_scale * self.units.kt
        h = self.dt / self.substeps
        acc = np.zeros(self.n)
        for _ in range(self.substeps):
            if mode == 'P':
                # drive PD on its own encoder (true direction, includes the bias)
                enc = self._encoder_counts()
                err_counts = self.dir_true * (counts_t - enc)
                err = err_counts / (self.units.counts_per_rad * self.counts_scale)
                motor_vel = self.qd
                tau = self.drive_kp * err - self.drive_kd * motor_vel + tau_cmd
            else:
                tau = tau_cmd
            # 0x6072: the drive clamps its total torque
            tau = np.clip(tau, -clamp_t, clamp_t)
            self.tau_drive = tau
            acc += tau
            self._integrate(tau, h)
            self.t += h
        self.tau_cycle = acc / self.substeps

    def _friction(self, qd, tau_other):
        """The friction law as a smooth function (reference; _integrate applies the
        Coulomb + stiction part as a bounded impulse instead)."""
        eps = 1e-3
        stribeck = self.coulomb + self.stiction * np.exp(-(qd / 0.02) ** 2)
        return stribeck * np.tanh(qd / eps) + self.viscous * qd

    def _integrate(self, tau_motor, h):
        q, qd = self.q, self.qd
        # transmission: backlash + compliance as a quasi-static deflection
        k = self.stiffness
        with np.errstate(divide='ignore', invalid='ignore'):
            elastic = np.where(np.isfinite(k), tau_motor / k, 0.0)
        self.deflection = elastic + 0.5 * self.backlash * np.tanh(tau_motor / 0.05)
        # hard stops
        tau_stop = np.zeros(self.n)
        over = q - self.stop_upper
        under = self.stop_lower - q
        tau_stop -= np.where(over > 0, 2000.0 * over + 20.0 * qd, 0.0)
        tau_stop += np.where(under > 0, 2000.0 * under - 20.0 * qd, 0.0)
        tau_net = tau_motor + tau_stop + self.tau_ext
        M = self.truth.mass_matrix(q) + np.diag(self.armature)
        bias = self.truth.rnea(q, qd, np.zeros(self.n))  # C qd + g
        # semi-implicit viscous term for stability
        Md = M + h * np.diag(self.viscous)
        Minv = np.linalg.inv(Md)
        qd_free = qd + h * (Minv @ (tau_net - bias - self.viscous * qd))
        # Coulomb + Stribeck stiction as a velocity-level impulse (time-stepping
        # friction): per joint at most h * F, never more than it takes to stop
        # the joint. Sticks while the net torque is under the breakaway, and an
        # explicit tanh friction cannot chatter a light joint into a creep.
        # The sticking set S is solved jointly through M (sliding joints take
        # the full impulse), dropping joints whose impulse would exceed h * F.
        level = self.coulomb + self.stiction * np.exp(-(qd / 0.02) ** 2)
        lim = h * level
        p = np.zeros(self.n)
        S = (lim > 0) & (np.abs(qd_free) <= lim * np.diag(Minv))
        for _ in range(4):
            p = np.where(S, 0.0, np.sign(qd_free) * lim)
            if not S.any():
                break
            F = ~S
            rhs = qd_free[S] - Minv[np.ix_(S, F)] @ p[F]
            p[S] = np.linalg.solve(Minv[np.ix_(S, S)], rhs)
            bad = S & (np.abs(p) > lim * (1 + 1e-9))
            if not bad.any():
                break
            S = S & ~bad
        p = np.clip(p, -lim, lim)
        qd_new = qd_free - Minv @ p
        self.qd = qd_new
        self.q = q + h * qd_new

    # ------------------------------------------------------------------ test-rig actions
    def set_payload(self, payload: Optional[Payload]) -> None:
        """payload.link: a link of a configured joint, or any frame (tool0, flange);
        the offset is in that link's / frame's coordinates."""
        self.payload = payload
        for k, j in enumerate(self.truth.jid):
            self.truth.model.inertias[j] = self._truth_inertias[k].copy()
        if payload is not None:
            links = self.truth.link_names()
            offset = np.array(payload.offset_m, float)
            if payload.link in links:
                j = self.truth.jid[links.index(payload.link)]
            else:
                f = self.truth.model.frames[self.truth.model.getFrameId(payload.link)]
                j = f.parentJoint
                offset = f.placement.act(offset)
            # a point-ish mass at the offset, in that link's frame
            extra = pin.Inertia(payload.mass_kg, offset, np.eye(3) * 1e-6)
            self.truth.model.inertias[j] = self.truth.model.inertias[j] + extra
        self.truth.data = self.truth.model.createData()

    def apply_external_torque(self, tau: Optional[np.ndarray]) -> None:
        """Sim-only: joint torques from outside (an operator's hands), applied to
        the links from the next cycle until changed; None or zeros releases.
        Not seen by the current sensor, as on the real drive."""
        self.tau_ext = np.zeros(self.n) if tau is None else np.array(tau, float)

    def apply_external_force(self, force: Optional[np.ndarray], frame: str = 'tool0',
                             torque: Optional[np.ndarray] = None) -> np.ndarray:
        """Sim-only: a world-frame force (N) and optional moment (Nm) at `frame`,
        mapped to joint torques through the true Jacobian at the current pose."""
        if force is None:
            self.apply_external_torque(None)
            return self.tau_ext
        w = np.concatenate([np.asarray(force, float), np.zeros(3) if torque is None else np.asarray(torque, float)])
        self.apply_external_torque(self.truth.jacobian(self.q, frame).T @ w)
        return self.tau_ext

    def place_in_jig(self, q_jig: np.ndarray) -> None:
        self.q = np.array(q_jig, float) + self.rng.normal(0, self.err.jig_reseat_sigma_rad, self.n)
        self.qd = np.zeros(self.n)
        self.tau_drive = np.zeros(self.n)
        self.tau_cycle = np.zeros(self.n)
        self.deflection = np.zeros(self.n)
        self.cmd = None
        self.queue = collections.deque([None] * max(self.err.delay_cycles, 0))

    def hand_guide(self, prompt: str, hint_q: Optional[np.ndarray] = None) -> JointState:
        """The 'operator' puts the arm at hint_q (true link angles) and lets go
        in gravity comp; the arm settles there under its own friction."""
        self.events.append({'t': self.t, 'event': 'hand_guide', 'prompt': prompt})
        if hint_q is not None:
            self.set_true_state(hint_q)
        self.send(Command(mode='T', q=None, kp=np.zeros(self.n), kd=np.zeros(self.n)))
        self.step(0.05)
        if hint_q is not None:           # the operator holds it there while confirming
            self.set_true_state(hint_q)
        return self.state()

    def drive_gains(self):
        return self.drive_kp.copy(), self.drive_kd.copy()

    def set_drive_gains(self, kp_scale: np.ndarray, kd_scale: np.ndarray) -> None:
        self.drive_kp = self.drive_kp0 * np.asarray(kp_scale, float)
        self.drive_kd = self.drive_kd0 * np.asarray(kd_scale, float)

    def set_true_state(self, q: np.ndarray) -> None:
        """Sim-only: put the arm somewhere (for hand guiding in the pivot test)."""
        self.q = np.array(q, float)
        self.qd = np.zeros(self.n)
        self.tau_drive = self.truth.gravity(self.q)
        self.tau_cycle = self.tau_drive.copy()
        k = self.stiffness
        with np.errstate(divide='ignore', invalid='ignore'):
            elastic = np.where(np.isfinite(k), self.tau_drive / k, 0.0)
        self.deflection = elastic + 0.5 * self.backlash * np.tanh(self.tau_drive / 0.05)

    def mocap(self):
        return SimMocap(self) if self.mocap_enabled else None

    def time_since_power_on(self) -> float:
        return self.t - self.power_on_t

    def true_joint_angles(self) -> np.ndarray:
        """Sim-only: true link angles (what a dial indicator / protractor reads)."""
        return self.q.copy()

    def true_tool_position(self, frame: Optional[str] = None) -> np.ndarray:
        """Where the tool really is (link angles; encoders see the motor side)."""
        return self.truth.tool_position(self.q, frame)


class SimMocap:
    """Marker cluster on the flange, seen in the mocap frame."""

    def __init__(self, sim: SimArm, base_in_mocap: Optional[pin.SE3] = None,
                 marker_in_flange: Optional[pin.SE3] = None, frame: str = 'flange'):
        self.sim = sim
        r = np.random.default_rng(1234)
        self.base_in_mocap = base_in_mocap or pin.SE3(pin.exp3(r.normal(0, 0.3, 3)), r.normal(0, 0.5, 3))
        self.marker_in_flange = marker_in_flange or pin.SE3(pin.exp3(r.normal(0, 0.2, 3)),
                                                            np.array([0.02, -0.01, 0.03]))
        self.frame = frame if sim.truth.model.existFrame(frame) else sim.truth.tool_frame
        self.sigma = sim.err.mocap_sigma_m
        self.length_bar_scale_error = 2e-5   # 20 ppm

    def marker_pose(self) -> pin.SE3:
        """True flange pose (link angles, i.e. motor angle minus deflection) plus noise."""
        sim = self.sim
        q_link = sim.q
        M = self.base_in_mocap * sim.truth.fk(q_link, self.frame) * self.marker_in_flange
        noise = pin.SE3(pin.exp3(sim.rng.normal(0, self.sigma / 0.05, 3)), sim.rng.normal(0, self.sigma, 3))
        out = M * noise
        out.translation = out.translation * (1 + self.length_bar_scale_error)
        return out

    def static_noise_3sigma_m(self, samples: int = 600) -> float:
        p = np.array([self.marker_pose().translation for _ in range(samples)])
        return float(3 * np.max(p.std(axis=0)))

    def length_bar_error_m(self, length: float = 0.5) -> float:
        return abs(length * self.length_bar_scale_error) + math.sqrt(2) * self.sigma
