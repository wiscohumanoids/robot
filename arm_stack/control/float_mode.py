"""`arm float`: teach / hand-guide mode. The arm floats in Mode T under gravity
compensation while the operator moves it; `--record` logs the session to MCAP
for kinesthetic teaching.

Per cycle, for each floating joint (Mode T, kp = 0 unless at a wall):

  tau = g(q) + g_payload(q)                     RT loop (gravity_ff, payload)
      + gain * Fc * ramp(|qd|) * sign(qd)       Coulomb help, tau_ff (this file)
      - damping * qd                            RT kd, optional
      + wall spring-damper near the soft limits RT kp / kd, optional

The Coulomb help has a velocity deadband and a smoothstep ramp, so at rest
it is zero and cannot start a drift, and gain < 1, so a moving joint always
keeps some net friction and stops when let go. `arm gravity-check` finds the
largest gain per joint that does not creep and recommends it as
`float_friction_gain` in the joint config. The RT law's own friction
feedforward (friction_ff_gain) is off: it has no deadband.

Locked joints (--lock) hold their pose with a stiff joint PD in the RT loop
plus a slow integrator here. The RT controller switches CSP/CST for the whole
arm at once, so a per-joint Mode P hold would need a per-joint mode in
ArmController; the PD is the Mode T stand-in.

Safety: any joint above 0.5 x max_vel latches damping on every floating joint
(no friction help) until all are below 0.25 x max_vel for 0.2 s; every
command carries the bring-up torque clamp; the RT watchdog is unchanged (no
command for 50 ms: gravity comp, payload included, plus damping).

Exit (Ctrl-C or --duration): damp to a stop, then a Mode P hold at the
current pose (the RT loop re-enters CSP from the actual position: bumpless).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..hw.client import ArmClient, Command, JointState

DEG = math.pi / 180.0

DEADBAND = 0.02               # rad/s: no Coulomb help below this
RAMP = 0.03                   # rad/s: smoothstep to full help over this
DEFAULT_FRICTION_GAIN = 0.6   # when the joint config has no float_friction_gain
MAX_FRICTION_GAIN = 0.95      # never compensate all of it
VEL_LIMIT_FRACTION = 0.5      # x max_vel: switch to damping
RESUME_FRACTION = 0.25        # x max_vel: all below this for RESUME_S -> float again
RESUME_S = 0.2
SAFETY_KD_SCALE = 3.0         # x max(kd, 0.2) (the watchdog damping): safety damping
WALL_MARGIN = 10 * DEG        # walls start this far inside the soft limits
WALL_TORQUE_FRACTION = 0.5    # wall spring reaches this x clamp at the soft limit
LOCK_KP_SCALE = 5.0           # locked joints: x the joint config gains
LOCK_KD_SCALE = 2.0
LOCK_KI = 5.0                 # 1/s, integrator on the lock target (static error -> 0)
LOCK_I_MAX = 5 * DEG
STOP_SPEED = 0.01             # rad/s: "stopped" on exit
EXIT_DAMP_S = 0.6             # at most this long damping to a stop on exit
EXIT_HOLD_S = 0.5             # Mode P hold streamed before returning


def coulomb_ramp(qd, deadband=DEADBAND, ramp=RAMP) -> np.ndarray:
    """sign(qd) * smoothstep((|qd| - deadband) / ramp): 0 at rest, 1 when moving."""
    qd = np.asarray(qd, float)
    a = np.clip((np.abs(qd) - deadband) / max(ramp, 1e-9), 0.0, 1.0)
    return np.sign(qd) * a * a * (3.0 - 2.0 * a)


def parse_payload(text: Optional[str]):
    """'KG@x,y,z' (m, tool0 frame) -> (kg, x, y, z); 'KG' -> at the tool0 origin;
    None -> None (no change); '0' -> (0, 0, 0, 0) (clear)."""
    if text is None:
        return None
    text = str(text).strip()
    if '@' in text:
        kg, xyz = text.split('@', 1)
        p = [float(v) for v in xyz.replace(' ', '').split(',')]
        if len(p) != 3:
            raise ValueError('--payload KG@x,y,z needs three coordinates, got %r' % text)
    else:
        kg, p = text, [0.0, 0.0, 0.0]
    kg = float(kg)
    if kg < 0 or not math.isfinite(kg):
        raise ValueError('--payload mass must be >= 0 kg, got %r' % text)
    return (kg, p[0], p[1], p[2])


def parse_lock(text: Optional[str], names: Sequence[str]) -> np.ndarray:
    """Joint lock mask from names or indices ('shoulder_pitch,elbow', '0,3') or a
    0/1 string with one digit per joint ('1100000')."""
    n = len(names)
    mask = np.zeros(n, bool)
    if not text:
        return mask
    text = text.strip()
    if len(text) == n and set(text) <= {'0', '1'}:
        return np.array([c == '1' for c in text])
    for tok in text.split(','):
        tok = tok.strip()
        if not tok:
            continue
        if tok in names:
            mask[list(names).index(tok)] = True
        elif tok.lstrip('-').isdigit() and 0 <= int(tok) < n:
            mask[int(tok)] = True
        else:
            raise ValueError('--lock: unknown joint %r (names: %s, or indices 0..%d)' % (tok, ', '.join(names), n - 1))
    return mask


def _per_joint(v, n, name) -> np.ndarray:
    a = np.asarray(v, float).ravel()
    if a.size == 1:
        return np.full(n, float(a[0]))
    if a.size != n:
        raise ValueError('%s: give 1 or %d values, got %d' % (name, n, a.size))
    return a.copy()


def configured_friction_gains(arm) -> np.ndarray:
    """Per-joint `float_friction_gain` from the joint config (gravity-check
    writes it), DEFAULT_FRICTION_GAIN where unset."""
    rows = {r.get('name'): r for r in (arm.raw.get('joints') or [])}
    out = []
    for nme in arm.names:
        v = (rows.get(nme) or {}).get('float_friction_gain')
        out.append(DEFAULT_FRICTION_GAIN if v is None else float(v))
    return np.array(out)


@dataclass
class FloatConfig:
    friction_gain: np.ndarray     # per joint, fraction of the model Coulomb (clipped < 1)
    coulomb: np.ndarray           # Nm, the calibrated friction model
    damping: np.ndarray           # Nm s/rad on floating joints (RT kd)
    lock: np.ndarray              # bool: held, not floating
    torque_clamp: np.ndarray      # Nm, bring-up limits
    max_vel: np.ndarray           # rad/s
    lower: np.ndarray             # soft limits, rad
    upper: np.ndarray
    lock_kp: np.ndarray
    lock_kd: np.ndarray
    safety_kd: np.ndarray
    wall_kp: np.ndarray
    wall_kd: np.ndarray
    payload: Optional[Sequence[float]] = None     # (kg, x, y, z) in tool0; sent every cycle
    walls: bool = True
    wall_margin: float = WALL_MARGIN
    deadband: float = DEADBAND
    ramp: float = RAMP
    vel_limit_fraction: float = VEL_LIMIT_FRACTION
    resume_fraction: float = RESUME_FRACTION

    @classmethod
    def from_arm(cls, arm, model, friction_gain=None, damping=0.0, lock=None, payload=None,
                 walls=True, **kw) -> 'FloatConfig':
        """model: nominal URDF + calibration with the soft limits applied (cli.build_model)."""
        n = arm.n
        gain = configured_friction_gains(arm) if friction_gain is None else _per_joint(friction_gain, n, 'friction gain')
        kp = np.array([j.kp for j in arm.joints], float)
        kd = np.array([j.kd for j in arm.joints], float)
        clamp = np.array([j.max_torque_nm for j in arm.joints], float)
        margin = kw.get('wall_margin', WALL_MARGIN)
        return cls(
            friction_gain=np.clip(gain, 0.0, MAX_FRICTION_GAIN),
            coulomb=np.asarray(getattr(model, 'coulomb', np.zeros(n)), float).copy(),
            damping=np.maximum(_per_joint(damping, n, 'damping'), 0.0),
            lock=np.zeros(n, bool) if lock is None else np.asarray(lock, bool).copy(),
            torque_clamp=clamp,
            max_vel=np.array([j.max_vel_rad_s for j in arm.joints], float),
            lower=np.asarray(model.lower, float).copy(), upper=np.asarray(model.upper, float).copy(),
            lock_kp=LOCK_KP_SCALE * kp, lock_kd=LOCK_KD_SCALE * kd,
            safety_kd=SAFETY_KD_SCALE * np.maximum(kd, 0.2),
            wall_kp=WALL_TORQUE_FRACTION * clamp / max(margin, 1e-3), wall_kd=kd.copy(),
            payload=payload, walls=walls, **kw)

    @property
    def n(self) -> int:
        return len(self.lock)

    def summary(self) -> Dict:
        return {'friction_gain': self.friction_gain.tolist(), 'coulomb_nm': self.coulomb.tolist(),
                'damping': self.damping.tolist(), 'lock': self.lock.astype(int).tolist(),
                'torque_clamp': self.torque_clamp.tolist(), 'payload': None if self.payload is None
                else [float(x) for x in self.payload], 'walls': bool(self.walls),
                'vel_limit_rad_s': (self.vel_limit_fraction * self.max_vel).tolist()}


class FloatController:
    """One Command per cycle from the measured state. Stateful: the lock
    integrator and the latched safety damping."""

    def __init__(self, cfg: FloatConfig, q0: np.ndarray):
        self.cfg = cfg
        self.lock_q = np.asarray(q0, float).copy()
        self.lock_i = np.zeros(cfg.n)
        self.damping_on = False
        self._calm_s = 0.0
        self.events: List[Dict] = []

    def _event(self, t, name, **kw):
        self.events.append({'t': float(t), 'event': name, **kw})

    def command(self, s: JointState, dt: float) -> Command:
        c = self.cfg
        q, qd = np.asarray(s.q, float), np.asarray(s.qd, float)
        free = ~c.lock
        # safety: velocity limit -> latched damping
        speed = np.abs(qd) / np.maximum(c.max_vel, 1e-6)
        if not self.damping_on and np.any(speed[free] > c.vel_limit_fraction):
            self.damping_on = True
            self._calm_s = 0.0
            j = int(np.argmax(np.where(free, speed, 0.0)))
            self._event(s.t, 'float_damping_on', joint=j, qd=float(qd[j]),
                        limit=float(c.vel_limit_fraction * c.max_vel[j]))
        elif self.damping_on:
            self._calm_s = self._calm_s + dt if np.all(speed[free] < c.resume_fraction) else 0.0
            if self._calm_s >= RESUME_S:
                self.damping_on = False
                self._event(s.t, 'float_damping_off')

        q_d = q.copy()
        kp = np.zeros(c.n)
        kd = c.damping.copy()
        tau_ff = np.zeros(c.n)
        if self.damping_on:
            kd = np.maximum(kd, c.safety_kd)
        else:
            tau_ff = c.friction_gain * c.coulomb * coulomb_ramp(qd, c.deadband, c.ramp)
        if c.walls:
            lo, hi = c.lower + c.wall_margin, c.upper - c.wall_margin
            past = (q < lo) | (q > hi)
            outward = ((q < lo) & (qd < 0)) | ((q > hi) & (qd > 0))
            q_d = np.where(past, np.clip(q, lo, hi), q_d)
            kp = np.where(past, c.wall_kp, kp)
            kd = np.where(outward, kd + c.wall_kd, kd)
        tau_ff = np.where(free, tau_ff, 0.0)
        # locked joints: stiff PD on the lock pose, plus a slow integrator
        if np.any(c.lock):
            self.lock_i = np.clip(self.lock_i + LOCK_KI * (self.lock_q - q) * dt, -LOCK_I_MAX, LOCK_I_MAX)
            q_d = np.where(c.lock, self.lock_q + self.lock_i, q_d)
            kp = np.where(c.lock, c.lock_kp, kp)
            kd = np.where(c.lock, c.lock_kd, kd)
        return Command(mode='T', q=q_d, qd=np.zeros(c.n), kp=kp, kd=kd, tau_ff=tau_ff,
                       torque_clamp=c.torque_clamp.copy(), gravity_ff=True, friction_ff_gain=0.0,
                       payload=c.payload)


@dataclass
class FloatResult:
    reason: str                               # 'duration', 'interrupt', 'stop'
    seconds: float
    q_start: np.ndarray
    q_end: np.ndarray                         # after the exit hold
    final: JointState
    events: List[Dict]
    max_speed_fraction: np.ndarray            # max |qd| / max_vel per joint
    max_torque_fraction: np.ndarray           # max |tau sent| / clamp per joint
    travel: np.ndarray                        # max |q - q_start| per joint, rad
    trace: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def damping_triggered(self) -> bool:
        return any(e['event'] == 'float_damping_on' for e in self.events)


def _send_hold(client: ArmClient, q, cfg: FloatConfig):
    client.send(Command(mode='P', q=np.asarray(q, float).copy(), qd=np.zeros(cfg.n),
                        torque_clamp=cfg.torque_clamp.copy(), payload=cfg.payload))


def exit_to_hold(client: ArmClient, cfg: FloatConfig, ctrl: FloatController, s: JointState,
                 hold_s: float = EXIT_HOLD_S) -> JointState:
    """Damp the floating joints to a stop, then hold the current pose in Mode P."""
    dt = client.cmd_dt
    still = 0.0
    for _ in range(max(1, int(EXIT_DAMP_S / dt))):
        cmd = ctrl.command(s, dt)
        cmd.kd = np.where(cfg.lock, cmd.kd, np.maximum(cmd.kd, cfg.safety_kd))
        cmd.tau_ff = np.zeros(cfg.n)
        client.send(cmd)
        s = client.step(dt)
        still = still + dt if np.max(np.abs(s.qd)) < STOP_SPEED else 0.0
        if still >= 0.05:
            break
    q_hold = s.q.copy()
    for _ in range(max(1, int(round(hold_s / dt)))):
        _send_hold(client, q_hold, cfg)
        s = client.step(dt)
    return s


def run_float(client: ArmClient, cfg: FloatConfig, duration: Optional[float] = None,
              operator: Optional[Callable] = None, log=None,
              stop: Optional[Callable[[JointState, float], bool]] = None,
              hold_s: float = EXIT_HOLD_S, keep_trace: bool = True,
              status: Optional[Callable[[float, JointState], None]] = None) -> FloatResult:
    """Float until `duration` (s, None = until Ctrl-C), `stop(state, t)` is true,
    or KeyboardInterrupt; then exit to a Mode P hold.

    operator(client, t, state): sim only, applies the operator's push each cycle
    (SimArm.apply_external_force / apply_external_torque); released on exit."""
    if np.all(cfg.lock):
        raise ValueError('every joint is locked: nothing to float')
    dt = client.cmd_dt
    s = client.state()
    t0 = s.t
    q0 = s.q.copy()
    ctrl = FloatController(cfg, q0)
    if log is not None:
        log.event('float_start', **cfg.summary())
    n = cfg.n
    vmax = np.zeros(n)
    tmax = np.zeros(n)
    travel = np.zeros(n)
    tr = {k: [] for k in ('t', 'q', 'qd', 'tau_cmd', 'tau_ff', 'damping', 'fallback', 'mode')} if keep_trace else {}
    reason = 'duration'
    wall0 = time.monotonic()
    last_status = -1.0
    n_events = 0
    try:
        while True:
            t = s.t - t0
            if duration is not None and t >= duration:
                break
            if operator is not None:
                operator(client, t, s)
            cmd = ctrl.command(s, dt)
            client.send(cmd)
            s = client.step(dt)
            vmax = np.maximum(vmax, np.abs(s.qd) / np.maximum(cfg.max_vel, 1e-6))
            if s.tau_cmd is not None:
                tmax = np.maximum(tmax, np.abs(s.tau_cmd) / np.maximum(cfg.torque_clamp, 1e-6))
            travel = np.maximum(travel, np.abs(s.q - q0))
            if keep_trace:
                tr['t'].append(s.t - t0)
                tr['q'].append(s.q.copy())
                tr['qd'].append(s.qd.copy())
                tr['tau_cmd'].append(np.full(n, np.nan) if s.tau_cmd is None else s.tau_cmd.copy())
                tr['tau_ff'].append(cmd.tau_ff.copy())
                tr['damping'].append(ctrl.damping_on)
                tr['fallback'].append(bool(s.fallback))
                tr['mode'].append(s.mode)
            if log is not None and len(ctrl.events) > n_events:
                for e in ctrl.events[n_events:]:
                    e = dict(e)
                    log.event(e.pop('event'), t_ns=int(e.pop('t') * 1e9), **e)
                n_events = len(ctrl.events)
            if status is not None and time.monotonic() - wall0 - last_status >= 1.0:
                last_status = time.monotonic() - wall0
                status(t, s)
            if stop is not None and stop(s, s.t - t0):
                reason = 'stop'
                break
    except KeyboardInterrupt:
        reason = 'interrupt'
    finally:
        if operator is not None and hasattr(client, 'apply_external_torque'):
            client.apply_external_torque(None)
    seconds = s.t - t0
    s = exit_to_hold(client, cfg, ctrl, s, hold_s)
    if log is not None:
        log.event('float_end', reason=reason, seconds=float(seconds), q=s.q.tolist(),
                  max_speed_fraction=vmax.tolist(), max_torque_fraction=tmax.tolist())
    trace = {k: np.array(v) for k, v in tr.items()} if keep_trace else {}
    return FloatResult(reason, float(seconds), q0, s.q.copy(), s, list(ctrl.events), vmax, tmax, travel, trace)


# --------------------------------------------------------------------------- sim operator
def push_profile(force, t_on: float, t_off: float, frame: str = 'tool0', torque=None,
                 rise_s: float = 0.1) -> Callable:
    """Sim operator: a world-frame force (N) at `frame` between t_on and t_off,
    with rise_s smooth ramps, re-mapped through the Jacobian every cycle."""
    force = np.asarray(force, float)
    torque = None if torque is None else np.asarray(torque, float)

    def operator(client, t, _s):
        if t < t_on or t > t_off:
            client.apply_external_torque(None)
            return
        a = min(1.0, (t - t_on) / rise_s, (t_off - t) / rise_s) if rise_s > 0 else 1.0
        client.apply_external_force(a * force, frame, None if torque is None else a * torque)
    return operator


# --------------------------------------------------------------------------- CLI
def cli_run(args, arm, model, client, recorder=None) -> int:
    """`arm float [--sim] [--damping D] [--friction-gain G] [--lock J,..] [--payload KG@x,y,z]
    [--duration S] [--record [PATH]]`"""
    lock = parse_lock(args.lock, arm.names)
    payload = parse_payload(args.payload)
    if payload is None:
        payload = (0.0, 0.0, 0.0, 0.0)      # no --payload: none mounted (clears a sticky one)
    damping = [float(x) for x in str(args.damping).split(',')] if args.damping is not None else 0.0
    gain = [float(x) for x in str(args.friction_gain).split(',')] if args.friction_gain is not None else None
    cfg = FloatConfig.from_arm(arm, model, friction_gain=gain, damping=damping, lock=lock, payload=payload)
    duration = args.duration
    operator = None
    if args.sim:
        if payload[0] > 0:            # hang it on the sim plant too
            from ..hw.client import Payload
            client.set_payload(Payload('tool0', payload[0], tuple(payload[1:])))
        if duration is None:
            duration = 4.0
        # the sim operator: 12 N sideways on the tool for 1.5 s, then let go
        operator = push_profile([0.0, 12.0, 0.0], 0.5, 2.0)
    if not np.any(cfg.coulomb > 0):
        print('  note: no friction model in the calibration (run arm friction): no Coulomb help')
    print('float: %s floating, %s locked; friction gain %s; damping %s Nm s/rad; payload %s; '
          'velocity limit %.2f rad/s; %s' % (
              ', '.join(n for n, l in zip(arm.names, lock) if not l) or '-',
              ', '.join(n for n, l in zip(arm.names, lock) if l) or 'none',
              np.round(cfg.friction_gain, 2).tolist(), np.round(cfg.damping, 3).tolist(),
              'none' if not payload[0] > 0 else '%.3f kg @ %s m (tool0)' % (payload[0], list(payload[1:])),
              float(np.min(cfg.vel_limit_fraction * cfg.max_vel)),
              'for %.1f s' % duration if duration is not None else 'Ctrl-C to stop'))

    def status(t, s):
        if not args.sim:
            print('  t %6.1f s  q (deg) %s%s' % (t, np.round(np.degrees(s.q), 1).tolist(),
                                                '  [fallback]' if s.fallback else ''))

    r = run_float(client, cfg, duration=duration, operator=operator, log=recorder,
                  keep_trace=False, status=status)
    print('float: %s after %.1f s; travel (deg) %s; peak speed %.0f%% of max_vel; peak torque %.0f%% of clamp%s'
          % (r.reason, r.seconds, np.round(np.degrees(r.travel), 1).tolist(),
             100 * float(np.max(r.max_speed_fraction)), 100 * float(np.max(r.max_torque_fraction)),
             '; safety damping %d x' % sum(e['event'] == 'float_damping_on' for e in r.events)
             if r.damping_triggered else ''))
    print('  holding in Mode P at %s deg' % np.round(np.degrees(r.q_end), 1).tolist())
    return 0
