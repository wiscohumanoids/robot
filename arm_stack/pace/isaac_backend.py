"""Isaac Lab PACE replay backend (spec stage 9, full arm).

Same contract as JointReplayBackend: `simulate(k, P)` replays log k for a
population P (pop x 7, PARAMS order) and returns the measured-position traces
(pop x T). The population is batched as Isaac Lab envs (num_envs = population;
a larger population runs in chunks, a smaller one is padded with copies of its
last row). Per env, for the replayed joint j:

  drive      ImplicitActuator / PhysX joint drive, stiffness = drive_kp and
             damping = drive_kd + viscous (the drive's velocity target is 0, so
             the drive's damping term is exactly kd*qd + viscous*qd)
  armature   PhysX joint armature
  coulomb    an effort -Fc*tanh(qd/eps) on the joint each step (PhysX joint
             friction is a unitless coefficient, not N m). eps is 0.01 rad/s as
             in JointReplayBackend, widened only where an explicit step would
             chatter (eps >= 2*Fc*dt/I).
  bias       the drive closes the loop on q + bias, so its target is
             q_cmd - bias, and the reported position is q + bias
  delay      a command ring buffer at the sim rate: the command read each step
             is the one pushed delay/sim_dt steps ago, round(delay/sim_dt)
             whole steps with the fractional remainder interpolated between
             the two neighbouring entries (fractional_delay=False rounds)

The other joints are held at the log's q_hold with their joint-config gains.
Gravity feedforward follows JointReplayBackend: the host's Mode P feedforward
(Pinocchio gravity at the position it measured, plus the model's friction
feedforward) is computed on the log grid and goes through the same delay
buffer as the command.

The replay consumes the logged arrays as they are: JointLog objects (what
stage 9 records) or the `arm export pace` npz files, whose `command` and
`position` arrays are joint-side radians, the units Isaac Lab uses. No
conversion step.

isaaclab is imported lazily; constructing the backend without it raises an
ImportError that says how to get it. The numpy parts (DelayBuffer, env
batching, parameter tables, the replay plan) need no simulator and are unit
tested with a fake isaaclab module (test/arm_stack/test_isaac_backend.py).
"""

from __future__ import annotations

import importlib.util
import json
import os
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from .backend import PARAMS, JointLog

COULOMB_EPS = 0.01         # rad/s, the tanh width JointReplayBackend uses
FF_COULOMB_EPS = 0.02      # rad/s, the host feedforward's width (JointReplayBackend)

INSTALL_HINT = (
    'Isaac Lab is not importable in this Python (no `isaaclab` module). Install Isaac Sim 5.x '
    '(pip install isaacsim[all,extscache]) and Isaac Lab 2.x (git clone '
    'https://github.com/isaac-sim/IsaacLab, then isaaclab.bat/.sh --install) and run with that '
    'Python, or use the built-in per-joint replay (--backend joint). Build the USD first with '
    'tools/isaac/import_urdf.py.')


def isaaclab_available() -> bool:
    try:
        return importlib.util.find_spec('isaaclab') is not None
    except (ImportError, ValueError):
        return False


def require_isaaclab():
    if not isaaclab_available():
        raise ImportError(INSTALL_HINT)


# --------------------------------------------------------------------------- pure numpy parts
class DelayBuffer:
    """Ring buffer of commands at the sim rate, read with a per-env fractional delay.

    push(x) stores this step's command (width,), shared by every env; read()
    returns (envs, width): the entry `delay_steps` pushes old, linearly
    interpolated between floor and ceil. Before the history fills, the oldest
    command is repeated (as np.interp clamps at the first sample).
    """

    def __init__(self, delay_steps: Sequence[float], width: int, fractional: bool = True):
        d = np.maximum(np.asarray(delay_steps, float), 0.0)
        if not fractional:
            d = np.round(d)
        self.n0 = np.floor(d + 1e-9).astype(int)
        self.frac = np.where(d - self.n0 > 1e-9, d - self.n0, 0.0)
        self.size = int(self.n0.max()) + 2
        self.width = width
        self.buf = None
        self.head = 0

    @staticmethod
    def steps(delay_s: Sequence[float], sim_dt: float) -> np.ndarray:
        return np.asarray(delay_s, float) / sim_dt

    def push(self, x):
        x = np.asarray(x, float).reshape(self.width)
        if self.buf is None:
            self.buf = np.repeat(x[None, :], self.size, axis=0)
            self.head = 0
            return
        self.head = (self.head + 1) % self.size
        self.buf[self.head] = x

    def read(self) -> np.ndarray:
        a = self.buf[(self.head - self.n0) % self.size]
        b = self.buf[(self.head - self.n0 - 1) % self.size]
        return (1.0 - self.frac)[:, None] * a + self.frac[:, None] * b


def env_batches(pop: int, num_envs: int) -> Iterator[Tuple[np.ndarray, int]]:
    """Map population rows onto envs: yields (row index per env, number of real rows).

    A partial last batch is padded with its last row; the padded envs' results
    are dropped.
    """
    for start in range(0, pop, num_envs):
        rows = np.arange(start, min(start + num_envs, pop))
        n = len(rows)
        if n < num_envs:
            rows = np.concatenate([rows, np.full(num_envs - n, rows[-1])])
        yield rows, n


def coulomb_eps(fc: np.ndarray, inertia: np.ndarray, sim_dt: float) -> np.ndarray:
    """tanh width for an explicit Coulomb effort that cannot chatter at this step."""
    return np.maximum(COULOMB_EPS, 2.0 * np.asarray(fc) * sim_dt / np.maximum(inertia, 1e-9))


def param_tables(P: np.ndarray, joint: int, hold_kp: np.ndarray, hold_kd: np.ndarray):
    """Per-env drive tables (envs x n joints, config order) for replaying `joint`.

    Returns dict: stiffness, damping, armature, coulomb, bias, delay_s.
    """
    P = np.atleast_2d(np.asarray(P, float))
    assert P.shape[1] == len(PARAMS), 'P must be pop x %d (%s)' % (len(PARAMS), ', '.join(PARAMS))
    col = {name: P[:, i] for i, name in enumerate(PARAMS)}
    E, n = len(P), len(hold_kp)
    stiffness = np.repeat(np.asarray(hold_kp, float)[None, :], E, axis=0)
    damping = np.repeat(np.asarray(hold_kd, float)[None, :], E, axis=0)
    armature = np.zeros((E, n))
    stiffness[:, joint] = col['drive_kp']
    damping[:, joint] = col['drive_kd'] + col['viscous']
    armature[:, joint] = col['armature']
    return dict(stiffness=stiffness, damping=damping, armature=armature, coulomb=col['coulomb'].copy(),
                bias=col['encoder_bias'].copy(), delay_s=col['delay_s'].copy())


class ReplayPlan:
    """One log resampled to the sim rate: command and feedforward channels.

    channels[:, 0] is the replayed joint's command; channels[:, 1:] the host's
    full-arm torque feedforward (zeros if gravity_ff is off), one row per sim
    step. Both go through the DelayBuffer together.
    """

    def __init__(self, model, log: JointLog, sim_dt: float, gravity_ff: bool = True):
        self.log = log
        self.joint = log.joint
        sub = log.dt / sim_dt
        self.substeps = int(round(sub))
        if self.substeps < 1 or abs(sub - self.substeps) > 0.01 * sub:
            raise ValueError('log dt %.4g s is not a whole number of sim steps (%.4g s)' % (log.dt, sim_dt))
        self.sim_dt = sim_dt
        T = len(log.q_cmd)
        self.T = T
        n = len(log.q_hold)
        t_log = np.arange(T) * log.dt
        t_sim = np.arange(T * self.substeps) * sim_dt
        j = self.joint
        ff = np.zeros((T, n))
        if gravity_ff:
            lo = min(log.q_meas.min(), log.q_cmd.min()) - 0.2
            hi = max(log.q_meas.max(), log.q_cmd.max()) + 0.2
            grid = np.linspace(lo, hi, 121)
            g = []
            for v in grid:
                q = np.array(log.q_hold, float)
                q[j] = v
                g.append(np.asarray(model.gravity(q), float))
            g = np.array(g)                                   # (121, n)
            for i in range(n):
                ff[:, i] = np.interp(log.q_meas, grid, g[:, i])
        fc0 = float(np.asarray(getattr(model, 'coulomb', np.zeros(n)))[j])
        fv0 = float(np.asarray(getattr(model, 'viscous', np.zeros(n)))[j])
        ff[:, j] += fc0 * np.tanh(log.qd_cmd / FF_COULOMB_EPS) + fv0 * log.qd_cmd
        chans = np.column_stack([log.q_cmd, ff])               # (T, 1+n) on the log grid
        self.channels = np.column_stack([np.interp(t_sim, t_log, chans[:, c]) for c in range(chans.shape[1])])
        self.q_hold = np.array(log.q_hold, float)


def load_pace_npz(path: str, joint: Union[int, str, None] = None, names: Optional[Sequence[str]] = None,
                  window: Optional[Tuple[int, int]] = None) -> JointLog:
    """A JointLog straight from an `arm export pace` npz (arrays used as stored).

    joint: index or name (names from the json sidecar); default the joint with
    the largest command amplitude. q_hold is the median command of the other
    joints, qd_cmd the command's time derivative (only the friction feedforward
    uses it).
    """
    d = np.load(path)
    cmd, pos = np.asarray(d['command'], float), np.asarray(d['position'], float)
    if window is not None:
        cmd, pos = cmd[window[0]:window[1]], pos[window[0]:window[1]]
    dt = 1.0 / float(d['sample_rate_hz'])
    if isinstance(joint, str):
        if names is None:
            side = os.path.splitext(path)[0] + '.json'
            with open(side) as f:
                names = json.load(f).get('joints')
        joint = list(names).index(joint)
    if joint is None:
        joint = int(np.argmax(np.asarray(d['amplitude_rad'], float)))
    hold = np.median(cmd, axis=0)
    lo, hi = cmd[:, joint].min(), cmd[:, joint].max()
    hold[joint] = 0.5 * (lo + hi)
    return JointLog(int(joint), dt, cmd[:, joint].copy(), np.gradient(cmd[:, joint], dt), pos[:, joint].copy(),
                    hold, (float(hold[joint]), float(0.5 * (hi - lo))), (0.0, 0.0))


def _from_sim(x) -> np.ndarray:
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return np.asarray(x, float)


def _to_sim(a, like):
    """numpy -> the simulator's array type (torch on the sim device in Isaac Lab)."""
    if hasattr(like, 'detach'):
        import torch
        return torch.as_tensor(np.asarray(a), dtype=like.dtype, device=like.device)
    return np.asarray(a, float)


# --------------------------------------------------------------------------- backend
class IsaacLabBackend:
    """PACE replay in Isaac Lab: full arm, one env per candidate.

    model: ArmModel (calibrated: its link masses/COMs are written to the sim
    bodies, its gravity drives the feedforward). logs: JointLog objects or
    `arm export pace` npz paths. hold_gains: (kp[n], kd[n]) for the held
    joints, default the joint-config Mode P gains.
    """

    def __init__(self, model, logs: List[Union[JointLog, str]], sim_dt: Optional[float] = None,
                 num_envs: int = 16, device: str = 'cuda:0', usd_path: Optional[str] = None,
                 hold_gains: Optional[Tuple[Sequence[float], Sequence[float]]] = None, gravity_ff: bool = True,
                 fractional_delay: bool = True, sync_inertials: bool = True, headless: bool = True):
        require_isaaclab()
        self.model = model
        self.logs = [load_pace_npz(l) if isinstance(l, str) else l for l in logs]
        self.sim_dt = float(sim_dt or min(l.dt for l in self.logs))
        self.num_envs = int(num_envs)
        self.device = device
        self.fractional_delay = fractional_delay
        self.joint_names = list(model.joint_names)
        n = len(self.joint_names)
        if hold_gains is None:
            from .isaac_cfg import config_gains
            g = config_gains()
            hold_gains = ([g.get(nm, {}).get('kp', 0.0) for nm in self.joint_names],
                          [g.get(nm, {}).get('kd', 0.0) for nm in self.joint_names])
        self.hold_kp = np.asarray(hold_gains[0], float).reshape(n)
        self.hold_kd = np.asarray(hold_gains[1], float).reshape(n)
        self.plans = [ReplayPlan(model, l, self.sim_dt, gravity_ff) for l in self.logs]
        self.inertia = [float(model.mass_matrix(l.q_hold)[l.joint, l.joint]) for l in self.logs]
        self._build(usd_path, headless)
        if sync_inertials:
            self._sync_inertials()

    # ------------------------------------------------------------------ sim setup
    def _build(self, usd_path, headless):
        from isaaclab.app import AppLauncher
        self._app = None
        if not _app_running():
            self._app = AppLauncher(headless=headless, device=self.device).app
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.utils import configclass

        from .isaac_cfg import USD_PATH, arm01_articulation_cfg
        usd = usd_path or USD_PATH
        if not os.path.exists(usd):
            raise FileNotFoundError('%s not found; run tools/isaac/import_urdf.py first' % usd)
        robot_cfg = arm01_articulation_cfg(usd_path=usd, joint_names=self.joint_names,
                                           prim_path='{ENV_REGEX_NS}/Robot')

        @configclass
        class _PaceSceneCfg(InteractiveSceneCfg):
            robot = robot_cfg

        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=self.sim_dt, device=self.device))
        self.scene = InteractiveScene(_PaceSceneCfg(num_envs=self.num_envs, env_spacing=1.5,
                                                    replicate_physics=True))
        self.sim.reset()
        self.robot = self.scene['robot']
        ids, names = self.robot.find_joints(self.joint_names, preserve_order=True)
        if list(names) != self.joint_names:
            raise ValueError('USD joints %s do not match the model %s' % (names, self.joint_names))
        self.jidx = list(ids)              # config order -> articulation dof index

    def _sync_inertials(self):
        """Write the calibrated link masses and COMs (ArmModel) to the sim bodies."""
        view = getattr(self.robot, 'root_physx_view', None)
        if view is None or not hasattr(self.model, 'link_names'):
            return
        links = self.model.link_names()
        bids, _ = self.robot.find_bodies(links, preserve_order=True)
        masses = view.get_masses().clone()
        coms = view.get_coms().clone()
        inertias = view.get_inertias().clone()
        for b, j in zip(bids, self.model.jid):
            I = self.model.model.inertias[j]
            old = float(masses[0, b])
            masses[:, b] = float(I.mass)
            coms[:, b, :3] = _to_sim(np.asarray(I.lever, float), coms)
            if old > 0:
                inertias[:, b] *= float(I.mass) / old
        import torch
        env_ids = torch.arange(self.num_envs, device=masses.device)
        view.set_masses(masses, env_ids)
        view.set_coms(coms, env_ids)
        view.set_inertias(inertias, env_ids)

    # ------------------------------------------------------------------ replay
    def simulate(self, k: int, P: np.ndarray) -> np.ndarray:
        """Replay log k for a population P (pop x 7, PARAMS order); returns q (pop x T)."""
        P = np.atleast_2d(np.asarray(P, float))
        out = np.empty((len(P), self.plans[k].T))
        for rows, n in env_batches(len(P), self.num_envs):
            out[rows[:n]] = self._run_batch(k, P[rows])[:n]
        return out

    def _run_batch(self, k: int, P: np.ndarray) -> np.ndarray:
        plan, robot = self.plans[k], self.robot
        j, E, n = plan.joint, len(P), len(self.joint_names)
        tab = param_tables(P, j, self.hold_kp, self.hold_kd)
        like = robot.data.joint_pos
        ids = self.jidx
        robot.write_joint_stiffness_to_sim(_to_sim(tab['stiffness'], like), joint_ids=ids)
        robot.write_joint_damping_to_sim(_to_sim(tab['damping'], like), joint_ids=ids)
        robot.write_joint_armature_to_sim(_to_sim(tab['armature'], like), joint_ids=ids)
        bias, fc = tab['bias'], tab['coulomb']
        eps = coulomb_eps(fc, self.inertia[k] + tab['armature'][:, j], self.sim_dt)
        q0 = np.repeat(plan.q_hold[None, :], E, axis=0)
        q0[:, j] = plan.log.q_meas[0] - bias                   # true angle = measured - bias
        robot.reset()
        robot.write_joint_state_to_sim(_to_sim(q0, like), _to_sim(np.zeros((E, n)), like), joint_ids=ids)
        buf = DelayBuffer(DelayBuffer.steps(tab['delay_s'], self.sim_dt), plan.channels.shape[1],
                          self.fractional_delay)
        target = q0.copy()
        target[:, j] = 0.0
        robot.set_joint_velocity_target(_to_sim(np.zeros((E, n)), like), joint_ids=ids)
        out = np.empty((E, plan.T))
        s = 0
        for t in range(plan.T):
            for _ in range(plan.substeps):
                buf.push(plan.channels[s])
                c = buf.read()                                  # (E, 1+n)
                target[:, j] = c[:, 0] - bias
                effort = c[:, 1:].copy()
                qd = _from_sim(robot.data.joint_vel)[:, ids][:, j]
                effort[:, j] -= fc * np.tanh(qd / eps)
                robot.set_joint_position_target(_to_sim(target, like), joint_ids=ids)
                robot.set_joint_effort_target(_to_sim(effort, like), joint_ids=ids)
                robot.write_data_to_sim()
                self.sim.step(render=False)
                robot.update(self.sim_dt)
                s += 1
            out[:, t] = _from_sim(robot.data.joint_pos)[:, ids][:, j] + bias
        return out

    def close(self):
        if self._app is not None:
            self._app.close()
            self._app = None


def _app_running() -> bool:
    try:
        import omni.kit.app
        return omni.kit.app.get_app() is not None
    except Exception:  # noqa: BLE001 - no Kit app yet
        return False
