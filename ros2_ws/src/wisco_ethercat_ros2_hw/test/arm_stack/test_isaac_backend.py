"""IsaacLabBackend: delay buffer, bias, parameter-to-env mapping, npz replay.

Runs without Isaac Lab: a fake `isaaclab` package (monkeypatched into
sys.modules) stands in for the simulator. Its articulation integrates each
joint on its own with the same scheme as JointReplayBackend, in a shuffled DOF
order, so the replay through the fake must reproduce JointReplayBackend
exactly. The real-sim smoke test runs only where isaaclab imports.
"""

import importlib.machinery
import json
import os
import sys
import types

import numpy as np
import pytest

from arm_stack.pace import isaac_backend as ib
from arm_stack.pace.backend import PARAMS, IsaacLabBackend, JointLog, JointReplayBackend
from arm_stack.pace.isaac_cfg import DRIVEN_JOINTS, MIMIC_JOINTS, URDF_PATH, prepare_urdf_for_isaac

NAMES = list(DRIVEN_JOINTS)
N = len(NAMES)
DT = 0.002


# --------------------------------------------------------------------------- fakes
class FakeModel:
    """Duck-typed ArmModel: diagonal inertia, optional constant gravity."""

    def __init__(self, inertia=None, g=None):
        self.joint_names = NAMES
        self.I = np.asarray(inertia if inertia is not None else np.linspace(0.08, 0.02, N), float)
        self.g = np.zeros(N) if g is None else np.asarray(g, float)
        self.coulomb = np.zeros(N)
        self.viscous = np.zeros(N)

    def mass_matrix(self, q):
        return np.diag(self.I)

    def gravity(self, q):
        return self.g.copy()


class _Cfg:
    def __init__(self, *args, **kw):
        self.args = args
        self.__dict__.update(kw)

    def replace(self, **kw):
        out = _Cfg(**self.__dict__)
        out.__dict__.update(kw)
        return out


class FakeArticulation:
    ORDER = [NAMES[i] for i in (3, 0, 6, 1, 5, 2, 4)]      # not the config order, like PhysX's BFS order

    def __init__(self, cfg, num_envs, inertia, gravity):
        self.cfg = cfg
        self.joint_names = list(self.ORDER)
        self.num_joints = N
        E = num_envs
        self.I = np.array([inertia[NAMES.index(n)] for n in self.joint_names])
        self.g = np.array([gravity[NAMES.index(n)] for n in self.joint_names])   # pose-independent
        z = lambda: np.zeros((E, N))   # noqa: E731
        self.data = types.SimpleNamespace(joint_pos=z(), joint_vel=z())
        self.stiffness, self.damping, self.armature = z(), z(), z()
        self.pos_target, self.vel_target, self.effort_target = z(), z(), z()
        self.steps = 0

    def find_joints(self, names, preserve_order=False):
        return [self.joint_names.index(n) for n in names], list(names)

    @staticmethod
    def _c(joint_ids):
        return slice(None) if joint_ids is None else list(joint_ids)

    def write_joint_stiffness_to_sim(self, v, joint_ids=None, env_ids=None):
        self.stiffness[:, self._c(joint_ids)] = v

    def write_joint_damping_to_sim(self, v, joint_ids=None, env_ids=None):
        self.damping[:, self._c(joint_ids)] = v

    def write_joint_armature_to_sim(self, v, joint_ids=None, env_ids=None):
        self.armature[:, self._c(joint_ids)] = v

    def write_joint_state_to_sim(self, pos, vel, joint_ids=None, env_ids=None):
        self.data.joint_pos[:, self._c(joint_ids)] = pos
        self.data.joint_vel[:, self._c(joint_ids)] = vel

    def set_joint_position_target(self, v, joint_ids=None, env_ids=None):
        self.pos_target[:, self._c(joint_ids)] = v

    def set_joint_velocity_target(self, v, joint_ids=None, env_ids=None):
        self.vel_target[:, self._c(joint_ids)] = v

    def set_joint_effort_target(self, v, joint_ids=None, env_ids=None):
        self.effort_target[:, self._c(joint_ids)] = v

    def reset(self, env_ids=None):
        pass

    def write_data_to_sim(self):
        pass

    def update(self, dt):
        pass

    def physics_step(self, h):
        # symplectic Euler, as JointReplayBackend; the drive is a PD on the targets
        d = self.data
        tau = (self.stiffness * (self.pos_target - d.joint_pos) + self.damping * (self.vel_target - d.joint_vel)
               + self.effort_target - self.g[None, :])
        d.joint_vel = d.joint_vel + h * tau / (self.I[None, :] + self.armature)
        d.joint_pos = d.joint_pos + h * d.joint_vel
        self.steps += 1


@pytest.fixture
def fake_isaaclab(monkeypatch):
    """Install a minimal isaaclab package; returns a dict the fakes report into."""
    state = {'inertia': np.linspace(0.08, 0.02, N), 'gravity': np.zeros(N), 'robot': None, 'sim': None}

    def mod(name):
        m = types.ModuleType(name)
        m.__spec__ = importlib.machinery.ModuleSpec(name, None)
        monkeypatch.setitem(sys.modules, name, m)
        return m

    root = mod('isaaclab')
    app = mod('isaaclab.app')
    sim = mod('isaaclab.sim')
    assets = mod('isaaclab.assets')
    actuators = mod('isaaclab.actuators')
    scene = mod('isaaclab.scene')
    utils = mod('isaaclab.utils')
    root.app, root.sim, root.assets, root.actuators, root.scene, root.utils = app, sim, assets, actuators, scene, utils

    class AppLauncher:
        def __init__(self, *a, **kw):
            self.app = types.SimpleNamespace(close=lambda: None)
    app.AppLauncher = AppLauncher

    class SimulationContext:
        def __init__(self, cfg):
            self.cfg = cfg
            state['sim'] = self

        def reset(self):
            pass

        def step(self, render=True):
            state['robot'].physics_step(self.cfg.dt)

    sim.SimulationContext = SimulationContext
    sim.SimulationCfg = _Cfg
    sim.UsdFileCfg = sim.RigidBodyPropertiesCfg = sim.ArticulationRootPropertiesCfg = _Cfg

    class ArticulationCfg(_Cfg):
        InitialStateCfg = _Cfg
    assets.ArticulationCfg = ArticulationCfg
    actuators.ImplicitActuatorCfg = _Cfg

    class InteractiveSceneCfg:
        def __init__(self, num_envs=1, env_spacing=1.0, replicate_physics=True):
            self.num_envs = num_envs

    class InteractiveScene:
        def __init__(self, cfg):
            self.robot = FakeArticulation(cfg.robot, cfg.num_envs, state['inertia'], state['gravity'])
            state['robot'] = self.robot

        def __getitem__(self, key):
            return getattr(self, key)

    scene.InteractiveSceneCfg, scene.InteractiveScene = InteractiveSceneCfg, InteractiveScene
    utils.configclass = lambda cls: cls
    assert ib.isaaclab_available()
    return state


@pytest.fixture
def usd(tmp_path):
    p = tmp_path / 'arm01.usd'
    p.write_text('#usda 1.0\n')
    return str(p)


def chirp_log(j=2, T=600, amp=0.15, hold=None):
    t = np.arange(T) * DT
    f = 0.5 + 6.0 * t / t[-1]
    q_cmd = 0.3 + amp * np.sin(2 * np.pi * np.cumsum(f) * DT)
    qd_cmd = np.gradient(q_cmd, DT)
    hold = np.linspace(0.1, 0.4, N) if hold is None else hold
    hold = hold.copy()
    hold[j] = 0.3
    return JointLog(j, DT, q_cmd, qd_cmd, q_cmd.copy(), hold, (0.3, amp), (0.5, 6.5))


def population(n, seed=0):
    rng = np.random.default_rng(seed)
    P = np.zeros((n, len(PARAMS)))
    c = {p: i for i, p in enumerate(PARAMS)}
    P[:, c['armature']] = rng.uniform(0.0, 0.02, n)
    P[:, c['coulomb']] = rng.uniform(0.0, 0.05, n)
    P[:, c['viscous']] = rng.uniform(0.0, 0.3, n)
    P[:, c['encoder_bias']] = 0.0
    P[:, c['delay_s']] = rng.uniform(0.0, 0.009, n)          # fractional steps
    P[:, c['drive_kp']] = rng.uniform(20.0, 80.0, n)
    P[:, c['drive_kd']] = rng.uniform(0.5, 2.0, n)
    return P


# --------------------------------------------------------------------------- pure parts
def test_delay_buffer_reads_whole_and_fractional_steps():
    buf = ib.DelayBuffer([0, 2, 2.5, 10], width=1)
    for v in range(21):
        buf.push([v])
    assert np.allclose(buf.read()[:, 0], [20, 18, 17.5, 10])
    # before the history fills, the first command is held (np.interp clamps)
    early = ib.DelayBuffer([10, 3.5], width=1)
    for v in range(3):
        early.push([v + 5])
    assert np.allclose(early.read()[:, 0], [5, 5])
    rounded = ib.DelayBuffer([2.6, 2.4], width=1, fractional=False)
    for v in range(10):
        rounded.push([v])
    assert np.allclose(rounded.read()[:, 0], [6, 7])
    assert np.allclose(ib.DelayBuffer.steps([0.004, 0.005], 0.002), [2.0, 2.5])


def test_delay_buffer_equals_interpolated_delay():
    rng = np.random.default_rng(3)
    x = np.cumsum(rng.normal(size=200))
    t = np.arange(200) * DT
    delays = np.array([0.0, 0.0013, 0.004, 0.0071])
    buf = ib.DelayBuffer(delays / DT, width=2)
    for s in range(200):
        buf.push([x[s], -x[s]])
        got = buf.read()
        want = np.interp(t[s] - delays, t, x)
        assert np.allclose(got[:, 0], want) and np.allclose(got[:, 1], -want)


def test_env_batches_cover_population_and_pad_last_batch():
    batches = list(ib.env_batches(10, 4))
    assert [n for _, n in batches] == [4, 4, 2]
    assert all(len(r) == 4 for r, _ in batches)
    assert list(batches[2][0]) == [8, 9, 9, 9]
    real = np.concatenate([r[:n] for r, n in batches])
    assert list(real) == list(range(10))
    assert [n for _, n in ib.env_batches(3, 8)] == [3]


def test_param_tables_map_params_to_the_replayed_joint():
    P = population(5)
    kp_h, kd_h = np.arange(1.0, N + 1), np.arange(N) * 0.1
    tab = ib.param_tables(P, 4, kp_h, kd_h)
    c = {p: i for i, p in enumerate(PARAMS)}
    for e in range(5):
        others = [i for i in range(N) if i != 4]
        assert np.allclose(tab['stiffness'][e, others], kp_h[others])
        assert np.allclose(tab['damping'][e, others], kd_h[others])
        assert np.allclose(tab['armature'][e, others], 0.0)
        assert tab['stiffness'][e, 4] == P[e, c['drive_kp']]
        assert tab['damping'][e, 4] == pytest.approx(P[e, c['drive_kd']] + P[e, c['viscous']])
        assert tab['armature'][e, 4] == P[e, c['armature']]
    assert np.allclose(tab['bias'], P[:, c['encoder_bias']])
    assert np.allclose(tab['delay_s'], P[:, c['delay_s']])
    with pytest.raises(AssertionError):
        ib.param_tables(P[:, :5], 0, kp_h, kd_h)


def test_coulomb_eps_widens_only_when_an_explicit_step_would_chatter():
    assert ib.coulomb_eps(np.array([0.05]), np.array([0.05]), DT)[0] == ib.COULOMB_EPS
    assert ib.coulomb_eps(np.array([2.0]), np.array([0.001]), DT)[0] == pytest.approx(2 * 2.0 * DT / 0.001)


# --------------------------------------------------------------------------- backend on the fake
def test_missing_isaaclab_gives_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, 'isaaclab', None)
    with pytest.raises(ImportError, match='isaaclab'):
        IsaacLabBackend(FakeModel(), [chirp_log()])


def test_stub_in_backend_returns_the_isaac_backend(fake_isaaclab, usd):
    be = IsaacLabBackend(FakeModel(), [chirp_log()], num_envs=3, usd_path=usd,
                         hold_gains=(np.full(N, 50.0), np.full(N, 1.0)))
    assert isinstance(be, ib.IsaacLabBackend)
    assert be.sim_dt == DT
    # config order -> fake DOF order
    assert [FakeArticulation.ORDER[i] for i in be.jidx] == NAMES


def test_fake_replay_matches_joint_replay(fake_isaaclab, usd):
    model = FakeModel()
    logs = [chirp_log(j=2), chirp_log(j=5, T=400)]
    P = population(7, seed=1)
    ref = JointReplayBackend(model, logs)
    be = ib.IsaacLabBackend(model, logs, num_envs=3, usd_path=usd, hold_gains=(np.full(N, 50.0), np.full(N, 1.0)))
    for k in range(2):
        want = ref.simulate(k, P)
        got = be.simulate(k, P)
        assert got.shape == want.shape == (7, len(logs[k].q_cmd))
        assert np.allclose(got, want, atol=1e-9), np.abs(got - want).max()
    # rows are independent of how they were batched
    alone = be.simulate(0, P[4:5])
    assert np.allclose(alone[0], be.simulate(0, P)[4])


def test_bias_offsets_target_and_reading_and_others_hold(fake_isaaclab, usd):
    T, c, j = 1500, 0.4, 3
    hold = np.linspace(-0.2, 0.5, N)
    log = JointLog(j, DT, np.full(T, c), np.zeros(T), np.full(T, c), hold.copy(), (c, 0.0), (0.0, 0.0))
    model = FakeModel(g=np.linspace(0.5, -0.5, N))
    fake_isaaclab['gravity'] = model.g          # the sim's gravity, which the host feedforward cancels
    kp_h = np.arange(10.0, 10.0 + N)
    be = ib.IsaacLabBackend(model, [log], num_envs=2, usd_path=usd, hold_gains=(kp_h, np.full(N, 1.0)))
    P = population(2)
    P[:, PARAMS.index('coulomb')] = 0.0
    P[:, PARAMS.index('encoder_bias')] = [0.01, -0.02]
    out = be.simulate(0, P)
    robot = fake_isaaclab['robot']
    idx = be.jidx
    q_true = robot.data.joint_pos[:, idx]
    # the drive servoes the biased reading to the command: true angle = c - bias, reading = c
    assert np.allclose(q_true[:, j], c - P[:, PARAMS.index('encoder_bias')], atol=1e-4)
    assert np.allclose(out[:, -1], c, atol=1e-4)
    assert np.allclose(robot.pos_target[:, idx][:, j], c - P[:, PARAMS.index('encoder_bias')])
    others = [i for i in range(N) if i != j]
    assert np.allclose(robot.pos_target[:, idx][:, others], hold[others])
    assert np.allclose(robot.stiffness[:, idx][:, others], kp_h[others])
    assert np.allclose(robot.stiffness[:, idx][:, j], P[:, PARAMS.index('drive_kp')])
    # host feedforward (gravity at the pose) reaches every joint
    assert np.allclose(robot.effort_target[:, idx], model.g[None, :])
    assert robot.steps == T     # one physics step per log sample at sim_dt = log dt


def test_substeps_when_log_is_slower_than_sim(fake_isaaclab, usd):
    log = chirp_log(T=200)
    log2 = JointLog(log.joint, 2 * DT, log.q_cmd, log.qd_cmd, log.q_meas, log.q_hold, log.sweep, log.freq)
    be = ib.IsaacLabBackend(FakeModel(), [log2], sim_dt=DT, num_envs=2, usd_path=usd,
                            hold_gains=(np.full(N, 50.0), np.full(N, 1.0)))
    out = be.simulate(0, population(2))
    assert out.shape == (2, 200) and np.all(np.isfinite(out))
    assert fake_isaaclab['robot'].steps == 400
    with pytest.raises(ValueError, match='whole number'):
        ib.ReplayPlan(FakeModel(), log2, 0.0015)


def test_replays_exported_npz_directly(fake_isaaclab, usd, tmp_path):
    T = 300
    rng = np.random.default_rng(0)
    command = np.tile(np.linspace(0.1, 0.4, N), (T, 1))
    command[:, 5] += 0.1 * np.sin(np.arange(T) * 0.05)
    position = command + 1e-3 * rng.normal(size=command.shape)
    rng_ = np.stack([command.min(axis=0), command.max(axis=0)], axis=1)
    path = str(tmp_path / 'pace.npz')
    np.savez_compressed(path, t=np.arange(T) * DT, command=command, position=position, velocity=0 * command,
                        torque=0 * command, sample_rate_hz=1.0 / DT, sweep_range_rad=rng_,
                        amplitude_rad=0.5 * (rng_[:, 1] - rng_[:, 0]))
    with open(str(tmp_path / 'pace.json'), 'w') as f:
        json.dump({'joints': NAMES}, f)
    log = ib.load_pace_npz(path)
    assert log.joint == 5 and log.dt == pytest.approx(DT)
    assert np.array_equal(log.q_cmd, command[:, 5]) and np.array_equal(log.q_meas, position[:, 5])
    assert ib.load_pace_npz(path, joint='wrist_pitch').joint == 5
    assert np.allclose(np.delete(log.q_hold, 5), np.delete(command[0], 5))
    be = ib.IsaacLabBackend(FakeModel(), [path], num_envs=4, usd_path=usd,
                            hold_gains=(np.full(N, 50.0), np.full(N, 1.0)))
    assert be.simulate(0, population(4)).shape == (4, T)


def test_pace_fit_selects_the_isaac_backend(fake_isaaclab, usd, monkeypatch):
    from arm_stack.calib import pace
    from arm_stack.config import load_arm
    from arm_stack.pace import isaac_cfg
    monkeypatch.setattr(isaac_cfg, 'USD_PATH', usd)
    arm = load_arm()
    fits = pace.fit(FakeModel(), arm, [chirp_log(T=200)], generations=2, popsize=6, backend='isaac',
                    drive_gains=(np.full(N, 50.0), np.full(N, 1.0)))
    assert len(fits) == 1 and np.isfinite(fits[0]['cost'])
    assert fake_isaaclab['robot'].steps > 0


# --------------------------------------------------------------------------- URDF prep, real sim
def test_prepared_urdf_fixes_mimic_joints_and_keeps_mass(tmp_path):
    import xml.etree.ElementTree as ET
    out = prepare_urdf_for_isaac(URDF_PATH, str(tmp_path / 'arm01_isaac.urdf'))
    root = ET.parse(out).getroot()
    joints = {j.get('name'): j for j in root.findall('joint')}
    for n in MIMIC_JOINTS:
        assert joints[n].get('type') == 'fixed' and joints[n].find('mimic') is None
    assert root.find('.//mimic') is None
    for mesh in root.iter('mesh'):
        assert os.path.isfile(mesh.get('filename')), mesh.get('filename')
    pin = pytest.importorskip('pinocchio')
    a, b = pin.buildModelFromUrdf(URDF_PATH), pin.buildModelFromUrdf(out)
    assert b.nq == N and [n for n in b.names[1:]] == NAMES
    mass = lambda m: sum(m.inertias[i].mass for i in range(m.njoints))   # noqa: E731
    assert mass(a) == pytest.approx(mass(b))


@pytest.mark.skipif(not ib.isaaclab_available(), reason='isaaclab not importable')
def test_real_isaac_lab_replay_smoke():
    from arm_stack.model.robot import ArmModel
    from arm_stack.pace.isaac_cfg import USD_PATH
    if not os.path.exists(USD_PATH):
        pytest.skip('no USD; run tools/isaac/import_urdf.py')
    model = ArmModel(URDF_PATH, NAMES)
    log = chirp_log(j=5, T=250, hold=np.clip(np.zeros(N), model.lower, model.upper))
    be = IsaacLabBackend(model, [log], num_envs=2)
    out = be.simulate(0, population(3))
    assert out.shape == (3, 250) and np.all(np.isfinite(out))
    assert np.max(np.abs(out - log.q_cmd[None, :])) < 0.1
