"""Plain-MuJoCo copy of what mjlab will simulate, for checks on the Mac (mjlab itself needs
Python < 3.14 and a GPU, neither available here).

Copied from mjlab 1.6.0 (utils/spec.py create_position_actuator, G1 FULL_COLLISION, velocity
task sim settings):
  - one <position> actuator per policy joint: force = kp*(target - q) - kd*qdot, clamped to
    +/- effort; targets NOT clamped to the joint range; joint armature set from the config
  - collisions: every *_collision geom contype=conaffinity=1; feet condim 3, friction 0.6,
    priority 1; everything else condim 1 (self-collision on)
  - timestep 0.005 s, implicitfast, Newton solver, 10 iterations, 20 line-search iterations,
    policy at 50 Hz (decimation 4)

Two actuator models (set ACTUATOR / DELAY_STEPS below, or pass to Sim):
  "position": mjlab BuiltinPositionActuatorCfg (MuJoCo <position>, PD inside the solver)
  "dc":       mjlab DcMotorActuatorCfg (PD torque computed every physics step, clipped to the motor's
              torque-speed line: min(peak, saturation * (1 - |w| / velocity_limit)), applied by a <motor>)
DELAY_STEPS: command delay in physics steps (5 ms each), like mjlab's delay_min_lag / delay_max_lag.
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
import robot_params as P  # noqa: E402

SIM_DT = 0.005
DECIMATION = 4
FOOT_FRICTION = 0.6
ACTUATOR = "dc"            # what the mjlab task uses (robot_cfg.py)
DELAY_STEPS = 0


def make_spec(fixed_base=False, base_height=None, floor=True, actuator="position"):
    spec = mujoco.MjSpec.from_file(str(P.TRAIN_XML))
    spec.option.timestep = SIM_DT
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    spec.option.iterations = 10
    spec.option.ls_iterations = 20

    for g in spec.geoms:
        if not g.name.endswith("_collision"):
            continue
        g.contype, g.conaffinity = 1, 1
        if g.name in P.FOOT_GEOMS:
            g.condim, g.priority = 3, 1
            g.friction = [FOOT_FRICTION, g.friction[1], g.friction[2]]
        else:
            g.condim, g.priority = 1, 0

    params = P.joint_params()
    for name, p in params.items():
        a = spec.add_actuator(name=name, target=name)
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.dyntype = mujoco.mjtDyn.mjDYN_NONE
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        if actuator == "position":
            a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            a.gainprm[0] = p["kp"]
            a.biasprm[1] = -p["kp"]
            a.biasprm[2] = -p["kd"]
        else:                                     # motor: ctrl = torque, computed in Sim.step_policy
            a.biastype = mujoco.mjtBias.mjBIAS_NONE
            a.gainprm[0] = 1.0
        a.inheritrange = 0.0
        a.ctrllimited = False
        a.forcelimited = True
        a.forcerange = [-p["effort"], p["effort"]]
        spec.joint(name).armature = p["armature"]

    if floor:
        spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
                                contype=1, conaffinity=1, condim=3, friction=[1.0, 0.005, 0.0001],
                                rgba=[0.5, 0.55, 0.6, 1])
        spec.worldbody.add_light(pos=[1, -1, 3], dir=[-0.3, 0.3, -1], diffuse=[0.9, 0.9, 0.9])
    if fixed_base:
        spec.delete(spec.joint("floating_base"))
        spec.body("pelvis").pos = [0, 0, base_height if base_height is not None else 1.2]
    return spec, params


class Sim:
    """Robot with mjlab-style actuators. ctrl is the joint target; policy action 0 = default pose."""

    def __init__(self, fixed_base=False, base_height=None, floor=True, actuator=None, delay_steps=None):
        self.actuator = actuator or ACTUATOR
        self.delay = DELAY_STEPS if delay_steps is None else delay_steps
        self.spec, self.params = make_spec(fixed_base, base_height, floor, self.actuator)
        self.m = self.spec.compile()
        self.d = mujoco.MjData(self.m)
        self.fixed = fixed_base
        self.names = [self.m.actuator(i).name for i in range(self.m.nu)]
        self.qadr = np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in self.names])
        self.vadr = np.array([self.m.jnt_dofadr[self.m.joint(n).id] for n in self.names])
        self.default = np.array([self.params[n]["default"] for n in self.names])
        self.scale = np.array([self.params[n]["action_scale"] for n in self.names])
        self.effort = np.array([self.params[n]["effort"] for n in self.names])
        self.rated = np.array([self.params[n]["rated"] for n in self.names])
        self.kp = np.array([self.params[n]["kp"] for n in self.names])
        self.kd = np.array([self.params[n]["kd"] for n in self.names])
        self.sat = np.array([self.params[n]["dc_saturation"] for n in self.names])
        self.vlim = np.array([self.params[n]["dc_velocity_limit"] for n in self.names])
        self.buf = []
        info = P.HERE / "train_model_info.json"
        import json
        self.pelvis_z = json.loads(info.read_text())["default_pelvis_height"]

    def reset(self, z_offset=0.002, quat=(1, 0, 0, 0), q=None):
        mujoco.mj_resetData(self.m, self.d)
        if not self.fixed:
            self.d.qpos[2] = self.pelvis_z + z_offset
            self.d.qpos[3:7] = quat
        self.d.qpos[self.qadr] = self.default if q is None else q
        self.d.ctrl[:] = self.default if self.actuator == "position" else 0.0
        self.buf = [self.default.copy()] * (self.delay + 1)
        mujoco.mj_forward(self.m, self.d)

    def dc_torque(self, target):
        """mjlab DcMotorActuator control law: PD, then the torque-speed clip (dc_motor_clip)."""
        q, v = self.d.qpos[self.qadr], self.d.qvel[self.vadr]
        tau = self.kp * (target - q) - self.kd * v
        v_corner = self.vlim * (1 + self.effort / self.sat)
        vc = np.clip(v, -v_corner, v_corner)
        hi = np.minimum(self.sat * (1 - vc / self.vlim), self.effort)
        lo = np.maximum(self.sat * (-1 - vc / self.vlim), -self.effort)
        return np.clip(tau, lo, hi)

    def step_policy(self, action=None, target=None):
        """One 50 Hz policy step: target = default + scale*action (or an explicit target)."""
        if target is None:
            target = self.default + (0 if action is None else self.scale * action)
        for _ in range(DECIMATION):
            self.physics_step(target)

    def physics_step(self, target):
        """One 5 ms physics step toward `target` (with command delay and the chosen actuator model)."""
        self.buf = self.buf[1:] + [np.array(target, dtype=float)]
        cmd = self.buf[0]                         # target from `delay` physics steps ago
        self.d.ctrl[:] = cmd if self.actuator == "position" else self.dc_torque(cmd)
        mujoco.mj_step(self.m, self.d)

    def motors_off(self):
        """Zero stiffness and damping (limp robot)."""
        self.m.actuator_gainprm[:, 0] = 0 if self.actuator == "position" else 1.0
        self.m.actuator_biasprm[:, 1:3] = 0
        self.kp[:] = 0; self.kd[:] = 0

    def tilt_deg(self):
        R = self.d.xmat[self.m.body("pelvis").id].reshape(3, 3)
        return float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))

    def joint_q(self):
        return self.d.qpos[self.qadr].copy()

    def torque(self):
        return self.d.actuator_force.copy()

    def self_contacts(self):
        """Contacts between two robot bodies (not the floor), with penetration depth."""
        out = []
        for c in self.d.contact[: self.d.ncon]:
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            if b1 == 0 or b2 == 0:
                continue
            out.append((self.m.body(b1).name, self.m.body(b2).name, float(c.dist)))
        return out

    def foot_forces(self):
        """Vertical floor force on each foot (N)."""
        f = {n: 0.0 for n in P.FOOT_BODIES}
        buf = np.zeros(6)
        for i, c in enumerate(self.d.contact[: self.d.ncon]):
            bodies = {self.m.body(self.m.geom_bodyid[c.geom1]).name, self.m.body(self.m.geom_bodyid[c.geom2]).name}
            for n in P.FOOT_BODIES:
                if n in bodies and "world" in bodies:
                    mujoco.mj_contactForce(self.m, self.d, i, buf)
                    f[n] += abs(buf[0])            # normal force (contact frame x)
        return f
