"""Rigid-body model of the arm: URDF load, Pinocchio FK, gravity, regressors.

Calibration results layer on top of the supplied URDF without editing it:
  * link_inertials (`arm gravity`): per-link mass and center of mass,
  * kinematic_overlay (mocap-calib): small per-joint origin translations and
    axis-tilt rotations,
  * friction (`arm friction`): Coulomb + viscous per joint.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence

import numpy as np
import pinocchio as pin

GRAVITY = np.array([0.0, 0.0, -9.81])


class ArmModel:

    """Pinocchio model of the arm: kinematics, gravity, mass matrix, friction, with calibration applied."""
    def __init__(self, urdf_path: str, joint_names: Sequence[str], tool_frame: str = 'tool0',
                 home: Optional[Sequence[float]] = None):
        """Build the model from the URDF and keep the named joints (others fixed)."""
        self.urdf_path = urdf_path
        # the arm's resting pose (joint config `home_pose_rad`); None: q = 0
        self.home = None if home is None else np.asarray(home, float)
        full = pin.buildModelFromUrdf(urdf_path)
        self.joint_names = list(joint_names)
        missing = [n for n in self.joint_names if not full.existJointName(n)]
        if missing:
            raise ValueError('joints not in %s: %s' % (urdf_path, missing))
        # Passive / mimic joints (the wrist parallelogram's crank and rod) are
        # locked at zero and their bodies merged into the parent link. The
        # parallelogram keeps the rod's orientation fixed, so this only moves
        # ~30 g by a few mm as the wrist pitches.
        lock = [full.getJointId(n) for n in full.names[1:] if n not in self.joint_names]
        self.locked_joints = [full.names[j] for j in lock]
        self.nominal = pin.buildReducedModel(full, lock, pin.neutral(full)) if lock else full
        self.nominal.gravity.linear = GRAVITY
        # Pinocchio orders joints by tree depth; map to config order.
        self.jid = [self.nominal.getJointId(n) for n in self.joint_names]
        self.qidx = [self.nominal.joints[j].idx_q for j in self.jid]
        self.vidx = [self.nominal.joints[j].idx_v for j in self.jid]
        if self.nominal.nq != len(self.joint_names):
            raise ValueError('model has %d dofs, config lists %d' % (self.nominal.nq, len(joint_names)))
        self.tool_frame = tool_frame
        self.lower = np.array([self.nominal.lowerPositionLimit[i] for i in self.qidx])
        self.upper = np.array([self.nominal.upperPositionLimit[i] for i in self.qidx])
        self.effort_limit = np.array([self.nominal.effortLimit[i] for i in self.vidx])
        self.velocity_limit = np.array([self.nominal.velocityLimit[i] for i in self.vidx])
        self.coulomb = np.zeros(len(self.joint_names))
        self.viscous = np.zeros(len(self.joint_names))
        # a copy: calibration layers edit self.model and read the nominal values,
        # so applying a calibration twice must not compound
        self.model = self.nominal.copy()
        self.data = self.model.createData()
        self.overlay = None

    @property
    def n(self) -> int:
        """Number of arm joints."""
        return len(self.joint_names)

    # ------------------------------------------------------------------ conversion
    def _q(self, q: Sequence[float]) -> np.ndarray:
        """Embed arm joint angles in the full configuration."""
        out = np.zeros(self.model.nq)
        out[self.qidx] = q
        return out

    def _v(self, v: Sequence[float]) -> np.ndarray:
        """Embed arm joint velocities in the full velocity vector."""
        out = np.zeros(self.model.nv)
        out[self.vidx] = v
        return out

    def _from_v(self, v: np.ndarray) -> np.ndarray:
        """Arm entries of a full velocity-sized vector."""
        return np.asarray(v)[self.vidx]

    def copy(self) -> 'ArmModel':
        """Independent copy (model, data and friction)."""
        other = copy.copy(self)
        other.model = self.model.copy()
        other.data = other.model.createData()
        other.coulomb = self.coulomb.copy()
        other.viscous = self.viscous.copy()
        return other

    # ------------------------------------------------------------------ calibration layers
    def link_names(self) -> List[str]:
        """Child link of each configured joint, in config order."""
        names = []
        for j in self.jid:
            child = [f for f in self.model.frames
                     if f.parentJoint == j and f.type == pin.FrameType.BODY]
            names.append(child[0].name if child else self.model.names[j])
        return names

    def apply_link_inertials(self, inertials: Dict[str, Dict], masses: Optional[Dict] = None):
        """Set mass/COM per link (COM in the child-link frame of its joint)."""
        masses = masses or {}
        for link, j in zip(self.link_names(), self.jid):
            entry = inertials.get(link) or {}
            old = self.model.inertias[j]
            mass = entry.get('mass_kg') or masses.get(link) or old.mass
            com = entry.get('com_m')
            com = np.array(com, float) if com is not None and None not in com else old.lever
            # keep the rotational inertia about the COM, rescaled with mass
            I_c = old.inertia * (mass / old.mass if old.mass > 0 else 1.0)
            self.model.inertias[j] = pin.Inertia(float(mass), com, I_c)
        self.data = self.model.createData()

    def apply_friction(self, friction: Dict[str, Dict]):
        """Set Coulomb and viscous friction from a calibration's friction table."""
        for i, n in enumerate(self.joint_names):
            f = friction.get(n) or {}
            self.coulomb[i] = f.get('coulomb_nm') or 0.0
            self.viscous[i] = f.get('viscous_nm_s_per_rad') or 0.0

    def apply_overlay(self, overlay: Optional[Dict]):
        """overlay: {joint: {'dxyz': [3], 'drot': [3] (rotation vector)}} on joint placements."""
        self.overlay = overlay
        if not overlay:
            return
        for n, j in zip(self.joint_names, self.jid):
            o = overlay.get(n)
            if not o:
                continue
            M = self.nominal.jointPlacements[j]
            dR = pin.exp3(np.array(o.get('drot', [0, 0, 0]), float))
            self.model.jointPlacements[j] = pin.SE3(M.rotation @ dR,
                                                     M.translation + np.array(o.get('dxyz', [0, 0, 0])))
        self.data = self.model.createData()

    def apply_calibration(self, calib: Dict, masses: Optional[Dict] = None):
        """Apply link inertials and masses, friction and the kinematic overlay from a calibration."""
        if calib.get('link_inertials') or masses:
            self.apply_link_inertials(calib.get('link_inertials') or {}, masses)
        if calib.get('friction'):
            self.apply_friction(calib['friction'])
        if calib.get('kinematic_overlay'):
            self.apply_overlay(calib['kinematic_overlay'])

    # ------------------------------------------------------------------ kinematics
    def axis_in_child(self, k: int) -> np.ndarray:
        """Unit rotation axis of configured joint k, in its own (child link) frame."""
        d = self.model.createData()
        pin.forwardKinematics(self.model, d, pin.neutral(self.model))
        # motion subspace of a 1-dof revolute joint: its angular part is the axis
        # (pinocchio returns it as (6,) or (6, 1) depending on the joint type)
        S = np.array(d.joints[self.jid[k]].S).reshape(6, -1)
        a = S[3:, 0]
        return a / np.linalg.norm(a)

    def fk(self, q: Sequence[float], frame: Optional[str] = None) -> pin.SE3:
        """Pose of `frame` (default the tool) at `q`."""
        frame = frame or self.tool_frame
        pin.framesForwardKinematics(self.model, self.data, self._q(q))
        return self.data.oMf[self.model.getFrameId(frame)].copy()

    def tool_position(self, q, frame=None) -> np.ndarray:
        """Position of `frame` (default the tool)."""
        return self.fk(q, frame).translation.copy()

    def jacobian(self, q, frame=None) -> np.ndarray:
        """6xn world-aligned Jacobian of `frame`, columns in config order."""
        frame = frame or self.tool_frame
        qq = self._q(q)
        pin.computeJointJacobians(self.model, self.data, qq)
        pin.framesForwardKinematics(self.model, self.data, qq)
        J = pin.getFrameJacobian(self.model, self.data, self.model.getFrameId(frame),
                                 pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        # eigenpy hands back 1-D arrays for some 1-dof shapes; keep (6, nv)
        return np.asarray(J).reshape(6, self.model.nv)[:, self.vidx]

    def ik_position(self, target: np.ndarray, q0: Sequence[float], frame=None, iters=200,
                    tol=1e-6, stay_inside=True) -> np.ndarray:
        """Damped least squares position IK (used by sim procedures)."""
        q = np.array(q0, float)
        for _ in range(iters):
            err = target - self.tool_position(q, frame)
            if np.linalg.norm(err) < tol:
                break
            J = self.jacobian(q, frame)[:3]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), err)
            q = q + dq
            if stay_inside:
                q = np.clip(q, self.lower, self.upper)
        return q

    def ik(self, target, q_seed: Sequence[float], method: str = 'clik', **opts):
        """6-DOF (or masked) IK of the tool frame; thin wrapper over arm_stack.model.ik.solve.

        target: pin.SE3, a 4x4 matrix, or a 3-vector (position only).
        method: 'clik' (default, null-space posture), 'lm', 'dls' or 'multistart'.
        Returns an ik.IKResult (q, success, pos_err, rot_err, iters, time); q is
        always inside the joint limits.
        """
        from . import ik as _ik
        return _ik.solve(self, target, q_seed, method=method, **opts)

    # ------------------------------------------------------------------ dynamics
    def gravity(self, q: Sequence[float]) -> np.ndarray:
        """Gravity torque at `q`."""
        return self._from_v(pin.computeGeneralizedGravity(self.model, self.data, self._q(q)))

    def rnea(self, q, qd, qdd) -> np.ndarray:
        """Inverse dynamics torque."""
        return self._from_v(pin.rnea(self.model, self.data, self._q(q), self._v(qd), self._v(qdd)))

    def friction(self, qd: Sequence[float], eps: float = 0.02) -> np.ndarray:
        """Friction torque (smooth Coulomb + viscous)."""
        qd = np.asarray(qd, float)
        return self.coulomb * np.tanh(qd / eps) + self.viscous * qd

    def aba(self, q, qd, tau) -> np.ndarray:
        """Forward dynamics: joint accelerations."""
        return self._from_v(pin.aba(self.model, self.data, self._q(q), self._v(qd), self._v(tau)))

    def mass_matrix(self, q) -> np.ndarray:
        """Joint-space mass matrix at `q`."""
        M = np.asarray(pin.crba(self.model, self.data, self._q(q))).reshape(self.model.nv, self.model.nv)
        M = np.triu(M) + np.triu(M, 1).T
        return M[np.ix_(self.vidx, self.vidx)]

    def gravity_first_moment_regressor(self, q) -> np.ndarray:
        """Static joint torques are linear in each link's first moment m*c.

        Returns Y (n x 3n) with tau_g = Y @ concat(m_k * c_k) + tau_g(masses only at
        their joint origins). Columns for link k are the torque from a unit first
        moment along x, y, z of that link's frame.
        """
        qq = self._q(q)
        Yfull = pin.computeJointTorqueRegressor(self.model, self.data, qq, np.zeros(self.model.nv),
                                                np.zeros(self.model.nv))
        Yfull = np.asarray(Yfull).reshape(self.model.nv, -1)    # 1-dof: eigenpy returns 1-D
        # pinocchio orders dynamic parameters per joint as
        # [m, mcx, mcy, mcz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz]
        Y = np.zeros((self.n, 3 * self.n))
        Y0 = np.zeros((self.n, self.n))
        for k, j in enumerate(self.jid):
            base = 10 * (j - 1)
            Y[:, 3 * k:3 * k + 3] = Yfull[np.ix_(self.vidx, range(base + 1, base + 4))]
            Y0[:, k] = Yfull[self.vidx, base]
        return Y, Y0

    def first_moments(self) -> np.ndarray:
        """Per-link mass x COM vectors, concatenated."""
        out = []
        for j in self.jid:
            I = self.model.inertias[j]
            out.extend(I.mass * I.lever)
        return np.array(out)

    def masses(self) -> np.ndarray:
        """Link masses."""
        return np.array([self.model.inertias[j].mass for j in self.jid])

    def set_first_moments(self, h: np.ndarray):
        """Set link COMs from first moments (masses unchanged)."""
        for k, j in enumerate(self.jid):
            I = self.model.inertias[j]
            com = np.asarray(h[3 * k:3 * k + 3]) / I.mass
            self.model.inertias[j] = pin.Inertia(I.mass, com, I.inertia)
        self.data = self.model.createData()

    def peak_gravity_torque(self, samples: int = 500, seed: int = 0) -> np.ndarray:
        """Largest gravity torque per joint over random configurations."""
        rng = np.random.default_rng(seed)
        peak = np.zeros(self.n)
        for _ in range(samples):
            q = rng.uniform(self.lower, self.upper)
            peak = np.maximum(peak, np.abs(self.gravity(q)))
        return peak
