"""Damped-least-squares IK for one G1 arm (7 joints) to a TCP pose (position + orientation).

Kinematics-only (mj_kinematics + mj_comPos on a scratch MjData), joint-limit clamping, and a
null-space pull toward a rest posture so the elbow stays in a natural place.
"""
import mujoco
import numpy as np

ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")


def quat_to_mat(q):
    """Rotation matrix of a quaternion (w, x, y, z)."""
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, float))
    return m.reshape(3, 3)


def mat_to_quat(R):
    """Quaternion (w, x, y, z) of a rotation matrix."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    return q


def rot_error(R_target, R_cur):
    """Axis-angle error vector (world frame) rotating R_cur onto R_target."""
    dR = R_target @ R_cur.T
    q = mat_to_quat(dR)
    if q[0] < 0:
        q = -q
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


class ArmIK:
    """Damped least-squares IK for one 7-joint arm."""
    def __init__(self, model, side, prefix="g1/"):
        """Joint and site addresses for one arm."""
        self.m = model
        self.d = mujoco.MjData(model)
        self.side = side
        self.jids = [model.joint(f"{prefix}{side}_{j}_joint").id for j in ARM]
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.jids])
        self.dadr = np.array([model.jnt_dofadr[j] for j in self.jids])
        self.lo = model.jnt_range[self.jids, 0] + 0.02
        self.hi = model.jnt_range[self.jids, 1] - 0.02
        self.site = model.site(f"{prefix}tcp_{side}").id
        self.rest = None

    def solve(self, qpos_full, pos, R, q_init=None, iters=60, rot_w=0.35, lam=0.03, tol=1e-3):
        """Joint angles that put the tool at (pos, R), starting from `q_init`."""
        d = self.d
        d.qpos[:] = qpos_full
        if q_init is not None:
            d.qpos[self.qadr] = q_init
        jp = np.zeros((3, self.m.nv))
        jr = np.zeros((3, self.m.nv))
        rest = self.rest if self.rest is not None else d.qpos[self.qadr].copy()
        err_n = np.inf
        for _ in range(iters):
            mujoco.mj_kinematics(self.m, d)
            mujoco.mj_comPos(self.m, d)
            p = d.site_xpos[self.site]
            Rc = d.site_xmat[self.site].reshape(3, 3)
            e = np.concatenate([pos - p, rot_w * rot_error(R, Rc)])
            err_n = np.linalg.norm(e[:3]) + np.linalg.norm(e[3:]) / max(rot_w, 1e-6) * 0.05
            if np.linalg.norm(e[:3]) < tol and np.linalg.norm(e[3:]) < rot_w * 0.02:
                break
            mujoco.mj_jacSite(self.m, d, jp, jr, self.site)
            J = np.vstack([jp[:, self.dadr], rot_w * jr[:, self.dadr]])
            JJt = J @ J.T + lam ** 2 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            # null-space pull to the rest posture
            N = np.eye(7) - np.linalg.pinv(J) @ J
            dq += N @ (0.1 * (rest - d.qpos[self.qadr]))
            d.qpos[self.qadr] = np.clip(d.qpos[self.qadr] + np.clip(dq, -0.25, 0.25), self.lo, self.hi)
        mujoco.mj_kinematics(self.m, d)
        pos_err = float(np.linalg.norm(pos - d.site_xpos[self.site]))
        rot_err = float(np.linalg.norm(rot_error(R, d.site_xmat[self.site].reshape(3, 3))))
        return d.qpos[self.qadr].copy(), pos_err, rot_err

    def solve_robust(self, qpos_full, pos, R, q_init=None, restarts=6, tol=0.008, seed=0, iters=150):
        """solve() from q_init, then from the rest posture, then from random postures (escapes local minima)."""
        rng = np.random.default_rng(seed)
        best = None
        inits = [q_init, self.rest] + [rng.uniform(self.lo, self.hi) * 0.6 + (self.rest if self.rest is not None else 0) * 0.4
                                        for _ in range(restarts)]
        for qi in inits:
            if qi is None:
                continue
            q, pe, re = self.solve(qpos_full, pos, R, q_init=qi, iters=iters)
            if best is None or pe + 0.05 * re < best[1] + 0.05 * best[2]:
                best = (q, pe, re)
            if pe < tol and re < 0.05:
                break
        return best


def solve_feasible(ik, qpos_full, pos, R, ok_fn, q_init=None, restarts=12, tol=0.008, seed=0, iters=150):
    """Like solve_robust, but keeps only solutions that also pass ok_fn(ik.d) (e.g. a collision check).
    Returns (q, pos_err, rot_err) of the first feasible solution, or None."""
    rng = np.random.default_rng(seed)
    inits = [q_init, ik.rest] + [rng.uniform(ik.lo, ik.hi) * 0.7 + (ik.rest if ik.rest is not None else 0) * 0.3
                                  for _ in range(restarts)]
    for qi in inits:
        if qi is None:
            continue
        q, pe, re = ik.solve(qpos_full, pos, R, q_init=qi, iters=iters)
        if pe < tol and re < 0.05:
            ik.d.qpos[ik.qadr] = q
            if ok_fn(ik.d):
                return q, pe, re
    return None
