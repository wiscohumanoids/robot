"""Kinematic-fit pieces shared by `arm mocap-calib` and `arm plate`.

Both stages identify the same joint-side parameters from measured tool poses:
joint zero offsets, the kinematic overlay (per-joint origin translation and
axis-tilt rotation increments on the current calibration), joint compliance
and backlash. They differ in what is measured (a full marker-cluster pose in
the mocap frame; a probe-tip position in a divot of a plate) and in the extra
transforms that tie the measurement to the arm (base-to-mocap and
marker-to-flange; plate pose and probe-tip offset).

A fit object is a parameter vector made of named blocks (`sl`, `labels`,
`scale`) over a copy of the model whose joint placements take the overlay
block. `select_parameters` keeps an independent, well scaled subset in a
priority order; `solve` runs Levenberg-Marquardt on that subset.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pinocchio as pin

from ..pace.lm import levenberg_marquardt

BACKLASH_EPS = 0.05           # Nm, the tanh width of the backlash state
OVERLAY_KEYS = ('dx', 'dy', 'dz', 'rx', 'ry', 'rz')


def mean_rotation(R) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.mean(R, axis=0))
    if np.linalg.det(U @ Vt) < 0:
        U[:, -1] *= -1
    return U @ Vt


def kabsch(A: np.ndarray, B: np.ndarray) -> pin.SE3:
    """Rigid transform T with B ~ T(A), points as rows (least squares)."""
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return pin.SE3(R, cb - R @ ca)


def se3(tw) -> pin.SE3:
    """Twist-like increment (rotation vector[3], translation[3]) as an SE3."""
    return pin.SE3(pin.exp3(np.asarray(tw[:3], float)), np.asarray(tw[3:], float))


def joint_blocks(names: Sequence[str]):
    """The joint-side blocks, in the order mocap-calib has always used."""
    n = len(names)
    return {
        'dzero': (['dzero:' + j for j in names], np.full(n, 1e-3)),
        'overlay': (['%s:%s' % (k, j) for j in names for k in OVERLAY_KEYS], np.full(6 * n, 1e-3)),
        'compliance': (['compliance:' + j for j in names], np.full(n, 2e-4)),
        'backlash': (['backlash:' + j for j in names], np.full(n, 1e-3)),
    }


class KinematicModel:
    """Parameter vector of named blocks over a model copy (all increments on
    the current calibration, except compliance and backlash, which are
    absolute). Subclasses define `residuals(x, *data)`."""

    def __init__(self, model, frame: str, blocks: Sequence[Tuple[str, Sequence[str], np.ndarray]]):
        self.model = model.copy()
        self.n = model.n
        self.frame = frame if model.model.existFrame(frame) else model.tool_frame
        self.fid = self.model.model.getFrameId(self.frame)
        self.jid = list(self.model.jid)
        self.placements0 = [self.model.model.jointPlacements[j].copy() for j in self.jid]
        self.sl: Dict[str, slice] = {}
        self.labels = []
        scales = []
        i = 0
        for key, labels, scale in blocks:
            self.sl[key] = slice(i, i + len(labels))
            self.labels += list(labels)
            scales.append(np.asarray(scale, float))
            i += len(labels)
        self.size = i
        # typical magnitudes, for scaling the Jacobian columns
        self.scale = np.concatenate(scales)

    def set_geometry(self, x):
        ov = x[self.sl['overlay']].reshape(self.n, 6)
        for k, j in enumerate(self.jid):
            M0 = self.placements0[k]
            self.model.model.jointPlacements[j] = pin.SE3(M0.rotation @ pin.exp3(ov[k, 3:]),
                                                           M0.translation + ov[k, :3])

    def link_angles(self, x, q, tau) -> np.ndarray:
        """Encoder angles to link angles: zero increment, compliance, backlash state."""
        c = x[self.sl['compliance']]
        b = x[self.sl['backlash']]
        dz = x[self.sl['dzero']]
        return q + dz - c * tau - 0.5 * b * np.tanh(tau / BACKLASH_EPS)

    def frame_pose(self, q_link) -> pin.SE3:
        """FK of the fit frame with the current geometry (call set_geometry first)."""
        m, d = self.model.model, self.model.data
        pin.framesForwardKinematics(m, d, self.model._q(q_link))
        return d.oMf[self.fid]

    def residuals(self, x, *data) -> np.ndarray:
        raise NotImplementedError

    def jacobian(self, x, *data) -> np.ndarray:
        r0 = self.residuals(x, *data)
        J = np.zeros((len(r0), self.size))
        for i in range(self.size):
            h = 1e-3 * self.scale[i]
            dx = np.zeros(self.size)
            dx[i] = h
            J[:, i] = (self.residuals(x + dx, *data) - r0) / h
        return J


def select_parameters(fit: KinematicModel, J: np.ndarray, priority: Sequence[str], tol: float = 2e-3,
                      exclude: Sequence[str] = (), details: bool = False):
    """Greedy independent-column selection in `priority` order on the scaled
    Jacobian: a parameter is kept only if it adds a direction the kept ones
    cannot produce. Returns (kept indices, condition number of the kept set);
    with details=True also each parameter's independent share (the part of
    its column the higher-priority kept ones cannot produce, 0..1; 0 for
    excluded or zero columns)."""
    Js = J * fit.scale
    order = [i for key in priority if key not in exclude for i in range(fit.size)[fit.sl[key]]]
    basis = np.zeros((Js.shape[0], 0))
    keep = []
    share = np.zeros(fit.size)
    for i in order:
        col = Js[:, i]
        norm = np.linalg.norm(col)
        if norm < 1e-12:
            continue
        resid = col - basis @ (basis.T @ col)
        share[i] = np.linalg.norm(resid) / norm
        if np.linalg.norm(resid) > tol * norm:
            keep.append(i)
            basis = np.hstack([basis, (resid / np.linalg.norm(resid))[:, None]])
    keep = sorted(keep)
    cond = condition(fit, J, keep)
    return (keep, cond, share) if details else (keep, cond)


def condition(fit: KinematicModel, J: np.ndarray, keep) -> float:
    """Condition number of the kept columns, scaled and then normalized."""
    Jk = (J * fit.scale)[:, keep]
    Jk = Jk / np.linalg.norm(Jk, axis=0)
    S = np.linalg.svd(Jk, compute_uv=False)
    return float(S[0] / S[-1])


def parameter_sigma(fit: KinematicModel, J: np.ndarray, keep, residual_rms: float) -> np.ndarray:
    """1-sigma of each kept parameter (physical units) from the Gauss-Newton
    covariance, with the fit's residual RMS as the measurement noise."""
    Jk = J[:, keep]
    cov = np.linalg.pinv(Jk.T @ Jk) * residual_rms ** 2
    out = np.full(fit.size, np.nan)
    out[list(keep)] = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return out


def solve(fit: KinematicModel, data: tuple, keep, x0=None, max_iter: int = 60):
    """LM on the kept parameters, scaled to order 1; the rest stay at x0."""
    x0 = np.zeros(fit.size) if x0 is None else x0.copy()
    keep = np.asarray(keep)

    def f(z):
        x = x0.copy()
        x[keep] += z * fit.scale[keep]
        return fit.residuals(x, *data)

    sol = levenberg_marquardt(f, np.zeros(len(keep)), max_iter=max_iter)
    x = x0.copy()
    x[keep] += sol.x * fit.scale[keep]
    return x


def rms(v) -> float:
    return float(np.sqrt(np.mean(np.square(v))))


def compose_overlay(old: Optional[dict], names, ov: np.ndarray) -> dict:
    out = {}
    for k, n in enumerate(names):
        o = (old or {}).get(n) or {}
        dxyz = np.array(o.get('dxyz', [0, 0, 0]), float) + ov[k, :3]
        R = pin.exp3(np.array(o.get('drot', [0, 0, 0]), float)) @ pin.exp3(ov[k, 3:])
        out[n] = {'dxyz': dxyz.tolist(), 'drot': pin.log3(R).tolist()}
    return out
