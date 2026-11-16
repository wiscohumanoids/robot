"""Inverse kinematics for the 7-DOF arm (tool frame `tool0` by default).

The arm is not a spherical-shoulder / spherical-wrist (SRS) design: J6 has
only +-35 deg and J7's axis is ~62 deg off the forearm, so there is no closed
form. Everything here is numerical, numpy + Pinocchio only.

One interface, `solve(model, target, q_seed, method=..., **opts) -> IKResult`:

  dls         damped least squares on the 6D twist error (pin.log6), damping
              adapted from Yoshikawa manipulability (Nakamura & Hanafusa).
  lm          Levenberg-Marquardt with error-proportional damping (Sugihara)
              and accept/reject; joint limits by projection with an active
              set (a joint pinned at a stop is dropped from the step).
  clik        weighted-least-norm CLIK (Chan & Dubey joint-limit weights) with
              null-space secondary objectives: rest posture, a joint-limit
              barrier, optionally pulling the elbow toward a direction.
              Uses the 7th DOF; the one to use for teleop.
  multistart  any of the above from the seed, then random restarts within
              the limits; keeps the best feasible solution.

Every solver keeps q inside [model.lower, model.upper] at every iterate.

Partial targets go through `mask` (or a 6-vector `weights`), expressed in the
target frame: 'full', 'position' (3-DOF), 'free_z' (5-DOF, tool spin about
the target z axis free). A 3-vector target means position-only.

`IKTracker` wraps a solver for streaming targets (teleop): warm start from
the last solution, joint-velocity rate limit, and a flag (plus the closest
feasible pose) when a target is out of reach.

`BatchKinematics` / `solve_batch` are a vectorized numpy FK, Jacobian and LM
over many targets at once (workspace maps, pose planning).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pinocchio as pin

DEG = math.pi / 180.0
TOL_POS = 1e-4            # m, success threshold (0.1 mm)
TOL_ROT = 0.1 * DEG       # rad, success threshold (0.1 deg)
POLISH = 0.1              # iterate until errors < POLISH * tolerance
ROT_WEIGHT = 0.3          # m per rad: rotation rows relative to position rows
LIMIT_EPS = 1e-9

MASKS = {
    'full': (1, 1, 1, 1, 1, 1),
    'position': (1, 1, 1, 0, 0, 0),
    'free_z': (1, 1, 1, 1, 1, 0),
}

Target = Union[pin.SE3, np.ndarray, Sequence[float]]


@dataclass
class IKResult:
    q: np.ndarray
    success: bool
    pos_err: float          # m (masked components)
    rot_err: float          # rad (masked components)
    iters: int
    time: float             # s
    status: str = ''        # converged | max_iters | stalled | timeout
    starts: int = 1         # restarts used (multistart)


# ----------------------------------------------------------------------------- helpers
def mid_posture(model) -> np.ndarray:
    """Middle of each joint's range: the default rest posture (max limit clearance)."""
    return 0.5 * (model.lower + model.upper)


def as_target(target: Target) -> Tuple[pin.SE3, bool]:
    """SE3, 4x4 array or 3-vector (position only) -> (SE3, position_only)."""
    if isinstance(target, pin.SE3):
        return target, False
    a = np.asarray(target, float)
    if a.shape == (4, 4):
        return pin.SE3(a[:3, :3].copy(), a[:3, 3].copy()), False
    if a.shape == (3,):
        return pin.SE3(np.eye(3), a.copy()), True
    raise ValueError('target must be pin.SE3, 4x4 or a 3-vector, got shape %s' % (a.shape,))


def weight_vector(mask='full', weights=None, rot_weight=ROT_WEIGHT) -> np.ndarray:
    """Row weights [px py pz rx ry rz] in the target frame (0 = free)."""
    if weights is not None:
        w = np.asarray(weights, float)
        if w.shape != (6,) or np.any(w < 0) or not np.any(w > 0):
            raise ValueError('weights must be 6 non-negative numbers, not all zero')
        return w
    if isinstance(mask, str):
        if mask not in MASKS:
            raise ValueError('mask must be one of %s' % sorted(MASKS))
        m = np.array(MASKS[mask], float)
    else:
        m = np.asarray(mask, float)
    return m * np.array([1, 1, 1, rot_weight, rot_weight, rot_weight])


def limit_barrier_gradient(q, lo, hi, margin=0.1):
    """Gradient of a barrier that is zero away from the stops and grows
    quadratically inside the last `margin` fraction of each joint's range."""
    rng = hi - lo
    d = margin * rng
    g = np.zeros_like(q)
    low = q < lo + d
    high = q > hi - d
    g[low] = -((lo + d - q) / d)[low]
    g[high] = ((q - (hi - d)) / d)[high]
    return g


def wln_gradient(q, lo, hi):
    """dH/dq of Chan & Dubey's H = sum (hi-lo)^2 / (4 (hi-q)(q-lo))."""
    a = np.maximum(hi - q, 1e-6)
    b = np.maximum(q - lo, 1e-6)
    return (hi - lo) ** 2 * (2 * q - hi - lo) / (4 * a * a * b * b)


class _Kin:
    """Tool pose and world-aligned Jacobian through Pinocchio, config order."""

    def __init__(self, arm, frame=None):
        self.model = arm.model
        self.data = arm.data
        self.fid = self.model.getFrameId(frame or arm.tool_frame)
        self.qidx = list(arm.qidx)
        self.vidx = list(arm.vidx)
        self.nq = self.model.nq
        self.nv = self.model.nv

    def _full(self, q):
        out = np.zeros(self.nq)
        out[self.qidx] = q
        return out

    def pose(self, q) -> pin.SE3:
        pin.forwardKinematics(self.model, self.data, self._full(q))
        return pin.updateFramePlacement(self.model, self.data, self.fid)

    def pose_jac(self, q):
        pin.computeJointJacobians(self.model, self.data, self._full(q))
        M = pin.updateFramePlacement(self.model, self.data, self.fid)
        J = pin.getFrameJacobian(self.model, self.data, self.fid,
                                 pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return M, np.asarray(J).reshape(6, self.nv)[:, self.vidx]


class _Problem:
    """Weighted residual r(q) (-> 0 at the target) and its Jacobian.

    mode 'split': r = [Rt^T (p - pt); log3(Rt^T R)], both in the target frame,
                  so a mask frees exactly target-frame axes.
    mode 'log6' : r = log6(T^-1 M), the body twist error, full pose only.
    Reported errors are always the split ones (|dp|, rotation angle).
    """

    def __init__(self, arm, target, mask='full', weights=None, frame=None, mode='split',
                 tol_pos=TOL_POS, tol_rot=TOL_ROT, rot_weight=ROT_WEIGHT):
        T, pos_only = as_target(target)
        if pos_only and weights is None and (isinstance(mask, str) and mask == 'full'):
            mask = 'position'
        self.T = T
        self.Tinv = T.inverse()
        self.RtT = T.rotation.T.copy()
        self.pt = T.translation.copy()
        self.w = weight_vector(mask, weights, rot_weight)
        self.rows = self.w > 0
        self.sw = self.w[self.rows]
        self.mp = self.rows[:3]
        self.mr = self.rows[3:]
        if mode == 'log6' and not np.all(self.rows):
            mode = 'split'
        self.mode = mode
        self.kin = _Kin(arm, frame)
        self.lo = arm.lower
        self.hi = arm.upper
        self.n = arm.n
        self.tol_pos = tol_pos
        self.tol_rot = tol_rot

    def errors(self, M):
        rp = self.RtT @ (M.translation - self.pt)
        A = self.RtT @ M.rotation
        rr = pin.log3(A)
        return rp, rr, A, float(np.linalg.norm(rp[self.mp])), float(np.linalg.norm(rr[self.mr]))

    def eval(self, q, jac=True):
        if jac:
            M, J = self.kin.pose_jac(q)
        else:
            M, J = self.kin.pose(q), None
        rp, rr, A, pe, re = self.errors(M)
        if self.mode == 'log6':
            B = self.Tinv * M
            r = pin.log6(B).vector
            if jac:
                R = M.rotation
                Jl = np.vstack((R.T @ J[:3], R.T @ J[3:]))
                Jr = pin.Jlog6(B) @ Jl
        else:
            r = np.concatenate((rp, rr))
            if jac:
                Jr = np.vstack((self.RtT @ J[:3], pin.Jlog3(A.T) @ self.RtT @ J[3:]))
        r = r[self.rows] * self.sw
        if jac:
            Jr = Jr[self.rows] * self.sw[:, None]
            return r, Jr, pe, re
        return r, None, pe, re

    def converged(self, pe, re, polish=POLISH):
        return pe < self.tol_pos * polish and re < self.tol_rot * polish

    def ok(self, pe, re):
        return pe < self.tol_pos and re < self.tol_rot

    def clip(self, q):
        return np.minimum(np.maximum(q, self.lo), self.hi)

    def blocked(self, q, dq):
        """Joints at a stop whose step pushes further out."""
        return ((q <= self.lo + LIMIT_EPS) & (dq < 0)) | ((q >= self.hi - LIMIT_EPS) & (dq > 0))


def _scale_step(dq, max_step):
    m = np.max(np.abs(dq))
    return dq * (max_step / m) if m > max_step else dq


def _result(P, q, pe, re, it, t0, status):
    return IKResult(q=np.array(q, float), success=bool(P.ok(pe, re)), pos_err=pe, rot_err=re,
                    iters=it, time=time.perf_counter() - t0, status=status)


# ----------------------------------------------------------------------------- (a) DLS
def solve_dls(model, target, q_seed, *, mask='full', weights=None, frame=None, mode='log6',
              max_iters=300, tol_pos=TOL_POS, tol_rot=TOL_ROT, lam_max=0.05, w0=1e-5,
              max_step=0.3, stall_iters=30, timeout=None, **_) -> IKResult:
    """Damped least squares: dq = -J^T (J J^T + lam^2 I)^-1 r, then clamp.

    Damping after Nakamura & Hanafusa: lam^2 = lam_max^2 (1 - w/w0)^2 when the
    manipulability w = sqrt(det J J^T) is below w0, else ~0 (Gauss-Newton).
    Joint limits by clamping only, so it can stall against a stop.
    """
    t0 = time.perf_counter()
    P = _Problem(model, target, mask, weights, frame, mode, tol_pos, tol_rot)
    q = P.clip(np.array(q_seed, float))
    best = (np.inf, q, np.inf, np.inf)
    since = 0
    status = 'max_iters'
    it = 0
    for it in range(max_iters + 1):
        r, J, pe, re = P.eval(q)
        E = float(r @ r)
        if E < best[0] * (1 - 1e-9):
            best = (E, q, pe, re)
            since = 0
        else:
            since += 1
        if P.converged(pe, re):
            status = 'converged'
            break
        if it == max_iters:
            break
        if since > stall_iters:
            status = 'stalled'
            break
        if timeout is not None and time.perf_counter() - t0 > timeout:
            status = 'timeout'
            break
        JJt = J @ J.T
        w = math.sqrt(max(np.linalg.det(JJt), 0.0))
        lam2 = lam_max ** 2 * (1 - w / w0) ** 2 if w < w0 else 0.0
        dq = -J.T @ np.linalg.solve(JJt + (lam2 + 1e-12) * np.eye(len(r)), r)
        q = P.clip(q + _scale_step(dq, max_step))
    if status == 'converged':
        return _result(P, q, pe, re, it, t0, status)
    _, q, pe, re = best
    return _result(P, q, pe, re, it, t0, status)


# ----------------------------------------------------------------------------- (b) LM
def solve_lm(model, target, q_seed, *, mask='full', weights=None, frame=None, mode='split',
             max_iters=150, tol_pos=TOL_POS, tol_rot=TOL_ROT, lam0=1.0, lam_bar=1e-6,
             max_step=0.5, timeout=None, **_) -> IKResult:
    """Levenberg-Marquardt, damping lam*E + lam_bar (Sugihara 2011), E = r.r/2.

    A step that does not lower E is rejected and lam grows; accepted steps
    shrink it. Limits: the trial point is projected into the box and joints
    pinned at a stop and pushing outward are removed from the step (active set).
    """
    t0 = time.perf_counter()
    P = _Problem(model, target, mask, weights, frame, mode, tol_pos, tol_rot)
    q = P.clip(np.array(q_seed, float))
    r, J, pe, re = P.eval(q)
    E = 0.5 * float(r @ r)
    lam = lam0
    n = P.n
    I = np.eye(n)
    status = 'max_iters'
    it = 0
    small = 0
    while True:
        if P.converged(pe, re):
            status = 'converged'
            break
        if it >= max_iters:
            break
        if timeout is not None and time.perf_counter() - t0 > timeout:
            status = 'timeout'
            break
        g = J.T @ r
        H = J.T @ J
        D = lam * E + lam_bar
        dq = -np.linalg.solve(H + D * I, g)
        blk = P.blocked(q, dq)
        if blk.any():
            free = ~blk
            if not free.any():
                status = 'stalled'
                break
            dq = np.zeros(n)
            dq[free] = -np.linalg.solve(H[np.ix_(free, free)] + D * np.eye(int(free.sum())), g[free])
        q_new = P.clip(q + _scale_step(dq, max_step))
        r_n, J_n, pe_n, re_n = P.eval(q_new)
        E_n = 0.5 * float(r_n @ r_n)
        it += 1
        if E_n < E:
            small = small + 1 if E - E_n < 1e-12 + 1e-6 * E else 0
            q, r, J, pe, re, E = q_new, r_n, J_n, pe_n, re_n, E_n
            lam = max(lam * 0.3, 1e-6)
            if small >= 8:
                status = 'stalled'
                break
        else:
            lam *= 8.0
            if lam > 1e10:
                status = 'stalled'
                break
    return _result(P, q, pe, re, it, t0, status)


# ----------------------------------------------------------------------------- (c) CLIK
def solve_clik(model, target, q_seed, *, mask='full', weights=None, frame=None, mode='split',
               max_iters=200, tol_pos=TOL_POS, tol_rot=TOL_ROT, gain=1.0, lam_max=0.05,
               w0=1e-5, max_step=0.3, q_rest=None, rest_weights=None, k_rest=0.05,
               k_limits=0.2, limit_margin=0.1, elbow_dir=None, k_elbow=0.0, elbow_frame='elbow',
               wln=True, min_iters=0, ns_band=20.0, ns_step=0.05, stall_iters=30, timeout=None,
               **_) -> IKResult:
    """Weighted-least-norm CLIK with null-space secondary objectives.

    dq = gain * J#_W (-r) + (I - J#_W J) dq0,  J#_W = W^-1 J^T (J W^-1 J^T + lam^2 I)^-1
      W   = diag(1 + |dH/dq|) on joints moving toward a stop (Chan & Dubey WLN),
      dq0 = -k_rest * rest_weights * (q - q_rest)          (rest posture, default mid-range)
            -k_limits * barrier gradient near the stops     (joint-limit avoidance)
            +k_elbow * J_elbow^T elbow_dir                  (optional elbow pull)
    The null space is the 7th DOF (the elbow swivel for a 6D task), so the
    secondary terms hold the elbow without disturbing the tool to first order.
    Joints pinned at a stop and pushing outward are dropped (active set).
    The null-space step is capped at ns_step (rad) and switched off once the
    tool is within ns_band x tolerance, so the last steps are pure Newton.
    `min_iters` forces that many (null-space) steps even from a converged
    seed, so a tracker keeps regulating the posture.
    """
    t0 = time.perf_counter()
    P = _Problem(model, target, mask, weights, frame, mode, tol_pos, tol_rot)
    lo, hi = P.lo, P.hi
    n = P.n
    q = P.clip(np.array(q_seed, float))
    q_rest = mid_posture(model) if q_rest is None else np.asarray(q_rest, float)
    rw = np.ones(n) if rest_weights is None else np.asarray(rest_weights, float)
    span = hi - lo
    elbow_kin = None
    if elbow_dir is not None and k_elbow:
        elbow_kin = _Kin(model, elbow_frame)
        elbow_dir = np.asarray(elbow_dir, float) / np.linalg.norm(elbow_dir)
    dq_prev = None
    best = (np.inf, q, np.inf, np.inf)
    since = 0
    status = 'max_iters'
    it = 0
    I = np.eye(n)
    for it in range(max_iters + 1):
        r, J, pe, re = P.eval(q)
        E = float(r @ r)
        if E < best[0] * (1 - 1e-9):
            best = (E, q, pe, re)
            since = 0
        else:
            since += 1
        if P.converged(pe, re) and it >= min_iters:
            status = 'converged'
            break
        if it == max_iters:
            break
        if since > stall_iters and it >= min_iters:
            status = 'stalled'
            break
        if timeout is not None and time.perf_counter() - t0 > timeout:
            status = 'timeout'
            break
        # joint weights
        if wln:
            dH = wln_gradient(q, lo, hi)
            toward = np.ones(n, bool) if dq_prev is None else (dq_prev * dH > 0)
            winv = 1.0 / np.where(toward, 1.0 + np.abs(dH), 1.0)
        else:
            winv = np.ones(n)
        # secondary objective (gradient step), scaled by the range. Off for the
        # last Newton steps near the target (after min_iters), so the null-space
        # motion's second-order effect on the tool does not set an error floor.
        near = pe < ns_band * P.tol_pos and re < ns_band * P.tol_rot
        use_ns = it < min_iters or not near
        if use_ns:
            dq0 = -k_rest * rw * (q - q_rest) - k_limits * span * limit_barrier_gradient(q, lo, hi, limit_margin)
            if elbow_kin is not None:
                _, Je = elbow_kin.pose_jac(q)
                dq0 = dq0 + k_elbow * (Je[:3].T @ elbow_dir)
        else:
            dq0 = np.zeros(n)
        for _pass in range(2):
            JW = J * winv[None, :]
            JJt = JW @ J.T
            w = math.sqrt(max(np.linalg.det(J @ J.T), 0.0))
            lam2 = lam_max ** 2 * (1 - w / w0) ** 2 if w < w0 else 0.0
            Jp = JW.T @ np.linalg.inv(JJt + (lam2 + 1e-12) * np.eye(len(r)))
            dq_task = -gain * (Jp @ r)
            dq_ns = (I - Jp @ J) @ (winv * dq0)
            dq = _scale_step(dq_task, max_step) + _scale_step(dq_ns, ns_step)
            blk = P.blocked(q, dq)
            if not blk.any() or _pass == 1:
                break
            winv = np.where(blk, 0.0, winv)
            dq0 = np.where(blk, 0.0, dq0)
        dq = _scale_step(np.where(P.blocked(q, dq), 0.0, dq), max_step)
        dq_prev = dq
        q = P.clip(q + dq)
    if status == 'converged':
        return _result(P, q, pe, re, it, t0, status)
    _, q, pe, re = best
    return _result(P, q, pe, re, it, t0, status)


# ----------------------------------------------------------------------------- (d) multi-start
def solve_multistart(model, target, q_seed, *, base='lm', n_starts=40, seed=0, rng=None,
                     select='first', time_budget=None, start_margin=0.0, **opts) -> IKResult:
    """Run `base` from the seed, then from random configurations inside the limits.

    select='first'  : return the first feasible solution (fastest),
    select='closest': try all starts, return the feasible one closest to the seed,
    select='rest'   : try all starts, return the feasible one closest to q_rest/mid.
    If none is feasible: the start with the smallest error (success=False).
    """
    t0 = time.perf_counter()
    fn = SOLVERS[base] if isinstance(base, str) else base
    rng = rng if rng is not None else np.random.default_rng(seed)
    q_seed = np.asarray(q_seed, float)
    ref = q_seed if select != 'rest' else (np.asarray(opts['q_rest'], float) if opts.get('q_rest') is not None
                                           else mid_posture(model))
    lo = model.lower + start_margin * (model.upper - model.lower)
    hi = model.upper - start_margin * (model.upper - model.lower)
    best_ok, best_any = None, None
    iters = 0
    starts = 0
    for k in range(n_starts + 1):
        q0 = q_seed if k == 0 else rng.uniform(lo, hi)
        res = fn(model, target, q0, **opts)
        iters += res.iters
        starts += 1
        if res.success:
            d = float(np.linalg.norm(res.q - ref))
            if best_ok is None or d < best_ok[0]:
                best_ok = (d, res)
            if select == 'first':
                break
        else:
            score = res.pos_err + ROT_WEIGHT * res.rot_err
            if best_any is None or score < best_any[0]:
                best_any = (score, res)
        if time_budget is not None and time.perf_counter() - t0 > time_budget:
            break
    res = best_ok[1] if best_ok is not None else best_any[1]
    out = IKResult(q=res.q, success=res.success, pos_err=res.pos_err, rot_err=res.rot_err,
                   iters=iters, time=time.perf_counter() - t0, status=res.status, starts=starts)
    return out


SOLVERS: Dict[str, Callable[..., IKResult]] = {
    'dls': solve_dls,
    'lm': solve_lm,
    'clik': solve_clik,
    'multistart': solve_multistart,
}


def solve(model, target: Target, q_seed, method: str = 'clik', **opts) -> IKResult:
    """IK of `target` (pin.SE3, 4x4, or a 3-vector for position only) from q_seed.

    method: 'dls' | 'lm' | 'clik' | 'multistart' (opts['base'] picks its inner solver).
    Common opts: mask ('full' | 'position' | 'free_z' | 6 flags), weights (6),
    frame, max_iters, tol_pos (m), tol_rot (rad), timeout (s).
    """
    if method not in SOLVERS:
        raise ValueError('unknown IK method %r (have %s)' % (method, sorted(SOLVERS)))
    return SOLVERS[method](model, target, q_seed, **opts)


# ----------------------------------------------------------------------------- tracker
@dataclass
class TrackResult:
    q: np.ndarray               # commanded joint position (inside limits, rate limited)
    reachable: bool             # the IK met tolerance for this target
    rate_limited: bool          # the step was scaled down to the velocity limit
    pos_err: float              # m, commanded pose vs target
    rot_err: float              # rad
    ik: IKResult = field(repr=False, default=None)


class IKTracker:
    """Streaming IK for teleop / task-space control.

    - warm starts from the last commanded solution,
    - limits joint velocity: the step to the IK solution is scaled uniformly
      (keeps the direction) so no joint exceeds max_vel * dt,
    - out-of-reach targets: returns the closest feasible pose the local solver
      reaches (a least-squares fit inside the limits) with reachable=False,
      instead of jumping to a far restart,
    - stall escape: when the local solver fails and the tracker has stopped
      moving (pinned at a joint limit in a local minimum, e.g. after stretching
      for an unreachable target), a multistart solve looks for the reachable
      solution closest to the current pose and the tracker heads there at the
      velocity limit. An unreachable target is tried once per position (again
      only after it moves `escape_retry_m`), so a held far target costs one solve.

    tracker = IKTracker(model, q_now, dt=0.01)
    out = tracker.update(target_se3)      # out.q -> joint command
    """

    def __init__(self, model, q0, dt: float = 0.01, method: str = 'clik', max_vel=None,
                 max_iters: int = 30, escape: bool = True, escape_starts: int = 20,
                 escape_retry_m: float = 0.01, **opts):
        self.model = model
        self.escape = escape
        self.escape_starts = int(escape_starts)
        self.escape_retry_m = float(escape_retry_m)
        self._escape_failed_at = None
        self.dt = float(dt)
        self.method = method
        if max_vel is None:
            max_vel = np.minimum(model.velocity_limit, 1.5)
        self.max_vel = np.broadcast_to(np.asarray(max_vel, float), (model.n,)).copy()
        self.opts = dict(opts)
        self.opts.setdefault('max_iters', max_iters)
        if method == 'clik':
            self.opts.setdefault('min_iters', 1)
        self.q = np.clip(np.asarray(q0, float), model.lower, model.upper)
        self.last: Optional[TrackResult] = None

    def reset(self, q):
        self.q = np.clip(np.asarray(q, float), self.model.lower, self.model.upper)
        self.last = None
        self._escape_failed_at = None

    STALL_RAD = 1e-4

    def _escape(self, target, res):
        """Multistart from a stalled local solve; res unchanged if nothing better."""
        pos = np.asarray(target.translation if hasattr(target, 'translation') else target, float)[:3]
        if self._escape_failed_at is not None and                 np.linalg.norm(pos - self._escape_failed_at) < self.escape_retry_m:
            return res
        opts = {k: v for k, v in self.opts.items() if k not in ('min_iters', 'max_iters')}
        alt = solve(self.model, target, self.q, method='multistart', n_starts=self.escape_starts,
                    select='closest', seed=0, **opts)
        if alt.success:
            self._escape_failed_at = None
            return alt
        self._escape_failed_at = pos
        return res

    def update(self, target: Target, dt: Optional[float] = None) -> TrackResult:
        dt = self.dt if dt is None else float(dt)
        res = solve(self.model, target, self.q, method=self.method, **self.opts)
        if self.escape and not res.success and np.max(np.abs(res.q - self.q)) < self.STALL_RAD:
            res = self._escape(target, res)
        dq = res.q - self.q
        lim = self.max_vel * dt
        ratio = float(np.max(np.abs(dq) / np.maximum(lim, 1e-12)))
        limited = ratio > 1.0
        if limited:
            dq = dq / ratio
        self.q = np.clip(self.q + dq, self.model.lower, self.model.upper)
        if limited:
            P = _Problem(self.model, target, self.opts.get('mask', 'full'), self.opts.get('weights'),
                         self.opts.get('frame'), 'split', self.opts.get('tol_pos') or TOL_POS,
                         self.opts.get('tol_rot') or TOL_ROT)
            _, _, pe, re = P.eval(self.q, jac=False)
        else:
            pe, re = res.pos_err, res.rot_err
        self.last = TrackResult(q=self.q.copy(), reachable=bool(res.success), rate_limited=limited,
                                pos_err=pe, rot_err=re, ik=res)
        return self.last


# ----------------------------------------------------------------------------- analysis
def manipulability(model, q, frame=None) -> float:
    """Yoshikawa sqrt(det(J J^T)) of the 6x7 world-aligned tool Jacobian."""
    J = model.jacobian(q, frame)
    return math.sqrt(max(np.linalg.det(J @ J.T), 0.0))


def swivel_angle(model, q, ref=(0.0, 0.0, -1.0)) -> float:
    """Elbow swivel: angle of the elbow point about the shoulder-wrist line,
    zero where the elbow points along `ref` (projected). Shoulder point: the
    shoulder_roll origin; elbow: the elbow origin; wrist: the wrist_pitch origin."""
    S = model.fk(q, 'shoulder_roll').translation
    E = model.fk(q, 'elbow').translation
    W = model.fk(q, 'wrist_pitch').translation
    u = W - S
    u = u / np.linalg.norm(u)
    e = (E - S) - ((E - S) @ u) * u
    r = np.asarray(ref, float)
    r = r - (r @ u) * u
    if np.linalg.norm(r) < 1e-9 or np.linalg.norm(e) < 1e-9:
        return 0.0
    r = r / np.linalg.norm(r)
    return math.atan2(float(np.cross(r, e) @ u), float(r @ e))


# ----------------------------------------------------------------------------- batched (numpy)
def _skew(a):
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], float)


def quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """(N,3,3) -> (N,4) unit quaternions [x y z w], w >= 0."""
    R = np.asarray(R)
    N = R.shape[0]
    q = np.empty((N, 4))
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    c0 = tr > 0
    c1 = (~c0) & (R[:, 0, 0] >= R[:, 1, 1]) & (R[:, 0, 0] >= R[:, 2, 2])
    c2 = (~c0) & (~c1) & (R[:, 1, 1] >= R[:, 2, 2])
    c3 = (~c0) & (~c1) & (~c2)
    s = np.sqrt(np.maximum(tr[c0] + 1.0, 1e-300)) * 2
    q[c0, 3] = 0.25 * s
    q[c0, 0] = (R[c0, 2, 1] - R[c0, 1, 2]) / s
    q[c0, 1] = (R[c0, 0, 2] - R[c0, 2, 0]) / s
    q[c0, 2] = (R[c0, 1, 0] - R[c0, 0, 1]) / s
    s = np.sqrt(np.maximum(1.0 + R[c1, 0, 0] - R[c1, 1, 1] - R[c1, 2, 2], 1e-300)) * 2
    q[c1, 3] = (R[c1, 2, 1] - R[c1, 1, 2]) / s
    q[c1, 0] = 0.25 * s
    q[c1, 1] = (R[c1, 0, 1] + R[c1, 1, 0]) / s
    q[c1, 2] = (R[c1, 0, 2] + R[c1, 2, 0]) / s
    s = np.sqrt(np.maximum(1.0 + R[c2, 1, 1] - R[c2, 0, 0] - R[c2, 2, 2], 1e-300)) * 2
    q[c2, 3] = (R[c2, 0, 2] - R[c2, 2, 0]) / s
    q[c2, 0] = (R[c2, 0, 1] + R[c2, 1, 0]) / s
    q[c2, 1] = 0.25 * s
    q[c2, 2] = (R[c2, 1, 2] + R[c2, 2, 1]) / s
    s = np.sqrt(np.maximum(1.0 + R[c3, 2, 2] - R[c3, 0, 0] - R[c3, 1, 1], 1e-300)) * 2
    q[c3, 3] = (R[c3, 1, 0] - R[c3, 0, 1]) / s
    q[c3, 0] = (R[c3, 0, 2] + R[c3, 2, 0]) / s
    q[c3, 1] = (R[c3, 1, 2] + R[c3, 2, 1]) / s
    q[c3, 2] = 0.25 * s
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 3] < 0] *= -1
    return q


def matrix_from_quat(q: np.ndarray) -> np.ndarray:
    """(N,4) [x y z w] -> (N,3,3)."""
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def log3_batch(R: np.ndarray) -> np.ndarray:
    """(N,3,3) -> (N,3) rotation vectors, through quaternions (safe near pi)."""
    q = quat_from_matrix(R)
    v = q[:, :3]
    s = np.linalg.norm(v, axis=1)
    ang = 2 * np.arctan2(s, q[:, 3])
    k = np.where(s > 1e-12, ang / np.maximum(s, 1e-300), 2.0)
    return v * k[:, None]


def jlog3_left_batch(phi: np.ndarray) -> np.ndarray:
    """Inverse left Jacobian of SO(3) at phi: d log(exp(x) R) / dx at x = 0."""
    th = np.linalg.norm(phi, axis=1)
    N = len(phi)
    K = np.zeros((N, 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -phi[:, 2], phi[:, 1]
    K[:, 1, 0], K[:, 1, 2] = phi[:, 2], -phi[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -phi[:, 1], phi[:, 0]
    thc = np.minimum(th, math.pi - 1e-3)
    small = th < 1e-4
    c = np.where(small, 1.0 / 12.0,
                 1.0 / np.maximum(thc, 1e-12) ** 2
                 - (1 + np.cos(thc)) / (2 * np.maximum(thc, 1e-12) * np.sin(np.maximum(thc, 1e-12))))
    return np.eye(3)[None] - 0.5 * K + c[:, None, None] * (K @ K)


class BatchKinematics:
    """Vectorized FK and world-aligned Jacobian of a serial chain (numpy).

    Built from the Pinocchio model inside an ArmModel (with any calibration
    overlay applied), so it agrees with ArmModel.fk / jacobian.
    """

    def __init__(self, arm, frame=None):
        model = arm.model
        self.n = arm.n
        self.lower = arm.lower.copy()
        self.upper = arm.upper.copy()
        self.order = [int(k) for k in np.argsort(arm.jid)]   # config index along the chain
        chain = [arm.jid[k] for k in self.order]
        prev = 0
        for j in chain:
            if model.parents[j] != prev:
                raise ValueError('BatchKinematics needs a serial chain')
            prev = j
        self.Rj = [np.array(model.jointPlacements[j].rotation) for j in chain]
        self.pj = [np.array(model.jointPlacements[j].translation) for j in chain]
        self.axes = [arm.axis_in_child(k) for k in self.order]
        self.K = [_skew(a) for a in self.axes]
        self.KK = [K @ K for K in self.K]
        fr = model.frames[model.getFrameId(frame or arm.tool_frame)]
        if fr.parentJoint not in chain:
            raise ValueError('frame %s is not on the chain' % (frame or arm.tool_frame))
        self.parent_pos = chain.index(fr.parentJoint)
        self.Rf = np.array(fr.placement.rotation)
        self.pf = np.array(fr.placement.translation)

    def fk(self, Q: np.ndarray, jacobian: bool = False):
        """Q (N,n) config order -> p (N,3), R (N,3,3)[, J (N,6,n) world-aligned]."""
        Q = np.atleast_2d(np.asarray(Q, float))
        N = Q.shape[0]
        R = np.broadcast_to(np.eye(3), (N, 3, 3)).copy()
        p = np.zeros((N, 3))
        Z = np.zeros((N, self.n, 3))
        Pj = np.zeros((N, self.n, 3))
        pt = Rt = None
        for i, k in enumerate(self.order):
            p = p + R @ self.pj[i]
            R = R @ self.Rj[i]
            Z[:, k] = R @ self.axes[i]
            Pj[:, k] = p
            s = np.sin(Q[:, k])[:, None, None]
            c = np.cos(Q[:, k])[:, None, None]
            R = R + s * (R @ self.K[i]) + (1 - c) * (R @ self.KK[i])
            if i == self.parent_pos:
                pt = p + R @ self.pf
                Rt = R @ self.Rf
                break
        if not jacobian:
            return pt, Rt
        J = np.zeros((N, 6, self.n))
        for i, k in enumerate(self.order[:self.parent_pos + 1]):
            J[:, :3, k] = np.cross(Z[:, k], pt - Pj[:, k])
            J[:, 3:, k] = Z[:, k]
        return pt, Rt, J


def manipulability_batch(J: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(np.linalg.det(J @ np.transpose(J, (0, 2, 1))), 0.0))


def solve_batch(bk: BatchKinematics, p_target: np.ndarray, R_target: Optional[np.ndarray],
                Q0: np.ndarray, mask='full', weights=None, max_iters: int = 40,
                tol_pos=TOL_POS, tol_rot=TOL_ROT, lam0=1.0, lam_bar=1e-6, max_step=0.5,
                stall_ratio: Optional[float] = None, stall_every: int = 6):
    """Vectorized LM (same scheme as solve_lm) for N targets at once.

    p_target (N,3), R_target (N,3,3) or None (position only), Q0 (N,n) seeds.
    Limits as in solve_lm (projection plus active set). stall_ratio: drop a
    target whose cost did not fall below stall_ratio x its value stall_every
    iterations earlier (saves time when most targets are unreachable).
    Returns Q (N,n), success (N,), pos_err (N,), rot_err (N,).
    """
    N = len(p_target)
    if R_target is None:
        R_target = np.broadcast_to(np.eye(3), (N, 3, 3))
        if weights is None and mask == 'full':
            mask = 'position'
    w = weight_vector(mask, weights)
    rows = w > 0
    sw = w[rows]
    RtT = np.transpose(R_target, (0, 2, 1))
    lo, hi = bk.lower, bk.upper
    n = bk.n

    def evaluate(Q, sel=slice(None)):
        p, R, J = bk.fk(Q, jacobian=True)
        Rs = RtT[sel]
        rp = np.einsum('nij,nj->ni', Rs, p - p_target[sel])
        A = Rs @ R
        rr = log3_batch(A)
        Jp = Rs @ J[:, :3]
        Jr = jlog3_left_batch(rr) @ (Rs @ J[:, 3:])
        r = np.concatenate((rp, rr), axis=1)[:, rows] * sw
        JJ = np.concatenate((Jp, Jr), axis=1)[:, rows] * sw[None, :, None]
        pe = np.linalg.norm(rp[:, rows[:3]], axis=1)
        re = np.linalg.norm(rr[:, rows[3:]], axis=1)
        return r, JJ, pe, re

    Q = np.clip(np.asarray(Q0, float).copy(), lo, hi)
    r, J, pe, re = evaluate(Q)
    E = 0.5 * np.sum(r * r, axis=1)
    lam = np.full(N, lam0)
    active = np.ones(N, bool)
    I = np.eye(n)
    E_ref = E.copy()
    for it in range(max_iters):
        done = (pe < tol_pos * POLISH) & (re < tol_rot * POLISH)
        active &= ~done
        if stall_ratio is not None and it and it % stall_every == 0:
            active &= ~(E > stall_ratio * E_ref)
            E_ref = E.copy()
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        Ji, ri, qi = J[idx], r[idx], Q[idx]
        H = np.transpose(Ji, (0, 2, 1)) @ Ji
        g = np.einsum('nji,nj->ni', Ji, ri)
        D = (lam[idx] * E[idx] + lam_bar)[:, None, None] * I
        dq = -np.linalg.solve(H + D, g[..., None])[..., 0]
        blk = ((qi <= lo + LIMIT_EPS) & (dq < 0)) | ((qi >= hi - LIMIT_EPS) & (dq > 0))
        if blk.any():
            free = (~blk).astype(float)
            Hf = H * free[:, :, None] * free[:, None, :] + D
            dq = -np.linalg.solve(Hf, (g * free)[..., None])[..., 0] * free
        m = np.max(np.abs(dq), axis=1)
        dq *= np.where(m > max_step, max_step / np.maximum(m, 1e-300), 1.0)[:, None]
        qn = np.clip(qi + dq, lo, hi)
        rn, Jn, pen, ren = evaluate(qn, idx)
        En = 0.5 * np.sum(rn * rn, axis=1)
        acc = En < E[idx]
        a = idx[acc]
        Q[a], r[a], J[a], pe[a], re[a], E[a] = qn[acc], rn[acc], Jn[acc], pen[acc], ren[acc], En[acc]
        lam[a] = np.maximum(lam[a] * 0.3, 1e-6)
        rj = idx[~acc]
        lam[rj] *= 8.0
        active[rj[lam[rj] > 1e10]] = False
    ok = (pe < tol_pos) & (re < tol_rot)
    return Q, ok, pe, re
