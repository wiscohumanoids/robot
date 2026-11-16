"""Levenberg-Marquardt least squares, numpy only (the stack avoids a scipy
dependency: the ROS 2 Humble image ships numpy 1.21, which current scipy
wheels reject)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LMResult:
    x: np.ndarray
    cost: float
    iterations: int
    converged: bool


def numeric_jacobian(f, x, r0=None, h=1e-6):
    r0 = f(x) if r0 is None else r0
    J = np.empty((len(r0), len(x)))
    for i in range(len(x)):
        dx = np.zeros_like(x)
        dx[i] = h * max(1.0, abs(x[i]))
        J[:, i] = (f(x + dx) - r0) / dx[i]
    return J


def levenberg_marquardt(f, x0, jac=None, max_iter=100, xtol=1e-10, ftol=1e-12, lam0=1e-3) -> LMResult:
    """Minimize 0.5 |f(x)|^2. `jac(x, r)` defaults to forward differences.
    Parameters should be scaled to order 1 by the caller."""
    x = np.asarray(x0, float).copy()
    jac = jac or (lambda z, r: numeric_jacobian(f, z, r))
    r = f(x)
    cost = 0.5 * float(r @ r)
    lam = lam0
    for it in range(1, max_iter + 1):
        J = jac(x, r)
        g = J.T @ r
        A = J.T @ J
        diag = np.maximum(np.diag(A), 1e-12)
        improved = False
        for _ in range(20):
            step = -np.linalg.solve(A + lam * np.diag(diag), g)
            x_new = x + step
            r_new = f(x_new)
            c_new = 0.5 * float(r_new @ r_new)
            if c_new < cost:
                improved = True
                break
            lam *= 10.0
        if not improved:
            return LMResult(x, cost, it, True)
        rel = (cost - c_new) / max(cost, 1e-300)
        x, r, cost = x_new, r_new, c_new
        lam = max(lam / 3.0, 1e-12)
        if np.linalg.norm(step) < xtol * (np.linalg.norm(x) + xtol) or rel < ftol:
            return LMResult(x, cost, it, True)
    return LMResult(x, cost, max_iter, False)
