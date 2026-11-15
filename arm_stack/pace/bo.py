"""Small Gaussian-process Bayesian optimizer (expected improvement), numpy only.

Used by `arm mocap-repeat` to tune Kp, Kd, friction feedforward gain, approach
direction and settle time with ISO 9283 RP as the objective. Trials are
expensive (each one is a short run on the arm), so a few tens of evaluations
is the budget; a dense GP over [0, 1]^d is plenty at that size.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np


class Space:
    """Named box dimensions; ('choice', [a, b]) dimensions are rounded to an index."""

    def __init__(self, dims: Dict[str, Tuple]):
        self.names = list(dims)
        self.dims = dims

    @property
    def d(self) -> int:
        return len(self.names)

    def to_params(self, u: np.ndarray) -> Dict[str, float]:
        out = {}
        for x, name in zip(u, self.names):
            spec = self.dims[name]
            if spec[0] == 'choice':
                options = spec[1]
                out[name] = options[min(int(x * len(options)), len(options) - 1)]
            elif spec[0] == 'log':
                lo, hi = math.log(spec[1]), math.log(spec[2])
                out[name] = float(math.exp(lo + x * (hi - lo)))
            else:
                out[name] = float(spec[1] + x * (spec[2] - spec[1]))
        return out


def _kernel(A, B, length):
    d = (A[:, None, :] - B[None, :, :]) / length
    return np.exp(-0.5 * np.sum(d * d, axis=2))


def _norm_cdf(z):
    return 0.5 * (1 + np.vectorize(math.erf)(z / math.sqrt(2)))


def _norm_pdf(z):
    return np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


class GPBO:
    """Minimize f over a Space. ask() -> unit vector, tell(u, y)."""

    def __init__(self, space: Space, n_init: int = 5, length: float = 0.3, noise: float = 0.05,
                 seed: int = 0):
        self.space = space
        self.n_init = n_init
        self.length = length
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.U: List[np.ndarray] = []
        self.y: List[float] = []

    def _posterior(self, X):
        U = np.array(self.U)
        y = np.array(self.y)
        mu0, sd0 = y.mean(), y.std() or 1.0
        z = (y - mu0) / sd0
        K = _kernel(U, U, self.length) + self.noise ** 2 * np.eye(len(U))
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, z))
        Ks = _kernel(X, U, self.length)
        mu = Ks @ alpha
        v = np.linalg.solve(L, Ks.T)
        var = np.maximum(1.0 - np.sum(v * v, axis=0), 1e-12)
        return mu * sd0 + mu0, np.sqrt(var) * sd0

    def ask(self) -> np.ndarray:
        if len(self.U) < self.n_init:
            return self.rng.uniform(0, 1, self.space.d)
        X = self.rng.uniform(0, 1, (4000, self.space.d))
        best = int(np.argmin(self.y))
        X[:400] = np.clip(self.U[best] + self.rng.normal(0, 0.08, (400, self.space.d)), 0, 1)
        mu, sd = self._posterior(X)
        imp = min(self.y) - mu
        z = imp / sd
        ei = imp * _norm_cdf(z) + sd * _norm_pdf(z)
        return X[int(np.argmax(ei))]

    def tell(self, u: Sequence[float], y: float) -> None:
        self.U.append(np.asarray(u, float))
        self.y.append(float(y))

    def best(self):
        i = int(np.argmin(self.y))
        return self.U[i], self.y[i]
