"""A small CMA-ES (Hansen's (mu/mu_w, lambda) with rank-one and rank-mu updates).

Box bounds are handled by mirroring samples into the box. Several independent
problems can run in lockstep (one per joint) so one batched simulation call
scores every candidate of every problem.
"""

from __future__ import annotations

import numpy as np


class CMAES:

    def __init__(self, x0, sigma0, lower, upper, popsize=None, seed=0):
        self.x = np.asarray(x0, float)
        self.n = len(self.x)
        self.lower = np.asarray(lower, float)
        self.upper = np.asarray(upper, float)
        self.sigma = float(sigma0)
        self.rng = np.random.default_rng(seed)
        n = self.n
        self.lam = popsize or 4 + int(3 * np.log(n))
        self.mu = self.lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / np.sum(self.w ** 2)
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chin = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.C = np.eye(n)
        self.B = np.eye(n)
        self.D = np.ones(n)
        self.gen = 0
        self.best_x = self.x.copy()
        self.best_f = np.inf
        self.history = []
        self.best_history = []      # best-so-far after each generation

    def ask(self):
        z = self.rng.standard_normal((self.lam, self.n))
        y = z @ (self.B * self.D).T
        X = self.x + self.sigma * y
        span = self.upper - self.lower
        # mirror into the box
        t = np.mod(X - self.lower, 2 * span)
        X = self.lower + np.where(t > span, 2 * span - t, t)
        return np.clip(X, self.lower, self.upper)

    def tell(self, X, f):
        f = np.asarray(f, float)
        order = np.argsort(f)
        if f[order[0]] < self.best_f:
            self.best_f = float(f[order[0]])
            self.best_x = X[order[0]].copy()
        self.history.append(float(f[order[0]]))
        self.best_history.append(self.best_f)
        old = self.x
        Xs = X[order[:self.mu]]
        self.x = self.w @ Xs
        y = (self.x - old) / self.sigma
        Cinvsqrt = self.B @ np.diag(1 / self.D) @ self.B.T
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * Cinvsqrt @ y
        self.gen += 1
        hsig = (np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * self.gen)) / self.chin
                < 1.4 + 2 / (self.n + 1))
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y
        Y = (Xs - old) / self.sigma
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * (Y.T * self.w) @ Y)
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chin - 1))
        self.sigma = float(min(self.sigma, 1e3))
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        D2, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.maximum(D2, 1e-20))

    def plateaued(self, rtol=0.01, window=15):
        """Best-so-far cost improved by less than rtol over the last `window` generations."""
        b = self.best_history
        if len(b) <= window:
            return False
        return (b[-window - 1] - b[-1]) <= rtol * max(abs(b[-1]), 1e-30)

    def converged(self, ftol=1e-9, window=10, sigma_tol=1e-4):
        if self.sigma * self.D.max() < sigma_tol:
            return True
        if len(self.history) > window:
            h = self.history[-window:]
            return (max(h) - min(h)) <= ftol * max(abs(min(h)), 1e-30)
        return False
