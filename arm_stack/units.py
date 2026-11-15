"""Unit conversion between drive values and SI (spec: torque conversion).

Every command passes through

    tau_permille = 1000 * tau_out / (Kt_out * I_rated)

with Kt measured at the output (calibration `kt_output_nm_per_a`, stage 4) and
I_rated from the joint config. Until stage 4 has run, the model table's
nominal Kt is used and `kt_source` says so. Raw values are kept next to
converted ones in every log so a unit fix can reprocess old data.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np

from .config import ArmConfig


class Units:

    def __init__(self, arm: ArmConfig, calibration: Optional[Dict] = None):
        calibration = calibration or {}
        kt_cal = calibration.get('kt_output_nm_per_a') or {}
        zeros = calibration.get('zero_offsets_rad') or {}
        self.names = arm.names
        self.direction = np.array([j.direction for j in arm.joints], float)
        self.counts_per_rad = np.array([j.counts_per_rev / (2 * math.pi) for j in arm.joints])
        self.rated_a = np.array([j.rated_current_a for j in arm.joints])
        self.kt = np.array([kt_cal.get(j.name) or j.kt_nominal_nm_per_a for j in arm.joints])
        self.kt_source = ['calibrated' if kt_cal.get(j.name) else 'nominal' for j in arm.joints]
        self.zero = np.array([float(zeros.get(n) or 0.0) for n in self.names])

    # position: q = direction * counts / counts_per_rad - zero
    def counts_to_rad(self, counts: Sequence[float]) -> np.ndarray:
        return self.direction * np.asarray(counts, float) / self.counts_per_rad - self.zero

    def rad_to_counts(self, q: Sequence[float]) -> np.ndarray:
        return np.round(self.direction * (np.asarray(q, float) + self.zero) * self.counts_per_rad)

    def counts_vel_to_rad(self, counts_per_s: Sequence[float]) -> np.ndarray:
        return self.direction * np.asarray(counts_per_s, float) / self.counts_per_rad

    # torque
    def nm_per_permille(self) -> np.ndarray:
        return self.kt * self.rated_a / 1000.0

    def nm_to_permille(self, tau: Sequence[float]) -> np.ndarray:
        return np.round(self.direction * np.asarray(tau, float) / self.nm_per_permille())

    def permille_to_nm(self, permille: Sequence[float]) -> np.ndarray:
        return self.direction * np.asarray(permille, float) * self.nm_per_permille()

    def permille_to_amps(self, permille: Sequence[float]) -> np.ndarray:
        return np.asarray(permille, float) * self.rated_a / 1000.0
