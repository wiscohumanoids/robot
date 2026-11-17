"""Ankle joint torque -> torque on each of the two ankle motors (numpy version).

tau_motor = solve(J(q)^T, tau_joint), with J = d(crank angle)/d(joint angle) bilinearly
interpolated from the grid in ankle_linkage.json (made by linkage_tools.py).
The mjlab penalty (training/locomotion/wisco_v1/wisco_mjlab/linkage_penalty.py) does the same in torch.
"""
import json

import numpy as np

from robot_params import LINKAGE_OUT

_INFO = json.loads(LINKAGE_OUT.read_text())
PITCH = np.array(_INFO["grid_pitch"])
ROLL = np.array(_INFO["grid_roll"])
GRID = {s: np.array(_INFO["sides"][s]["grid"]) for s in ("left", "right")}


def jacobian(side, pitch, roll):
    """Crank-to-joint Jacobian at (pitch, roll), bilinearly interpolated from the precomputed grid."""
    g = GRID[side]
    p = np.clip(pitch, PITCH[0], PITCH[-1]); r = np.clip(roll, ROLL[0], ROLL[-1])
    i = min(np.searchsorted(PITCH, p) - 1, len(PITCH) - 2); i = max(i, 0)
    k = min(np.searchsorted(ROLL, r) - 1, len(ROLL) - 2); k = max(k, 0)
    tp = (p - PITCH[i]) / (PITCH[i + 1] - PITCH[i]); tr = (r - ROLL[k]) / (ROLL[k + 1] - ROLL[k])
    return ((1 - tp) * (1 - tr) * g[i, k] + tp * (1 - tr) * g[i + 1, k]
            + (1 - tp) * tr * g[i, k + 1] + tp * tr * g[i + 1, k + 1])


def motor_torques(side, pitch, roll, tau_pitch, tau_roll):
    """Per-motor torques for ankle joint torques."""
    J = jacobian(side, pitch, roll)
    return np.linalg.solve(J.T, np.array([tau_pitch, tau_roll]))
