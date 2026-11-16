"""Shared pieces for the calibration stages."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..config import ArmConfig
from ..hw.client import ArmClient
from ..model.robot import ArmModel

DEG = math.pi / 180.0


@dataclass
class StageContext:
    arm: ArmConfig
    client: ArmClient
    calibration: Dict[str, Any]
    model: ArmModel                  # nominal URDF + current calibration
    sim: bool
    quick: bool = False              # fewer poses/cycles (tests, smoke runs)
    log: Optional[Any] = None        # log.recorder.Recorder
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def note(self, text: str):
        if self.log is not None:
            self.log.event('note', text=text)


@dataclass
class StageResult:
    stage: str
    passed: bool
    metrics: Dict[str, Any]
    writes: Dict[str, Any]           # calibration fields to write
    lines: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    joint_config_updates: Dict[str, Any] = field(default_factory=dict)
    skipped: bool = False            # not applicable to this arm (counts as passed)

    def report(self) -> str:
        head = '%s: %s' % (self.stage, 'SKIP' if self.skipped else ('PASS' if self.passed else 'FAIL'))
        out = [head] + ['  ' + l for l in self.lines]
        if self.flags:
            out += ['  flag: ' + f for f in self.flags]
        return '\n'.join(out)


def skipped(stage: str, why: str) -> StageResult:
    return StageResult(stage, True, {'skipped': True, 'reason': why}, {}, [why], skipped=True)


def safe_home(model: ArmModel) -> np.ndarray:
    """The arm's home pose (joint config `home_pose_rad`, else q = 0), 3 deg inside the limits.

    arm01's q = 0 is the arm stretched out forward, its worst gravity load, so its home is
    the hanging CAD capture pose."""
    margin = 3 * DEG
    margin = np.minimum(margin, 0.2 * (model.upper - model.lower))
    home = getattr(model, 'home', None)
    base = np.zeros(model.n) if home is None else home
    return np.clip(base, model.lower + margin, model.upper - margin)


def random_poses(model: ArmModel, count: int, rng, margin=10 * DEG, spread=1.0) -> np.ndarray:
    margin = np.minimum(margin, 0.2 * (model.upper - model.lower))
    lo = model.lower + margin
    hi = model.upper - margin
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo) * spread
    return rng.uniform(mid - half, mid + half, (count, model.n))


def r_squared(y, yhat) -> float:
    y = np.asarray(y)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def feasible_poses(model: ArmModel, count: int, rng, max_torque, fraction=0.6, margin=10 * DEG,
                   spread=0.8, extra_mass: float = 0.0, tries: int = 50):
    """Random poses whose gravity load (plus `extra_mass` kg at the tool) stays
    under `fraction` of each joint's torque limit, so Mode P holds them."""
    limit = fraction * np.asarray(max_torque, float)
    out = []
    for _ in range(count * tries):
        q = random_poses(model, 1, rng, margin=margin, spread=spread)[0]
        g = np.abs(model.gravity(q))
        if extra_mass:
            J = model.jacobian(q)[:3]
            g = g + np.abs(J.T @ np.array([0, 0, 9.81 * extra_mass]))
        if np.all(g < limit):
            out.append(q)
            if len(out) == count:
                break
    return np.array(out)


def mount_payload(ctx: StageContext, payload) -> None:
    """Hang (or remove, payload=None) a test mass: the sim does it, an operator on hardware."""
    try:
        ctx.client.set_payload(payload)
    except NotImplementedError:
        if payload is None:
            input('Remove the test mass, then press Enter ')
        else:
            input('Hang %.2f kg on %s at %s m, then press Enter '
                  % (payload.mass_kg, payload.link, list(payload.offset_m)))


def chain_skip(name: str, client) -> Optional[str]:
    """`arm all`: the kinematics and repeatability stages that do not apply to
    this client's measurement source, with the reason (None: run it). With
    mocap: mocap-calib and mocap-repeat; without: plate and the cube."""
    if name not in ('plate', 'repeat', 'mocap-calib', 'mocap-repeat'):
        return None
    mocap = client.mocap() is not None
    if mocap and name == 'plate':
        return 'skipped: mocap present, mocap-calib does the kinematics (run `arm plate` alone to use the plate)'
    if mocap and name == 'repeat':
        return 'skipped: mocap present, mocap-repeat measures repeatability'
    if not mocap and name == 'mocap-calib':
        return 'skipped: no mocap, the kinematics come from `arm plate`'
    if not mocap and name == 'mocap-repeat':
        return ('skipped: no mocap, so no repeatability tuning (it needs a source that sees every pose every '
                'cycle without an operator); `arm repeat` and `arm validate` measure RP with the indicator cube')
    return None
