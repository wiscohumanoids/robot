"""Joint config and versioned calibration files (spec: "Repo layout and configs").

Joint config: one YAML per arm, checked in (config/arms/<arm>.yaml).
Calibration file: one per arm per calibration run, never edited by hand,
config/calib/<arm>/<date>_v<N>.yaml. Each stage writes a new version that
copies every field it did not measure, so one version number names one set of
values.
"""

from __future__ import annotations

import copy
import datetime as _dt
import glob
import hashlib
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _share_root() -> str:
    """Source checkout or installed share dir, whichever has config/."""
    if os.path.isdir(os.path.join(PACKAGE_ROOT, 'config')):
        return PACKAGE_ROOT
    try:
        from ament_index_python.packages import get_package_share_directory
        return get_package_share_directory('wisco_ethercat_ros2_hw')
    except Exception:  # noqa: BLE001
        return PACKAGE_ROOT


SHARE = _share_root()


def resolve(path: str) -> str:
    """Absolute path; relative paths are relative to the package root."""
    return path if os.path.isabs(path) else os.path.join(SHARE, path)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def file_hash(path: str) -> str:
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def git_sha() -> Optional[str]:
    try:
        out = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=PACKAGE_ROOT,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


# --------------------------------------------------------------------------- joint config
@dataclass
class JointSpec:
    name: str
    bus_position: int
    model: str
    direction: int
    counts_per_rev: int
    rated_current_a: float
    kt_nominal_nm_per_a: float
    max_torque_nm: float
    max_vel_rad_s: float
    peak_torque_nm: float
    kp: float
    kd: float
    nulls: List[str] = field(default_factory=list)  # fields that fell back to the model table
    serial: Optional[str] = None                    # actuator serial (bench rig runs)


@dataclass
class ArmConfig:
    arm: str
    path: str
    urdf: str
    control_mode: str
    cycle_us: int
    joints: List[JointSpec]
    link_masses_kg: Dict[str, Optional[float]]
    raw: Dict[str, Any]

    @property
    def names(self) -> List[str]:
        return [j.name for j in self.joints]

    @property
    def n(self) -> int:
        return len(self.joints)

    @property
    def dt(self) -> float:
        return self.cycle_us * 1e-6

    def hash(self) -> str:
        return file_hash(self.path)


def arm_path(name_or_path: str) -> str:
    """`bench01` -> config/arms/bench01.yaml; paths pass through."""
    if os.sep in name_or_path or '/' in name_or_path or name_or_path.endswith(('.yaml', '.yml')):
        return name_or_path
    return os.path.join('config', 'arms', name_or_path + '.yaml')


def load_arm(path: str = 'config/arms/arm01.yaml',
             models_path: str = 'config/actuator_models.yaml',
             model: Optional[str] = None, serial: Optional[str] = None) -> ArmConfig:
    """model / serial override every joint row's `model` / `serial` (bench rig: one
    joint, whichever actuator is mounted today)."""
    path = resolve(arm_path(path))
    raw = load_yaml(path)
    models = load_yaml(resolve(models_path)).get('models', {})
    if model is not None or serial is not None:
        raw = copy.deepcopy(raw)
        for row in raw['joints']:
            if model is not None:
                if model not in models:
                    raise ValueError('unknown actuator model %r (config/actuator_models.yaml has %s)'
                                     % (model, sorted(models)))
                row['model'] = model
            if serial is not None:
                row['serial'] = str(serial)
    joints = []
    for row in sorted(raw['joints'], key=lambda r: int(r['bus_position'])):
        model = models[row['model']]
        nulls = []

        def pick(key, fallback, row=row, nulls=nulls):
            value = row.get(key)
            if value is None:
                nulls.append(key)
                return fallback
            return value

        gains = row.get('gains') or {}
        rated = float(pick('rated_current_a', model.get('rated_current_a', 10.0)))
        # The RT clamps may be raised from the bring-up limits toward the datasheet
        # ratings after Phase 1, never past them.
        for key, cap_key in (('max_torque_nm', 'rated_torque_nm'),
                             ('max_vel_rad_s', 'no_load_velocity_rad_s')):
            value, cap = row.get(key), model.get(cap_key)
            if value is not None and cap is not None and float(value) > float(cap):
                raise ValueError('%s: %s %.3g is above the %s datasheet %s %.3g (%s)'
                                 % (row['name'], key, float(value), row['model'], cap_key,
                                    float(cap), path))
        joints.append(JointSpec(
            name=row['name'],
            bus_position=int(row['bus_position']),
            model=row['model'],
            direction=int(row.get('direction', 1)),
            counts_per_rev=int(pick('counts_per_rev', model['counts_per_rev'])),
            rated_current_a=rated,
            kt_nominal_nm_per_a=float(model.get('kt_output_nm_per_a',
                                                1000.0 * model['effort_nm_per_permille'] / rated)),
            max_torque_nm=float(pick('max_torque_nm', model['bringup_effort_limit_nm'])),
            max_vel_rad_s=float(pick('max_vel_rad_s', model['bringup_velocity_limit_rad_s'])),
            peak_torque_nm=float(model['peak_effort_nm']),
            kp=float(gains.get('kp', 0.0)),
            kd=float(gains.get('kd', 0.0)),
            nulls=nulls,
            serial=None if row.get('serial') is None else str(row['serial']),
        ))
    return ArmConfig(
        arm=raw['arm'], path=path, urdf=resolve(raw.get('urdf', 'urdf/arm01/arm01.urdf')),
        control_mode=raw.get('control_mode', 'arm'),
        cycle_us=int((raw.get('bus') or {}).get('cycle_us', 1000)),
        joints=joints, link_masses_kg=dict(raw.get('link_masses_kg') or {}), raw=raw)


# --------------------------------------------------------------------------- calibration
CALIB_FIELDS = (
    'zero_offsets_rad', 'soft_limits_rad', 'kt_output_nm_per_a', 'link_inertials', 'friction',
    'latency_ms', 'joint_stiffness_nm_per_rad', 'kinematic_overlay', 'pace', 'validation',
)


def empty_calibration(arm: ArmConfig) -> Dict[str, Any]:
    names = arm.names
    return {
        'version': 0,
        'arm_serial': arm.arm,
        'date': None,
        'git_sha': None,
        'joint_config_hash': None,
        'stage': None,
        'zero_offsets_rad': {n: 0.0 for n in names},
        'soft_limits_rad': {n: [None, None] for n in names},
        'kt_output_nm_per_a': {n: None for n in names},
        'link_inertials': {},
        'friction': {n: {'coulomb_nm': None, 'viscous_nm_s_per_rad': None} for n in names},
        'latency_ms': None,
        'joint_stiffness_nm_per_rad': {n: None for n in names},
        'kinematic_overlay': None,
        'pace': {'armature': {}, 'viscous': {}, 'friction': {}, 'encoder_bias': {},
                 'delay_steps': None},
        'validation': {'pivot_spread_mm': None, 'repeatability_rp_mm': None,
                       'hold_drift_deg': None, 'replay_rms_deg': None},
    }


class CalibrationStore:
    """config/calib/<arm>/<date>_v<N>.yaml, append-only."""

    PATTERN = re.compile(r'_v(\d+)\.yaml$')

    def __init__(self, arm: ArmConfig, root: Optional[str] = None, serial: Optional[str] = None):
        """serial: one actuator on the bench rig; its versions go to <arm>/<serial>/
        and carry `actuator_serial`, since Kt and friction vary unit to unit."""
        self.arm = arm
        self.serial = serial
        self.dir = os.path.join(root or resolve('config/calib'), arm.arm)
        if serial:
            self.dir = os.path.join(self.dir, re.sub(r'[^A-Za-z0-9_.-]', '_', str(serial)))

    def versions(self) -> List[str]:
        files = glob.glob(os.path.join(self.dir, '*_v*.yaml'))
        return sorted(files, key=lambda p: int(self.PATTERN.search(p).group(1)))

    def latest_path(self) -> Optional[str]:
        v = self.versions()
        return v[-1] if v else None

    def latest(self) -> Dict[str, Any]:
        path = self.latest_path()
        return load_yaml(path) if path else empty_calibration(self.arm)

    def write(self, stage: str, updates: Dict[str, Any], date: Optional[str] = None) -> str:
        """New version = latest with `updates` merged in; returns its path."""
        base = copy.deepcopy(self.latest())
        unknown = set(updates) - set(CALIB_FIELDS)
        if unknown:
            raise ValueError('not calibration fields: %s' % sorted(unknown))
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                merged = dict(base[key])
                merged.update(value)
                base[key] = merged
            else:
                base[key] = value
        base['version'] = int(base.get('version') or 0) + 1
        base['date'] = date or _dt.date.today().isoformat()
        base['git_sha'] = git_sha()
        base['joint_config_hash'] = self.arm.hash()
        base['stage'] = stage
        if self.serial:
            base['actuator_serial'] = str(self.serial)
            base['actuator_model'] = self.arm.joints[0].model if self.arm.n == 1 else None
        os.makedirs(self.dir, exist_ok=True)
        path = os.path.join(self.dir, '%s_v%d.yaml' % (base['date'], base['version']))
        with open(path, 'w') as f:
            f.write('# Written by `arm %s`. Never edit by hand; rerun the stage.\n' % stage)
            yaml.safe_dump(_plain(base), f, sort_keys=False)
        return path


def calibration_hash(path: Optional[str]) -> Optional[str]:
    return file_hash(path) if path else None


def _plain(obj):
    """numpy scalars/arrays -> plain Python for YAML."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if np is not None and isinstance(obj, np.ndarray):
        return [_plain(v) for v in obj.tolist()]
    if np is not None and isinstance(obj, np.generic):
        return obj.item()
    return obj
