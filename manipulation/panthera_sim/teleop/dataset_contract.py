"""Provenance and timing shared by rendering, conversion, and ACT deployment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HZ = 30
DEFAULT_RENDERED = ROOT / 'VLA-Adapter/data/robot_arm_learning_rendered_30hz'
DEFAULT_DATASET = ROOT / 'outputs/lerobot/panthera_stack_30hz'
DEFAULT_CHECKPOINT = ROOT / 'outputs/act/panthera_stack_30hz'


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_signature(path: Path) -> dict:
    return {name: file_hash(path / name) for name in ('data.npz', 'meta.json')}


def render_signature(hz: float, size: int, quality: int) -> dict:
    files = [ROOT / 'teleop/render_vla_dataset.py', Path(__file__),
             ROOT / 'sim/panthera_env.py', ROOT / 'sim/dynamics.py',
             *sorted((ROOT / 'sim/panthera').glob('*.xml'))]
    return {'version': 2, 'sample_hz': hz, 'image_size': [size, size],
            'jpeg_quality': quality,
            'implementation': {str(p.relative_to(ROOT)): file_hash(p) for p in files}}


def validate_rendered(root: Path) -> dict:
    manifest = json.loads((root / 'manifest.json').read_text())
    settings = manifest['render_signature']
    if settings != render_signature(settings['sample_hz'], settings['image_size'][0], settings['jpeg_quality']):
        raise ValueError('Rendered dataset uses outdated rendering code/settings; re-render it.')
    for name in manifest['episodes']:
        saved = json.loads((root / name / 'source.json').read_text())
        if saved['render_signature'] != settings or saved['source_signature'] != source_signature(Path(saved['source'])):
            raise ValueError(f'{name}: source changed since rendering; re-render it.')
    return manifest


def resolve_control_hz(checkpoint: Path, requested: float | None, dataset_hz: float) -> float:
    path = checkpoint / 'deployment.json'
    # Historical ACT checkpoints in this project were trained at 10 Hz.
    trained_hz = float(json.loads(path.read_text())['fps']) if path.exists() else 10.0
    if requested is not None and not np.isclose(requested, trained_hz):
        raise ValueError(f'Checkpoint expects {trained_hz:g} Hz, not {requested:g} Hz.')
    if not np.isclose(dataset_hz, trained_hz):
        raise ValueError(f'Dataset is {dataset_hz:g} Hz but checkpoint expects {trained_hz:g} Hz; select its original dataset.')
    return trained_hz


class PhysicsClock:
    """Distribute fractional physics ticks without long-term control-rate drift."""
    def __init__(self, hz: float, dt: float):
        if not np.isfinite(hz) or hz <= 0 or hz * dt > 1:
            raise ValueError('Control rate must be positive and no faster than physics.')
        self.ratio = 1.0 / (hz * dt)
        self.ticks = 0
        self.steps = 0

    def next_steps(self) -> int:
        self.ticks += 1
        target = round(self.ticks * self.ratio)
        count = target - self.steps
        self.steps = target
        return count
