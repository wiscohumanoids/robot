"""Pinned native Panthera and checkpoint quantile contracts."""
from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

CAMERAS = ('overhead', 'wrist_port2', 'wrist_port3')
STATE_NAMES = tuple(f'{side}.joint{i}' if i < 7 else f'{side}.gripper_rad'
                    for side in ('left', 'right') for i in range(1, 8))
IMAGE_CROPS = {'overhead': [176, 224, 432, 480],
               'wrist_port2': [40, 0, 520, 480], 'wrist_port3': [80, 0, 560, 480]}
OBS_DIM = 14 + len(CAMERAS) * 3 * 16 * 16


class ActionNormalizer:
    """LeRobot QUANTILES formula with exact checkpoint q01/q99, without clipping.

    Zero-width dimensions require the exact explicitly saved processor epsilon;
    this adapter never invents statistics or an epsilon scale.
    """
    def __init__(self, q01, q99, zero_width_epsilon=None):
        self.q01 = np.asarray(q01, dtype=np.float32)
        self.q99 = np.asarray(q99, dtype=np.float32)
        if (self.q01.shape != (14,) or self.q99.shape != (14,) or
                not np.isfinite(self.q01).all() or not np.isfinite(self.q99).all() or
                np.any(self.q99 < self.q01)):
            raise ValueError('Require finite, nondegenerate checkpoint q01/q99 for all 14 dimensions')
        self.denominator = self.q99 - self.q01
        if np.any(self.denominator == 0):
            if zero_width_epsilon is None or not np.isfinite(zero_width_epsilon) or zero_width_epsilon <= 0:
                raise ValueError('Zero-width checkpoint quantiles require the explicitly saved processor epsilon')
            self.denominator = np.where(self.denominator == 0, np.float32(zero_width_epsilon), self.denominator)
        self.zero_width_epsilon = zero_width_epsilon

    @classmethod
    def from_stats(cls, stats):
        if not isinstance(stats, dict) or 'q01' not in stats or 'q99' not in stats:
            raise ValueError('Checkpoint q01/q99 are mandatory; identity normalization is unavailable')
        if stats.get('mode', 'QUANTILES') != 'QUANTILES':
            raise ValueError('This adapter requires the saved QUANTILES normalization mode')
        return cls(stats['q01'], stats['q99'], zero_width_epsilon=stats.get('eps'))

    def _values(self, values):
        values = np.asarray(values, dtype=np.float32)
        if values.ndim < 1 or values.shape[-1] != 14 or not np.isfinite(values).all():
            raise ValueError('Expected finite native/normalized values with final dimension 14')
        return values

    def normalize(self, values):
        values = self._values(values)
        return 2.0 * (values - self.q01) / self.denominator - 1.0

    def denormalize(self, values):
        values = self._values(values)
        return (values + 1.0) * self.denominator / 2.0 + self.q01

    def as_dict(self):
        result = {'q01': self.q01.tolist(), 'q99': self.q99.tolist(), 'mode': 'QUANTILES'}
        if self.zero_width_epsilon is not None:
            result['eps'] = self.zero_width_epsilon
        return result


@dataclass(frozen=True)
class ReplaySpec:
    gamma: float = .99
    C: int = 1
    N: int = 8
    obs_dim: int = OBS_DIM
    checkpoint_id: str = ''
    processor_id: str = ''

    def __post_init__(self):
        if not 0 < self.gamma <= 1 or not np.isfinite(self.gamma):
            raise ValueError('gamma must be finite in (0,1]')
        if any(type(x) is not int or x < 1 for x in (self.C, self.N, self.obs_dim)):
            raise ValueError('C, N, and obs_dim must be positive integers')
        if not self.checkpoint_id or not self.processor_id:
            raise ValueError('Pinned checkpoint_id and processor_id are mandatory')

    def as_dict(self):
        return asdict(self)


def normalization_fingerprint(normalization):
    return hashlib.sha256(json.dumps(normalization, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()
