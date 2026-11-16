"""Deterministic compact critic observations; the actor retains its native images."""
from pathlib import Path

import numpy as np
from PIL import Image

from .contracts import CAMERAS


class ObservationEncoder:
    def __init__(self, state_normalizer, pool_size=16):
        if type(pool_size) is not int or pool_size < 1 or 256 % pool_size:
            raise ValueError('pool_size must divide the 256 pixel image size')
        self.state_normalizer = state_normalizer
        self.pool_size = pool_size
        self.obs_dim = 14 + len(CAMERAS) * 3 * pool_size * pool_size

    def encode(self, observation, root=None):
        state = np.asarray(observation.get('state'), dtype=np.float32)
        if state.shape != (14,):
            raise ValueError('Observation requires native 14D measured state')
        features = [self.state_normalizer.normalize(state)]
        images = observation.get('images', {})
        if set(images) != set(CAMERAS):
            raise ValueError('Require exactly overhead, wrist_port2, and wrist_port3 images')
        for camera in CAMERAS:
            image = images[camera]
            if isinstance(image, str):
                if root is None:
                    raise ValueError('Image paths require an explicit ledger root')
                base = Path(root).resolve()
                path = (base / image).resolve()
                if not path.is_relative_to(base):
                    raise ValueError('Image reference escapes ledger root')
                with Image.open(path) as source:
                    if source.mode != 'RGB':
                        raise ValueError('Ledger image must be RGB')
                    image = np.asarray(source)
            image = np.asarray(image)
            if image.shape != (256, 256, 3) or image.dtype != np.uint8:
                raise ValueError('Require exact cropped 256x256 RGB uint8 images')
            size = self.pool_size
            block = 256 // size
            pooled = image.astype(np.float32).reshape(size, block, size, block, 3).mean(axis=(1, 3)) / 255.0
            features.append(pooled.transpose(2, 0, 1).reshape(-1))
        result = np.concatenate(features).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError('Encoded observation is nonfinite')
        return result
