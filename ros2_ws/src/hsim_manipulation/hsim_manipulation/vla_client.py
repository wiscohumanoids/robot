"""HTTP client for services/vla_server (GR00T). Standard library + numpy only, so it runs in the ROS image.

Images are sent as PNG built with zlib (no Pillow/OpenCV needed). A 320x240 frame is ~150-250 kB as PNG;
fine over the SSH tunnel at one request per action chunk (every 0.4 s).
"""
import base64
import json
import struct
import urllib.request
import zlib

import numpy as np


def png_b64(rgb):
    """RGB uint8 (H, W, 3) -> base64 PNG."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, _ = rgb.shape
    raw = b''.join(b'\x00' + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data) & 0xFFFFFFFF)
    png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)) \
        + chunk(b'IDAT', zlib.compress(raw, 3)) + chunk(b'IEND', b'')
    return base64.b64encode(png).decode()


def resize_nearest(rgb, w, h):
    """Nearest-neighbour resize of an RGB array to (w, h); returns the input when it already has that size."""
    H, W, _ = rgb.shape
    if (W, H) == (w, h):
        return rgb
    ys = (np.arange(h) * H / h).astype(int)
    xs = (np.arange(w) * W / w).astype(int)
    return rgb[ys][:, xs]


class VlaClient:
    """Minimal HTTP client for services/vla_server (health and act)."""
    def __init__(self, url, timeout=10.0):
        """Server URL and timeout."""
        self.url = url.rstrip('/')
        self.timeout = timeout

    def _call(self, path, payload=None):
        """GET `path` (POST when `payload` is given, as JSON) and return the decoded JSON reply."""
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(self.url + path, data=data, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def health(self):
        """Server health."""
        return self._call('/health')

    def act(self, instruction, ego_rgb, state):
        """state: {key: list[float]} -> (actions {key: (H, D) array}, server latency ms)."""
        r = self._call('/act', {'instruction': instruction, 'images': {'ego_view': png_b64(ego_rgb)},
                                'state': {k: [float(x) for x in v] for k, v in state.items()}})
        if 'error' in r:
            raise RuntimeError(r['error'])
        return {k: np.asarray(v, dtype=np.float64) for k, v in r['actions'].items()}, float(r.get('latency_ms', 0.0))
