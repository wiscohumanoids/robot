"""The GPU VM's perception server (services/perception_server: Grounding DINO + SAM 2.1) as a Segmenter.

Same wire format as the ROS client (`perception_server/protocol.py`): the colour image, the query as the one prompt,
masks back as run-length encodings. No depth is sent, so the server returns 2D masks only; the geometry here turns
them into positions. The server's kitchen post-processing (3D size priors, the SigLIP crop check against kitchen
vocabulary) is switched off per request: it knows nothing about blocks on a table. Mask NMS stays on.

The server listens on the VM's 127.0.0.1:8100; reach it with `ssh -N -L 8100:localhost:8100 $VM_HOST`.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import urllib.error
import urllib.request

from perception.base import Instance
from robots.base import Observation

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / 'services' / 'perception_server'))
from perception_server import protocol, rle  # noqa: E402

DEFAULT_URL = 'http://127.0.0.1:8100'          # the SSH tunnel's local end


class ServerSegmenter:
    """POST /detect to the perception server."""

    name = 'server'

    def __init__(self, url: str | None = None, timeout: float = 20.0, box_threshold: float | None = None):
        self.url = (url or os.environ.get('PERCEPTION_SERVER_URL') or DEFAULT_URL).rstrip('/')
        self.timeout = timeout
        self.box_threshold = box_threshold

    def health(self) -> dict:
        """The server's /health reply (raises if it cannot be reached)."""
        with urllib.request.urlopen(self.url + '/health', timeout=5) as r:
            return json.loads(r.read())

    def segment(self, obs: Observation, query: str) -> list[Instance]:
        """Detections for the query, with masks, largest first."""
        req = {'rgb_jpeg_b64': protocol.encode_rgb_jpeg(obs.rgb), 'prompts': [query], 'return_masks': True,
               'postprocess': {'priors': False, 'verify': False, 'mask_nms': True}}
        if self.box_threshold is not None:
            req['box_threshold'] = self.box_threshold
        body = json.dumps(req).encode()
        http = urllib.request.Request(self.url + '/detect', data=body, headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(http, timeout=self.timeout) as r:
                reply = json.loads(r.read())
        except urllib.error.URLError as e:
            raise ConnectionError(f'perception server at {self.url} unreachable ({e}); is the SSH tunnel up?') from e
        out = []
        for d in reply.get('detections', []):
            if d.get('mask_rle'):
                mask = rle.decode(d['mask_rle']).astype(bool)
                if mask.shape == obs.rgb.shape[:2] and mask.any():
                    out.append(Instance(mask, d['label'], float(d['score'])))
        return sorted(out, key=lambda i: -int(i.mask.sum()))


if __name__ == '__main__':                     # check the tunnel: python perception/server_client.py
    print(ServerSegmenter().health())
