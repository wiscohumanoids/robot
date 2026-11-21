"""The GPU VM's perception servers (services/perception_server) as Segmenters.

Two servers with one API, side by side on the VM:

  dino_sam2   Grounding DINO (text -> boxes) + SAM 2.1 (box -> mask)   127.0.0.1:8100   PERCEPTION_SERVER_URL
  sam3        SAM 3 (text -> every matching instance's mask)           127.0.0.1:8101   SAM3_SERVER_URL

Same wire format as the ROS client (`perception_server/protocol.py`): the colour image, the query as the one prompt,
masks back as run-length encodings. No depth is sent, so the servers return 2D masks only; the geometry here turns
them into positions. The kitchen post-processing (3D size priors, the SigLIP crop check against kitchen vocabulary)
is switched off per request: it knows nothing about blocks on a table. Mask NMS stays on.

Both listen on the VM's localhost; tools/tunnel.sh forwards both ports from this machine.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import urllib.error
import urllib.request

from perception.base import COLOURS, Instance, words
from robots.base import Observation

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / 'services' / 'perception_server'))
from perception_server import protocol, rle  # noqa: E402

# The colours prompted against a colour query for Grounding DINO (see `colour_distractors`).
DISTRACTOR_COLOURS = ['red', 'green', 'blue', 'yellow', 'orange', 'white', 'black']

# name -> (URL variable, default URL (the tunnel's local end), the backend /health must report, description)
SERVERS = {
    'dino_sam2': ('PERCEPTION_SERVER_URL', 'http://127.0.0.1:8100', 'grounded_sam2', 'Grounding DINO + SAM 2.1'),
    'sam3': ('SAM3_SERVER_URL', 'http://127.0.0.1:8101', 'sam3', 'SAM 3'),
}


class ServerSegmenter:
    """POST /detect to one of the perception servers."""

    def __init__(self, name: str = 'dino_sam2', url: str | None = None, timeout: float = 30.0,
                 box_threshold: float | None = None, colour_distractors: bool | None = None):
        env, default, self.backend, self.description = SERVERS[name]
        self.name = name
        self.url = (url or os.environ.get(env) or default).rstrip('/')
        self.timeout = timeout
        self.box_threshold = box_threshold          # None: the server's default (DINO 0.30, SAM 3 0.5)
        self.last_timing: dict = {}                 # the last reply's timing_ms (decode, detect, segment, total)
        # Grounding DINO scores "red cube" nearly as high on a green cube: alone it returns every cube for a colour
        # query. Prompting the other colours too ("green cube", "blue cube", ...) as the server's distractors lets
        # them win the cubes they describe better. SAM 3 tells colours apart by itself.
        self.colour_distractors = (name == 'dino_sam2') if colour_distractors is None else colour_distractors
        self._checked = False

    def health(self) -> dict:
        """The server's /health reply (raises ConnectionError if it cannot be reached)."""
        try:
            with urllib.request.urlopen(self.url + '/health', timeout=5) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, OSError) as e:
            raise ConnectionError(f'{self.description} server at {self.url} unreachable ({e}): is the tunnel up '
                                  f'(tools/tunnel.sh) and the server running on the VM?') from e

    def _check(self) -> None:
        """Once: the server answers and is the backend this segmenter is named for (catches swapped ports)."""
        h = self.health()
        if h.get('backend') != self.backend:
            raise ConnectionError(f'{self.url} runs backend {h.get("backend")!r}, expected {self.backend!r} '
                                  f'({self.description})')
        self._checked = True

    def segment(self, obs: Observation, query: str) -> list[Instance]:
        """Detections for the query, with masks, largest first."""
        if not self._checked:
            self._check()
        req = {'rgb_jpeg_b64': protocol.encode_rgb_jpeg(obs.rgb), 'prompts': [query], 'return_masks': True,
               'postprocess': {'priors': False, 'verify': False, 'mask_nms': True}}
        if self.box_threshold is not None:
            req['box_threshold'] = self.box_threshold
        q = words(query)
        if self.colour_distractors and any(w in COLOURS for w in q):
            req['distractors'] = [' '.join(c if w in COLOURS else w for w in q) for c in DISTRACTOR_COLOURS
                                  if c not in q]
            req['distractor_mode'] = 'incontext'            # one prompt: the colours compete for each box
        body = json.dumps(req).encode()
        http = urllib.request.Request(self.url + '/detect', data=body, headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(http, timeout=self.timeout) as r:
                reply = json.loads(r.read())
        except (urllib.error.URLError, OSError) as e:
            raise ConnectionError(f'{self.description} server at {self.url} failed ({e})') from e
        self.last_timing = reply.get('timing_ms', {})
        out = []
        for d in reply.get('detections', []):
            if d.get('mask_rle'):
                mask = rle.decode(d['mask_rle']).astype(bool)
                if mask.shape == obs.rgb.shape[:2] and mask.any():
                    out.append(Instance(mask, d['label'], float(d['score'])))
        return sorted(out, key=lambda i: -int(i.mask.sum()))


if __name__ == '__main__':                     # check both servers: .venv/bin/python -m perception.server_client
    for n in SERVERS:
        try:
            print(n, ServerSegmenter(n).health())
        except ConnectionError as e:
            print(n, 'DOWN:', e)
