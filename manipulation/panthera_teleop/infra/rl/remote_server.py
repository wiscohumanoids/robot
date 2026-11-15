#!/usr/bin/env python3
"""Local EXPO collection proxy for an already loaded frozen pi0.5 backend.

This module imports no model framework, initializes no CUDA context, and makes
no network request at construction. Initial collection samples exactly one real
50-step chunk and executes a configured prefix. Offline callers can sample eight independent
backend draws at the same recorded observation, without creating another model.
"""
import argparse
import base64
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = json.loads((ROOT / 'infra/inference/contract.json').read_text())
CAMERAS = tuple(CONTRACT['image_crops_xyxy'])
NAMES = [name for side in ('left', 'right') for name in
         [*[f'{side}.joint{i}' for i in range(1, 7)], f'{side}.gripper_rad']]
HORIZON, ACTION_DIM = 50, 14


class BackendChanged(RuntimeError):
    """Backend no longer matches the frozen, locally attested base manifest."""


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _finite_vector(values, length):
    return (isinstance(values, list) and len(values) == length and
            all(type(value) in (int, float) and math.isfinite(value) for value in values))


def read_manifest(path):
    manifest = json.loads(Path(path).resolve(strict=True).read_text())
    if (manifest.get('schema_version') != 1 or manifest.get('synthetic') is not False or
            manifest.get('verified') is not True or
            manifest.get('source_kind') != 'verified_deployed_pi05_checkpoint'):
        raise ValueError('Require a nonsynthetic frozen-base manifest schema version 1')
    checkpoint = manifest.get('checkpoint', manifest.get('checkpoint_path'))
    if not isinstance(checkpoint, str) or not checkpoint.startswith('/'):
        raise ValueError('Manifest checkpoint must be the exact absolute deployed checkpoint path')
    if manifest.get('contract') != CONTRACT:
        raise ValueError('Manifest differs from the native Panthera camera/task/action contract')
    files = manifest.get('file_manifest', manifest.get('file_hashes'))
    required = {'config.json', 'deployment.json', 'model.safetensors',
                'policy_preprocessor.json', 'policy_postprocessor.json'}
    if not isinstance(files, dict) or not required.issubset(files):
        raise ValueError('Require verified checkpoint, deployment and processor file hashes')
    if any(not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts or
           not isinstance(digest, str) or len(digest) != 64 or
           any(char not in '0123456789abcdef' for char in digest) for name, digest in files.items()):
        raise ValueError('Invalid checkpoint file hash manifest')
    # Match PI05Adapter identity exactly, including its JSON separator spelling.
    checkpoint_id = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    processors = {name: digest for name, digest in files.items()
                  if 'processor' in name or 'normalizer' in name}
    processor_id = hashlib.sha256(json.dumps(processors, sort_keys=True).encode()).hexdigest()
    if manifest.get('checkpoint_id') != checkpoint_id or manifest.get('processor_id') != processor_id:
        raise ValueError('Manifest checkpoint/processor identities do not match their verified file hashes')
    normalization = manifest.get('normalization', {})
    if (normalization.get('checkpoint_id') != checkpoint_id or
            normalization.get('processor_id') != processor_id or
            manifest.get('normalization_sha256') != canonical_hash(normalization)):
        raise ValueError('Manifest normalization is not bound to this exact checkpoint/processor')
    for feature in ('action', 'state'):
        stats = normalization.get(feature, {})
        if (stats.get('mode') != 'QUANTILES' or
                not _finite_vector(stats.get('q01'), ACTION_DIM) or
                not _finite_vector(stats.get('q99'), ACTION_DIM) or
                any(upper < lower for lower, upper in zip(stats['q01'], stats['q99']))):
            raise ValueError(f'Require genuine saved 14D checkpoint quantiles for {feature}')
        if any(upper == lower for lower, upper in zip(stats['q01'], stats['q99'])):
            if type(stats.get('eps')) not in (int, float) or not math.isfinite(stats['eps']) or stats['eps'] <= 0:
                raise ValueError('Constant quantile dimensions require the saved processor epsilon')
    manifest = dict(manifest, checkpoint=checkpoint, file_manifest=files)
    return manifest


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        raise BackendChanged('Backend redirects are forbidden')


class BackendClient:
    """Bounded loopback-only requests; environment proxies and redirects disabled."""
    def __init__(self, backend, timeout=30.):
        parsed = urlsplit(backend)
        if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', '::1', 'localhost') or
                parsed.username is not None or parsed.password is not None or
                parsed.path not in ('', '/') or parsed.query or parsed.fragment):
            raise ValueError('Backend must be an HTTP loopback service root')
        if not math.isfinite(timeout) or not 0 < timeout <= 120:
            raise ValueError('Backend timeout must be in (0,120] seconds')
        self.backend = backend.rstrip('/')
        self.timeout = timeout
        self.opener = build_opener(ProxyHandler({}), _NoRedirect())

    def request(self, path, body=None):
        if path not in ('/health', '/infer'):
            raise ValueError('Unsupported backend endpoint')
        data = None if body is None else json.dumps(body, allow_nan=False).encode()
        request = Request(self.backend + path, data=data, headers={'Content-Type': 'application/json'})
        with self.opener.open(request, timeout=min(self.timeout, 5.) if body is None else self.timeout) as response:
            data = response.read(4_000_001)
            if len(data) > 4_000_000:
                raise BackendChanged('Backend response exceeds size limit')
            result = json.loads(data)
        if not isinstance(result, dict):
            raise BackendChanged('Backend response must be a JSON object')
        return result


class Runtime:
    def __init__(self, manifest, backend='http://127.0.0.1:8765', *, client=None, timeout=30., prefix_length=1):
        if type(prefix_length) is not int or not 1 <= prefix_length <= HORIZON:
            raise ValueError('Prefix length must be an integer from 1 to 50')
        self.chunk_size = prefix_length
        self.manifest_path = str(Path(manifest).resolve(strict=True))
        self.manifest_sha256 = hashlib.sha256(Path(self.manifest_path).read_bytes()).hexdigest()
        self.manifest = read_manifest(self.manifest_path)
        self.client = client if client is not None else BackendClient(backend, timeout)
        self.backend = backend
        self.lock = threading.Lock()
        self.checkpoint = self.manifest['checkpoint']
        self.checkpoint_id = self.manifest['checkpoint_id']
        self.processor_id = self.manifest['processor_id']
        self.normalization = self.manifest['normalization']
        self.normalizer_hash = self.manifest['normalization_sha256']
        self.policy_bundle_id = canonical_hash(dict(checkpoint_id=self.checkpoint_id,
            processor_id=self.processor_id, normalization_sha256=self.normalizer_hash,
            rl_model_id=None, C=self.chunk_size, collection_candidates=1))

    def _verify_backend(self):
        if hashlib.sha256(Path(self.manifest_path).read_bytes()).hexdigest() != self.manifest_sha256:
            raise BackendChanged('Frozen-base manifest changed after proxy startup')
        health = self.client.request('/health')
        if health.get('checkpoint') != self.checkpoint:
            raise BackendChanged(f'Backend checkpoint changed: expected {self.checkpoint}, '
                                 f'received {health.get("checkpoint")}. Re-verify the deployed '
                                 'checkpoint and start a separately pinned collection session.')
        if (health.get('status') != 'ready' or health.get('mock', False) is not False or
                health.get('checkpoint') != self.checkpoint or health.get('contract') != CONTRACT or
                health.get('names') != NAMES):
            raise BackendChanged('Backend health/path/contract differs from the attested frozen base')
        for key, expected in (('checkpoint_id', self.checkpoint_id), ('processor_id', self.processor_id)):
            if health.get(key) is not None and health[key] != expected:
                raise BackendChanged(f'Backend {key} differs from the frozen-base manifest')
        expected_service = self.manifest.get('backend_service_identity')
        if health.get('backend_service_identity') is not None and health['backend_service_identity'] != expected_service:
            raise BackendChanged('Backend service identity changed')
        return health

    def health(self):
        backend_health = self._verify_backend()
        return dict(status='ready', checkpoint=self.checkpoint, contract=CONTRACT, names=NAMES,
            checkpoint_id=self.checkpoint_id, processor_id=self.processor_id,
            normalizer_hash=self.normalizer_hash, normalization=self.normalization,
            rl_model_id=None, training_objective=None,
            policy_bundle_id=self.policy_bundle_id, mode='frozen_base_remote_collection',
            num_candidates=1, offline_num_candidates=8, chunk_size=self.chunk_size, fps=CONTRACT['fps'], mock=False,
            backend=self.backend, manifest_sha256=self.manifest_sha256,
            backend_service_identity_attested=self.manifest.get('backend_service_identity'),
            live_identity_checks='checkpoint_path_contract' if backend_health.get('checkpoint_id') is None
                                 else 'checkpoint_path_contract_checkpoint_processor',
            live_weight_identity_verified=(backend_health.get('checkpoint_id') == self.checkpoint_id and
                                           backend_health.get('processor_id') == self.processor_id),
            seed_controlled=False, action_transform='checkpoint_postprocess_and_source_gripper_clamp')

    def _observation(self, payload):
        if not isinstance(payload, dict) or 'sequence' not in payload:
            raise ValueError('Expected an inference request with sequence')
        if not _finite_vector(payload.get('state'), ACTION_DIM):
            raise ValueError('Expected 14 finite native measured positions')
        images = payload.get('images')
        if not isinstance(images, dict) or set(images) != set(CAMERAS):
            raise ValueError('Expected all three named cropped camera JPEGs')
        if payload.get('task', CONTRACT['task']) != CONTRACT['task']:
            raise ValueError('Observation task differs from the frozen base')
        from PIL import Image
        for camera, encoded in images.items():
            if not isinstance(encoded, str):
                raise ValueError('Camera JPEG payload must be base64 text')
            with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as image:
                if image.format != 'JPEG' or image.mode != 'RGB' or image.size != (256, 256):
                    raise ValueError('Expected exact cropped 256x256 RGB JPEGs')
                image.load()
        exact = dict(state=payload['state'], images=images, task=CONTRACT['task'])
        digest = canonical_hash(exact)
        observation_id = payload.get('observation_id', digest)
        if not isinstance(observation_id, str) or not observation_id:
            raise ValueError('observation_id must be nonempty text')
        # The backend does not implement seed control. Do not forward a seed or
        # report one as reproducible evidence for its independent runtime noise.
        forwarded = dict(sequence=payload['sequence'], state=payload['state'], images=images)
        return forwarded, observation_id, digest

    def _chunk(self, response, sequence):
        if (response.get('sequence') != sequence or response.get('fps') != CONTRACT['fps'] or
                response.get('mock') is not False or response.get('checkpoint') != self.checkpoint):
            raise BackendChanged('Inference checkpoint/sequence/rate/mock status mismatch')
        actions = response.get('actions')
        if (not isinstance(actions, list) or len(actions) != HORIZON or
                any(not _finite_vector(row, ACTION_DIM) for row in actions)):
            raise BackendChanged('Backend must return exactly 50 finite physical 14D actions')
        if any(not 0 <= row[index] <= 2 for row in actions for index in (6, 13)):
            raise BackendChanged('Backend violates its source gripper-clamp contract')
        corrections = response.get('limit_corrections')
        if not isinstance(corrections, list):
            raise BackendChanged('Backend must preserve its source gripper correction records')
        seen = set()
        for correction in corrections:
            if not isinstance(correction, dict):
                raise BackendChanged('Malformed backend gripper correction')
            action, index = correction.get('action'), correction.get('index')
            raw, command = correction.get('raw_target_rad'), correction.get('command_target_rad')
            if (type(action) is not int or not 1 <= action <= HORIZON or
                    type(index) is not int or index not in (6, 13) or (action, index) in seen or
                    type(raw) not in (int, float) or not math.isfinite(raw) or
                    type(command) not in (int, float) or not math.isfinite(command) or
                    0 <= raw <= 2 or command != min(2., max(0., raw)) or
                    command != actions[action - 1][index]):
                raise BackendChanged('Source gripper correction differs from returned physical command')
            seen.add((action, index))
        return actions

    def sample_candidates(self, payload, num_candidates=8):
        """Offline serial draws; never duplicate a single response into N candidates."""
        if type(num_candidates) is not int or not 1 <= num_candidates <= 8:
            raise ValueError('Candidate count must be an integer from 1 to 8')
        forwarded, observation_id, digest = self._observation(payload)
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('Inference busy')
        try:
            chunks, corrections = [], []
            for _ in range(num_candidates):
                self._verify_backend()
                response = self.client.request('/infer', forwarded)
                chunks.append(self._chunk(response, forwarded['sequence']))
                corrections.append(response.get('limit_corrections', []))
            provenance = dict(kind='pi05_inference', sampling='independent',
                checkpoint_id=self.checkpoint_id, processor_id=self.processor_id,
                observation_id=observation_id, observation_hash=digest,
                observation_hash_kind='native_state_exact_jpeg_transport',
                backend=self.backend, sampling_backend='independent_runtime_noise', seed_controlled=False,
                action_transform='checkpoint_postprocess_and_source_gripper_clamp',
                manifest_sha256=self.manifest_sha256,
                identity_verification='verified_manifest_and_live_checkpoint_path_contract')
            return dict(sequence=payload['sequence'], observation_id=observation_id,
                actions=chunks[0][:self.chunk_size], selected_physical_chunk=chunks[0][:self.chunk_size],
                current_physical_chunk=chunks[0][:self.chunk_size], base_actions=chunks,
                candidates_provenance=provenance, checkpoint=self.checkpoint,
                checkpoint_id=self.checkpoint_id, processor_id=self.processor_id,
                normalizer_hash=self.normalizer_hash, rl_model_id=None, training_objective=None,
                policy_bundle_id=self.policy_bundle_id, selected_candidate=0, selected_is_edited=False,
                num_candidates=num_candidates, chunk_size=self.chunk_size, fps=CONTRACT['fps'], mock=False,
                seed_controlled=False, request_seed_ignored='seed' in payload,
                limit_corrections=[record for record in corrections[0] if record['action'] <= self.chunk_size],
                candidate_limit_corrections=corrections)
        finally:
            self.lock.release()

    def infer(self, payload):
        return self.sample_candidates(payload, num_candidates=1)


def make_handler(runtime):
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body):
            data = json.dumps(body, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path != '/health':
                return self.send(404, {})
            try:
                self.send(200, runtime.health())
            except Exception as exc:
                self.send(503, dict(status='failed', error=str(exc)))

        def do_POST(self):
            if self.path != '/infer':
                return self.send(404, {})
            try:
                size = int(self.headers.get('Content-Length', 0))
                if not 0 < size <= 2_000_000:
                    raise ValueError('Invalid body size')
                self.connection.settimeout(5)
                self.send(200, runtime.infer(json.loads(self.rfile.read(size))))
            except (ValueError, KeyError, TypeError) as exc:
                self.send(400, dict(error=str(exc)))
            except Exception as exc:
                self.send(503, dict(error=str(exc)))
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', default='http://127.0.0.1:8765')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--manifest', help='Verified local frozen-base JSON manifest')
    mode.add_argument('--follow-checkpoint', type=Path, help='Automatically verify deployed checkpoints; cache directory')
    parser.add_argument('--ssh-host', default='admin@3.83.15.102', help='Read-only checkpoint verification host')
    parser.add_argument('--port', type=int, default=8767)
    parser.add_argument('--prefix-length', type=int, default=17, help='Executed prefix length 1..50 (default: 17)')
    parser.add_argument('--timeout', type=float, default=30., help='Bounded backend inference timeout seconds')
    args = parser.parse_args()
    if args.port in (8765, 8766):
        parser.error('Use a separate proxy port; 8765/8766 belong to existing inference services')
    if args.follow_checkpoint:
        from .auto_remote import AutoRuntime
        runtime = AutoRuntime(args.follow_checkpoint, args.backend, ssh_host=args.ssh_host,
                              timeout=args.timeout, prefix_length=args.prefix_length)
    else:
        runtime = Runtime(args.manifest, args.backend, timeout=args.timeout, prefix_length=args.prefix_length)
    ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(runtime)).serve_forever()


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT))
    main()
