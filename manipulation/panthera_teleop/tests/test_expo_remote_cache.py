"""Remote candidate-cache checks using an explicit in-memory backend fixture.

No real /infer request, model framework import, hardware, or AWS operation.
"""
import base64
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from infra.rl.contracts import ActionNormalizer, CAMERAS, normalization_fingerprint
from infra.rl.remote_server import BackendChanged, CONTRACT, NAMES, Runtime
from scripts.cache_expo_candidates import cache_candidates


def jpeg(color):
    output = io.BytesIO()
    Image.new('RGB', (256, 256), color).save(output, format='JPEG', quality=95)
    return output.getvalue()


class BackendFixture:
    """Explicit arithmetic fixture: every request has a distinct real response."""
    def __init__(self, manifest):
        self.manifest = manifest
        self.requests = []
        self.draws = 0
        self.changed_after = None

    def request(self, path, body=None):
        self.requests.append((path, deepcopy(body)))
        checkpoint = self.manifest['checkpoint']
        if self.changed_after is not None and self.draws >= self.changed_after:
            checkpoint += '-changed'
        if path == '/health':
            return dict(status='ready', checkpoint=checkpoint, contract=CONTRACT, names=NAMES)
        self.draws += 1
        actions = np.full((50, 14), self.draws / 100., dtype=np.float32)
        actions[:, [6, 13]] = .5
        actions[0, 6], actions[-1, 13] = 0., 2.
        return dict(sequence=body['sequence'], fps=20, checkpoint=checkpoint, mock=False,
                    actions=actions.tolist(), limit_corrections=[
                        dict(action=1, index=6, raw_target_rad=-self.draws / 100., command_target_rad=0.),
                        dict(action=50, index=13, raw_target_rad=2. + self.draws / 100., command_target_rad=2.)])


class RemoteCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Self-contained fixture; no actual model or workspace attestation is
        # required. All HTTP traffic is injected and outputs are temporary.
        self.manifest_path = self.root / 'fixture-base-manifest.json'
        files = {name: hashlib.sha256(name.encode()).hexdigest() for name in (
            'config.json', 'deployment.json', 'model.safetensors',
            'policy_preprocessor.json', 'policy_postprocessor.json')}
        checkpoint_id = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        processors = {name: digest for name, digest in files.items() if 'processor' in name}
        processor_id = hashlib.sha256(json.dumps(processors, sort_keys=True).encode()).hexdigest()
        normalization = dict(checkpoint_id=checkpoint_id, processor_id=processor_id,
            action=dict(q01=[-.25] * 14, q99=[1.5] * 14, mode='QUANTILES', eps=1e-8),
            state=dict(q01=[-2.] * 14, q99=[2.] * 14, mode='QUANTILES', eps=1e-8))
        self.manifest = dict(schema_version=1, synthetic=False, test_fixture=True, verified=True,
            source_kind='verified_deployed_pi05_checkpoint',
            checkpoint='/fixture/immutable/pretrained_model', contract=CONTRACT, file_manifest=files,
            checkpoint_id=checkpoint_id, processor_id=processor_id, normalization=normalization,
            normalization_sha256=normalization_fingerprint(normalization))
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.client = BackendFixture(self.manifest)
        self.session = self.root / 'fixture-session'
        self.session.mkdir()
        observations = []
        for index, identity in enumerate(('fixture-current', 'fixture-next')):
            images, hashes = {}, {}
            for camera in CAMERAS:
                data = jpeg((index + 20, 30, 40))
                relative = f'images/{identity}/{camera}.jpg'
                path = self.session / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                images[camera] = relative
                hashes[camera] = hashlib.sha256(data).hexdigest()
            observations.append(dict(observation_id=identity, state=[float(index)] * 14,
                                     images=images, image_sha256=hashes))
        self.observations = observations
        self.ledger = self.session / 'decisions.jsonl'
        self.ledger.write_text(json.dumps(dict(observation=observations[0], next_observation=observations[1])) + '\n')
        self.output = self.root / 'candidate-cache'

    def runtime(self, manifest, backend):
        return Runtime(manifest, backend=backend, client=self.client)

    def cache(self):
        return cache_candidates([self.session], None, self.output, backend='http://127.0.0.1:8765',
                                manifest=self.manifest_path, remote_runtime_factory=self.runtime)

    def test_eight_independent_draws_at_each_exact_saved_current_and_next_observation(self):
        original = self.ledger.read_bytes()
        result = self.cache()
        self.assertEqual(result['observations'], 2)
        draws = [body for path, body in self.client.requests if path == '/infer']
        self.assertEqual(len(draws), 16)
        for group, observation in ((draws[:8], self.observations[0]), (draws[8:], self.observations[1])):
            for body in group:
                self.assertEqual(body['state'], observation['state'])
                self.assertEqual(body['sequence'], observation['observation_id'])
                self.assertNotIn('seed', body)
                for camera in CAMERAS:
                    expected = (self.session / observation['images'][camera]).read_bytes()
                    self.assertEqual(base64.b64decode(body['images'][camera]), expected)
        rows = [json.loads(line) for line in (self.output / 'candidates.jsonl').read_text().splitlines()]
        for row in rows:
            candidates = np.asarray(row['base_actions'])
            self.assertEqual(candidates.shape, (8, 50, 14))
            self.assertEqual(len(set(candidates[:, 0, 0])), 8)
            self.assertEqual(len(row['candidate_limit_corrections']), 8)
            self.assertEqual(row['action_transform'], 'checkpoint_postprocess_and_source_gripper_clamp')
        self.assertEqual(self.ledger.read_bytes(), original)

    def test_cache_retains_physical_actions_and_exact_saved_normalization_without_seed_claim(self):
        result = self.cache()
        self.assertIsNone(result['seed'])
        self.assertIs(result['seed_controlled'], False)
        self.assertIs(result['base_actions_normalized'], False)
        self.assertEqual(result['action_units'], 'rad')
        saved = json.loads((self.output / 'normalization.json').read_text())
        self.assertEqual(saved, self.manifest['normalization'])
        self.assertEqual(result['normalization_sha256'], normalization_fingerprint(saved))
        normalizer = ActionNormalizer.from_stats(saved['action'])
        row = json.loads((self.output / 'candidates.jsonl').read_text().splitlines()[0])
        native = np.asarray(row['base_actions'], dtype=np.float32)
        np.testing.assert_allclose(normalizer.denormalize(normalizer.normalize(native)), native, atol=2e-7)
        self.assertEqual(row['candidates_provenance']['sampling_backend'], 'independent_runtime_noise')
        self.assertIs(row['candidates_provenance']['seed_controlled'], False)
        self.assertNotIn('seed', row['candidates_provenance'])

    def test_changed_checkpoint_mid_sampling_rejects_and_removes_partial_cache(self):
        self.client.changed_after = 3
        with self.assertRaisesRegex(BackendChanged, 'Backend checkpoint changed'):
            self.cache()
        self.assertEqual(self.client.draws, 3)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob('.candidate-cache.partial-*')))

    def test_corrupt_exact_jpeg_rejects_before_any_backend_request(self):
        path = self.session / self.observations[0]['images'][CAMERAS[0]]
        path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.cache()
        self.assertEqual(self.client.requests, [])

    def test_remote_cli_rejects_seed_control_claim_and_does_not_request_backend(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/cache_expo_candidates.py'),
            '--session', str(self.session), '--output', str(self.output), '--backend', 'http://127.0.0.1:1',
            '--manifest', str(self.manifest_path), '--seed', '7'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('omit --checkpoint and --seed', result.stderr)
        self.assertFalse(self.output.exists())

    def test_local_and_remote_inputs_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, 'no local checkpoint'):
            cache_candidates([self.session], '/unused/local/checkpoint', self.output,
                             backend='http://127.0.0.1:1', manifest=self.manifest_path)


if __name__ == '__main__':
    unittest.main()
