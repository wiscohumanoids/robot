"""Automatic checkpoint following with fake health, inference and CPU resolvers."""
import base64
import hashlib
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from PIL import Image

from infra.rl.auto_remote import AutoRuntime
from infra.rl.remote_server import CONTRACT, NAMES, canonical_hash


class FakeBackend:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.calls = []
        self.contract = CONTRACT
        self.block_first = False
        self.started = threading.Event()
        self.release = threading.Event()
        self.overlap = threading.Event()
        self.active = 0
        self.lock = threading.Lock()

    def request(self, path, body=None):
        self.calls.append(path)
        checkpoint = self.checkpoint
        if path == '/health':
            return dict(status='ready', checkpoint=checkpoint, contract=self.contract, names=NAMES)
        with self.lock:
            self.active += 1
            if self.active > 1:
                self.overlap.set()
        try:
            if self.block_first and not self.started.is_set():
                self.started.set()
                if not self.release.wait(2):
                    raise AssertionError('Fake backend release timed out')
            return dict(sequence=body['sequence'], checkpoint=checkpoint, fps=20,
                        mock=False, actions=[[.01] * 14 for _ in range(50)], limit_corrections=[])
        finally:
            with self.lock:
                self.active -= 1


class AutoRemoteTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / 'manifest-cache'
        self.cache.mkdir()
        self.backend = FakeBackend('/test-only/base-A')

    def manifest(self, checkpoint, marker, directory=None):
        files = {name: hashlib.sha256((marker + name).encode()).hexdigest() for name in
                 ('config.json', 'deployment.json', 'model.safetensors',
                  'policy_preprocessor.json', 'policy_postprocessor.json')}
        cid = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        processors = {name: digest for name, digest in files.items() if 'processor' in name}
        pid = hashlib.sha256(json.dumps(processors, sort_keys=True).encode()).hexdigest()
        norm = dict(checkpoint_id=cid, processor_id=pid,
                    action=dict(q01=[-1.] * 14, q99=[1.] * 14, mode='QUANTILES'),
                    state=dict(q01=[-2.] * 14, q99=[2.] * 14, mode='QUANTILES'))
        value = dict(schema_version=1, synthetic=False, verified=True,
                     source_kind='verified_deployed_pi05_checkpoint', checkpoint=checkpoint,
                     checkpoint_id=cid, processor_id=pid, file_manifest=files,
                     normalization=norm, normalization_sha256=canonical_hash(norm), contract=CONTRACT)
        path = (directory or self.cache) / (marker + '.json')
        path.write_text(json.dumps(value))
        return path, value

    def payload(self):
        stream = io.BytesIO()
        Image.new('RGB', (256, 256), '#173042').save(stream, 'JPEG')
        image = base64.b64encode(stream.getvalue()).decode()
        return dict(sequence=5, state=[0.] * 14,
                    images={camera: image for camera in CONTRACT['image_crops_xyxy']})

    def runtime(self, resolver):
        return AutoRuntime(self.cache, ssh_host='test-only-no-ssh', client=self.backend, resolver=resolver)

    def wait(self, predicate):
        deadline = time.perf_counter() + 2
        while not predicate():
            if time.perf_counter() > deadline:
                self.fail('Background fixture did not finish')
            time.sleep(.005)

    def test_cached_manifests_follow_changed_path_and_ids_without_inference_or_resolver(self):
        _, first = self.manifest('/test-only/base-A', 'A')
        _, second = self.manifest('/test-only/base-B', 'B')
        def forbidden(checkpoint):
            raise AssertionError('Verified cache should avoid SSH resolver')
        runtime = self.runtime(forbidden)
        self.assertEqual(self.backend.calls, [])
        self.assertEqual(runtime.health()['checkpoint_id'], first['checkpoint_id'])
        self.backend.checkpoint = '/test-only/base-B'
        health = runtime.health()
        self.assertEqual(health['checkpoint_id'], second['checkpoint_id'])
        self.assertEqual(health['processor_id'], second['processor_id'])
        self.assertEqual(health['chunk_size'], 17)
        self.assertTrue(health['checkpoint_following'])
        self.assertNotIn('/infer', self.backend.calls)
        response = runtime.infer(self.payload())
        self.assertEqual(response['checkpoint_id'], second['checkpoint_id'])
        self.assertEqual(response['candidates_provenance']['checkpoint_id'], second['checkpoint_id'])
        self.assertEqual(len(response['actions']), 17)
        self.assertEqual(len(response['base_actions'][0]), 50)

    def test_unknown_base_is_loading_until_background_verification_completes(self):
        release = threading.Event()
        self.addCleanup(release.set)
        started = threading.Event()
        calls = []
        def resolver(checkpoint):
            calls.append(checkpoint)
            started.set()
            if not release.wait(2):
                raise AssertionError('Resolver fixture release timed out')
            return self.manifest(checkpoint, 'new')[0]
        runtime = self.runtime(resolver)
        self.assertEqual(runtime.health()['status'], 'loading')
        self.assertTrue(started.wait(1))
        self.assertEqual(runtime.health()['status'], 'loading')
        with self.assertRaisesRegex(RuntimeError, 'verifying'):
            runtime.infer(self.payload())
        self.assertEqual(calls, ['/test-only/base-A'])
        self.assertNotIn('/infer', self.backend.calls)
        release.set()
        self.wait(lambda: runtime.pending is None)
        self.assertEqual(runtime.health()['status'], 'ready')

    def test_bad_backend_contract_rejects_before_resolver_and_inference(self):
        self.backend.contract = {}
        runtime = self.runtime(lambda _: self.fail('Bad backend must not resolve'))
        with self.assertRaisesRegex(RuntimeError, 'contract'):
            runtime.health()
        self.assertEqual(self.backend.calls, ['/health'])

    def test_wrong_checkpoint_resolver_is_rejected_and_failure_retry_is_throttled(self):
        calls = []
        wrong, _ = self.manifest('/test-only/wrong-base', 'wrong', self.root)
        def resolver(checkpoint):
            calls.append(checkpoint)
            return wrong
        runtime = self.runtime(resolver)
        with patch('infra.rl.auto_remote.time.monotonic', return_value=100.):
            self.assertEqual(runtime.health()['status'], 'loading')
            self.wait(lambda: runtime.pending is None)
            for _ in range(4):
                self.assertEqual(runtime.health()['status'], 'loading')
            self.assertEqual(calls, ['/test-only/base-A'])
            self.assertIsNone(runtime.runtime)
            self.assertIn('different checkpoint', runtime.error)
            self.assertNotIn('/infer', self.backend.calls)

    def test_switched_runtime_cannot_overlap_prior_inflight_inference(self):
        self.manifest('/test-only/base-A', 'A')
        self.manifest('/test-only/base-B', 'B')
        runtime = self.runtime(lambda _: self.fail('Cache is already verified'))
        self.backend.block_first = True
        results, errors = [], []
        def infer():
            try:
                results.append(runtime.infer(self.payload()))
            except Exception as exc:
                errors.append(exc)
        first = threading.Thread(target=infer)
        second = threading.Thread(target=infer)
        first.start()
        self.assertTrue(self.backend.started.wait(1))
        try:
            self.backend.checkpoint = '/test-only/base-B'
            second.start()
            time.sleep(.05)
            self.assertFalse(self.backend.overlap.is_set(), 'Checkpoint swap allowed concurrent old/new inference')
        finally:
            self.backend.release.set()
            first.join(2)
            if second.ident is not None:
                second.join(2)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertTrue(results)
        self.assertTrue(all(isinstance(error, RuntimeError) for error in errors))
        self.assertEqual(runtime.infer(self.payload())['checkpoint'], '/test-only/base-B')


if __name__ == '__main__':
    unittest.main()
