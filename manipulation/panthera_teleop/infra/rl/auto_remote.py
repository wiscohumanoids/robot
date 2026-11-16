"""Follow deployed bases automatically, preserving exact per-request provenance."""
from pathlib import Path
import threading
import time

from .manifest_audit import audit_manifest
from .remote_server import BackendClient, CONTRACT, NAMES, Runtime, read_manifest


class AutoRuntime:
    def __init__(self, directory, backend='http://127.0.0.1:8765', *, ssh_host,
                 prefix_length=17, timeout=30., client=None, resolver=None):
        if type(prefix_length) is not int or not 1 <= prefix_length <= 50:
            raise ValueError('Prefix length must be an integer from 1 to 50')
        self.directory = Path(directory)
        self.backend, self.prefix_length, self.timeout = backend, prefix_length, timeout
        self.client = client or BackendClient(backend, timeout)
        self.resolver = resolver or (lambda checkpoint: audit_manifest(checkpoint, self.directory, ssh_host))
        self.lock = threading.Lock()
        self.runtime = None
        self.pending = None
        self.error = None
        self.retry_after = 0.

    def _runtime(self, path):
        return Runtime(path, self.backend, client=self.client,
                       timeout=self.timeout, prefix_length=self.prefix_length)

    def _verify_new(self, checkpoint):
        try:
            runtime = self._runtime(self.resolver(checkpoint))
            if runtime.checkpoint != checkpoint:
                raise ValueError('Verifier returned a different checkpoint')
            runtime.health()
            with self.lock:
                self.runtime, self.error = runtime, None
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
                self.retry_after = time.monotonic() + 30.
        finally:
            with self.lock:
                self.pending = None

    def _resolve(self):
        health = self.client.request('/health')
        if (health.get('status') != 'ready' or health.get('contract') != CONTRACT or
                health.get('names') != NAMES or health.get('mock', False) is not False):
            raise RuntimeError('Backend is not ready or has a different robot contract')
        checkpoint = health.get('checkpoint')
        if not isinstance(checkpoint, str) or not checkpoint.startswith('/'):
            raise RuntimeError('Backend checkpoint is missing')
        with self.lock:
            if self.runtime is not None and self.runtime.checkpoint == checkpoint:
                return self.runtime
            for path in sorted(self.directory.glob('*.json')):
                try:
                    manifest = read_manifest(path)
                except (ValueError, OSError, KeyError, TypeError):
                    continue
                if manifest['checkpoint'] == checkpoint:
                    self.runtime = self._runtime(path)
                    return self.runtime
            if self.pending is None and time.monotonic() >= self.retry_after:
                self.pending, self.error = checkpoint, None
                threading.Thread(target=self._verify_new, args=(checkpoint,), daemon=True).start()
            return None

    def health(self):
        runtime = self._resolve()
        if runtime is None:
            return dict(status='loading', contract=CONTRACT,
                        error=self.error or 'Automatically verifying deployed checkpoint; retry shortly',
                        checkpoint_following=True)
        return dict(runtime.health(), checkpoint_following=True)

    def infer(self, payload):
        # Strict runtimes have separate locks. Keep a single inference lease
        # across checkpoint swaps so an old/new pair cannot overlap the backend.
        with self.lock:
            if not hasattr(self, '_infer_lock'):
                self._infer_lock = threading.Lock()
            infer_lock = self._infer_lock
        if not infer_lock.acquire(blocking=False):
            raise RuntimeError('Inference busy')
        try:
            runtime = self._resolve()
            if runtime is None:
                raise RuntimeError('Automatically verifying deployed checkpoint; retry shortly')
            return runtime.infer(payload)
        finally:
            infer_lock.release()
