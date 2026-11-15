"""Local dashboard and timestamped camera capture."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
import time
from urllib.parse import urlsplit

import cv2


class Cameras:
    def __init__(self, store, config):
        self.store = store
        self.config = config
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.latest = {}
        self.errors = {}
        self.threads = []
        for name, device in config['cameras'].items():
            thread = threading.Thread(target=self.capture, args=(name, device), daemon=True, name=name)
            thread.start()
            self.threads.append(thread)

    def capture(self, name, device):
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        try:
            if not cap.isOpened():
                raise RuntimeError(f'Cannot open {device}; camera may be busy')
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera_resolution'][0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera_resolution'][1])
            cap.set(cv2.CAP_PROP_FPS, self.config['camera_fps'])
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            sequence = 0
            while not self.stop.is_set():
                started = time.monotonic()
                ok, frame = cap.read()
                timestamp = time.monotonic_ns()
                unix_ns = time.time_ns()
                if not ok:
                    raise RuntimeError('Camera capture failed')
                ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.config['jpeg_quality']])
                if not ok:
                    raise RuntimeError('JPEG encoding failed')
                payload = dict(camera=name, device=device, sequence=sequence, monotonic_ns=timestamp,
                               unix_ns=unix_ns, width=frame.shape[1], height=frame.shape[0], jpeg=encoded.tobytes())
                with self.lock:
                    self.latest[name] = payload
                self.store.submit('frame', payload)
                sequence += 1
                self.stop.wait(max(0, 1/self.config['camera_fps']-(time.monotonic()-started)))
        except Exception as exc:
            with self.lock:
                self.errors[name] = str(exc)
        finally:
            cap.release()

    def status(self):
        now = time.monotonic_ns()
        with self.lock:
            return {name: dict(device=device,
                               ok=name in self.latest and name not in self.errors and
                                  (now-self.latest[name]['monotonic_ns']) < 1_000_000_000,
                               error=self.errors.get(name),
                               sequence=self.latest.get(name, {}).get('sequence', 0),
                               age_s=(now-self.latest[name]['monotonic_ns'])/1e9 if name in self.latest else None)
                    for name, device in self.config['cameras'].items()}

    def jpeg(self, name):
        with self.lock:
            return self.latest.get(name, {}).get('jpeg')

    def recording_frames(self):
        with self.lock:
            return dict(self.latest)

    def wait_ready(self, timeout=10):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self.status()
            if any(c['error'] for c in state.values()):
                raise RuntimeError(f'Camera preflight failed: {state}')
            if all(c['ok'] and c['sequence'] >= 3 for c in state.values()):
                return
            time.sleep(0.1)
        raise RuntimeError(f'Camera preflight timed out: {self.status()}')

    def close(self):
        self.stop.set()
        for thread in self.threads:
            thread.join(timeout=2)


class Dashboard:
    def __init__(self, store, cameras, port, demo=False):
        self.token = secrets.token_urlsafe(32)
        self.store = store
        self.cameras = cameras
        self.url = f'http://127.0.0.1:{port}'
        dashboard = self
        html = Path(__file__).with_name('web').joinpath('index.html').read_bytes()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send(self, code, body, mime='application/json'):
                self.send_response(code)
                self.send_header('Content-Type', mime)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def trusted_host(self):
                return self.headers.get('Host') in (f'127.0.0.1:{port}', f'localhost:{port}')

            def do_GET(self):
                if not self.trusted_host():
                    self.send(403, b'{}')
                    return
                path = urlsplit(self.path).path
                if path == '/':
                    self.send(200, html, 'text/html; charset=utf-8')
                elif path == '/api/status':
                    status = dict(store.snapshot(), cameras=cameras.status() if cameras else {},
                                  token=dashboard.token, demo=demo)
                    self.send(200, json.dumps(status).encode())
                elif path.startswith('/camera/') and cameras:
                    jpg = cameras.jpeg(path.split('/')[-1])
                    self.send(200 if jpg else 503, jpg or b'', 'image/jpeg')
                else:
                    self.send(404, b'{}')

            def do_POST(self):
                if (not self.trusted_host() or self.headers.get('X-Session-Token') != dashboard.token
                    or self.headers.get('Origin') not in (None, dashboard.url, f'http://localhost:{port}')):
                    self.send(403, b'{"error":"Invalid session"}')
                    return
                try:
                    size = int(self.headers.get('Content-Length', 0))
                    if not 0 < size < 2048:
                        raise ValueError('Invalid request size')
                    payload = json.loads(self.rfile.read(size))
                    if self.path != '/api/decision' or payload.get('label') not in ('kept', 'review', 'discarded'):
                        raise ValueError('Unknown action')
                    if payload.get('episode_id') != store.snapshot()['active'] or not payload.get('episode_id'):
                        raise ValueError('Episode changed; action was not applied')
                    store.submit('decision', payload)
                    self.send(202, b'{"accepted":true}')
                except (ValueError, RuntimeError, TypeError) as exc:
                    self.send(409, json.dumps({'error': str(exc)}).encode())

        self.server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True, name='dashboard')
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
