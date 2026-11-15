"""Replay saved joint targets through the existing controller and audit feedback.

This source has its own checkpoint identity. It never opens robot devices; the
inference dashboard remains the only controller, with its usual limits and Stop.
Start the dashboard with --url http://127.0.0.1:8766 and a separate --data-dir.
"""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import sys
import threading
import time
from urllib.request import Request, urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
from remote_inference import CONTRACT, validate_chunk


def call(base, path, body=None, token=None):
    headers = {'Content-Type': 'application/json'}
    if token:
        headers['X-Session-Token'] = token
    req = Request(base + path, data=None if body is None else json.dumps(body).encode(), headers=headers)
    with urlopen(req, timeout=5) as response:
        return json.load(response)


class ReplaySource:
    def __init__(self, manifest):
        manifest = Path(manifest).resolve()
        self.manifest = json.loads(manifest.read_text())
        if self.manifest['fps'] != CONTRACT['fps']:
            raise ValueError('Recording rate does not match the controller')
        self.episodes = {}
        digest = hashlib.sha256(manifest.read_bytes())
        import yaml
        configs = [yaml.safe_load((ROOT / f'config/robot-configs/{side}.yaml').read_text())
                   for side in ('left', 'right')]
        lower = np.concatenate([np.r_[c['robot']['joint_limits']['lower'], c['robot']['gripper_limits']['lower']]
                                for c in configs])
        upper = np.concatenate([np.r_[c['robot']['joint_limits']['upper'], c['robot']['gripper_limits']['upper']]
                                for c in configs])
        for item in self.manifest['episodes']:
            path = manifest.parent / item['file']
            digest.update(path.read_bytes())
            with np.load(path, allow_pickle=False) as saved:
                states, actions = saved['states'], saved['actions']
            if states.shape != actions.shape or states.ndim != 2 or states.shape[1] != 14:
                raise ValueError('Malformed saved episode')
            # The existing small closed-stop gripper tolerance is applied by the
            # same validator as policy inference. All joint limits stay strict.
            validated = []
            for start in range(0, len(actions), 50):
                response = dict(sequence=0, fps=CONTRACT['fps'], actions=actions[start:start+50].tolist())
                validated.extend(validate_chunk(response, 0, states[0], lower, upper))
            self.episodes[item['episode']] = dict(info=item, initial=states[0], actions=np.asarray(validated))
        self.checkpoint = 'recorded-episode-replay:' + digest.hexdigest()
        self.token = secrets.token_urlsafe(24)
        self.selected = None
        self.cursor = 0
        self.cache = {}
        self.lock = threading.Lock()

    def select(self, episode):
        with self.lock:
            if episode not in self.episodes:
                raise ValueError('Episode is not in the prepared manifest')
            self.selected, self.cursor, self.cache = episode, 0, {}
            return self.episodes[episode]['info']

    def infer(self, body):
        with self.lock:
            sequence = body['sequence']
            state = np.asarray(body['state'], dtype=float)
            if state.shape != (14,) or not np.isfinite(state).all():
                raise ValueError('Invalid measured state')
            if self.selected is None:
                raise RuntimeError('Select a recorded episode before replay')
            if sequence in self.cache:
                raise RuntimeError('Repeated inference sequence; reselect the episode after canceling')
            episode = self.episodes[self.selected]
            if self.cursor >= len(episode['actions']):
                raise RuntimeError('Recorded episode exhausted; arms remain held')
            if self.cursor == 0:
                error = np.abs(state - episode['initial'])
                tolerance = np.full(14, .06)
                tolerance[[6, 13]] = .10
                if np.any(error > tolerance):
                    raise RuntimeError('Reset to the recorded starting pose before replay: max error '
                                       f'{error.max():.4f} rad')
            start = self.cursor
            self.cursor = min(start + 50, len(episode['actions']))
            actions = episode['actions'][start:self.cursor]
            # The final request can be shorter than 50. The runner uses exactly
            # this remaining count, so no waypoint is dropped or repeated.
            result = dict(sequence=sequence, fps=CONTRACT['fps'], checkpoint=self.checkpoint,
                          mock=False, source='saved_demonstration', episode=self.selected,
                          first_frame=start, next_frame=self.cursor, actions=actions.tolist())
            self.cache[sequence] = True
            return result


def serve(manifest, port):
    source = ReplaySource(manifest)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass
        def send_json(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def trusted(self):
            return self.headers.get('Host') in (f'127.0.0.1:{port}', f'localhost:{port}')
        def do_GET(self):
            if not self.trusted():
                return self.send_json(403, {})
            if self.path == '/health':
                return self.send_json(200, dict(status='ready', contract=CONTRACT,
                    checkpoint=source.checkpoint, source='saved_demonstration'))
            if self.path == '/status':
                return self.send_json(200, dict(episode=source.selected, cursor=source.cursor,
                                                token=source.token, episodes=list(source.episodes)))
            self.send_json(404, {})
        def do_POST(self):
            if not self.trusted():
                return self.send_json(403, {})
            try:
                size = int(self.headers.get('Content-Length', 0))
                if not 0 < size < 2000000:
                    raise ValueError('Invalid request size')
                self.connection.settimeout(3)
                body = json.loads(self.rfile.read(size))
                if self.path == '/select':
                    if self.headers.get('X-Session-Token') != source.token:
                        return self.send_json(403, {})
                    return self.send_json(200, source.select(body['episode']))
                if self.path == '/infer':
                    return self.send_json(200, source.infer(body))
                self.send_json(404, {})
            except (ValueError, KeyError, RuntimeError) as exc:
                self.send_json(409, dict(error=str(exc)))
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    print(json.dumps(dict(url=f'http://127.0.0.1:{port}', checkpoint=source.checkpoint,
                          episodes=list(source.episodes))), flush=True)
    server.serve_forever()


def run_episode(source_url, dashboard_url, episode, output):
    status = call(dashboard_url, '/api/status')
    expected = call(source_url, '/health')['checkpoint']
    if status['checkpoint'] != expected or status['mode'] != 'holding' or status['playing']:
        raise RuntimeError('Dashboard must be connected to the replay source and holding')
    if status['rollout_active']:
        raise RuntimeError('Finish the existing rollout before starting replay')
    source = call(source_url, '/status')
    info = call(source_url, '/select', dict(episode=episode), source['token'])
    token = status['token']
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / f'episode_{episode:06}-status.jsonl'
    stopped = threading.Event()
    lease_failed = threading.Event()
    def heartbeat():
        while not stopped.is_set():
            try:
                call(dashboard_url, '/api/heartbeat', {}, token)
            except Exception:
                lease_failed.set()
                return
            stopped.wait(.25)
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    started = False
    result = dict(episode=episode, frames=info['frames'], completed=False)
    try:
        call(dashboard_url, '/api/heartbeat', {}, token)
        call(dashboard_url, '/api/collection/begin',
             dict(operator='Codex replay audit', scene=f'recorded episode {episode}; tracking diagnostic'), token)
        deadline = time.monotonic() + 5
        while not call(dashboard_url, '/api/status')['rollout_active']:
            if time.monotonic() > deadline:
                raise RuntimeError('Replay recording did not start')
            time.sleep(.05)
        started = True
        with log_path.open('x') as log:
            for first in range(0, info['frames'], 50):
                state = call(dashboard_url, '/api/status')
                if state['mode'] != 'holding' or state['error'] or lease_failed.is_set():
                    raise RuntimeError(f'Replay interrupted before frame {first}: {state["error"]}')
                previous = state['steps_completed']
                count = min(50, info['frames'] - first)
                call(dashboard_url, '/api/step', dict(count=count, blend=0), token)
                deadline = time.monotonic() + 90
                while True:
                    state = call(dashboard_url, '/api/status')
                    log.write(json.dumps(dict(monotonic_ns=time.monotonic_ns(), first_frame=first,
                                              state={k:v for k,v in state.items() if k not in ('token','prediction')}))+'\n')
                    log.flush()
                    if state['error'] or lease_failed.is_set() or not state['connected']:
                        raise RuntimeError(f'Replay interrupted at frame {first}: {state["error"]}')
                    if state['steps_completed'] > previous:
                        break
                    if state['mode'] == 'holding' or time.monotonic() > deadline:
                        raise RuntimeError('Replay canceled or timed out')
                    time.sleep(.05)
                print(f'Episode {episode}: {first+count}/{info["frames"]} frames', flush=True)
        result['completed'] = True
    except BaseException as exc:
        result['error'] = str(exc)
        raise
    finally:
        try:
            if started and call(dashboard_url, '/api/status')['rollout_active']:
                call(dashboard_url, '/api/heartbeat', {}, token)
                call(dashboard_url, '/api/collection/finish', dict(label='review', success=False,
                    quality=1, notes=f'Saved demonstration replay diagnostic, episode {episode}; '
                                    f'completed={result["completed"]}; excluded from training'), token)
                deadline = time.monotonic() + 5
                while call(dashboard_url, '/api/status')['rollout_active'] and time.monotonic() < deadline:
                    time.sleep(.05)
        finally:
            try:
                call(dashboard_url, '/api/stop', {}, token)
            except Exception as exc:
                result['stop_error'] = str(exc)
            stopped.set()
            thread.join(2)
            (output / f'episode_{episode:06}-result.json').write_text(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('serve')
    p.add_argument('manifest', type=Path)
    p.add_argument('--port', type=int, default=8766)
    p = sub.add_parser('run')
    p.add_argument('episode', type=int)
    p.add_argument('--source', default='http://127.0.0.1:8766')
    p.add_argument('--dashboard', default='http://127.0.0.1:8091')
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'serve':
        serve(args.manifest, args.port)
    else:
        run_episode(args.source, args.dashboard, args.episode, args.output)


if __name__ == '__main__':
    main()
