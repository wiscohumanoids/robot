#!/usr/bin/env python3
"""Local, hardware-free review trimmer. Run from any directory."""
import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import sys
import threading
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
from hitl_recording import data_timestamp, data_origin


class ReviewStore:
    def __init__(self, source, annotations, template):
        self.source = source.resolve()
        self.annotations = annotations.resolve()
        self.template = template
        self.lock = threading.Lock()

    def spec(self):
        if self.annotations.exists():
            return json.loads(self.annotations.read_text())
        spec = json.loads(self.template.read_text())
        spec.update(episodes=[], source_kind='operator_review_real_recordings',
                    trim_method='Manual recovery-only temporal crop using synchronized camera review')
        for key in ('recording_session_local_date', 'recording_timezone'):
            spec.pop(key, None)
        return spec

    @staticmethod
    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        temporary.replace(path)

    def review_state(self):
        path = self.annotations.with_suffix('.review-state.json')
        return json.loads(path.read_text()) if path.exists() else {}

    def episode_path(self, identifier):
        if Path(identifier).name != identifier or identifier in ('.', '..'):
            raise ValueError('Unknown episode')
        roots = [self.source]
        if self.source.name in ('kept', 'review', 'discarded'):
            roots = [self.source.parent / c for c in ('review', 'kept', 'discarded')]
        matches = [root / identifier for root in roots if (root / identifier / 'metadata.json').is_file()]
        if len(matches) != 1:
            raise ValueError('Unknown or ambiguous episode')
        path = matches[0].resolve()
        if path.parent not in [r.resolve() for r in roots]:
            raise ValueError('Invalid episode path')
        return path

    def episode(self, identifier):
        path = self.episode_path(identifier)
        metadata = json.loads((path / 'metadata.json').read_text())
        origin = data_origin(metadata)
        groups = {}
        with (path / 'frames.jsonl').open() as stream:
            for line in stream:
                if line.strip():
                    frame = json.loads(line)
                    groups.setdefault(frame['camera'], []).append({
                        't': (data_timestamp(frame) - origin) / 1e9, 'path': frame['path']})
        for frames in groups.values():
            frames.sort(key=lambda f: f['t'])
        # Match the exporter's state clock, including recovered/incomplete runs.
        last = None
        with (path / 'states.jsonl').open() as stream:
            for line in stream:
                if line.strip():
                    last = json.loads(line)
        duration = (data_timestamp(last) - origin) / 1e9 if last else 0
        return dict(id=identifier, duration=duration, complete=metadata.get('complete', False),
                    cameras=groups, label=metadata.get('label'),
                    notes=self.review_state().get(identifier, {}).get('notes', None))

    def listing(self):
        edits = {e['id']: e for e in self.spec()['episodes']}
        state = self.review_state()
        roots = [self.source]
        if self.source.name in ('kept', 'review', 'discarded'):
            roots.append(self.source.parent / 'kept')
        identifiers = {p.name for root in roots if root.exists()
                       for p in root.iterdir() if p.is_dir() and (p / 'metadata.json').exists()}
        identifiers.update(state)
        identifiers.update(edits)
        rows = []
        for identifier in sorted(identifiers):
            try:
                path = self.episode_path(identifier)
            except ValueError:
                continue
            metadata = json.loads((path / 'metadata.json').read_text())
            rows.append(dict(id=identifier, edit=edits.get(identifier),
                             label=metadata.get('label', path.parent.name),
                             notes=state.get(identifier, {}).get('notes', None)))
        return dict(episodes=rows, annotations=str(self.annotations))

    def update_details(self, identifier, payload):
        with self.lock:
            path = self.episode_path(identifier)
            category = payload.get('category')
            if category is not None and category not in ('kept', 'review', 'discarded'):
                raise ValueError('Invalid category')
            state = self.review_state()
            details = state.setdefault(identifier, {})
            if 'notes' in payload:
                if not isinstance(payload['notes'], str) or len(payload['notes']) > 4000:
                    raise ValueError('Notes must be text up to 4000 characters')
                details['notes'] = payload['notes']
            metadata = json.loads((path / 'metadata.json').read_text())
            if category is not None and category != metadata.get('label'):
                if self.source.name not in ('kept', 'review', 'discarded'):
                    raise ValueError('Recategorization requires a kept/review/discarded source folder')
                destination = self.source.parent / category / identifier
                if destination.exists():
                    raise ValueError('Destination episode already exists')
                destination.parent.mkdir(parents=True, exist_ok=True)
                path.rename(destination)
                try:
                    metadata['label'] = category
                    self.write_json(destination / 'metadata.json', metadata)
                except OSError:
                    destination.rename(path)
                    raise
                spec = self.spec()
                if category == 'discarded':
                    spec['episodes'] = [e for e in spec['episodes'] if e['id'] != identifier]
                else:
                    for entry in spec['episodes']:
                        if entry['id'] == identifier:
                            entry['source_category'] = category
                self.write_json(self.annotations, spec)
            details['updated_utc'] = datetime.now(timezone.utc).isoformat()
            self.write_json(self.annotations.with_suffix('.review-state.json'), state)
            return dict(id=identifier, notes=details.get('notes', None), label=metadata.get('label'))

    def save(self, identifier, payload):
        with self.lock:
            episode = self.episode(identifier)
            if episode['label'] == 'discarded':
                raise ValueError('Move this run to review or kept before saving a crop')
            start, end = float(payload['start_s']), float(payload['end_s'])
            if not all(math.isfinite(x) for x in (start, end)) or not 0 <= start < end <= episode['duration']:
                raise ValueError('Choose a valid interval within the run')
            if end - start < 0.6:
                raise ValueError('Keep at least 0.6 seconds for the 20 Hz exporter')
            spec = self.spec()
            entry = dict(id=identifier, start_s=start, end_s=end,
                         reason=str(payload.get('reason', ''))[:4000],
                         annotation_source='review_runs synchronized camera editor',
                         original_end_s=episode['duration'], review_approved=True,
                         source_category=episode['label'],
                         edited_utc=datetime.now(timezone.utc).isoformat())
            spec['episodes'] = sorted([e for e in spec['episodes'] if e['id'] != identifier] + [entry], key=lambda e: e['id'])
            self.annotations.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.annotations.with_suffix(self.annotations.suffix + '.tmp')
            temporary.write_text(json.dumps(spec, indent=2, allow_nan=False) + '\n')
            temporary.replace(self.annotations)
            return entry


def handler_for(store):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, value, status=200):
            data = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            route = unquote(urlparse(self.path).path)
            try:
                if route == '/api/runs':
                    return self.respond(store.listing())
                if route.startswith('/api/run/'):
                    return self.respond(store.episode(route[len('/api/run/'):]))
                if route.startswith('/frame/'):
                    identifier, relative = route[len('/frame/'):].split('/', 1)
                    root = store.episode_path(identifier)
                    path = (root / relative).resolve()
                    if not path.is_relative_to(root) or path.suffix.lower() != '.jpg':
                        raise ValueError('Invalid frame path')
                    data = path.read_bytes()
                    content_type = 'image/jpeg'
                elif route == '/':
                    data = (ROOT / 'app/web/review.html').read_bytes()
                    content_type = 'text/html; charset=utf-8'
                else:
                    return self.respond({'error': 'Not found'}, 404)
                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except (ValueError, KeyError, OSError) as exc:
                self.respond({'error': str(exc)}, 400)

        def do_POST(self):
            # Reject cross-site form/JavaScript writes to this local tool.
            if self.headers.get('Origin') not in (None, f"http://{self.headers.get('Host')}"):
                return self.respond({'error': 'Invalid origin'}, 403)
            if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                return self.respond({'error': 'JSON required'}, 415)
            route = unquote(urlparse(self.path).path)
            try:
                if not route.startswith('/api/run/'):
                    return self.respond({'error': 'Not found'}, 404)
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= 16384:
                    raise ValueError('Invalid request size')
                payload = json.loads(self.rfile.read(length))
                identifier = route[len('/api/run/'):]
                if identifier.endswith('/details'):
                    self.respond(store.update_details(identifier[:-len('/details')], payload))
                else:
                    self.respond(store.save(identifier, payload))
            except (ValueError, KeyError, TypeError, OSError) as exc:
                self.respond({'error': str(exc)}, 400)

        def log_message(self, *_):
            pass
    return Handler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/review')
    parser.add_argument('--annotations', type=Path, default=ROOT / 'config/postprocessing-review.json')
    parser.add_argument('--template', type=Path, default=ROOT / 'config/postprocessing-gear-pin-sept30.json')
    parser.add_argument('--port', type=int, default=8091)
    args = parser.parse_args()
    store = ReviewStore(args.source, args.annotations, args.template)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(store))
    print(f'Review trimmer: http://127.0.0.1:{args.port}\nAnnotations: {store.annotations}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
