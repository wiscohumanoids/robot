#!/usr/bin/env python3
"""Watch the agent work in the sim: the arm in MuJoCo, and every step of the stack beside it.

    .venv/bin/python tools/viewer.py              # opens http://127.0.0.1:8765

The page shows the scene (drag to orbit, scroll to zoom, or switch to the overhead / wrist camera) and a timeline of
everything that happens once you type an instruction: the system prompt, tools and request options sent to Claude;
each turn's new content (the tool results and images the model is shown); Claude's reply (summarized thinking, text,
the tool call, tokens and cost); then the tool running: camera images, segmentation, the measured positions, the
grasp plan, each straight-line move and its tracking error, wrist-camera corrections, gripper widths, and the result
that goes back to the model.

Controls: pause at each turn (before each request and each tool) or at each motion, step with Next, pause or stop
at any time, sim speed. "Scripted" plays a fixed red-on-green plan with no API call, to try the viewer for free.
Runs are logged to runs/viewer_<time>/ like tools/run_agent.py's.

One process: the simulation and rendering stay on the main thread (MuJoCo's GL context lives there); the web server
answers on background threads; a Claude request runs on a worker thread while the main thread keeps rendering.
Binds to 127.0.0.1 only.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import pathlib
import queue
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robots.sim_robot import SimRobot  # noqa: E402  (sets MUJOCO_GL before mujoco loads)

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from core.build import SEGMENTERS, load_config, load_dotenv, make_segmenter, make_skills  # noqa: E402
from core.camera import pose_error  # noqa: E402
from core.tasks import TASKS  # noqa: E402
from llm.loop import AgentRun, StopRun  # noqa: E402

PAGE = pathlib.Path(__file__).with_name('viewer.html')
SCENES = ['scene.xml', 'scene_two_blocks.xml']
FRAME = (960, 640)                       # width, height of the streamed view
# USD per million tokens: input, output, cache read, cache write (5-minute). From Anthropic's pricing for the model.
PRICES = {'claude-opus-5-5': (4.0, 20.0, 0.20, 5.0)}
# Where to pause in each mode (the run waits before these events until Next).
PAUSE_AT = {'off': set(), 'turn': {'llm_request', 'tool_start'},
            'motion': {'llm_request', 'tool_start', 'move_start', 'gripper_start'}}


def cost(model: str | None, usage: dict) -> float | None:
    """The request's cost in USD, if the model's price is known."""
    p = PRICES.get(model or '')
    if p is None or not usage:
        return None
    u = {k: (v or 0) for k, v in usage.items()}
    return (u.get('input_tokens', 0) * p[0] + u.get('output_tokens', 0) * p[1]
            + u.get('cache_read_input_tokens', 0) * p[2] + u.get('cache_creation_input_tokens', 0) * p[3]) / 1e6


class BackgroundPlanner:
    """Runs the planner's request on a worker thread; the main thread keeps rendering (and can stop) meanwhile."""

    def __init__(self, planner, viewer: 'Viewer'):
        self.planner, self.viewer = planner, viewer
        self.model = getattr(planner, 'model', None)
        self.effort = getattr(planner, 'effort', None)
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def request_options(self) -> dict:
        f = getattr(self.planner, 'request_options', None)
        return f() if f else {}

    def create(self, system, tools, messages):
        fut = self._pool.submit(self.planner.create, system, tools, messages)
        self.viewer.set_status(state='thinking')
        while not fut.done():
            self.viewer.idle()
            if self.viewer.stop_requested:
                raise StopRun()             # the request finishes on its own; its reply is dropped
            time.sleep(0.03)
        self.viewer.set_status(state='acting')
        return fut.result()


class Viewer:
    """The sim, the agent run, and the state the web page reads."""

    def __init__(self, config: dict, args):
        self.config, self.args = config, args
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)          # new events or status
        self.frame_cond = threading.Condition()
        self.events: list[str] = []                          # JSON, in order; replayed to each new page
        self.images: dict[str, bytes] = {}                   # id -> JPEG
        self.status: dict = dict(state='idle', mode='off', speed=1.0, turn=0, cost=0.0, tokens={}, paused=False)
        self.status_version = 0
        self.frame, self.frame_seq = b'', 0
        self.commands: queue.Queue = queue.Queue()
        self.view = dict(camera='free', azimuth=150.0, elevation=-28.0, distance=1.25, lookat=[0.36, 0.0, 0.12])
        self.view_dirty = True
        self.paused = False
        self.stop_requested = False
        self.robot: SimRobot | None = None
        self.task = None
        self.task_fresh = False                               # the scene is the task's, untouched since reset
        self.renderer: mujoco.Renderer | None = None
        self.free_cam = mujoco.MjvCamera()
        self._last_render = 0.0
        self._last_status = 0.0
        self._next_tick = None

    # ---------- events and status ----------
    def _store(self, jpeg: bytes) -> str:
        key = hashlib.sha1(jpeg).hexdigest()[:16]
        self.images.setdefault(key, jpeg)
        return key

    def _clean(self, x, key: str = ''):
        """Make an event JSON-ready: images become ids served at /img/<id>, arrays become lists."""
        if isinstance(x, np.ndarray):
            if x.ndim == 3 and x.dtype == np.uint8:
                ok, buf = cv2.imencode('.jpg', cv2.cvtColor(x, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
                return {'image': self._store(buf.tobytes()), 'size': [x.shape[1], x.shape[0]]}
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if key == 'signature' and isinstance(x, str):
            return f'<{len(x)} characters>'
        if isinstance(x, dict):
            src = x.get('source')
            if x.get('type') == 'image' and isinstance(src, dict) and src.get('type') == 'base64':
                return {'type': 'image', 'image': self._store(base64.b64decode(src['data'])),
                        'media_type': src.get('media_type')}
            return {k: self._clean(v, k) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [self._clean(v) for v in x]
        if hasattr(x, 'model_dump'):
            return self._clean(x.model_dump(exclude_none=True))
        if x is None or isinstance(x, (str, int, float, bool)):
            return x
        return str(x)

    def push(self, kind: str, data: dict) -> None:
        """Add an event to the timeline."""
        if kind == 'tool_result':
            data = {**data, 'images': [{'caption': c, **self._clean(img)} for c, img in data.get('images', [])]}
        line = json.dumps(dict(kind=kind, t=round(time.time(), 3), **self._clean(data)))
        with self.cond:
            self.events.append(line)
            self.cond.notify_all()

    def set_status(self, **kw) -> None:
        with self.cond:
            self.status.update(kw)
            self.status_version += 1
            self.cond.notify_all()

    def on_event(self, kind: str, data: dict) -> None:
        """Listener for the agent loop and the skills: show the event, keep the totals, then pause where asked (so
        what is about to be sent or done is on the page while it waits)."""
        if kind == 'llm_response':
            c = cost(data.get('model') or self.status.get('model'), data.get('usage') or {})
            data = {**data, 'cost': c}
            tokens = dict(self.status['tokens'])
            for k, v in (data.get('usage') or {}).items():
                tokens[k] = tokens.get(k, 0) + (v or 0)
            self.set_status(turn=data['turn'], cost=self.status['cost'] + (c or 0), tokens=tokens)
        self.push(kind, data)
        if self.stop_requested and kind != 'run_end':
            raise StopRun()
        if kind in PAUSE_AT[self.status['mode']]:
            self.pause_here(kind)

    def pause_here(self, before: str) -> None:
        self.paused = True
        self.set_status(paused=True, paused_before=before)
        self.wait_while_paused()

    def wait_while_paused(self) -> None:
        while self.paused and not self.stop_requested:
            self.idle()
            time.sleep(0.03)
        self.set_status(paused=False, paused_before=None)
        self._next_tick = None
        if self.stop_requested:
            raise StopRun()

    # ---------- rendering ----------
    def render(self) -> None:
        """Render the selected view and publish it as the latest frame."""
        if self.robot is None:
            return
        model, data = self.robot.sim.model, self.robot.sim.data
        if self.renderer is None:
            self.renderer = mujoco.Renderer(model, height=FRAME[1], width=FRAME[0])
            mujoco.mjv_defaultFreeCamera(model, self.free_cam)
        with self.lock:
            v = dict(self.view)
            self.view_dirty = False
        if v['camera'] == 'free':
            c = self.free_cam
            c.azimuth, c.elevation, c.distance = v['azimuth'], v['elevation'], v['distance']
            c.lookat[:] = v['lookat']
            self.renderer.update_scene(data, camera=c)
        else:
            self.renderer.update_scene(data, camera=v['camera'])
        rgb = self.renderer.render()
        ok, buf = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
        with self.frame_cond:
            self.frame, self.frame_seq = buf.tobytes(), self.frame_seq + 1
            self.frame_cond.notify_all()
        self._last_render = time.monotonic()

    def idle(self) -> None:
        """Keep the view live while nothing in the sim moves (the camera may still orbit)."""
        if self.view_dirty and time.monotonic() - self._last_render > 1 / 30:
            self.render()

    def on_tick(self) -> None:
        """After each control tick: a frame (at most 30 per second), real-time pacing at the chosen speed."""
        if self.stop_requested:
            raise StopRun()
        now = time.monotonic()
        if now - self._last_render >= 1 / 30:
            self.render()
        dt = 1.0 / self.robot.hz / float(self.status['speed'])
        if self._next_tick is None or now - self._next_tick > 0.5:
            self._next_tick = now
        self._next_tick += dt
        delay = self._next_tick - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        if self.paused:
            self.wait_while_paused()
        if now - self._last_status > 0.2:
            self._last_status = now
            self.set_status(sim_time=round(float(self.robot.sim.data.time), 2))

    # ---------- commands (main thread) ----------
    def reset(self, scene: str | None, task: str | None, seed: int, camera_error: bool) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self.task = TASKS[task] if task else None
        scene = self.task.scene if self.task else (scene or self.config['scene'])
        self.robot = SimRobot({**self.config, 'scene': scene}, seed=seed)
        if self.task and self.task.setup:
            self.task.setup(self.robot)
        if camera_error:
            self.robot.camera_errors['overhead'] = pose_error(0.005, 1.0, np.random.default_rng(seed))
        self.robot.on_tick = self.on_tick
        self.task_fresh = self.task is not None
        truth = {n: [round(float(c) * 100, 1) for c in p] for n, (p, _) in self.robot.truth_objects().items()}
        self.push('reset', dict(scene=scene, task=task, seed=seed, camera_error=camera_error,
                                instruction=self.task.instruction if self.task else None, truth_cm=truth))
        self.set_status(scene=scene, task=task, seed=seed, sim_time=round(float(self.robot.sim.data.time), 2))
        self.view_dirty = True
        self.render()

    def make_planner(self, planner: str, model: str | None, effort: str | None):
        if planner == 'scripted':
            from llm.scripted import RedOnGreenScript
            return RedOnGreenScript()
        from llm.claude import ClaudePlanner
        return ClaudePlanner(model=model or None, effort=effort or None)

    def run(self, instruction: str, planner: str, model: str | None, effort: str | None, segmenter: str) -> None:
        if self.robot is None:
            self.reset(None, None, 0, False)
        self.stop_requested = False
        self.set_status(state='acting', turn=0)
        try:
            p = BackgroundPlanner(self.make_planner(planner, model, effort), self)
            self.set_status(model=p.model, effort=p.effort)
            skills = make_skills(self.robot, make_segmenter(segmenter, self.config), self.config)
            skills.on_event = self.on_event
            log_dir = ROOT / 'runs' / f'viewer_{time.strftime("%Y%m%d_%H%M%S")}'
            self.push('log_dir', dict(path=str(log_dir.relative_to(ROOT))))
            AgentRun(skills, p, self.config, log_dir, max_turns=self.args.max_turns,
                     on_event=self.on_event).run(instruction)
            if self.task and self.task_fresh:
                self.push('judged', dict(task=self.task.name, metrics=self.task.check(self.robot.truth_objects())))
        except StopRun:
            self.push('run_end', dict(outcome=dict(stop='stopped')))
        except Exception as e:                                     # shown on the page, the viewer keeps running
            self.push('error', dict(message=f'{type(e).__name__}: {e}', traceback=traceback.format_exc()))
        finally:
            self.task_fresh = False
            self.paused = False
            self.stop_requested = False
            self.set_status(state='idle', paused=False, paused_before=None)

    def serve(self) -> None:
        """The main loop: commands that change the sim run here, on the main thread."""
        self.reset(None, self.args.task, self.args.seed, False)
        while True:
            try:
                cmd = self.commands.get(timeout=0.03)
            except queue.Empty:
                self.idle()
                continue
            op = cmd.pop('op')
            if op == 'reset':
                self.set_status(cost=0.0, tokens={}, turn=0)
                try:
                    self.reset(cmd.get('scene'), cmd.get('task') or None, int(cmd.get('seed') or 0),
                               bool(cmd.get('camera_error')))
                except Exception as e:
                    self.push('error', dict(message=f'reset failed: {type(e).__name__}: {e}',
                                            traceback=traceback.format_exc()))
                finally:
                    self.set_status(state='idle')
            elif op == 'run':
                self.run(cmd['instruction'], cmd.get('planner', 'claude'), cmd.get('model'), cmd.get('effort'),
                         cmd.get('segmenter', 'sim_truth'))

    def command(self, cmd: dict) -> dict:
        """A command from the page (web thread). Run controls act at once; scene changes queue for the main thread."""
        op = cmd.get('op')
        busy = self.status['state'] != 'idle'
        if op in ('reset', 'run'):
            if busy:
                return dict(ok=False, error='a run is in progress: stop it first')
            if op == 'run' and not str(cmd.get('instruction', '')).strip():
                return dict(ok=False, error='type an instruction')
            self.set_status(state='starting' if op == 'run' else 'resetting')
            self.commands.put(dict(cmd))
        elif op == 'pause':
            if busy:
                self.paused = True
                self.set_status(paused=True, paused_before='now')
        elif op == 'next':
            self.paused = False
        elif op == 'stop':
            if busy:
                self.stop_requested = True
                self.paused = False
        elif op == 'mode':
            if cmd.get('mode') in PAUSE_AT:
                self.set_status(mode=cmd['mode'])
                if cmd['mode'] == 'off':
                    self.paused = False
        elif op == 'speed':
            self.set_status(speed=float(np.clip(float(cmd.get('speed', 1)), 0.1, 8)))
        elif op == 'view':
            with self.lock:
                v = self.view
                if cmd.get('camera') in ('free', 'overhead', 'wrist'):
                    v['camera'] = cmd['camera']
                if cmd.get('home'):
                    v.update(azimuth=150.0, elevation=-28.0, distance=1.25, lookat=[0.36, 0.0, 0.12])
                v['azimuth'] = (v['azimuth'] + float(cmd.get('d_azimuth', 0))) % 360
                v['elevation'] = float(np.clip(v['elevation'] + float(cmd.get('d_elevation', 0)), -89, -2))
                v['distance'] = float(np.clip(v['distance'] * float(cmd.get('zoom', 1)), 0.3, 4))
                self.view_dirty = True
        else:
            return dict(ok=False, error=f'unknown command {op!r}')
        return dict(ok=True)

    def segmenters(self) -> dict:
        """{name: {label, up, detail}}: the VM servers are asked for /health (through tools/tunnel.sh)."""
        from perception.server_client import ServerSegmenter
        out = {}
        for name, label in SEGMENTERS.items():
            up, detail = True, ''
            if name != 'sim_truth':
                try:
                    h = ServerSegmenter(name).health()
                    detail = f'{h.get("backend")}, {h.get("requests")} requests'
                except ConnectionError as e:
                    up, detail = False, str(e)
            out[name] = dict(label=label, up=up, detail=detail)
        return out

    def info(self) -> dict:
        import os
        return dict(tasks={n: dict(instruction=t.instruction, scene=t.scene) for n, t in TASKS.items()},
                    scenes=SCENES, has_key=bool(os.environ.get('ANTHROPIC_API_KEY')),
                    segmenters=self.segmenters(), segmenter=self.args.segmenter,
                    model=os.environ.get('PANTHERA_AGENT_MODEL') or 'claude-opus-5-5',
                    effort=os.environ.get('PANTHERA_AGENT_EFFORT') or 'medium')


def handler(viewer: Viewer):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *a):
            pass

        def _send(self, body: bytes, ctype: str, code: int = 200):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            try:
                if self.path in ('/', '/index.html'):
                    self._send(PAGE.read_bytes(), 'text/html; charset=utf-8')
                elif self.path == '/info':
                    self._send(json.dumps(viewer.info()).encode(), 'application/json')
                elif self.path.startswith('/img/'):
                    img = viewer.images.get(self.path[5:])
                    self._send(img, 'image/jpeg') if img else self._send(b'not found', 'text/plain', 404)
                elif self.path == '/frame.mjpg':
                    self._mjpeg()
                elif self.path == '/events':
                    self._events()
                else:
                    self._send(b'not found', 'text/plain', 404)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            if self.path != '/cmd':
                return self._send(b'not found', 'text/plain', 404)
            n = int(self.headers.get('Content-Length', 0))
            try:
                reply = viewer.command(json.loads(self.rfile.read(n) or b'{}'))
            except Exception as e:
                reply = dict(ok=False, error=str(e))
            self._send(json.dumps(reply).encode(), 'application/json')

        def _mjpeg(self):
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            seq = -1
            while True:
                with viewer.frame_cond:
                    viewer.frame_cond.wait_for(lambda: viewer.frame_seq != seq, timeout=2)
                    seq, data = viewer.frame_seq, viewer.frame
                if data:
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n' % len(data)
                                     + data + b'\r\n')
                    self.wfile.flush()

        def _events(self):
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            sent, version = 0, -1
            while True:
                with viewer.cond:
                    viewer.cond.wait_for(lambda: len(viewer.events) > sent or viewer.status_version != version,
                                         timeout=10)
                    new = viewer.events[sent:]
                    sent = len(viewer.events)
                    status = None
                    if viewer.status_version != version:
                        version, status = viewer.status_version, json.dumps(viewer.status)
                out = ''.join(f'data: {e}\n\n' for e in new)
                if status:
                    out += f'event: status\ndata: {status}\n\n'
                self.wfile.write((out or ': ping\n\n').encode())
                self.wfile.flush()

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--task', choices=sorted(TASKS), help='set up this task at start')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-turns', type=int, default=40)
    ap.add_argument('--segmenter', choices=list(SEGMENTERS), default='sim_truth', help='preselected on the page')
    ap.add_argument('--no-browser', action='store_true')
    args = ap.parse_args()
    load_dotenv()
    viewer = Viewer(load_config(), args)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler(viewer))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{args.port}'
    print(f'viewer: {url}  (Ctrl-C to quit)', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        viewer.serve()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
