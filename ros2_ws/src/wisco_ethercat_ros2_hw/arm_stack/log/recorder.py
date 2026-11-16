"""One MCAP file per session (spec: Logging and data formats).

Channels (JSON-encoded, with a JSON schema each):
  joint_state  every cycle   t_ns, q, qd, tau (Nm), tau_raw_permille, counts_raw, status
  joint_cmd    every cycle   t_ns, mode, q, qd, kp, kd, tau_ff (Nm), tau_sent_permille
  events       on change     mode switches, faults, limit hits, clamp activations, estop, notes
  meta         once          calibration version and hash, joint config hash, git sha,
                             time since power on, operator, arm

Raw drive values sit next to converted ones so a later unit fix can
reprocess old logs. Every reader here (`load_session`) rejects a session whose
meta has no calibration version.

On the real arm the RT loop pushes into a lock-free ring and a non-RT thread
writes the file; here `RecordingClient` records whatever the procedure sees
through the ArmClient API (every cycle on the sim, every state message on ROS).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from ..hw.client import ArmClient, Command, JointState

SCHEMAS = {
    'joint_state': {'type': 'object', 'properties': {
        't_ns': {'type': 'integer'}, 'q': {'type': 'array'}, 'qd': {'type': 'array'},
        'tau': {'type': 'array'}, 'tau_raw_permille': {'type': 'array'},
        'counts_raw': {'type': 'array'}, 'status': {'type': 'array'}, 'mode': {'type': 'string'},
        'fallback': {'type': 'boolean'}}},
    'joint_cmd': {'type': 'object', 'properties': {
        't_ns': {'type': 'integer'}, 'mode': {'type': 'string'}, 'q': {'type': ['array', 'null']},
        'qd': {'type': ['array', 'null']}, 'kp': {'type': ['array', 'null']},
        'kd': {'type': ['array', 'null']}, 'tau_ff': {'type': ['array', 'null']},
        'tau_sent_permille': {'type': ['array', 'null']}}},
    'events': {'type': 'object', 'properties': {'t_ns': {'type': 'integer'}, 'event': {'type': 'string'}}},
    'meta': {'type': 'object', 'properties': {'calibration_version': {'type': ['integer', 'null']}}},
}


class SessionRejected(ValueError):
    pass


def _list(v):
    if v is None:
        return None
    return np.asarray(v, float).tolist()


class Recorder:
    """Writes one session. Use as a context manager or call close()."""

    def __init__(self, path: str, meta: Dict[str, Any]):
        try:
            from mcap.writer import Writer
        except ImportError as e:  # pragma: no cover
            raise ImportError('pip install mcap (or apt ros-<distro>-mcap-vendor)') from e
        if meta.get('calibration_version') is None:
            raise SessionRejected('meta needs calibration_version (spec: sessions without one are rejected)')
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self._f = open(path, 'wb')
        self._w = Writer(self._f)
        self._w.start(profile='', library='arm_stack')
        self._ch = {}
        for topic, schema in SCHEMAS.items():
            sid = self._w.register_schema(name=topic, encoding='jsonschema',
                                          data=json.dumps(schema).encode())
            self._ch[topic] = self._w.register_channel(topic=topic, message_encoding='json',
                                                      schema_id=sid)
        self._last_mode = None
        self._write('meta', {**meta, 't_ns': time.monotonic_ns()}, time.monotonic_ns())

    def _write(self, topic, msg, t_ns):
        self._w.add_message(self._ch[topic], log_time=int(t_ns), publish_time=int(t_ns), sequence=0,
                            data=json.dumps(msg).encode())

    def joint_state(self, s: JointState):
        t = int(s.t * 1e9)
        self._write('joint_state', {
            't_ns': t, 'q': _list(s.q), 'qd': _list(s.qd), 'tau': _list(s.tau),
            'tau_raw_permille': _list(s.raw_permille), 'counts_raw': _list(s.raw_counts),
            'status': [int(x) for x in s.status], 'mode': s.mode, 'fallback': bool(s.fallback)}, t)
        if s.mode != self._last_mode:
            if self._last_mode is not None:
                self.event('mode_switch', t_ns=t, **{'from': self._last_mode, 'to': s.mode})
            self._last_mode = s.mode

    def joint_cmd(self, t: float, cmd: Command, tau_sent_permille=None):
        t_ns = int(t * 1e9)
        self._write('joint_cmd', {
            't_ns': t_ns, 'mode': cmd.mode, 'q': _list(cmd.q), 'qd': _list(cmd.qd), 'kp': _list(cmd.kp),
            'kd': _list(cmd.kd), 'tau_ff': _list(cmd.tau_ff), 'tau_sent_permille': _list(tau_sent_permille)},
            t_ns)

    def event(self, event: str, t_ns: Optional[int] = None, **fields):
        t_ns = int(t_ns if t_ns is not None else time.monotonic_ns())
        self._write('events', {'t_ns': t_ns, 'event': event, **_plain(fields)}, t_ns)

    def close(self):
        if self._w is not None:
            self._w.finish()
            self._f.close()
            self._w = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _plain(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            v = v.tolist()
        elif isinstance(v, np.generic):
            v = v.item()
        out[k] = v
    return out


class RecordingClient(ArmClient):
    """ArmClient proxy that records every command sent and state stepped to."""

    def __init__(self, inner: ArmClient, recorder: Recorder, every: int = 1):
        """every: record one cycle in `every` (1 on hardware; the sim uses 10 to stay fast)."""
        self.inner = inner
        self.rec = recorder
        self.every = max(1, int(every))
        self._k = 0
        self.n, self.dt, self.names = inner.n, inner.dt, inner.names
        self.cmd_dt = getattr(inner, 'cmd_dt', inner.dt)
        self._events_seen = 0

    def __getattr__(self, name):          # set_true_state, truth, units, ... on the sim
        return getattr(self.inner, name)

    def state(self) -> JointState:
        return self.inner.state()

    def send(self, cmd: Command) -> None:
        self.inner.send(cmd)
        if self._k % self.every:
            return
        sent = None
        out = getattr(self.inner, '_last_out', None)
        units = getattr(self.inner, 'units', None)
        if out is not None and units is not None:
            sent = units.nm_to_permille(out.tau)
        self.rec.joint_cmd(self.inner.state().t, cmd, sent)

    def step(self, seconds: float = 0.0) -> JointState:
        s = self.inner.step(seconds)
        if self._k % self.every == 0:
            self.rec.joint_state(s)
        self._k += 1
        events = getattr(self.inner, 'events', None)
        if events is not None:
            for e in events[self._events_seen:]:
                e = dict(e)
                self.rec.event(e.pop('event'), t_ns=int(e.pop('t', 0) * 1e9), **e)
            self._events_seen = len(events)
        return s

    def set_payload(self, payload):
        self.rec.event('payload', mass_kg=getattr(payload, 'mass_kg', 0.0))
        return self.inner.set_payload(payload)

    def place_in_jig(self, q_jig):
        return self.inner.place_in_jig(q_jig)

    def mocap(self):
        return self.inner.mocap()

    def hand_guide(self, prompt, hint_q=None):
        self.rec.event('hand_guide', prompt=prompt)
        s = self.inner.hand_guide(prompt, hint_q)
        self.rec.joint_state(s)
        return s

    def drive_gains(self):
        return self.inner.drive_gains()

    def set_drive_gains(self, kp_scale, kd_scale):
        self.inner.set_drive_gains(kp_scale, kd_scale)
        self.rec.event('drive_gains', kp_scale=np.asarray(kp_scale).tolist(),
                       kd_scale=np.asarray(kd_scale).tolist())

    def time_since_power_on(self) -> float:
        return self.inner.time_since_power_on()

    def close(self):
        self.rec.close()
        self.inner.close()


# --------------------------------------------------------------------------- reading
@dataclass
class Session:
    meta: Dict[str, Any]
    state: Dict[str, np.ndarray]
    cmd: List[Dict[str, Any]]
    events: List[Dict[str, Any]]


def iter_messages(path: str) -> Iterator:
    from mcap.reader import make_reader
    with open(path, 'rb') as f:
        for _schema, channel, message in make_reader(f).iter_messages():
            yield channel.topic, json.loads(message.data)


def load_session(path: str) -> Session:
    """Read a session; raises SessionRejected without a calibration version."""
    meta, states, cmds, events = None, [], [], []
    for topic, msg in iter_messages(path):
        if topic == 'meta':
            meta = msg
        elif topic == 'joint_state':
            states.append(msg)
        elif topic == 'joint_cmd':
            cmds.append(msg)
        elif topic == 'events':
            events.append(msg)
    if not meta or meta.get('calibration_version') is None:
        raise SessionRejected('%s: no calibration version in meta' % path)
    keys = ('t_ns', 'q', 'qd', 'tau', 'tau_raw_permille', 'counts_raw')
    state = {k: np.array([m[k] for m in states]) for k in keys} if states else {}
    return Session(meta, state, cmds, events)


def command_array(session: Session, key: str = 'q') -> Dict[str, np.ndarray]:
    """Commands with `key` set, as arrays (t_ns, values)."""
    rows = [c for c in session.cmd if c.get(key) is not None]
    return {'t_ns': np.array([c['t_ns'] for c in rows]), key: np.array([c[key] for c in rows])}
