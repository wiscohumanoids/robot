"""ROS-free pieces of the integration and stress harness (unit-tested with plain pytest).

The ROS drivers in scripts/ only record what they see into a Timeline and send inputs;
every decision (was that the right status order? is that plan valid? what would a fake
behavior tree say now?) lives here.
"""
import math
import random
import re
from dataclasses import dataclass

from task_planner import skill_catalog
from task_planner.llm_planner import EDGE_MARGIN_M, MAP_MAX_M, MAP_MIN_M

RUNNING = 'RUNNING'


@dataclass
class Event:
    t: float            # seconds since the run started
    kind: str           # 'task' | 'planner' | 'plan' | 'held' | 'sent' (what we sent)
    text: str
    data: object = None


class Timeline:
    """Everything the driver saw, in order. /task_status is stored on change only
    (it repeats at 1 Hz); every message time is kept in task_times for heartbeat checks."""

    def __init__(self):
        self.events = []
        self.task_times = []
        self.last_pose = None       # (x, y, yaw)

    def add(self, t, kind, text, data=None):
        self.events.append(Event(t, kind, text, data))
        return len(self.events) - 1

    def add_task(self, t, text):
        self.task_times.append(t)
        last = self.last('task')
        if last is None or last.text != text:
            self.add(t, 'task', text)

    def add_held(self, t, text):
        last = self.last('held')
        if last is None or last.text != text:
            self.add(t, 'held', text)

    def of(self, kind, start=0):
        return [e for e in self.events[start:] if e.kind == kind]

    def texts(self, kind, start=0):
        return [e.text for e in self.of(kind, start)]

    def last(self, kind):
        for e in reversed(self.events):
            if e.kind == kind:
                return e
        return None

    def count(self, kind, prefix, start=0):
        return sum(1 for e in self.of(kind, start) if e.text.startswith(prefix))

    def find(self, kind, prefix, start=0):
        """Index (into events) of the first matching event at or after `start`, else None."""
        for i in range(start, len(self.events)):
            e = self.events[i]
            if e.kind == kind and e.text.startswith(prefix):
                return i
        return None

    def plans(self):
        """Distinct published plans (a resend repeats a plan_id): list of the Event.data dicts."""
        seen, out = set(), []
        for e in self.of('plan'):
            pid = e.data.get('plan_id') if e.data else None
            if pid and pid in seen:
                continue
            seen.add(pid)
            out.append(e.data)
        return out


# ---------------------------------------------------------------- reading the task status
def task_texts(tl):
    return tl.texts('task')


def run_starts(tl):
    """How many times the tree went from not-running to RUNNING (= plans it started)."""
    starts, previous_running = 0, False
    for text in task_texts(tl):
        running = text.startswith(RUNNING)
        if running and not previous_running:
            starts += 1
        previous_running = running
    return starts


def finished(tl):
    """Final statuses of the plans that ran: each non-RUNNING status that follows a RUNNING one."""
    results, was_running = [], False
    for text in task_texts(tl):
        running = text.startswith(RUNNING)
        if was_running and not running:
            results.append(text)
        was_running = running
    return results


def max_gap(times):
    return max((b - a for a, b in zip(times, times[1:])), default=0.0)


def heartbeat_problems(tl, limit_s=2.5):
    """/task_status must keep arriving (the tree re-sends it at 1 Hz)."""
    gap = max_gap(tl.task_times)
    return [f'/task_status went quiet for {gap:.1f} s (limit {limit_s} s)'] if gap > limit_s else []


def near(pose, goal, tol):
    return pose is not None and math.hypot(pose[0] - goal[0], pose[1] - goal[1]) <= tol


# ---------------------------------------------------------------- reading the planner status
_LINE = re.compile(r'^(\w+) "(.*?)"(?=$|: | \()(?:: (.*)| (\(.*\)))?$', re.S)   # re.S: intents may hold newlines


def classify(text):
    """'NO_PLAN "pick x": reason' -> ('NO_PLAN', 'pick x', 'reason');
    'REMOVED "b" (was #2)' -> ('REMOVED', 'b', '(was #2)');
    'CLEARED: 3 removed' -> ('CLEARED', '', '3 removed'). Unknown text -> ('?', '', text)."""
    match = _LINE.match(text)
    if match:
        return match.group(1), match.group(2), match.group(3) or match.group(4) or ''
    head, sep, rest = text.partition(': ')
    if sep and head.replace('_', '').isupper():
        return head, '', rest
    return '?', '', text


def parse_queue_line(text):
    """'QUEUE: 1. "a" | 2. "b"' -> ['a', 'b']; 'QUEUE: empty' -> []."""
    body = text.partition(': ')[2]
    if body.strip() == 'empty':
        return []
    return [m.group(2) for m in re.finditer(r'(\d+)\. "(.*?)"(?: \| |$)', body)]


def kinds(tl, start=0):
    """Planner status kinds in order, e.g. ['QUEUED', 'PLANNING', 'SENT', 'DONE']."""
    return [classify(t)[0] for t in tl.texts('planner', start)]


def planner_count(tl, kind, start=0):
    return sum(1 for k in kinds(tl, start) if k == kind)


# ---------------------------------------------------------------- plan validity
def world_objects():
    """id -> object dict for what perception_stub reports (cube_1 plus scene.py)."""
    from perception_stub import scene
    objs = {'cube_1': {'object_id': 'cube_1', 'label': 'cube', 'x': 1.5, 'y': 0.3, 'z': scene.TABLE_Z}}
    for o in scene.extra_objects():
        objs[o['object_id']] = o
    return objs


def validate_plan(skills, objects=None):
    """Problems with a published plan (list of {name, object_id, x, y, z}); [] if fine.
    Checks what must hold for ANY plan the planner publishes, LLM or not."""
    objects = objects if objects is not None else world_objects()
    if not skills:
        return ['empty plan was published']
    problems = []
    low, high = MAP_MIN_M + EDGE_MARGIN_M, MAP_MAX_M - EDGE_MARGIN_M
    for n, step in enumerate(skills, start=1):
        for p in skill_catalog.check_step(step):
            problems.append(f'step {n}: {p}')
        if step.get('object_id') and step['object_id'] not in objects:
            problems.append(f"step {n}: unknown object {step['object_id']!r}")
        spec = skill_catalog.get(step.get('name'))
        if spec and spec.needs_pose:
            x, y = step.get('x'), step.get('y')
            if x is None or y is None or not (low <= x <= high and low <= y <= high):
                problems.append(f'step {n}: goal ({x}, {y}) is outside the map')
    problems += skill_catalog.check_hand(skills, '')       # the stress fake tree never moves the hand
    return problems


# ---------------------------------------------------------------- statistics
def percentile(values, p):
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, math.ceil(p / 100.0 * len(ordered)) - 1))
    return ordered[k]


def describe_latencies(values):
    if not values:
        return 'no samples'
    return (f'n={len(values)} min={min(values):.2f}s p50={percentile(values, 50):.2f}s '
            f'p95={percentile(values, 95):.2f}s max={max(values):.2f}s')


def latencies(tl, from_kind='QUEUED', to_kind='SENT'):
    """Seconds from each `from_kind` planner line to the next `to_kind` line, matched in order."""
    pending, out = [], []
    for e in tl.of('planner'):
        kind = classify(e.text)[0]
        if kind == from_kind:
            pending.append(e.t)
        elif kind == to_kind and pending:
            out.append(e.t - pending.pop(0))
    return out


# ---------------------------------------------------------------- a fake behavior tree
class FakeTree:
    """Stands in for the behavior tree in the stress tests, so the planner can be driven
    through every tree behavior (slow, failing, silent, stale) without Nav2.

    modes: 'normal'      runs each plan for run_s, fails it with probability fail_rate
           'always_fail' every plan fails
           'hang'        starts a plan and never finishes it
           'never_start' accepts plans and never starts them (status stays `initial`)
    Like the real tree it holds its last result until the next plan starts and ignores
    a plan_id it has already seen. `overlaps` records plans that arrived while it was busy.
    """

    def __init__(self, mode='normal', run_s=0.3, fail_rate=0.0, seed=1, initial='IDLE', dedupe=True):
        self.mode, self.run_s, self.fail_rate = mode, run_s, fail_rate
        self.dedupe = dedupe
        self.rng = random.Random(seed)
        self.initial = initial
        self.reset(mode, run_s, fail_rate, initial)

    def reset(self, mode=None, run_s=None, fail_rate=None, initial=None):
        self.mode = mode or self.mode
        self.run_s = self.run_s if run_s is None else run_s
        self.fail_rate = self.fail_rate if fail_rate is None else fail_rate
        self.initial = initial or 'IDLE'
        self._status = self.initial
        self._seen = set()
        self._queue = []
        self._current = None
        self.plans_run = []
        self.overlaps = []
        self.duplicates = 0

    @property
    def busy(self):
        return self._current is not None or bool(self._queue)

    def on_plan(self, plan_id, skills, now):
        """A /skill_sequence message arrived. Returns False if it was a duplicate or empty."""
        if not skills:
            return False
        if self.dedupe and plan_id:
            if plan_id in self._seen:
                self.duplicates += 1
                return False
            self._seen.add(plan_id)
        if self.busy and self.mode != 'never_start':
            self.overlaps.append(plan_id)
        if self.mode != 'never_start':
            self._queue.append((plan_id, skills))
        return True

    def status(self, now):
        """The /task_status text at time `now` (advances the fake tree)."""
        if self._current is None and self._queue and self.mode != 'never_start':
            plan_id, skills = self._queue.pop(0)
            fail = self.mode == 'always_fail' or self.rng.random() < self.fail_rate
            self._current = {'id': plan_id, 'skills': skills, 'start': now, 'fail': fail}
        cur = self._current
        if cur is None:
            return self._status
        n = len(cur['skills'])
        elapsed = now - cur['start']
        if self.mode != 'hang' and elapsed >= self.run_s:
            self.plans_run.append(cur['id'])
            self._status = (f"FAILED 1/{n} {cur['skills'][0]['name']}: injected failure (attempt 3/3)"
                            if cur['fail'] else 'SUCCEEDED')
            self._current = None
            return self._status
        index = 1 + min(n - 1, int(elapsed / max(self.run_s, 1e-6) * n)) if self.mode != 'hang' else 1
        self._status = f"RUNNING {index}/{n} {cur['skills'][index - 1]['name']}"
        return self._status
