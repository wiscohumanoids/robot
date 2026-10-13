"""Intent queue + when-to-plan decisions (no rclpy, unit-testable).

The queue holds raw user text. Nothing is planned when it is typed: an intent is
planned only when the behavior tree is idle (nothing running, nothing about to run),
so the plan is built from where the robot and the objects are NOW, not where they
were when the user asked.

    submit(text)            -> queue (no size limit for now; see MAX_QUEUED)
    on_status(text)         <- every /task_status message
    tick()                  -> if the tree is idle and the queue is not empty:
                               take the oldest intent, plan it with the current world,
                               publish the plan, then wait for the tree to finish it

When the tree reports FAILED for a plan we sent, the same intent is replanned ONCE
(replan_fn gets the intent, the old plan, the failure text and the current world)
before the next queued intent is looked at. A second failure is final.

Phases: READY -> PLANNING -> AWAITING_START (plan sent, tree not RUNNING yet) ->
RUNNING (tree busy with it) -> READY. The tree keeps showing the previous
SUCCEEDED/FAILED until it starts the next plan, so "status is idle" alone would
let a second plan be sent too early; AWAITING_START closes that gap.

/skill_sequence is not delivered to a tree that is not subscribed yet, so a plan can
be lost. If the tree has not started the plan RESEND_AFTER_S after a send, the same
plan (same plan_id, which the tree uses to ignore duplicates) is sent again, up to
MAX_SENDS times in total. Then the intent is given up (reported as NO_PLAN).

DONE is reported only when the tree says SUCCEEDED. If a plan we sent stops being
RUNNING with any other status that is not FAILED (e.g. IDLE: the tree was restarted
mid-plan), the outcome is unknown: it is reported as GAVE_UP and not replanned.

The tree re-sends its status at 1 Hz. If nothing arrives for STATUS_TIMEOUT_S while a
plan is in flight, the tree is taken for dead: the plan is given up (GAVE_UP), and
nothing more is planned until a status is heard again.
"""
import collections
import math
import threading
import time
import uuid

MAX_QUEUED = None       # no limit for now. Set a number to reject intents beyond it (REJECTED)
RESEND_AFTER_S = 2.0
MAX_SENDS = 3
STATUS_TIMEOUT_S = 5.0  # the tree's heartbeat is 1 Hz; None disables the check
NO_STATUS_LOG_S = 10.0
TICK_S = 0.1

READY, PLANNING, AWAITING_START, RUNNING = 'READY', 'PLANNING', 'AWAITING_START', 'RUNNING'


def is_busy(status):
    """True while the tree is running a plan ('RUNNING 2/3 pick ...')."""
    return status.startswith('RUNNING')


def is_success(status):
    return status.startswith('SUCCEEDED')


def format_report(state, intent, detail=''):
    """One line for /planner_status: QUEUED, PLANNING, SENT, NO_PLAN, REJECTED, DONE,
    RETRYING, REPLANNING, RESENDING or GAVE_UP,
    e.g. 'NO_PLAN "pick the ball": no such object' ."""
    text = f'{state} "{intent}"'
    return f'{text}: {detail}' if detail else text


def yaw_from_quaternion(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class Dispatcher:
    """plan_fn(intent, world) -> object with .plan (list) and .error (str);
    world = {'robot': {'x','y','yaw'} or None, 'objects': [...]}, read at dispatch time.
    publish(intent, plan, plan_id) sends the plan to the behavior tree (again with the
    same plan_id on a resend).
    report(text) tells the user what happened to their intent (see format_report).
    replan_fn(intent, old_plan, failure, world) -> like plan_fn; None disables replanning."""

    def __init__(self, plan_fn, publish, get_world, log=lambda msg: None,
                 report=lambda text: None, replan_fn=None, now=time.monotonic, max_queued=MAX_QUEUED, resend_after_s=RESEND_AFTER_S,
                 max_sends=MAX_SENDS, new_id=lambda: uuid.uuid4().hex[:8],
                 status_timeout_s=STATUS_TIMEOUT_S):
        self._plan_fn = plan_fn
        self._publish = publish
        self._get_world = get_world
        self._log = log
        self._report = report
        self._replan_fn = replan_fn
        self._now = now
        self._max_queued = max_queued
        self._resend_after_s = resend_after_s
        self._max_sends = max_sends
        self._new_id = new_id
        self._status_timeout_s = status_timeout_s
        self._lock = threading.Lock()
        self._queue = collections.deque()
        self._phase = READY
        self._status_seen = False
        self._busy = False
        self._status_at = 0.0   # when the last /task_status arrived
        self._deadline = 0.0
        self._last_wait_log = None
        self._current = None    # {'intent','plan','plan_id','sends','retried'}: the plan in flight
        self._retry = None      # {'intent', 'plan', 'failure'}: failed once, replan next
        self.status = None      # last /task_status text

    # -------------------------------------------------------------- inputs
    def submit(self, text):
        """Queue raw user text. Returns False if it was empty or the queue is full."""
        text = (text or '').strip()
        if not text:
            self._log('ignoring an empty /user_intent')
            return False
        with self._lock:
            if self._max_queued is not None and len(self._queue) >= self._max_queued:
                self._log(f'intent queue full ({self._max_queued}), rejected: "{text}"')
                self._say(format_report('REJECTED', text, f'queue full ({self._max_queued})'))
                return False
            self._queue.append(text)
            depth = len(self._queue)
        self._log(f'intent queued ({depth} waiting): "{text}"')
        self._say(format_report('QUEUED', text, f'{depth} waiting'))
        return True

    def on_status(self, text):
        with self._lock:
            self.status = text
            self._status_seen = True
            self._status_at = self._now()
            self._busy = is_busy(text)
            if self._phase == AWAITING_START and self._busy:
                self._phase = RUNNING
            elif self._phase == RUNNING and not self._busy:
                self._phase = READY
                finished = self._current
                self._current = None
                self._finish(finished, text)

    def _finish(self, job, status):
        """The tree finished `job` (lock held): report it, or queue a one-time replan."""
        if job is None:
            return
        intent = job['intent']
        if is_success(status):
            self._say(format_report('DONE', intent))
        elif not status.startswith('FAILED'):
            # Neither result: the tree dropped the plan (e.g. restarted). What the robot
            # did is unknown, so no DONE and no replan from a guessed state.
            reason = f'behavior tree stopped without a result (status: {status})'
            self._log(f'"{intent}": {reason}')
            self._say(format_report('GAVE_UP', intent, reason))
        elif job['retried'] or self._replan_fn is None:
            self._say(format_report('GAVE_UP', intent, status))
        else:
            self._retry = {'intent': intent, 'plan': job['plan'], 'failure': status}
            self._say(format_report('RETRYING', intent, status))

    # -------------------------------------------------------------- queue editing
    # Positions are 1-based and match the order shown by show(): 1 is the next to run.
    # These only touch intents that are still waiting; the plan the tree is running is
    # not affected.
    def clear(self):
        """Drop every waiting intent and any pending retry. Returns the removed intents."""
        with self._lock:
            removed = list(self._queue)
            self._queue.clear()
            had_retry = self._retry is not None
            self._retry = None
        text = f'{len(removed)} removed' if removed else 'queue was empty'
        if had_retry:
            text += ' (and the pending retry)'
        self._log(f'queue cleared: {text}')
        self._say(f'CLEARED: {text}')
        return removed

    def remove(self, position):
        """Drop the waiting intent at `position` (1 = next to run). Returns it, or None
        if there is no such position."""
        with self._lock:
            size = len(self._queue)
            if not isinstance(position, int) or not 1 <= position <= size:
                removed = None
            else:
                removed = self._queue[position - 1]
                del self._queue[position - 1]
        if removed is None:
            self._log(f'cannot remove #{position}: queue has {size}')
            self._say(f'REMOVE_FAILED: no intent #{position} (queue has {size})')
        else:
            self._log(f'removed #{position}: "{removed}"')
            self._say(f'REMOVED "{removed}" (was #{position})')
        return removed

    def show(self):
        """Report the waiting intents. Returns them."""
        waiting = self.pending()
        listing = ' | '.join(f'{i}. "{t}"' for i, t in enumerate(waiting, start=1))
        self._say(f'QUEUE: {listing}' if waiting else 'QUEUE: empty')
        return waiting

    def command(self, text):
        """Run a text command from /planner_command: `clear`, `remove <n>`, `list`."""
        words = (text or '').lower().split()
        if words == ['clear']:
            self.clear()
        elif words == ['list']:
            self.show()
        elif len(words) == 2 and words[0] == 'remove' and words[1].isdigit():
            self.remove(int(words[1]))
        else:
            self._log(f'unknown planner command: "{text}"')
            self._say(f'COMMAND_FAILED "{(text or "").strip()}": expected clear | remove <n> | list')

    # -------------------------------------------------------------- state
    @property
    def phase(self):
        with self._lock:
            return self._phase

    def __len__(self):
        with self._lock:
            return len(self._queue)

    def pending(self):
        with self._lock:
            return list(self._queue)

    # -------------------------------------------------------------- the decision
    def tick(self):
        """One decision. Returns True if an intent was taken from the queue."""
        self._check_heartbeat()
        self._resend_or_give_up()
        with self._lock:
            if self._phase != READY or not (self._queue or self._retry):
                return False
            if not self._status_seen or self._busy:
                self._log_waiting()
                return False
            retry, self._retry = self._retry, None
            intent = retry['intent'] if retry else self._queue.popleft()
            self._phase = PLANNING
        self._plan_and_send(intent, retry)
        return True

    def _check_heartbeat(self):
        """No /task_status for too long: the tree is gone. Give up the plan in flight and
        wait for a status before planning anything else."""
        with self._lock:
            if (self._status_timeout_s is None or not self._status_seen
                    or self._now() - self._status_at < self._status_timeout_s):
                return
            self._status_seen = False
            self._busy = False
            self._last_wait_log = None
            job = self._current if self._phase in (AWAITING_START, RUNNING) else None
            if job is not None:
                was_running = self._phase == RUNNING
                self._phase = READY
                self._current = None
            self._log(f'no /task_status for {self._status_timeout_s:g} s: behavior tree not running?')
            if job is not None:
                state = 'while the plan was running' if was_running else 'after the plan was sent'
                self._say(format_report(
                    'GAVE_UP', job['intent'],
                    f'no /task_status for {self._status_timeout_s:g} s {state} (outcome unknown)'))

    def _resend_or_give_up(self):
        """The tree has not started the plan we sent: send it again, or give up."""
        resend = None
        with self._lock:
            if self._phase != AWAITING_START or self._now() < self._deadline:
                return
            job = self._current
            if job is not None and job['sends'] < self._max_sends:
                job['sends'] += 1
                self._deadline = self._now() + self._resend_after_s
                resend = dict(job)
            else:
                self._phase = READY
                self._current = None
                if job is not None:
                    reason = f"behavior tree did not start the plan ({job['sends']} sends)"
                    self._log(f'"{job["intent"]}": {reason}')
                    self._say(format_report('NO_PLAN', job['intent'], reason))
        if resend is None:
            return
        self._log(f'resending plan {resend["plan_id"]} (send {resend["sends"]}/{self._max_sends})')
        self._say(format_report('RESENDING', resend['intent'],
                                f'tree has not started it (send {resend["sends"]}/{self._max_sends})'))
        try:
            self._publish(resend['intent'], resend['plan'], resend['plan_id'])
        except Exception as exc:
            self._log(f'resending the plan for "{resend["intent"]}" failed: {exc}')

    def _log_waiting(self):
        if self._status_seen:
            return
        now = self._now()
        if self._last_wait_log is None or now - self._last_wait_log >= NO_STATUS_LOG_S:
            self._last_wait_log = now
            self._log('waiting for /task_status from the behavior tree before planning')

    def _say(self, text):
        try:
            self._report(text)
        except Exception as exc:  # reporting must never break planning
            self._log(f'could not report "{text}": {exc}')

    def _plan_and_send(self, intent, retry=None):
        self._say(format_report('REPLANNING' if retry else 'PLANNING', intent))
        try:
            world = self._get_world()
            if retry:
                result = self._replan_fn(intent, retry['plan'], retry['failure'], world)
            else:
                result = self._plan_fn(intent, world)
            plan, error = result.plan, result.error
        except Exception as exc:  # a planner bug must not wedge the queue
            plan, error = [], f'planner error: {exc}'
        if not plan:
            reason = error or 'empty plan'
            self._log(f'no plan for "{intent}": {reason}')
            self._say(format_report('NO_PLAN', intent, f'replan failed: {reason}' if retry else reason))
            self._set_phase(READY)
            return
        with self._lock:                    # before publishing: the tree may answer at once
            self._phase = AWAITING_START
            self._deadline = self._now() + self._resend_after_s
            plan_id = self._new_id()
            self._current = {'intent': intent, 'plan': plan, 'plan_id': plan_id, 'sends': 1,
                             'retried': retry is not None}
        try:
            self._publish(intent, plan, plan_id)
        except Exception as exc:
            self._log(f'publishing the plan for "{intent}" failed: {exc}')
            self._say(format_report('NO_PLAN', intent, f'publishing failed: {exc}'))
            with self._lock:
                self._phase = READY
                self._current = None
            return
        self._say(format_report('SENT', intent, f'{len(plan)} steps'))

    def _set_phase(self, phase):
        with self._lock:
            self._phase = phase

    def run_forever(self, stop_event, period_s=TICK_S):
        while not stop_event.is_set():
            try:
                self.tick()
            except Exception as exc:
                self._log(f'dispatcher error: {exc}')
            stop_event.wait(period_s)
