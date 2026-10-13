"""Main flow of the behavior tree (no rclpy, unit-testable).

    take the oldest plan from the queue (wait if there is none)
    for each skill in the plan:
        up to MAX_ATTEMPTS attempts of run_skill(), each limited to SKILL_TIMEOUT_S,
        RETRY_BACKOFF_S apart
        all attempts failed -> publish FAILED, drop the rest of the plan
    all skills worked -> publish SUCCEEDED
    next plan

The result status (SUCCEEDED/FAILED) stays up until the next plan starts;
IDLE is only the status before the first plan.
"""
import collections
import threading
import time

from behavior_tree.skills import run_skill

MAX_ATTEMPTS = 3
SKILL_TIMEOUT_S = 60.0
RETRY_BACKOFF_S = 1.0   # pause before another attempt: a server that just failed or rejected
                        # a goal (e.g. still busy with the cancelled one) needs a moment
SEEN_IDS_KEPT = 64      # plan ids remembered, to ignore a plan the planner sends twice


def format_status(state, index=None, total=None, skill=None, detail=''):
    """One-line status for /task_status, e.g. 'RUNNING 2/3 pick',
    'SUCCEEDED', 'FAILED 2/3 pick: timed out after 60 s (attempt 3/3)'."""
    text = state
    if index is not None and total is not None:
        text += f' {index}/{total}'
    if skill:
        text += f' {skill}'
    if detail:
        text += f': {detail}'
    return text


class PlanQueue:
    """Thread-safe FIFO of plans (each plan is a list of Skill)."""

    def __init__(self):
        self._items = collections.deque()
        self._cond = threading.Condition()

    def put(self, plan):
        with self._cond:
            self._items.append(plan)
            self._cond.notify()

    def take(self, timeout_s):
        """Oldest plan, or None if none arrives within timeout_s."""
        with self._cond:
            if not self._items:
                self._cond.wait(timeout_s)
            return self._items.popleft() if self._items else None

    def __len__(self):
        with self._cond:
            return len(self._items)


def run_plan(plan, ctx, set_status, max_attempts=MAX_ATTEMPTS, timeout_s=SKILL_TIMEOUT_S,
             run=run_skill, backoff_s=RETRY_BACKOFF_S, sleep=time.sleep):
    """Runs every skill in order. Returns True if the whole plan worked."""
    total = len(plan)
    for index, skill in enumerate(plan, start=1):
        for attempt in range(1, max_attempts + 1):
            if attempt > 1 and backoff_s > 0:
                sleep(backoff_s)
            detail = f'attempt {attempt}/{max_attempts}' if attempt > 1 else ''
            set_status(format_status('RUNNING', index, total, skill.name, detail))
            outcome = run(ctx, skill, timeout_s)
            if outcome.ok:
                break
            if not outcome.retryable or attempt == max_attempts:
                reason = outcome.reason or 'failed'
                if outcome.retryable:
                    reason += f' (attempt {attempt}/{max_attempts})'
                set_status(format_status('FAILED', index, total, skill.name, reason))
                return False
    set_status(format_status('SUCCEEDED'))
    return True


class BehaviorRunner:
    """Owns the queue and the flow. The ROS node feeds it with submit() and
    runs run_forever() on a worker thread."""

    def __init__(self, ctx, publish_status, log=lambda msg: None,
                 max_attempts=MAX_ATTEMPTS, timeout_s=SKILL_TIMEOUT_S,
                 backoff_s=RETRY_BACKOFF_S, sleep=time.sleep):
        self._ctx = ctx
        self._publish = publish_status
        self._log = log
        self._max_attempts = max_attempts
        self._timeout_s = timeout_s
        self._backoff_s = backoff_s
        self._sleep = sleep
        self.queue = PlanQueue()
        self._seen_ids = collections.deque(maxlen=SEEN_IDS_KEPT)
        self.status = format_status('IDLE')

    def set_status(self, text):
        self.status = text
        self._log(f'task_status: {text}')
        self._publish(text)

    def submit(self, plan, plan_id=''):
        """Queue a plan. An empty plan is ignored (returns False), and so is a plan
        whose non-empty plan_id was already seen (the planner resends until it sees
        us start, so duplicates are expected)."""
        if not plan:
            self._log('empty plan received, ignored')
            return False
        if plan_id:
            if plan_id in self._seen_ids:
                self._log(f'duplicate plan {plan_id} ignored')
                return False
            self._seen_ids.append(plan_id)
        self.queue.put(plan)
        return True

    def step(self, wait_s=0.2):
        """Run at most one queued plan. Returns True if a plan was run."""
        plan = self.queue.take(wait_s)
        if plan is None:
            return False
        try:
            run_plan(plan, self._ctx, self.set_status, self._max_attempts, self._timeout_s,
                     backoff_s=self._backoff_s, sleep=self._sleep)
        except Exception as exc:  # never leave the status stuck on RUNNING
            self.set_status(format_status('FAILED', detail=f'unexpected error: {exc}'))
        return True

    def run_forever(self, stop_event):
        while not stop_event.is_set():
            try:
                self.step()
            except Exception as exc:  # e.g. publishing failed while reporting; keep serving plans
                self._log(f'behavior loop error: {exc}')
