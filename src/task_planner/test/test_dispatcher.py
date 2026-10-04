"""When does an intent get planned? Dispatcher with a fake planner, fake tree status and
a fake clock. No ROS, no threads (tick() is called by hand)."""
import threading
from types import SimpleNamespace

import pytest

from task_planner import dispatcher as d
from task_planner.dispatcher import Dispatcher

PLAN = [{'name': 'pick', 'object_id': 'cube_1', 'x': 1.0, 'y': 2.0, 'z': 0.0}]
IDLE, RUN, OK, FAIL = 'IDLE', 'RUNNING 1/1 pick', 'SUCCEEDED', 'FAILED 1/1 pick: x (attempt 3/3)'


class Rig:
    def __init__(self, plan=PLAN, error='', replan=True, **kwargs):
        self.t = 0.0
        self.logs, self.planned, self.sent, self.reports, self.ids = [], [], [], [], []
        self.world = {'robot': {'x': 0.0, 'y': 0.0, 'yaw': 0.0}, 'objects': []}
        self.plan, self.error = plan, error
        self.replans = []
        self.replan_plan = [{'name': 'navigate_to', 'object_id': '', 'x': 0.5, 'y': 0.5, 'z': 0.0}]
        self.replan_error = ''

        def replan_fn(intent, old_plan, failure, world):
            self.replans.append((intent, old_plan, failure, world))
            return SimpleNamespace(plan=self.replan_plan, error=self.replan_error)

        def plan_fn(intent, world):
            self.planned.append((intent, world))
            return SimpleNamespace(plan=self.plan, error=self.error)

        self.d = Dispatcher(plan_fn, self._publish, lambda: self.world,
                            log=self.logs.append, report=self.reports.append,
                            replan_fn=replan_fn if replan else None,
                            now=lambda: self.t, **kwargs)

    def _publish(self, intent, plan, plan_id):
        self.sent.append((intent, plan))
        self.ids.append(plan_id)

    def intents(self):
        return [i for i, _ in self.sent]


def test_nothing_is_planned_before_a_status_has_been_heard():
    r = Rig()
    r.d.submit('pick up the cube')
    assert not r.d.tick() and r.sent == [] and len(r.d) == 1


def test_idle_tree_gets_the_oldest_intent_first():
    r = Rig()
    r.d.on_status(IDLE)
    r.d.submit('first')
    r.d.submit('second')
    assert r.d.tick()
    assert r.intents() == ['first'] and r.d.pending() == ['second']
    assert r.sent[0][1] == PLAN


@pytest.mark.parametrize('status', [IDLE, OK, FAIL])
def test_every_non_running_status_counts_as_idle(status):
    r = Rig()
    r.d.on_status(status)
    r.d.submit('go')
    assert r.d.tick() and r.intents() == ['go']


def test_a_running_tree_is_never_given_another_plan():
    r = Rig()
    r.d.on_status(RUN)
    r.d.submit('go')
    assert not r.d.tick() and r.sent == [] and len(r.d) == 1


def test_empty_queue_does_nothing():
    r = Rig()
    r.d.on_status(IDLE)
    assert not r.d.tick() and r.planned == []


def test_second_intent_waits_for_the_first_to_start_and_finish():
    r = Rig()
    r.d.on_status(IDLE)
    r.d.submit('one')
    r.d.submit('two')
    r.d.tick()
    assert r.d.phase == d.AWAITING_START
    r.d.on_status(OK)                       # stale status from before: tree has not started yet
    assert not r.d.tick() and r.intents() == ['one']
    r.d.on_status(RUN)
    assert r.d.phase == d.RUNNING and not r.d.tick()
    r.d.on_status(RUN)
    assert not r.d.tick() and r.intents() == ['one']
    r.d.on_status(OK)
    assert r.d.phase == d.READY
    assert r.d.tick() and r.intents() == ['one', 'two']


def test_a_failed_plan_run_releases_the_next_intent_once_its_retry_is_done():
    r = Rig(replan=False)                       # no replanner: the failure is final
    r.d.on_status(IDLE)
    r.d.submit('one')
    r.d.submit('two')
    r.d.tick()
    r.d.on_status(RUN)
    r.d.on_status(FAIL)
    assert r.d.tick() and r.intents() == ['one', 'two']


def test_the_world_is_read_when_the_intent_is_planned_not_when_it_was_typed():
    r = Rig()
    r.d.on_status(RUN)
    r.d.submit('go')
    r.world = {'robot': {'x': 3.0, 'y': 4.0, 'yaw': 1.0}, 'objects': ['moved']}   # robot moved meanwhile
    r.d.on_status(OK)
    r.d.tick()
    assert r.planned[0][1]['robot']['x'] == 3.0 and r.planned[0][1]['objects'] == ['moved']


def test_a_plan_the_tree_does_not_start_is_resent_with_the_same_id():
    r = Rig(resend_after_s=2.0, max_sends=3)
    r.d.on_status(IDLE)
    r.d.submit('one')
    r.d.tick()
    assert r.ids and len(r.sent) == 1
    r.t = 1.9
    r.d.tick()
    assert len(r.sent) == 1                       # not yet
    r.t = 2.0
    r.d.tick()
    assert len(r.sent) == 2 and r.ids[0] == r.ids[1] and r.sent[0] == r.sent[1]
    assert r.reports[-1] == 'RESENDING "one": tree has not started it (send 2/3)'


def test_no_resend_once_the_tree_has_started_the_plan():
    r = Rig()
    r.d.on_status(IDLE)
    r.d.submit('one')
    r.d.tick()
    r.d.on_status(RUN)
    r.t = 100.0
    r.d.tick()
    assert len(r.sent) == 1


def test_after_max_sends_the_intent_is_given_up_and_reported():
    r = Rig(resend_after_s=2.0, max_sends=3)
    r.d.on_status(IDLE)
    r.d.submit('one')
    r.d.submit('two')
    r.d.tick()
    for t in (2.0, 4.0):
        r.t = t
        r.d.tick()
    assert r.intents() == ['one', 'one', 'one']
    r.t = 6.0
    r.d.tick()                                    # gives up, then carries on with the queue
    assert 'NO_PLAN "one": behavior tree did not start the plan (3 sends)' in r.reports
    assert r.intents() == ['one', 'one', 'one', 'two']
    assert r.ids[3] != r.ids[0]                   # a new plan, a new id


def test_a_resend_that_raises_is_logged_and_retried_later():
    r = Rig(resend_after_s=2.0, max_sends=3)
    r.d.on_status(IDLE)
    r.d.submit('one')
    r.d.tick()
    real = r.d._publish
    r.d._publish = lambda i, p, pid: (_ for _ in ()).throw(RuntimeError('dds down'))
    r.t = 2.0
    r.d.tick()
    assert any('resending the plan' in m and 'dds down' in m for m in r.logs)
    r.d._publish = real
    r.t = 4.0
    r.d.tick()
    assert len(r.sent) == 2


def test_every_plan_gets_a_unique_id():
    r = Rig()
    r.d.on_status(IDLE)
    for text in ('a', 'b'):
        r.d.submit(text)
        r.d.tick()
        r.d.on_status(RUN)
        r.d.on_status(OK)
    assert len(set(r.ids)) == 2 and all(r.ids)


def test_a_replan_gets_a_new_id():
    r = Rig()
    start_and_fail(r)
    r.d.tick()
    assert len(r.ids) == 2 and r.ids[0] != r.ids[1]


def test_a_busy_tree_is_waited_for_not_resent_to():
    r = Rig(resend_after_s=1.0)
    r.d.on_status(IDLE)
    r.d.submit('one')
    r.d.tick()
    r.d.on_status(RUN)
    r.t = 500.0
    r.d.tick()
    assert r.intents() == ['one']


# -------------------------------------------------------------- planning failures
@pytest.mark.parametrize('plan, error', [([], 'cannot do that: no'), ([], '')])
def test_no_plan_publishes_nothing_logs_the_reason_and_moves_on(plan, error):
    r = Rig(plan=plan, error=error)
    r.d.on_status(IDLE)
    r.d.submit('bad')
    r.d.submit('next')
    r.d.tick()
    assert r.sent == [] and r.d.phase == d.READY
    assert any('no plan for "bad"' in m for m in r.logs)
    r.plan = PLAN
    assert r.d.tick() and r.intents() == ['next']


def test_a_planner_crash_does_not_wedge_the_queue():
    r = Rig()
    r.d._plan_fn = lambda intent, world: (_ for _ in ()).throw(RuntimeError('boom'))
    r.d.on_status(IDLE)
    r.d.submit('x')
    r.d.tick()
    assert r.d.phase == d.READY and any('boom' in m for m in r.logs)


def test_a_failing_publish_returns_to_ready():
    r = Rig()
    r.d._publish = lambda i, p, pid: (_ for _ in ()).throw(RuntimeError('dds down'))
    r.d.on_status(IDLE)
    r.d.submit('x')
    r.d.tick()
    assert r.d.phase == d.READY and any('dds down' in m for m in r.logs)


# -------------------------------------------------------------- the queue
@pytest.mark.parametrize('text', ['', '   ', None, '\n'])
def test_empty_intents_are_ignored(text):
    r = Rig()
    assert r.d.submit(text) is False and len(r.d) == 0


def test_intents_are_trimmed():
    r = Rig()
    r.d.on_status(IDLE)
    r.d.submit('  pick up the cup \n')
    r.d.tick()
    assert r.intents() == ['pick up the cup']


def test_the_queue_is_capped_and_rejects_extra_intents():
    r = Rig(max_queued=2)
    assert r.d.submit('a') and r.d.submit('b')
    assert r.d.submit('c') is False
    assert r.d.pending() == ['a', 'b'] and any('queue full' in m for m in r.logs)


def test_room_frees_up_once_an_intent_is_taken():
    r = Rig(max_queued=1)
    r.d.on_status(IDLE)
    r.d.submit('a')
    r.d.tick()
    assert r.d.submit('b')


def test_default_cap_is_ten():
    r = Rig()
    assert [r.d.submit(str(i)) for i in range(11)] == [True] * 10 + [False]


def test_waiting_for_the_first_status_is_logged_once_per_interval():
    r = Rig()
    r.d.submit('x')
    r.d.tick()
    r.d.tick()
    assert sum('waiting for /task_status' in m for m in r.logs) == 1
    r.t = d.NO_STATUS_LOG_S
    r.d.tick()
    assert sum('waiting for /task_status' in m for m in r.logs) == 2


# -------------------------------------------------------------- helpers
@pytest.mark.parametrize('status, busy', [
    ('RUNNING 1/2 navigate_to', True), ('RUNNING 1/2 navigate_to: attempt 2/3', True),
    ('IDLE', False), ('SUCCEEDED', False), ('FAILED 1/2 pick: timed out after 60 s (attempt 3/3)', False)])
def test_is_busy(status, busy):
    assert d.is_busy(status) is busy


def test_every_status_the_tree_can_publish_is_classified():
    from behavior_tree.bt_logic import format_status
    assert d.is_busy(format_status('RUNNING', 1, 2, 'pick', 'attempt 2/3'))
    for state in ('IDLE', 'SUCCEEDED'):
        assert not d.is_busy(format_status(state))
    assert not d.is_busy(format_status('FAILED', 1, 2, 'pick', 'x'))


def test_yaw_from_quaternion():
    import math
    assert d.yaw_from_quaternion(0, 0, 0, 1) == 0
    assert abs(d.yaw_from_quaternion(0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)) - math.pi / 2) < 1e-9


def test_run_forever_ticks_until_stopped():
    r = Rig()
    r.d.on_status(IDLE)
    r.d.submit('x')
    stop = threading.Event()
    r.d._publish = lambda i, p, pid: (r.sent.append((i, p)), stop.set())
    r.d.run_forever(stop, period_s=0.01)
    assert r.intents() == ['x']


def test_run_forever_survives_an_exception_in_tick():
    r = Rig()
    stop = threading.Event()
    calls = []

    def tick():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError('oops')
        stop.set()
    r.d.tick = tick
    r.d.run_forever(stop, period_s=0.01)
    assert len(calls) == 2 and any('oops' in m for m in r.logs)


# -------------------------------------------------------------- what the user is told (/planner_status)
def test_format_report():
    assert d.format_report('PLANNING', 'go') == 'PLANNING "go"'
    assert d.format_report('NO_PLAN', 'go', 'no such object') == 'NO_PLAN "go": no such object'


def test_a_successful_intent_reports_queued_planning_sent_in_order():
    r = Rig()
    r.d.on_status(IDLE)
    r.d.submit('pick up the cup')
    r.d.tick()
    assert r.reports == ['QUEUED "pick up the cup": 1 waiting',
                         'PLANNING "pick up the cup"',
                         'SENT "pick up the cup": 1 steps']


def test_queued_reports_how_many_are_waiting():
    r = Rig()
    r.d.submit('a')
    r.d.submit('b')
    assert r.reports == ['QUEUED "a": 1 waiting', 'QUEUED "b": 2 waiting']


def test_a_failed_plan_tells_the_user_why():
    r = Rig(plan=[], error='cannot do that: no such object')
    r.d.on_status(IDLE)
    r.d.submit('pick the ball')
    r.d.tick()
    assert r.reports[-1] == 'NO_PLAN "pick the ball": cannot do that: no such object'


def test_an_empty_plan_without_a_reason_still_reports_one():
    r = Rig(plan=[], error='')
    r.d.on_status(IDLE)
    r.d.submit('x')
    r.d.tick()
    assert r.reports[-1] == 'NO_PLAN "x": empty plan'


def test_a_planner_crash_is_reported():
    r = Rig()
    r.d._plan_fn = lambda intent, world: (_ for _ in ()).throw(RuntimeError('boom'))
    r.d.on_status(IDLE)
    r.d.submit('x')
    r.d.tick()
    assert r.reports[-1] == 'NO_PLAN "x": planner error: boom'


def test_a_failing_publish_is_reported_and_not_called_sent():
    r = Rig()
    r.d._publish = lambda i, p, pid: (_ for _ in ()).throw(RuntimeError('dds down'))
    r.d.on_status(IDLE)
    r.d.submit('x')
    r.d.tick()
    assert r.reports[-1] == 'NO_PLAN "x": publishing failed: dds down'
    assert not any(m.startswith('SENT') for m in r.reports)


def test_a_rejected_intent_is_reported():
    r = Rig(max_queued=1)
    r.d.submit('a')
    r.d.submit('b')
    assert r.reports[-1] == 'REJECTED "b": queue full (1)'


def test_an_empty_intent_is_not_reported():
    r = Rig()
    r.d.submit('  ')
    assert r.reports == []


def test_a_broken_report_channel_does_not_stop_planning():
    r = Rig()
    r.d._report = lambda text: (_ for _ in ()).throw(RuntimeError('pub failed'))
    r.d.on_status(IDLE)
    assert r.d.submit('x') is True
    r.d.tick()
    assert r.intents() == ['x'] and any('could not report' in m for m in r.logs)


# -------------------------------------------------------------- replanning after a tree failure
def start_and_fail(r, intent='go'):
    r.d.on_status(IDLE)
    r.d.submit(intent)
    r.d.tick()
    r.d.on_status(RUN)
    r.d.on_status(FAIL)


def test_a_failed_plan_is_replanned_once_with_the_intent_old_plan_and_failure():
    r = Rig()
    start_and_fail(r)
    assert r.d.tick()
    intent, old_plan, failure, world = r.replans[0]
    assert (intent, old_plan, failure) == ('go', PLAN, FAIL)
    assert r.sent[-1] == ('go', r.replan_plan)
    assert len(r.planned) == 1                       # the first-time planner was not called again


def test_the_replan_uses_the_world_at_that_moment():
    r = Rig()
    start_and_fail(r)
    r.world = {'robot': {'x': 9.0, 'y': 9.0, 'yaw': 0.0}, 'objects': ['now']}
    r.d.tick()
    assert r.replans[0][3]['robot']['x'] == 9.0 and r.replans[0][3]['objects'] == ['now']


def test_the_replan_goes_before_the_next_queued_intent():
    r = Rig()
    r.d.on_status(IDLE)
    r.d.submit('one')
    r.d.submit('two')
    r.d.tick()
    r.d.on_status(RUN)
    r.d.on_status(FAIL)
    r.d.tick()
    assert r.intents() == ['one', 'one']
    r.d.on_status(RUN)
    r.d.on_status(OK)
    r.d.tick()
    assert r.intents() == ['one', 'one', 'two']


def test_the_replan_waits_until_the_tree_is_idle_like_any_plan():
    r = Rig()
    start_and_fail(r)
    r.d.on_status(RUN)                                   # tree busy again (foreign plan)
    assert not r.d.tick() and r.replans == []


def test_a_second_failure_is_final_no_third_attempt():
    r = Rig()
    start_and_fail(r)
    r.d.tick()
    r.d.on_status(RUN)
    r.d.on_status(FAIL)
    assert not r.d.tick() and len(r.replans) == 1
    assert r.reports[-1] == f'GAVE_UP "go": {FAIL}'


def test_the_next_intent_runs_after_the_retry_also_failed():
    r = Rig()
    r.d.on_status(IDLE)
    r.d.submit('one')
    r.d.submit('two')
    r.d.tick()
    r.d.on_status(RUN); r.d.on_status(FAIL)
    r.d.tick()
    r.d.on_status(RUN); r.d.on_status(FAIL)
    r.d.tick()
    assert r.intents() == ['one', 'one', 'two'] and len(r.replans) == 1


def test_each_intent_gets_its_own_one_retry():
    r = Rig()
    r.d.on_status(IDLE)
    r.d.submit('one')
    r.d.submit('two')
    for _ in range(2):                                   # one fails, retry fails; two fails, retry
        r.d.tick(); r.d.on_status(RUN); r.d.on_status(FAIL)
        r.d.tick(); r.d.on_status(RUN); r.d.on_status(FAIL)
    assert [i for i, *_ in r.replans] == ['one', 'two']


def test_a_successful_plan_is_never_replanned():
    r = Rig()
    r.d.on_status(IDLE)
    r.d.submit('go')
    r.d.tick(); r.d.on_status(RUN); r.d.on_status(OK)
    assert not r.d.tick() and r.replans == [] and r.reports[-1] == 'DONE "go"'


def test_a_successful_retry_is_done():
    r = Rig()
    start_and_fail(r)
    r.d.tick(); r.d.on_status(RUN); r.d.on_status(OK)
    assert r.reports[-1] == 'DONE "go"' and not r.d.tick()


def test_a_failure_reports_retrying_then_replanning_then_sent():
    r = Rig()
    start_and_fail(r)
    r.d.tick()
    assert r.reports[-3:] == [f'RETRYING "go": {FAIL}', 'REPLANNING "go"', 'SENT "go": 1 steps']


def test_a_replan_that_cannot_be_made_is_reported_and_final():
    r = Rig()
    r.replan_plan, r.replan_error = [], 'cannot do that: unreachable'
    start_and_fail(r)
    r.d.tick()
    assert r.reports[-1] == 'NO_PLAN "go": replan failed: cannot do that: unreachable'
    assert r.d.phase == d.READY and not r.d.tick()


def test_a_replanner_crash_does_not_wedge_the_queue():
    r = Rig()
    r.d._replan_fn = lambda *a: (_ for _ in ()).throw(RuntimeError('boom'))
    start_and_fail(r)
    r.d.submit('next')
    r.d.tick()
    assert 'replan failed: planner error: boom' in r.reports[-1]
    assert r.d.tick() and r.intents()[-1] == 'next'


def test_without_a_replanner_a_failure_is_final():
    r = Rig(replan=False)
    start_and_fail(r)
    assert not r.d.tick() and r.reports[-1] == f'GAVE_UP "go": {FAIL}'


def test_a_failure_from_a_plan_we_did_not_send_is_ignored():
    r = Rig()
    r.d.on_status(RUN)
    r.d.on_status(FAIL)
    assert not r.d.tick() and r.reports == [] and r.replans == []


def test_a_plan_the_tree_never_started_is_not_replanned_later():
    r = Rig(resend_after_s=1.0, max_sends=1)
    r.d.on_status(IDLE)
    r.d.submit('go')
    r.d.tick()
    r.t = 2.0
    r.d.tick()                                           # gives up
    r.d.on_status(FAIL)                                  # a stale failure arrives afterwards
    assert not r.d.tick() and r.replans == []


def test_a_retried_plan_that_is_never_started_does_not_loop():
    r = Rig(resend_after_s=1.0, max_sends=1)
    start_and_fail(r)
    r.d.tick()
    r.t = 5.0
    assert not r.d.tick() and len(r.replans) == 1


# -------------------------------------------------------------- editing the queue
def filled(*texts, **kwargs):
    r = Rig(**kwargs)
    for text in texts:
        r.d.submit(text)
    r.reports.clear()
    return r


def test_clear_empties_the_queue_and_returns_what_was_removed():
    r = filled('a', 'b', 'c')
    assert r.d.clear() == ['a', 'b', 'c']
    assert len(r.d) == 0 and r.reports == ['CLEARED: 3 removed']


def test_clearing_an_empty_queue_says_so():
    r = filled()
    assert r.d.clear() == [] and r.reports == ['CLEARED: queue was empty']


def test_cleared_intents_are_never_planned():
    r = filled('a', 'b')
    r.d.clear()
    r.d.on_status(IDLE)
    assert not r.d.tick() and r.sent == []


def test_new_intents_work_after_a_clear():
    r = filled('a')
    r.d.clear()
    r.d.on_status(IDLE)
    r.d.submit('b')
    r.d.tick()
    assert r.intents() == ['b']


def test_clear_frees_the_capacity():
    r = filled('a', 'b', max_queued=2)
    r.d.clear()
    assert r.d.submit('c') and r.d.submit('d')


def test_clear_also_drops_a_pending_retry():
    r = Rig()
    start_and_fail(r)                                    # retry is waiting
    r.d.submit('next')
    r.reports.clear()
    r.d.clear()
    assert r.reports == ['CLEARED: 1 removed (and the pending retry)']
    assert not r.d.tick() and r.replans == [] and r.intents() == ['go']


def test_clear_leaves_the_running_plan_alone():
    r = Rig()
    r.d.on_status(IDLE)
    r.d.submit('one')
    r.d.submit('two')
    r.d.tick()
    r.d.on_status(RUN)
    r.d.clear()
    r.d.on_status(FAIL)                                  # the running plan failing still gets its retry
    assert r.d.tick() and r.replans[0][0] == 'one'


def test_remove_takes_out_one_intent_by_position():
    r = filled('a', 'b', 'c')
    assert r.d.remove(2) == 'b'
    assert r.d.pending() == ['a', 'c']
    assert r.reports == ['REMOVED "b" (was #2)']


@pytest.mark.parametrize('position, expected', [(1, ['b', 'c']), (3, ['a', 'b'])])
def test_remove_first_and_last(position, expected):
    r = filled('a', 'b', 'c')
    r.d.remove(position)
    assert r.d.pending() == expected


@pytest.mark.parametrize('position', [0, 4, -1, 99, None, 'x', 1.5])
def test_remove_with_a_bad_position_changes_nothing_and_says_why(position):
    r = filled('a', 'b', 'c')
    assert r.d.remove(position) is None
    assert r.d.pending() == ['a', 'b', 'c']
    assert r.reports == [f'REMOVE_FAILED: no intent #{position} (queue has 3)']


def test_remove_from_an_empty_queue():
    r = filled()
    assert r.d.remove(1) is None and r.reports == ['REMOVE_FAILED: no intent #1 (queue has 0)']


def test_a_removed_intent_is_not_planned_and_the_rest_keep_their_order():
    r = filled('a', 'b', 'c')
    r.d.remove(2)
    r.d.on_status(IDLE)
    for _ in range(2):
        r.d.tick()
        r.d.on_status(RUN)
        r.d.on_status(OK)
    assert r.intents() == ['a', 'c']


def test_show_lists_waiting_intents_in_order():
    r = filled('a', 'b')
    assert r.d.show() == ['a', 'b']
    assert r.reports == ['QUEUE: 1. "a" | 2. "b"']


def test_show_on_an_empty_queue():
    r = filled()
    r.d.show()
    assert r.reports == ['QUEUE: empty']


def test_the_intent_being_planned_or_running_is_not_in_the_list():
    r = filled('a', 'b')
    r.d.on_status(IDLE)
    r.d.tick()
    assert r.d.show() == ['b']


@pytest.mark.parametrize('text, queue_after', [
    ('clear', []), ('CLEAR', []), ('  clear  ', []),
    ('remove 1', ['b', 'c']), ('Remove 3', ['a', 'b']), ('remove   2', ['a', 'c']),
    ('list', ['a', 'b', 'c']),
])
def test_text_commands(text, queue_after):
    r = filled('a', 'b', 'c')
    r.d.command(text)
    assert r.d.pending() == queue_after and r.reports


@pytest.mark.parametrize('text', ['', None, 'remove', 'remove x', 'remove -1', 'remove 1 2', 'clear all',
                                  'delete 1', 'remove 1.5'])
def test_bad_commands_change_nothing_and_report_the_syntax(text):
    r = filled('a', 'b')
    r.d.command(text)
    assert r.d.pending() == ['a', 'b']
    assert len(r.reports) == 1 and r.reports[0].startswith('COMMAND_FAILED')
    assert 'clear | remove <n> | list' in r.reports[0]


def test_remove_zero_by_text_is_reported_as_a_failed_remove():
    r = filled('a')
    r.d.command('remove 0')
    assert r.reports == ['REMOVE_FAILED: no intent #0 (queue has 1)']


def test_queue_editing_is_thread_safe_against_submits():
    r = Rig(max_queued=10000)
    stop = threading.Event()

    def spam():
        while not stop.is_set():
            r.d.submit('x')

    thread = threading.Thread(target=spam)
    thread.start()
    for _ in range(200):
        r.d.clear()
        r.d.remove(1)
        r.d.show()
    stop.set()
    thread.join()
    r.d.clear()
    assert len(r.d) == 0
