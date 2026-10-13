import threading

from behavior_tree.bt_logic import (MAX_ATTEMPTS, RETRY_BACKOFF_S, SKILL_TIMEOUT_S, BehaviorRunner,
                                     PlanQueue, format_status, run_plan)
from behavior_tree.skills import Outcome, Skill
from bt_fakes import FakeCtx, no_sleep


def run(plan, ctx):
    statuses = []
    ok = run_plan(plan, ctx, statuses.append, sleep=no_sleep)
    return ok, statuses


def test_status_formatting():
    assert format_status('IDLE') == 'IDLE'
    assert format_status('RUNNING', 2, 3, 'pick') == 'RUNNING 2/3 pick'
    assert format_status('FAILED', 1, 2, 'navigate_to', 'goal rejected') == (
        'FAILED 1/2 navigate_to: goal rejected')


def test_defaults_are_three_attempts_and_sixty_seconds():
    assert MAX_ATTEMPTS == 3 and SKILL_TIMEOUT_S == 60.0


def test_successful_plan_runs_skills_in_order_then_succeeds():
    ctx = FakeCtx()
    ok, statuses = run([Skill('navigate_to', pose='P'), Skill('pick', 'cube_1')], ctx)
    assert ok
    assert statuses == ['RUNNING 1/2 navigate_to', 'RUNNING 2/2 pick', 'SUCCEEDED']
    assert [c[0] for c in ctx.calls] == ['navigate', 'manipulate']
    assert all(c[-1] == 60.0 for c in ctx.calls)


def test_a_failed_attempt_is_retried_and_can_still_succeed():
    ctx = FakeCtx(Outcome(False, 'timed out after 60 s'), Outcome(True))
    ok, statuses = run([Skill('navigate_to')], ctx)
    assert ok
    assert statuses == ['RUNNING 1/1 navigate_to', 'RUNNING 1/1 navigate_to: attempt 2/3', 'SUCCEEDED']
    assert len(ctx.calls) == 2


def test_three_failed_attempts_fail_the_plan_and_drop_the_rest():
    ctx = FakeCtx(Outcome(False, 'action aborted'))
    ok, statuses = run([Skill('navigate_to'), Skill('pick', 'cube_1')], ctx)
    assert not ok
    assert len(ctx.calls) == 3          # 3 attempts, and the pick never ran
    assert statuses[-1] == 'FAILED 1/2 navigate_to: action aborted (attempt 3/3)'
    assert 'SUCCEEDED' not in statuses


# ---------------------------------------------------------------- pause between attempts
def pauses_of(plan, ctx, **kwargs):
    pauses = []
    ok = run_plan(plan, ctx, lambda s: None, sleep=pauses.append, **kwargs)
    return ok, pauses


def test_there_is_a_pause_before_every_retry_and_none_after_the_last_attempt():
    assert RETRY_BACKOFF_S == 1.0
    ok, pauses = pauses_of([Skill('pick', 'c')], FakeCtx(Outcome(False, 'goal rejected')))
    assert not ok and pauses == [RETRY_BACKOFF_S] * (MAX_ATTEMPTS - 1)


def test_no_pause_when_nothing_fails_or_when_the_failure_is_final():
    assert pauses_of([Skill('navigate_to'), Skill('pick', 'c')], FakeCtx()) == (True, [])
    assert pauses_of([Skill('dance')], FakeCtx()) == (False, [])


def test_the_pause_comes_between_the_failed_attempt_and_the_next_one():
    events = []

    class Ctx(FakeCtx):
        def navigate(self, pose, timeout_s):
            events.append('call')
            return super().navigate(pose, timeout_s)
    ctx = Ctx(Outcome(False, 'timed out after 60 s'), Outcome(True))
    run_plan([Skill('navigate_to')], ctx, lambda s: None, sleep=lambda s: events.append(f'sleep {s:g}'),
             backoff_s=2.5)
    assert events == ['call', 'sleep 2.5', 'call']


def test_a_zero_backoff_never_sleeps():
    assert pauses_of([Skill('pick', 'c')], FakeCtx(Outcome(False, 'x')), backoff_s=0) == (False, [])


def test_the_runner_passes_its_backoff_and_sleep_on():
    pauses = []
    runner = BehaviorRunner(FakeCtx(Outcome(False, 'x')), lambda s: None, backoff_s=0.25, sleep=pauses.append)
    runner.submit([Skill('pick', 'c')])
    runner.step(0)
    assert pauses == [0.25, 0.25]


def test_unknown_skill_fails_immediately_without_retries():
    ctx = FakeCtx()
    ok, statuses = run([Skill('dance')], ctx)
    assert not ok and ctx.calls == []
    assert statuses == ['RUNNING 1/1 dance', 'FAILED 1/1 dance: unknown skill: dance']


def test_failure_on_a_later_skill_reports_its_position():
    ctx = FakeCtx(Outcome(True), Outcome(False, 'manipulation reported failure: x'))
    ok, statuses = run([Skill('navigate_to'), Skill('pick', 'c')], ctx)
    assert not ok
    assert statuses[-1].startswith('FAILED 2/2 pick: manipulation reported failure: x')


def test_plan_queue_is_fifo_and_take_times_out_when_empty():
    q = PlanQueue()
    assert q.take(0.01) is None
    q.put('a'); q.put('b')
    assert len(q) == 2
    assert q.take(0) == 'a' and q.take(0) == 'b'


def test_plan_queue_wakes_a_waiting_taker():
    q = PlanQueue()
    threading.Timer(0.05, q.put, args=('late',)).start()
    assert q.take(2.0) == 'late'


def test_runner_runs_queued_plans_one_after_another_and_keeps_the_last_result():
    published = []
    runner = BehaviorRunner(FakeCtx(), published.append)
    assert runner.status == 'IDLE'
    assert runner.submit([Skill('navigate_to')]) and runner.submit([Skill('pick', 'c')])
    assert runner.step(0) and runner.step(0)
    assert not runner.step(0)                      # queue empty: nothing runs
    assert published == ['RUNNING 1/1 navigate_to', 'SUCCEEDED', 'RUNNING 1/1 pick', 'SUCCEEDED']
    assert runner.status == 'SUCCEEDED'            # held until the next plan


def test_a_failed_plan_does_not_block_the_next_one():
    ctx = FakeCtx(Outcome(False, 'no'), Outcome(False, 'no'), Outcome(False, 'no'), Outcome(True))
    runner = BehaviorRunner(ctx, lambda s: None, sleep=no_sleep)
    runner.submit([Skill('navigate_to')])
    runner.submit([Skill('pick', 'c')])
    runner.step(0)
    assert runner.status.startswith('FAILED')
    runner.step(0)
    assert runner.status == 'SUCCEEDED'


def test_empty_plan_is_ignored():
    runner = BehaviorRunner(FakeCtx(), lambda s: None)
    assert runner.submit([]) is False
    assert not runner.step(0)


def test_unexpected_error_does_not_leave_status_stuck():
    def bad_publish(text):
        if text.startswith('RUNNING'):
            raise RuntimeError('publish failed')
    runner = BehaviorRunner(FakeCtx(), bad_publish)
    runner.submit([Skill('navigate_to')])
    runner.step(0)
    assert runner.status.startswith('FAILED: unexpected error')


# ---------------------------------------------------------------- duplicate plans (planner resends)
def _runner():
    from behavior_tree.bt_logic import BehaviorRunner
    logs = []
    return BehaviorRunner(None, lambda text: None, log=logs.append), logs


def test_a_plan_with_a_seen_id_is_ignored():
    runner, logs = _runner()
    plan = [object()]
    assert runner.submit(plan, 'abc') is True
    assert runner.submit(plan, 'abc') is False
    assert len(runner.queue) == 1 and any('duplicate plan abc' in m for m in logs)


def test_different_ids_are_all_queued():
    runner, _ = _runner()
    assert runner.submit([1], 'a') and runner.submit([1], 'b')
    assert len(runner.queue) == 2


def test_plans_without_an_id_are_never_treated_as_duplicates():
    runner, _ = _runner()
    assert runner.submit([1]) and runner.submit([1])
    assert len(runner.queue) == 2


def test_an_id_stays_remembered_after_its_plan_ran():
    runner, _ = _runner()
    runner.submit([1], 'a')
    runner.queue.take(0)
    assert runner.submit([1], 'a') is False


def test_only_the_most_recent_ids_are_remembered():
    from behavior_tree.bt_logic import SEEN_IDS_KEPT
    runner, _ = _runner()
    for i in range(SEEN_IDS_KEPT + 1):
        runner.submit([1], f'id{i}')
    assert runner.submit([1], 'id0') is True            # forgotten
    assert runner.submit([1], f'id{SEEN_IDS_KEPT}') is False


def test_an_empty_plan_is_still_ignored_and_its_id_not_remembered():
    runner, _ = _runner()
    assert runner.submit([], 'x') is False
    assert runner.submit([1], 'x') is True
