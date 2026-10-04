import threading

from behavior_tree.bt_logic import (MAX_ATTEMPTS, SKILL_TIMEOUT_S, BehaviorRunner, PlanQueue,
                                     format_status, run_plan)
from behavior_tree.skills import Outcome, Skill
from bt_fakes import FakeCtx


def run(plan, ctx):
    statuses = []
    ok = run_plan(plan, ctx, statuses.append)
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
    runner = BehaviorRunner(ctx, lambda s: None)
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
