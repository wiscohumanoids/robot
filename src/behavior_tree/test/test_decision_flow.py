"""Every decision in the behavior tree's flowchart, exercised with a fake ctx (no ROS).

    queue empty?            -> TestQueueAndRunner
    skills left in plan?    -> TestPlanProgress
    which skill name?       -> TestSkillSwitch
    succeeded within 60 s?  -> TestAttempts (the timeout itself is in test_action_call.py)
    attempt < 3?            -> TestAttempts
    status text             -> TestStatusText, TestStatusStream
"""
import itertools
import threading
import time

import pytest

from behavior_tree.bt_logic import (MAX_ATTEMPTS, SKILL_TIMEOUT_S, BehaviorRunner, PlanQueue,
                                     format_status, run_plan)
from behavior_tree.skills import SKILLS, Outcome, Skill, manipulate, navigate_to, run_skill
from bt_fakes import FakeCtx, no_sleep

NAMES = ['navigate_to', 'pick', 'place', 'handover']
PATTERNS = [''.join(p) for p in itertools.product('SF', repeat=3)]   # S = success, F = failure


def script(pattern, reason='boom'):
    return [Outcome(True) if c == 'S' else Outcome(False, reason) for c in pattern]


def run(plan, ctx, **kwargs):
    statuses = []
    kwargs.setdefault('sleep', no_sleep)
    ok = run_plan(plan, ctx, statuses.append, **kwargs)
    return ok, statuses


def wait_until(condition, timeout_s=3.0):
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


def called_names(ctx):
    """Skill name behind each ctx call: 'navigate_to' for nav, the task for manipulation."""
    return ['navigate_to' if c[0] == 'navigate' else c[1] for c in ctx.calls]


# ---------------------------------------------------------------- constants
def test_defaults_are_three_attempts_and_sixty_seconds():
    assert MAX_ATTEMPTS == 3
    assert SKILL_TIMEOUT_S == 60.0


# ---------------------------------------------------------------- status text
class TestStatusText:
    def test_state_only(self):
        assert format_status('IDLE') == 'IDLE'
        assert format_status('SUCCEEDED') == 'SUCCEEDED'

    def test_index_and_total_are_shown_together_only(self):
        assert format_status('RUNNING', 2, 3) == 'RUNNING 2/3'
        assert format_status('RUNNING', 2) == 'RUNNING'
        assert format_status('RUNNING', total=3) == 'RUNNING'

    def test_zero_index_is_still_shown(self):
        assert format_status('RUNNING', 0, 3) == 'RUNNING 0/3'

    def test_skill_and_detail(self):
        assert format_status('RUNNING', 2, 3, 'pick') == 'RUNNING 2/3 pick'
        assert format_status('RUNNING', skill='pick') == 'RUNNING pick'
        assert format_status('FAILED', detail='x') == 'FAILED: x'
        assert format_status('FAILED', 1, 2, 'navigate_to', 'goal rejected') == (
            'FAILED 1/2 navigate_to: goal rejected')

    def test_empty_skill_and_detail_are_omitted(self):
        assert format_status('RUNNING', 1, 2, '', '') == 'RUNNING 1/2'


# ---------------------------------------------------------------- the skill switch
class TestSkillSwitch:
    def test_table_is_exactly_the_four_skills(self):
        assert set(SKILLS) == set(NAMES)

    def test_table_points_at_the_right_functions(self):
        assert SKILLS['navigate_to'] is navigate_to
        for name in ('pick', 'place', 'handover'):
            assert SKILLS[name] is manipulate

    def test_navigate_to_uses_only_the_pose(self):
        ctx = FakeCtx()
        run_skill(ctx, Skill('navigate_to', object_id='ignored', pose='P'), 60.0)
        assert ctx.calls == [('navigate', 'P', 60.0)]

    @pytest.mark.parametrize('name', ['pick', 'place', 'handover'])
    def test_manipulation_skills_pass_name_object_pose_and_timeout(self, name):
        ctx = FakeCtx()
        run_skill(ctx, Skill(name, 'cube_1', 'P'), 12.5)
        assert ctx.calls == [('manipulate', name, 'cube_1', 'P', 12.5)]

    def test_skill_defaults(self):
        s = Skill('pick')
        assert s.object_id == '' and s.pose is None

    def test_outcome_defaults(self):
        o = Outcome(True)
        assert o.ok and o.reason == '' and o.retryable is True

    def test_outcome_is_passed_through_untouched(self):
        for outcome in (Outcome(True), Outcome(False, 'why'), Outcome(False, 'why', retryable=False)):
            assert run_skill(FakeCtx(outcome), Skill('navigate_to'), 60.0) is outcome
            assert run_skill(FakeCtx(outcome), Skill('pick'), 60.0) is outcome

    @pytest.mark.parametrize('name', ['dance', 'Pick', 'PICK', ' pick', 'pick ', 'navigate',
                                      'Navigate_To', 'navigate_to_pose', ''])
    def test_unknown_names_fail_without_touching_the_robot(self, name):
        ctx = FakeCtx()
        out = run_skill(ctx, Skill(name), 60.0)
        assert not out.ok and not out.retryable
        assert out.reason == f'unknown skill: {name}'
        assert ctx.calls == []

    @pytest.mark.parametrize('name', NAMES)
    def test_a_crashing_context_becomes_a_retryable_failed_outcome(self, name):
        class Boom:
            def navigate(self, pose, timeout_s):
                raise RuntimeError('kaput')

            def manipulate(self, task, object_id, pose, timeout_s):
                raise ValueError('kaput')
        out = run_skill(Boom(), Skill(name), 60.0)
        assert not out.ok and out.retryable and out.reason.endswith('kaput')
        assert out.reason.startswith('error: ')


# ---------------------------------------------------------------- attempts (succeeded? / attempt < 3?)
class TestAttempts:
    @pytest.mark.parametrize('pattern', PATTERNS)
    def test_every_success_failure_pattern_over_three_attempts(self, pattern):
        ctx = FakeCtx(*script(pattern))
        ok, statuses = run([Skill('navigate_to', pose='P')], ctx)
        first_success = pattern.find('S')
        base = 'RUNNING 1/1 navigate_to'
        if first_success >= 0:
            attempts = first_success + 1
            assert ok
            assert statuses == ([base] + [f'{base}: attempt {a}/3' for a in range(2, attempts + 1)]
                                + ['SUCCEEDED'])
        else:
            attempts = 3
            assert not ok
            assert statuses == [base, f'{base}: attempt 2/3', f'{base}: attempt 3/3',
                                'FAILED 1/1 navigate_to: boom (attempt 3/3)']
        assert len(ctx.calls) == attempts           # stops at the first success, never exceeds 3

    @pytest.mark.parametrize('name', NAMES)
    @pytest.mark.parametrize('pattern', PATTERNS)
    def test_every_skill_gets_the_same_retry_behaviour(self, name, pattern):
        ctx = FakeCtx(*script(pattern))
        ok, _ = run([Skill(name, 'obj', 'P')], ctx)
        assert ok == ('S' in pattern)
        assert set(called_names(ctx)) == {name}                      # every attempt hit the same skill
        assert {c[0] for c in ctx.calls} == {'navigate' if name == 'navigate_to' else 'manipulate'}

    def test_each_attempt_gets_the_full_timeout(self):
        ctx = FakeCtx(*script('FFS'))
        run([Skill('navigate_to')], ctx)
        assert [c[-1] for c in ctx.calls] == [60.0, 60.0, 60.0]

    def test_attempt_counter_restarts_for_every_skill(self):
        ctx = FakeCtx(*script('FS') + script('FFS'))
        ok, statuses = run([Skill('navigate_to'), Skill('pick', 'c')], ctx)
        assert ok and len(ctx.calls) == 5
        assert statuses == [
            'RUNNING 1/2 navigate_to', 'RUNNING 1/2 navigate_to: attempt 2/3',
            'RUNNING 2/2 pick', 'RUNNING 2/2 pick: attempt 2/3', 'RUNNING 2/2 pick: attempt 3/3',
            'SUCCEEDED']

    def test_the_failure_reason_is_the_last_attempts_reason(self):
        ctx = FakeCtx(Outcome(False, 'first'), Outcome(False, 'second'), Outcome(False, 'third'))
        _, statuses = run([Skill('navigate_to')], ctx)
        assert statuses[-1] == 'FAILED 1/1 navigate_to: third (attempt 3/3)'

    def test_a_crash_is_retried_and_can_still_succeed(self):
        class FlakyCtx(FakeCtx):
            def navigate(self, pose, timeout_s):
                self.calls.append(('navigate', pose, timeout_s))
                if len(self.calls) == 1:
                    raise RuntimeError('first attempt crashed')
                return Outcome(True)
        ctx = FlakyCtx()
        ok, statuses = run([Skill('navigate_to')], ctx)
        assert ok and len(ctx.calls) == 2 and statuses[-1] == 'SUCCEEDED'

    def test_a_context_that_always_crashes_fails_after_three_attempts(self):
        class Boom:
            calls = 0

            def navigate(self, pose, timeout_s):
                Boom.calls += 1
                raise RuntimeError('kaput')
        ok, statuses = run([Skill('navigate_to')], Boom())
        assert not ok and Boom.calls == 3
        assert statuses[-1] == 'FAILED 1/1 navigate_to: error: kaput (attempt 3/3)'

    # --- configurable
    @pytest.mark.parametrize('limit', [1, 2, 3, 5])
    def test_max_attempts_is_respected(self, limit):
        ctx = FakeCtx(Outcome(False, 'boom'))
        ok, statuses = run([Skill('navigate_to')], ctx, max_attempts=limit)
        assert not ok and len(ctx.calls) == limit
        assert statuses[-1] == f'FAILED 1/1 navigate_to: boom (attempt {limit}/{limit})'

    def test_timeout_value_is_passed_to_every_call(self):
        ctx = FakeCtx(Outcome(False, 'boom'))
        run([Skill('navigate_to')], ctx, timeout_s=12.5)
        assert {c[-1] for c in ctx.calls} == {12.5}

    # --- custom run function: the retryable flag and empty reasons
    def test_non_retryable_failure_stops_after_one_call_and_has_no_attempt_suffix(self):
        calls = []

        def fn(ctx, skill, timeout_s):
            calls.append(skill.name)
            return Outcome(False, 'nope', retryable=False)
        ok, statuses = run([Skill('x')], None, run=fn)
        assert not ok and calls == ['x']
        assert statuses[-1] == 'FAILED 1/1 x: nope'

    def test_empty_reasons_become_the_word_failed(self):
        retry = lambda ctx, s, t: Outcome(False, '')
        once = lambda ctx, s, t: Outcome(False, '', retryable=False)
        assert run([Skill('x')], None, run=retry)[1][-1] == 'FAILED 1/1 x: failed (attempt 3/3)'
        assert run([Skill('x')], None, run=once)[1][-1] == 'FAILED 1/1 x: failed'

    def test_a_non_retryable_failure_is_not_retried_even_with_attempts_left(self):
        ctx = FakeCtx()
        ok, statuses = run([Skill('dance'), Skill('pick', 'c')], ctx, max_attempts=10)
        assert not ok and ctx.calls == []
        assert len([s for s in statuses if s.startswith('RUNNING')]) == 1


# ---------------------------------------------------------------- skills left in the plan?
class TestPlanProgress:
    def test_empty_plan_called_directly_just_succeeds(self):
        ok, statuses = run([], FakeCtx())          # BehaviorRunner never passes one in
        assert ok and statuses == ['SUCCEEDED']

    @pytest.mark.parametrize('size', [1, 2, 3, 4])
    def test_a_good_plan_runs_every_skill_in_order_then_succeeds(self, size):
        names = [NAMES[i % len(NAMES)] for i in range(size)]
        ctx = FakeCtx()
        ok, statuses = run([Skill(n, 'o', 'P') for n in names], ctx)
        assert ok
        assert called_names(ctx) == names
        assert statuses == [f'RUNNING {i}/{size} {n}' for i, n in enumerate(names, 1)] + ['SUCCEEDED']

    @pytest.mark.parametrize('size, fail_at', [(s, f) for s in (1, 2, 3) for f in range(1, s + 1)])
    def test_failure_at_each_position_drops_the_rest_of_the_plan(self, size, fail_at):
        names = [NAMES[i % len(NAMES)] for i in range(size)]
        ctx = FakeCtx(*([Outcome(True)] * (fail_at - 1) + [Outcome(False, 'boom')] * 3))
        ok, statuses = run([Skill(n) for n in names], ctx)
        assert not ok
        assert called_names(ctx) == names[:fail_at - 1] + [names[fail_at - 1]] * 3   # nothing after it ran
        assert statuses[-1] == f'FAILED {fail_at}/{size} {names[fail_at - 1]}: boom (attempt 3/3)'
        assert 'SUCCEEDED' not in statuses

    def test_unknown_skill_at_each_position(self):
        for pos in range(3):
            plan = [Skill('navigate_to'), Skill('pick', 'c'), Skill('place', 'c')]
            plan[pos] = Skill('dance')
            ctx = FakeCtx()
            ok, statuses = run(plan, ctx)
            assert not ok
            assert len(ctx.calls) == pos                   # earlier skills ran, dance and later did not
            assert statuses[-1] == f'FAILED {pos + 1}/3 dance: unknown skill: dance'
            assert statuses.count('RUNNING %d/3 dance' % (pos + 1)) == 1   # never retried


# ---------------------------------------------------------------- status stream invariants
SCENARIOS = {
    'one skill ok': ([Skill('navigate_to')], 'S'),
    'one skill fails': ([Skill('navigate_to')], 'FFF'),
    'retry then ok': ([Skill('navigate_to'), Skill('pick', 'c')], 'FSS'),
    'second fails': ([Skill('navigate_to'), Skill('pick', 'c')], 'SFFF'),
    'unknown': ([Skill('navigate_to'), Skill('dance')], 'S'),
}


class TestStatusStream:
    @pytest.mark.parametrize('scenario', list(SCENARIOS))
    def test_exactly_one_terminal_status_and_it_is_last(self, scenario):
        plan, pattern = SCENARIOS[scenario]
        _, statuses = run(plan, FakeCtx(*script(pattern)))
        terminals = [s for s in statuses if s.startswith(('SUCCEEDED', 'FAILED'))]
        assert len(terminals) == 1 and statuses[-1] == terminals[0]

    @pytest.mark.parametrize('scenario', list(SCENARIOS))
    def test_running_indexes_never_go_backwards_and_stay_in_range(self, scenario):
        plan, pattern = SCENARIOS[scenario]
        _, statuses = run(plan, FakeCtx(*script(pattern)))
        indexes = [int(s.split()[1].split('/')[0]) for s in statuses if s.startswith('RUNNING')]
        assert indexes == sorted(indexes)
        assert all(1 <= i <= len(plan) for i in indexes)

    @pytest.mark.parametrize('scenario', list(SCENARIOS))
    def test_every_status_is_one_line_and_starts_with_a_known_state(self, scenario):
        plan, pattern = SCENARIOS[scenario]
        _, statuses = run(plan, FakeCtx(*script(pattern)))
        for s in statuses:
            assert '\n' not in s
            assert s.split()[0].rstrip(':') in ('RUNNING', 'SUCCEEDED', 'FAILED')   # smoke_task.py relies on this


# ---------------------------------------------------------------- queue empty?
class TestQueueAndRunner:
    # --- PlanQueue
    def test_queue_is_fifo(self):
        q = PlanQueue()
        for i in range(100):
            q.put(i)
        assert len(q) == 100
        assert [q.take(0) for _ in range(100)] == list(range(100))
        assert len(q) == 0

    def test_take_on_an_empty_queue_returns_none_after_the_timeout(self):
        q = PlanQueue()
        assert q.take(0) is None
        start = time.monotonic()
        assert q.take(0.1) is None
        assert time.monotonic() - start >= 0.08

    def test_take_returns_immediately_when_something_is_waiting(self):
        q = PlanQueue()
        q.put('a')
        start = time.monotonic()
        assert q.take(5.0) == 'a'
        assert time.monotonic() - start < 1.0

    def test_take_wakes_up_when_a_plan_arrives(self):
        q = PlanQueue()
        threading.Timer(0.05, q.put, args=('late',)).start()
        assert q.take(3.0) == 'late'

    def test_many_producers_lose_nothing_and_keep_their_own_order(self):
        q = PlanQueue()

        def produce(pid):
            for i in range(50):
                q.put((pid, i))
        threads = [threading.Thread(target=produce, args=(p,)) for p in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        got = [q.take(0) for _ in range(200)]
        assert q.take(0) is None
        assert sorted(got) == sorted((p, i) for p in range(4) for i in range(50))
        for p in range(4):
            assert [i for pid, i in got if pid == p] == list(range(50))

    def test_waiting_takers_each_get_a_different_plan(self):
        q = PlanQueue()
        got = []
        lock = threading.Lock()

        def taker():
            item = q.take(3.0)
            with lock:
                got.append(item)
        threads = [threading.Thread(target=taker) for _ in range(3)]
        for t in threads:
            t.start()
        time.sleep(0.05)
        for i in range(3):
            q.put(i)
        for t in threads:
            t.join(3.0)
        assert sorted(got) == [0, 1, 2]

    # --- BehaviorRunner, driven step by step
    def test_starts_idle_and_an_empty_queue_runs_nothing(self):
        published = []
        runner = BehaviorRunner(FakeCtx(), published.append)
        assert runner.status == 'IDLE'
        assert runner.step(0) is False
        assert published == [] and runner.status == 'IDLE'

    def test_empty_plan_is_ignored_and_not_queued(self):
        logs = []
        runner = BehaviorRunner(FakeCtx(), lambda s: None, log=logs.append)
        assert runner.submit([]) is False
        assert len(runner.queue) == 0 and runner.step(0) is False
        assert any('empty plan' in m for m in logs)

    def test_a_good_plan_is_accepted(self):
        runner = BehaviorRunner(FakeCtx(), lambda s: None)
        assert runner.submit([Skill('navigate_to')]) is True
        assert len(runner.queue) == 1

    def test_plans_run_one_after_another_in_arrival_order(self):
        ctx = FakeCtx()
        runner = BehaviorRunner(ctx, lambda s: None)
        for label in ('A', 'B', 'C'):
            runner.submit([Skill('navigate_to', pose=label)])
        while runner.step(0):
            pass
        assert [c[1] for c in ctx.calls] == ['A', 'B', 'C']

    def test_each_step_runs_exactly_one_plan(self):
        ctx = FakeCtx()
        runner = BehaviorRunner(ctx, lambda s: None)
        runner.submit([Skill('navigate_to', pose='A')])
        runner.submit([Skill('navigate_to', pose='B')])
        assert runner.step(0) is True
        assert len(ctx.calls) == 1 and len(runner.queue) == 1

    def test_the_result_stays_published_until_the_next_plan(self):
        published = []
        runner = BehaviorRunner(FakeCtx(), published.append)
        runner.submit([Skill('navigate_to')])
        runner.step(0)
        assert runner.status == 'SUCCEEDED'
        runner.step(0)                               # nothing queued
        assert runner.status == 'SUCCEEDED' and published[-1] == 'SUCCEEDED'

    def test_a_failed_plan_does_not_block_the_next_plan(self):
        ctx = FakeCtx(*script('FFF') + [Outcome(True)])
        runner = BehaviorRunner(ctx, lambda s: None, sleep=no_sleep)
        runner.submit([Skill('navigate_to')])
        runner.submit([Skill('pick', 'c')])
        runner.step(0)
        assert runner.status.startswith('FAILED 1/1 navigate_to')
        runner.step(0)
        assert runner.status == 'SUCCEEDED'

    def test_failure_drops_only_that_plans_remaining_skills_not_the_queue(self):
        ctx = FakeCtx(*script('FFF') + [Outcome(True)])
        runner = BehaviorRunner(ctx, lambda s: None, sleep=no_sleep)
        runner.submit([Skill('navigate_to'), Skill('pick', 'c')])      # plan 1: fails at skill 1
        runner.submit([Skill('place', 'c')])                           # plan 2
        runner.step(0)
        assert len(runner.queue) == 1
        runner.step(0)
        assert called_names(ctx) == ['navigate_to'] * 3 + ['place']

    def test_every_status_goes_through_publish_and_log(self):
        published, logs = [], []
        runner = BehaviorRunner(FakeCtx(), published.append, log=logs.append)
        runner.submit([Skill('navigate_to')])
        runner.step(0)
        assert published == ['RUNNING 1/1 navigate_to', 'SUCCEEDED']
        assert logs == ['task_status: RUNNING 1/1 navigate_to', 'task_status: SUCCEEDED']

    def test_runner_settings_reach_the_skills(self):
        ctx = FakeCtx(Outcome(False, 'boom'))
        runner = BehaviorRunner(ctx, lambda s: None, max_attempts=2, timeout_s=7.0, sleep=no_sleep)
        runner.submit([Skill('navigate_to')])
        runner.step(0)
        assert len(ctx.calls) == 2 and {c[-1] for c in ctx.calls} == {7.0}
        assert runner.status == 'FAILED 1/1 navigate_to: boom (attempt 2/2)'

    def test_unexpected_error_while_running_ends_in_failed_not_stuck_running(self):
        def bad_publish(text):
            if text.startswith('RUNNING'):
                raise RuntimeError('publish failed')
        runner = BehaviorRunner(FakeCtx(), bad_publish)
        runner.submit([Skill('navigate_to')])
        assert runner.step(0) is True
        assert runner.status == 'FAILED: unexpected error: publish failed'

    # --- BehaviorRunner on its own thread, like the node uses it
    def test_run_forever_works_through_the_queue_and_stops_on_request(self):
        published = []
        runner = BehaviorRunner(FakeCtx(), published.append)
        stop = threading.Event()
        worker = threading.Thread(target=runner.run_forever, args=(stop,), daemon=True)
        worker.start()
        runner.submit([Skill('navigate_to')])
        runner.submit([Skill('pick', 'c')])
        assert wait_until(lambda: published.count('SUCCEEDED') == 2)
        stop.set()
        worker.join(2.0)
        assert not worker.is_alive()

    def test_run_forever_does_not_start_when_already_stopped(self):
        runner = BehaviorRunner(FakeCtx(), lambda s: None)
        stop = threading.Event()
        stop.set()
        worker = threading.Thread(target=runner.run_forever, args=(stop,), daemon=True)
        worker.start()
        worker.join(1.0)
        assert not worker.is_alive()

    def test_a_plan_that_arrives_while_busy_waits_its_turn(self):
        release = threading.Event()
        started = threading.Event()

        class BlockingCtx(FakeCtx):
            def navigate(self, pose, timeout_s):
                self.calls.append(('navigate', pose, timeout_s))
                if pose == 'A':
                    started.set()
                    release.wait(3.0)
                return Outcome(True)
        ctx = BlockingCtx()
        published = []
        runner = BehaviorRunner(ctx, published.append)
        stop = threading.Event()
        worker = threading.Thread(target=runner.run_forever, args=(stop,), daemon=True)
        worker.start()
        runner.submit([Skill('navigate_to', pose='A')])
        assert started.wait(3.0)
        runner.submit([Skill('navigate_to', pose='B')])          # arrives while A is running
        assert len(runner.queue) == 1                            # queued, not dropped
        assert [c[1] for c in ctx.calls] == ['A']                # and not started yet
        release.set()
        assert wait_until(lambda: published.count('SUCCEEDED') == 2)
        assert [c[1] for c in ctx.calls] == ['A', 'B']
        stop.set()
        worker.join(2.0)

    def test_the_loop_survives_a_publisher_that_always_fails(self):
        logs = []

        def broken(text):
            raise RuntimeError('publisher is gone')
        runner = BehaviorRunner(FakeCtx(), broken, log=logs.append)
        stop = threading.Event()
        worker = threading.Thread(target=runner.run_forever, args=(stop,), daemon=True)
        worker.start()
        runner.submit([Skill('navigate_to')])
        runner.submit([Skill('pick', 'c')])
        assert wait_until(lambda: len(runner.queue) == 0)
        assert wait_until(lambda: sum('behavior loop error' in m for m in logs) >= 2)
        assert worker.is_alive()
        stop.set()
        worker.join(2.0)
        assert not worker.is_alive()
