"""The harness's own decisions (Timeline, status readers, plan validation, FakeTree)."""
import pytest

from integration_tests import logic
from integration_tests.logic import FakeTree, Timeline


def tl_with(*pairs):
    tl = Timeline()
    for i, (kind, text) in enumerate(pairs):
        if kind == 'task':
            tl.add_task(float(i), text)
        else:
            tl.add(float(i), kind, text)
    return tl


# ------------------------------------------------------------ Timeline
def test_task_status_is_stored_on_change_but_every_message_time_is_kept():
    tl = Timeline()
    for t, text in [(0, 'IDLE'), (1, 'IDLE'), (2, 'RUNNING 1/1 navigate_to'), (3, 'RUNNING 1/1 navigate_to')]:
        tl.add_task(t, text)
    assert tl.texts('task') == ['IDLE', 'RUNNING 1/1 navigate_to']
    assert tl.task_times == [0, 1, 2, 3]


def test_held_is_stored_on_change_only():
    tl = Timeline()
    for t, text in [(0, ''), (1, ''), (2, 'cube_1'), (3, 'cube_1'), (4, '')]:
        tl.add_held(t, text)
    assert tl.texts('held') == ['', 'cube_1', '']


def test_find_and_count_respect_the_start_index():
    tl = tl_with(('planner', 'SENT "a": 1 steps'), ('sent', 'x'), ('planner', 'SENT "b": 1 steps'))
    assert tl.find('planner', 'SENT') == 0
    assert tl.find('planner', 'SENT', 1) == 2
    assert tl.find('planner', 'NO_PLAN') is None
    assert tl.count('planner', 'SENT') == 2 and tl.count('planner', 'SENT', 1) == 1


def test_last_returns_the_most_recent_of_a_kind():
    tl = tl_with(('planner', 'a'), ('task', 'IDLE'), ('planner', 'b'))
    assert tl.last('planner').text == 'b' and tl.last('held') is None


def test_plans_are_distinct_by_plan_id_so_a_resend_counts_once():
    tl = Timeline()
    for pid in ('p1', 'p1', 'p2', 'p1'):
        tl.add(0.0, 'plan', pid, {'plan_id': pid, 'intent': 'x', 'skills': []})
    assert [p['plan_id'] for p in tl.plans()] == ['p1', 'p2']


# ------------------------------------------------------------ status readers
STATUSES = ['IDLE', 'RUNNING 1/2 navigate_to', 'RUNNING 2/2 pick', 'SUCCEEDED',
            'RUNNING 1/1 navigate_to', 'RUNNING 1/1 navigate_to: attempt 2/3',
            'FAILED 1/1 navigate_to: action aborted (attempt 3/3)']


def status_tl(statuses=STATUSES):
    tl = Timeline()
    for i, s in enumerate(statuses):
        tl.add_task(float(i), s)
    return tl


def test_run_starts_counts_plans_not_steps_or_attempts():
    assert logic.run_starts(status_tl()) == 2


def test_run_starts_when_two_plans_run_back_to_back_without_an_idle_gap():
    assert logic.run_starts(status_tl(['RUNNING 1/1 a', 'SUCCEEDED', 'RUNNING 1/1 a', 'SUCCEEDED'])) == 2


def test_finished_lists_the_result_of_each_plan_that_ran():
    assert logic.finished(status_tl()) == ['SUCCEEDED', 'FAILED 1/1 navigate_to: action aborted (attempt 3/3)']


def test_a_leftover_result_before_any_running_is_not_a_finished_plan():
    assert logic.finished(status_tl(['SUCCEEDED', 'RUNNING 1/1 a', 'SUCCEEDED'])) == ['SUCCEEDED']


def test_nothing_ran():
    assert logic.run_starts(status_tl(['IDLE'])) == 0 and logic.finished(status_tl(['IDLE'])) == []


def test_heartbeat_gap_detection():
    tl = Timeline()
    tl.task_times = [0, 1, 2, 5.5]
    assert 'went quiet for 3.5 s' in logic.heartbeat_problems(tl)[0]
    tl.task_times = [0, 1, 2, 3]
    assert logic.heartbeat_problems(tl) == []
    assert logic.heartbeat_problems(Timeline()) == []


def test_near():
    assert logic.near((1.0, 1.0, 0.0), (1.3, 1.0), 0.5)
    assert not logic.near((1.0, 1.0, 0.0), (3.0, 1.0), 0.5)
    assert not logic.near(None, (0, 0), 1.0)


# ------------------------------------------------------------ planner status lines
@pytest.mark.parametrize('text, expected', [
    ('QUEUED "pick up the cube": 1 waiting', ('QUEUED', 'pick up the cube', '1 waiting')),
    ('PLANNING "go"', ('PLANNING', 'go', '')),
    ('SENT "go": 2 steps', ('SENT', 'go', '2 steps')),
    ('NO_PLAN "x": cannot do that: no "such" thing', ('NO_PLAN', 'x', 'cannot do that: no "such" thing')),
    ('REJECTED "x": queue full (10)', ('REJECTED', 'x', 'queue full (10)')),
    ('REMOVED "b" (was #2)', ('REMOVED', 'b', '(was #2)')),
    ('RETRYING "x": FAILED 1/2 pick: boom (attempt 3/3)', ('RETRYING', 'x', 'FAILED 1/2 pick: boom (attempt 3/3)')),
    ('DONE "x"', ('DONE', 'x', '')),
    ('CLEARED: 3 removed', ('CLEARED', '', '3 removed')),
    ('QUEUE: empty', ('QUEUE', '', 'empty')),
    ('REMOVE_FAILED: no intent #5 (queue has 3)', ('REMOVE_FAILED', '', 'no intent #5 (queue has 3)')),
    ('COMMAND_FAILED "bad": expected clear | remove <n> | list', ('COMMAND_FAILED', 'bad', 'expected clear | remove <n> | list')),
    ('hello', ('?', '', 'hello')),
    ('SENT "pick up\nand then\ndance": 2 steps', ('SENT', 'pick up\nand then\ndance', '2 steps')),
])
def test_classify(text, expected):
    assert logic.classify(text) == expected


@pytest.mark.parametrize('text, expected', [
    ('QUEUE: empty', []),
    ('QUEUE: 1. "a"', ['a']),
    ('QUEUE: 1. "pick up the cup" | 2. "put the cube down" | 3. "go to x=1 y=1"',
     ['pick up the cup', 'put the cube down', 'go to x=1 y=1']),
])
def test_parse_queue_line(text, expected):
    assert logic.parse_queue_line(text) == expected


def test_kinds_and_counts():
    tl = tl_with(('planner', 'QUEUED "a": 1 waiting'), ('planner', 'PLANNING "a"'), ('planner', 'SENT "a": 2 steps'),
                 ('task', 'RUNNING 1/2 a'), ('planner', 'DONE "a"'))
    assert logic.kinds(tl) == ['QUEUED', 'PLANNING', 'SENT', 'DONE']
    assert logic.planner_count(tl, 'SENT') == 1 and logic.planner_count(tl, 'SENT', 3) == 0


def test_latencies_pair_queued_with_the_next_sent_in_order():
    tl = Timeline()
    for t, text in [(0.0, 'QUEUED "a": 1 waiting'), (0.5, 'QUEUED "b": 2 waiting'), (1.0, 'SENT "a": 1 steps'),
                    (4.0, 'SENT "b": 1 steps')]:
        tl.add(t, 'planner', text)
    assert logic.latencies(tl) == [1.0, 3.5]


def test_percentile_and_description():
    assert logic.percentile([], 95) == 0.0
    assert logic.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) == 5
    assert logic.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10
    assert logic.describe_latencies([]) == 'no samples'
    assert 'p95=10.00s' in logic.describe_latencies(list(range(1, 11)))


# ------------------------------------------------------------ plan validation
def step(name, object_id='', x=None, y=None, z=0.0):
    out = {'name': name, 'object_id': object_id, 'z': z}
    if x is not None:
        out['x'], out['y'] = x, y
    return out


def test_world_objects_include_the_cube_and_the_scene():
    objs = logic.world_objects()
    assert set(objs) == {'cube_1', 'ball_1', 'cup_1', 'box_1', 'bottle_1'}
    assert objs['cube_1']['x'] == 1.5


def test_valid_plans_have_no_problems():
    assert logic.validate_plan([step('navigate_to', x=2.0, y=-3.0), step('pick', 'cup_1', 3.5, -2.5, 0.8)]) == []


@pytest.mark.parametrize('plan, fragment', [
    ([], 'empty plan'),
    ([step('dance')], 'unknown skill'),
    ([step('pick', 'ghost_9')], 'unknown object'),
    ([step('pick')], 'needs an object_id'),
    ([step('navigate_to', x=50.0, y=0.0)], 'outside the map'),
    ([step('navigate_to', x=4.9, y=0.0)], 'outside the map'),
    ([step('navigate_to')], 'needs a target pose'),
    ([step('place', 'cube_1')], 'not holding'),
    ([step('pick', 'cube_1'), step('pick', 'cup_1')], 'already holding'),
])
def test_invalid_plans_are_reported(plan, fragment):
    problems = logic.validate_plan(plan)
    assert problems and any(fragment in p for p in problems)


# ------------------------------------------------------------ FakeTree
PLAN = [{'name': 'pick', 'object_id': 'cube_1', 'x': 1.5, 'y': 0.3, 'z': 0.8}]


def test_a_fresh_fake_tree_is_idle():
    assert FakeTree().status(0.0) == 'IDLE'


def test_normal_mode_runs_a_plan_then_holds_its_result():
    t = FakeTree(run_s=1.0)
    t.on_plan('p1', PLAN, 0.0)
    assert t.status(0.1) == 'RUNNING 1/1 pick'      # starts at 0.1, runs for 1.0 s
    assert t.status(1.0) == 'RUNNING 1/1 pick'
    assert t.status(1.2) == 'SUCCEEDED'
    assert t.status(50.0) == 'SUCCEEDED'            # held until the next plan, like the real tree
    assert t.plans_run == ['p1']


def test_progress_through_a_multi_step_plan():
    t = FakeTree(run_s=2.0)
    two = [{'name': 'navigate_to', 'object_id': '', 'x': 1.0, 'y': 1.0, 'z': 0.0}] + PLAN
    t.on_plan('p', two, 0.0)
    assert t.status(0.0) == 'RUNNING 1/2 navigate_to'
    assert t.status(1.1) == 'RUNNING 2/2 pick'


def test_plans_run_one_at_a_time_in_order():
    t = FakeTree(run_s=1.0)
    t.on_plan('a', PLAN, 0.0)
    t.on_plan('b', PLAN, 0.0)
    assert t.overlaps == ['b']                        # b arrived while a was queued/running: a planner bug
    assert t.status(0.1) == 'RUNNING 1/1 pick'
    assert t.status(1.2) == 'SUCCEEDED'
    assert t.status(1.3) == 'RUNNING 1/1 pick'        # b starts only after a finished
    assert t.status(2.5) == 'SUCCEEDED'
    assert t.plans_run == ['a', 'b']


def test_a_plan_sent_to_an_idle_tree_is_not_an_overlap():
    t = FakeTree(run_s=0.5)
    t.on_plan('a', PLAN, 0.0)
    t.status(0.1)
    t.status(0.6)
    t.on_plan('b', PLAN, 1.0)
    assert t.overlaps == []


def test_duplicate_plan_ids_and_empty_plans_are_ignored():
    t = FakeTree()
    assert t.on_plan('a', PLAN, 0.0) is True
    assert t.on_plan('a', PLAN, 0.1) is False and t.duplicates == 1
    assert t.on_plan('b', [], 0.1) is False
    assert t.on_plan('', PLAN, 0.2) is True and t.on_plan('', PLAN, 0.3) is True    # no id: never a duplicate


def test_always_fail_mode_fails_every_plan_with_the_trees_wording():
    t = FakeTree(mode='always_fail', run_s=0.2)
    t.on_plan('a', PLAN, 0.0)
    t.status(0.0)
    assert t.status(0.3) == 'FAILED 1/1 pick: injected failure (attempt 3/3)'


def test_fail_rate_is_deterministic_for_a_seed():
    def outcomes(seed):
        t = FakeTree(run_s=0.1, fail_rate=0.5, seed=seed)
        out, now = [], 0.0
        for i in range(20):
            t.on_plan(f'p{i}', PLAN, now)
            t.status(now)
            now += 0.2
            out.append(t.status(now))
        return out
    assert outcomes(3) == outcomes(3)
    assert any(o == 'SUCCEEDED' for o in outcomes(3)) and any(o.startswith('FAILED') for o in outcomes(3))


def test_hang_mode_never_finishes():
    t = FakeTree(mode='hang')
    t.on_plan('a', PLAN, 0.0)
    assert t.status(0.0) == 'RUNNING 1/1 pick' and t.status(9999.0) == 'RUNNING 1/1 pick'
    assert t.busy


def test_never_start_mode_keeps_the_initial_status_and_is_never_busy():
    t = FakeTree(mode='never_start', initial='SUCCEEDED')
    assert t.on_plan('a', PLAN, 0.0) is True
    assert t.status(100.0) == 'SUCCEEDED' and not t.busy and t.overlaps == []


def test_reset_clears_everything():
    t = FakeTree(run_s=0.1)
    t.on_plan('a', PLAN, 0.0)
    t.status(0.0)
    t.status(0.2)
    t.reset('hang', 1.0, 0.0, 'IDLE')
    assert t.status(0.0) == 'IDLE' and t.plans_run == [] and t.overlaps == [] and not t.busy
    assert t.on_plan('a', PLAN, 0.0) is True                  # the id is forgotten too


def test_the_fake_trees_statuses_are_ones_the_planner_understands():
    from task_planner.dispatcher import is_busy
    t = FakeTree(run_s=0.2)
    t.on_plan('a', PLAN, 0.0)
    assert is_busy(t.status(0.0)) and not is_busy(t.status(0.3)) and not is_busy(FakeTree().status(0.0))
