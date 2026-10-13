"""The scenario table and its checks: each check passes on the story it expects and fails on a
broken one, so a green integration run really means something."""
import os

import pytest

from integration_tests import scenarios
from integration_tests.logic import Timeline

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def build(statuses=(), planner=(), held=(), pose=None, plans=()):
    tl = Timeline()
    t = 0.0
    for s in statuses:
        tl.add_task(t, s)
        t += 1.0
    for text in planner:
        tl.add(t, 'planner', text)
        t += 0.1
    for h in held:
        tl.add_held(t, h)
    for pid, intent, skills in plans:
        tl.add(t, 'plan', pid, {'plan_id': pid, 'intent': intent, 'skills': skills})
    tl.last_pose = pose
    tl.task_times = [float(i) for i in range(0, int(t) + 1)]
    return tl


# ------------------------------------------------------------ the table
def test_scenario_names_are_unique_and_layers_known():
    names = [s.name for s in scenarios.ALL]
    assert len(names) == len(set(names))
    assert {s.layer for s in scenarios.ALL} <= set(scenarios.LAYERS)


def test_every_map_exists():
    for s in scenarios.ALL:
        base = os.path.join(ROOT, 'src', 'nav2_config', 'maps', s.map)
        assert os.path.isfile(base + '.yaml'), f'{s.name}: no map {s.map}'


def test_every_scenario_is_documented_and_has_a_real_timeout():
    for s in scenarios.ALL:
        assert s.description and s.timeout_s >= 60 and callable(s.done) and callable(s.check)


def test_actions_are_well_formed():
    sizes = {'plan': 3, 'intent': 2, 'command': 2, 'sleep': 2, 'wait_for': 4}
    for s in scenarios.ALL:
        for action in s.actions:
            assert action[0] in sizes and len(action) == sizes[action[0]], (s.name, action)
            if action[0] == 'wait_for':
                assert action[1] in ('planner', 'task', 'held')


def test_tree_layer_scenarios_publish_plans_and_planner_layers_type_intents():
    for s in scenarios.ALL:
        kinds = {a[0] for a in s.actions}
        if s.layer == 'tree_nav2':
            assert 'plan' in kinds and 'intent' not in kinds, s.name
        else:
            assert 'intent' in kinds and 'plan' not in kinds, s.name


def test_scripted_plans_only_use_goals_inside_the_map_and_known_objects():
    from integration_tests.logic import validate_plan
    for s in scenarios.TREE_NAV2:
        for action in s.actions:
            if action[0] == 'plan' and s.name != 'unknown_skill_fails_fast' and action[2]:
                assert validate_plan(action[2]) == [], (s.name, validate_plan(action[2]))


def test_select_default_skips_llm_and_names_pick_exactly():
    default = scenarios.select()
    assert default and all(s.layer != 'llm' for s in default)
    assert any(s.layer == 'llm' for s in scenarios.select(include_llm=True))
    assert [s.name for s in scenarios.select(['nav_then_pick'])] == ['nav_then_pick']
    with pytest.raises(KeyError):
        scenarios.select(['nope'])


def test_the_readme_lists_every_scenario():
    readme = open(os.path.join(ROOT, 'src', 'integration_tests', 'README.md')).read()
    for s in scenarios.ALL:
        assert f'`{s.name}`' in readme, f'{s.name} is not in the README'


# ------------------------------------------------------------ checks accept the right story ...
GOOD_NAV = ['IDLE', 'RUNNING 1/1 navigate_to', 'SUCCEEDED']


def test_nav_check_accepts_a_good_run_and_rejects_bad_ones():
    check = scenarios.BY_NAME['nav_reaches_goal'].check
    assert check(build(GOOD_NAV, pose=(3.9, 0.1, 0.0))) == []
    assert check(build(GOOD_NAV, pose=(0.0, 0.0, 0.0)))                                  # never arrived
    assert check(build(['IDLE', 'RUNNING 1/1 navigate_to', 'FAILED 1/1 navigate_to: x (attempt 3/3)'], pose=(3.9, 0, 0)))
    assert check(build(['IDLE'], pose=(3.9, 0, 0)))                                      # never ran


def test_a_quiet_status_topic_fails_every_check():
    tl = build(GOOD_NAV, pose=(3.9, 0.1, 0.0))
    tl.task_times = [0.0, 1.0, 9.0]
    assert any('went quiet' in p for p in scenarios.BY_NAME['nav_reaches_goal'].check(tl))


def test_nav_then_pick_needs_the_hand_to_hold_the_cube():
    check = scenarios.BY_NAME['nav_then_pick'].check
    story = ['IDLE', 'RUNNING 1/2 navigate_to', 'RUNNING 2/2 pick', 'SUCCEEDED']
    assert check(build(story, held=['', 'cube_1'], pose=(2.0, -2.1, 0))) == []
    assert check(build(story, held=[''], pose=(2.0, -2.1, 0)))
    assert check(build(story[:-1] + ['FAILED 2/2 pick: x'], held=['', 'cube_1'], pose=(2.0, -2.1, 0)))


def test_pick_and_place_needs_hold_then_release():
    check = scenarios.BY_NAME['pick_and_place'].check
    story = ['IDLE', 'RUNNING 1/4 navigate_to', 'RUNNING 4/4 place', 'SUCCEEDED']
    assert check(build(story, held=['', 'cube_1', ''], pose=(-2.0, 2.0, 0))) == []
    assert check(build(story, held=['', 'cube_1'], pose=(-2.0, 2.0, 0)))                 # never let go
    assert check(build(story, held=[''], pose=(-2.0, 2.0, 0)))                           # never held it


def test_unreachable_needs_three_attempts_then_the_failure():
    check = scenarios.BY_NAME['unreachable_retries_then_fails'].check
    story = ['IDLE', 'RUNNING 1/1 navigate_to', 'RUNNING 1/1 navigate_to: attempt 2/3',
             'RUNNING 1/1 navigate_to: attempt 3/3', 'FAILED 1/1 navigate_to: action aborted (attempt 3/3)']
    assert check(build(story)) == []
    assert check(build(story[:3] + [story[4]]))                                          # skipped an attempt
    assert check(build(story[:4] + ['SUCCEEDED']))


def test_failed_plan_drops_rest_check():
    check = scenarios.BY_NAME['failed_plan_drops_rest_next_plan_runs'].check
    story = ['IDLE', 'RUNNING 1/2 navigate_to', 'FAILED 1/2 navigate_to: x (attempt 3/3)',
             'RUNNING 1/1 navigate_to', 'SUCCEEDED']
    assert check(build(story, held=[''])) == []
    assert check(build(story[:3] + ['RUNNING 2/2 pick'] + story[3:], held=['', 'cube_1']))    # the pick ran anyway
    assert check(build(story[:3]))                                                       # second plan never ran


def test_duplicate_check_wants_exactly_one_start():
    check = scenarios.BY_NAME['duplicate_plan_id_runs_once'].check
    assert check(build(['IDLE', 'RUNNING 1/1 navigate_to', 'SUCCEEDED'])) == []
    assert check(build(['IDLE', 'RUNNING 1/1 navigate_to', 'SUCCEEDED', 'RUNNING 1/1 navigate_to', 'SUCCEEDED']))


def test_unknown_skill_check_wants_no_retry():
    check = scenarios.BY_NAME['unknown_skill_fails_fast'].check
    assert check(build(['IDLE', 'RUNNING 1/1 dance', 'FAILED 1/1 dance: unknown skill: dance'])) == []
    assert check(build(['IDLE', 'RUNNING 1/1 dance', 'RUNNING 1/1 dance: attempt 2/3',
                        'FAILED 1/1 dance: unknown skill: dance (attempt 3/3)']))


NAV_PICK = [{'name': 'navigate_to', 'object_id': '', 'x': 0.9, 'y': 0.3, 'z': 0.0},
            {'name': 'pick', 'object_id': 'cube_1', 'x': 1.5, 'y': 0.3, 'z': 0.8}]
CANNED_STORY = ['QUEUED "pick up the cube": 1 waiting', 'PLANNING "pick up the cube"',
                'SENT "pick up the cube": 2 steps', 'DONE "pick up the cube"']


def test_canned_pick_check():
    check = scenarios.BY_NAME['canned_pick_the_cube'].check
    good = build(['IDLE', 'RUNNING 1/2 navigate_to', 'RUNNING 2/2 pick', 'SUCCEEDED'], CANNED_STORY, ['', 'cube_1'],
                 plans=[('p1', 'pick up the cube', NAV_PICK)])
    assert check(good) == []
    wrong_order = build(['IDLE', 'RUNNING 1/2 navigate_to', 'SUCCEEDED'], [CANNED_STORY[1], CANNED_STORY[0]] + CANNED_STORY[2:],
                        ['', 'cube_1'], plans=[('p1', 'pick up the cube', NAV_PICK)])
    assert check(wrong_order)
    invalid = [dict(NAV_PICK[0], x=60.0), NAV_PICK[1]]
    assert check(build(['IDLE', 'RUNNING 1/2 navigate_to', 'SUCCEEDED'], CANNED_STORY, ['', 'cube_1'],
                       plans=[('p1', 'pick up the cube', invalid)]))


def test_queue_editing_check():
    check = scenarios.BY_NAME['canned_queue_editing'].check
    said = ['SENT "pick up the cube": 2 steps',
            'QUEUE: 1. "pick up the cup" | 2. "pick up the ball" | 3. "put the cube down"',
            'REMOVED "pick up the ball" (was #2)',
            'QUEUE: 1. "pick up the cup" | 2. "put the cube down"',
            'CLEARED: 2 removed', 'QUEUE: empty', 'DONE "pick up the cube"']
    assert check(build(['IDLE', 'RUNNING 1/2 a', 'SUCCEEDED'], said, plans=[('p', 'pick up the cube', NAV_PICK)])) == []
    broken = [t.replace('(was #2)', '(was #1)') for t in said]
    assert check(build(['IDLE', 'RUNNING 1/2 a', 'SUCCEEDED'], broken, plans=[('p', 'pick up the cube', NAV_PICK)]))
    assert check(build(['IDLE', 'RUNNING 1/2 a', 'SUCCEEDED'], said + ['SENT "pick up the cup": 2 steps'],
                       plans=[('p', 'pick up the cube', NAV_PICK)]))                    # a cleared intent was sent


def test_queue_has_no_cap_check():
    check = scenarios.BY_NAME['canned_queue_has_no_cap'].check
    waiting = [f'pick up the cube {i}' for i in range(1, 13)]
    said = [f'QUEUED "i{i}": {i} waiting' for i in range(13)]
    said += ['QUEUE: ' + ' | '.join(f'{i}. "{t}"' for i, t in enumerate(waiting, start=1)), 'CLEARED: 12 removed']
    assert check(build(['IDLE'], said)) == []
    assert check(build(['IDLE'], said[:12] + ['REJECTED "x": queue full (10)'] + said[13:]))      # something was rejected
    assert check(build(['IDLE'], said[:-2] + ['QUEUE: 1. "a"', 'CLEARED: 12 removed']))          # list was wrong


def test_llm_infeasible_check_wants_three_refusals_and_no_plan():
    check = scenarios.BY_NAME['llm_impossible_requests'].check
    no = ['NO_PLAN "x": cannot do that: no'] * 3
    assert check(build(['IDLE'], no)) == []
    assert check(build(['IDLE'], no[:2]))
    assert check(build(['IDLE', 'RUNNING 1/1 a'], no))
    assert check(build(['IDLE'], no, plans=[('p', 'x', NAV_PICK)]))


def test_llm_replan_check():
    check = scenarios.BY_NAME['llm_replan_after_failure'].check
    ok = ['SENT "go": 1 steps', 'RETRYING "go": FAILED 1/1 navigate_to: x (attempt 3/3)', 'REPLANNING "go"',
          'NO_PLAN "go": replan failed: cannot do that: sealed']
    statuses = ['IDLE', 'RUNNING 1/1 navigate_to', 'FAILED 1/1 navigate_to: x (attempt 3/3)']
    plan = [('p1', 'go', [{'name': 'navigate_to', 'object_id': '', 'x': 4.0, 'y': 0.0, 'z': 0.0}])]
    assert check(build(statuses, ok, plans=plan)) == []
    assert check(build(statuses, ok + ['REPLANNING "go"'], plans=plan))                  # a second replan
    assert check(build(statuses, [t for t in ok if not t.startswith('RETRYING')], plans=plan))


def test_goal_of_returns_the_last_navigation_goal():
    plan = {'skills': [{'name': 'navigate_to', 'x': 1.0, 'y': 2.0}, {'name': 'pick'}, {'name': 'navigate_to', 'x': 3.0, 'y': 4.0}]}
    assert scenarios.goal_of(plan) == (3.0, 4.0)
    assert scenarios.goal_of({'skills': [{'name': 'pick'}]}) is None
