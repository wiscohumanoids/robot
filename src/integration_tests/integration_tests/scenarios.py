"""Integration scenarios: the behavior tree, the task planner and Nav2 working together
(ROS-free definitions; scripts/integration_driver.py plays them).

Layers (what is under test, and what it costs):
  tree_nav2  planner off. The driver publishes /skill_sequence itself, exactly as the planner
             would, so tree + Nav2 + manipulation are tested deterministically.
  canned     planner on WITHOUT an API key (its canned fallback), so planner + tree + Nav2
             are tested end to end, deterministically and for free.
  llm        planner on WITH the API key from .env: a real model call per intent. Slower, costs
             money, and the model is not deterministic, so it only runs with --llm.

An action is one of:
  ('plan', plan_id, [skill dicts])   publish a SkillSequence
  ('intent', text)                   publish /user_intent
  ('command', text)                  publish /planner_command
  ('sleep', seconds)
  ('wait_for', kind, prefix, timeout_s)   wait for an event of `kind` (planner|task|held) whose text
                                          starts with `prefix`, seen after the last thing we sent
After the actions the driver waits until done(timeline) (or timeout_s), then settle_s more
seconds (to catch late duplicates), then runs check(timeline) -> list of failure strings.
"""
from dataclasses import dataclass, field
from typing import Callable, List

from integration_tests import logic
from integration_tests.logic import (classify, finished, heartbeat_problems, near, parse_queue_line,
                                      planner_count, run_starts, task_texts, validate_plan, world_objects)

REACH_TOL_M = 0.6


@dataclass
class Scenario:
    name: str
    layer: str
    map: str
    description: str
    actions: List[tuple]
    done: Callable
    check: Callable
    timeout_s: float = 240.0
    settle_s: float = 2.0


# ---------------------------------------------------------------- building blocks
def nav(x, y):
    return {'name': 'navigate_to', 'object_id': '', 'x': float(x), 'y': float(y), 'z': 0.0}


def at_object(name, object_id):
    o = world_objects()[object_id]
    return {'name': name, 'object_id': object_id, 'x': o['x'], 'y': o['y'], 'z': o['z']}


def pick(object_id='cube_1'):
    return at_object('pick', object_id)


def place(object_id='cube_1'):
    return at_object('place', object_id)


def finished_at_least(n):
    return lambda tl: len(finished(tl)) >= n


def planner_at_least(kind, n):
    return lambda tl: planner_count(tl, kind) >= n


def held_values(tl):
    return tl.texts('held')


def expect(failures, condition, message):
    if not condition:
        failures.append(message)


def published_plans_valid(tl):
    problems = []
    for plan in tl.plans():
        for p in validate_plan(plan['skills']):
            problems.append(f"plan for \"{plan.get('intent', '')}\": {p}")
    return problems


def goal_of(plan):
    """(x, y) of the last navigate_to in a plan, else None."""
    goals = [(s['x'], s['y']) for s in plan['skills'] if s['name'] == 'navigate_to']
    return goals[-1] if goals else None


def common(tl):
    return heartbeat_problems(tl)


# ---------------------------------------------------------------- tree + Nav2 (planner off)
def check_nav_reaches_goal(tl):
    f = common(tl)
    expect(f, 'RUNNING 1/1 navigate_to' in task_texts(tl), 'never saw RUNNING 1/1 navigate_to')
    expect(f, finished(tl)[-1:] == ['SUCCEEDED'], f'final status {finished(tl)[-1:]}, wanted SUCCEEDED')
    expect(f, near(tl.last_pose, (4.0, 0.0), REACH_TOL_M), f'robot ended at {tl.last_pose}, not near (4, 0)')
    return f


def check_nav_then_pick(tl):
    f = common(tl)
    texts = task_texts(tl)
    expect(f, 'RUNNING 1/2 navigate_to' in texts and 'RUNNING 2/2 pick' in texts, f'status sequence was {texts}')
    expect(f, finished(tl)[-1:] == ['SUCCEEDED'], f'final status {finished(tl)[-1:]}')
    expect(f, near(tl.last_pose, (2.0, -2.0), REACH_TOL_M), f'robot ended at {tl.last_pose}, not near (2, -2)')
    expect(f, held_values(tl)[-1:] == ['cube_1'], f'hand ended as {held_values(tl)[-1:]}, wanted cube_1')
    return f


def check_pick_and_place(tl):
    f = common(tl)
    expect(f, finished(tl)[-1:] == ['SUCCEEDED'], f'final status {finished(tl)[-1:]}')
    expect(f, 'cube_1' in held_values(tl), f'the hand never held the cube: {held_values(tl)}')
    expect(f, held_values(tl)[-1:] == [''], f'hand ended as {held_values(tl)[-1:]}, wanted empty')
    expect(f, near(tl.last_pose, (-2.0, 2.0), REACH_TOL_M), f'robot ended at {tl.last_pose}, not near (-2, 2)')
    return f


def check_unreachable_retries(tl):
    f = common(tl)
    texts = task_texts(tl)
    expect(f, any('navigate_to: attempt 2/3' in t for t in texts), f'no second attempt in {texts}')
    expect(f, any('navigate_to: attempt 3/3' in t for t in texts), f'no third attempt in {texts}')
    last = finished(tl)[-1:] or ['']
    expect(f, last[0].startswith('FAILED 1/1 navigate_to') and '(attempt 3/3)' in last[0], f'final status {last}')
    return f


def check_failed_plan_drops_rest(tl):
    f = common(tl)
    results = finished(tl)
    expect(f, len(results) >= 2, f'expected two finished plans, got {results}')
    if len(results) >= 2:
        expect(f, results[0].startswith('FAILED 1/2 navigate_to'), f'first plan ended {results[0]}')
        expect(f, results[1] == 'SUCCEEDED', f'second plan ended {results[1]}')
    expect(f, 'RUNNING 2/2 pick' not in task_texts(tl), 'the dropped pick of the failed plan ran anyway')
    expect(f, held_values(tl)[-1:] in ([''], []), f'hand ended as {held_values(tl)[-1:]} (the pick must not run)')
    return f


def check_duplicate_runs_once(tl):
    f = common(tl)
    expect(f, run_starts(tl) == 1, f'the tree started {run_starts(tl)} plans, wanted exactly 1')
    expect(f, finished(tl) == ['SUCCEEDED'], f'results {finished(tl)}')
    return f


def check_two_in_order(tl):
    f = common(tl)
    expect(f, finished(tl) == ['SUCCEEDED', 'SUCCEEDED'], f'results {finished(tl)}')
    expect(f, run_starts(tl) == 2, f'started {run_starts(tl)} plans, wanted 2')
    expect(f, near(tl.last_pose, (-2.0, 2.0), REACH_TOL_M), f'robot ended at {tl.last_pose}, not near (-2, 2)')
    return f


def check_unknown_skill(tl):
    f = common(tl)
    last = finished(tl)[-1:] or ['']
    expect(f, last[0] == 'FAILED 1/1 dance: unknown skill: dance', f'final status {last}')
    expect(f, not any('attempt' in t for t in task_texts(tl)), 'an unknown skill must not be retried')
    return f


def check_empty_plan_ignored(tl):
    f = common(tl)
    expect(f, run_starts(tl) == 1, f'started {run_starts(tl)} plans, wanted 1 (the empty one is ignored)')
    expect(f, finished(tl) == ['SUCCEEDED'], f'results {finished(tl)}')
    return f


TREE_NAV2 = [
    Scenario('nav_reaches_goal', 'tree_nav2', 'obstacle_1',
             'navigate_to (4, 0) around a box: RUNNING then SUCCEEDED, robot at the goal',
             [('plan', 't1', [nav(4, 0)])], finished_at_least(1), check_nav_reaches_goal, 180),
    Scenario('nav_then_pick', 'tree_nav2', 'obstacle_1',
             'navigate then pick: both steps run, the hand ends up holding cube_1',
             [('plan', 't2', [nav(2, -2), pick('cube_1')])], finished_at_least(1), check_nav_then_pick, 240,
             settle_s=3.0),
    Scenario('pick_and_place', 'tree_nav2', 'obstacle_1',
             'navigate, pick, navigate, place: the hand holds the cube and then lets go',
             [('plan', 't3', [nav(2, -2), pick('cube_1'), nav(-2, 2), place('cube_1')])],
             finished_at_least(1), check_pick_and_place, 300, settle_s=3.0),
    Scenario('unreachable_retries_then_fails', 'tree_nav2', 'unreachable',
             'Nav2 aborts a goal in a sealed box: 3 attempts, then FAILED 1/1 (attempt 3/3)',
             [('plan', 't4', [nav(4, 0)])], finished_at_least(1), check_unreachable_retries, 420),
    Scenario('failed_plan_drops_rest_next_plan_runs', 'tree_nav2', 'unreachable',
             'plan A fails at step 1 (its pick never runs), plan B behind it still runs',
             [('plan', 't5a', [nav(4, 0), pick('cube_1')]), ('plan', 't5b', [nav(-2, 0)])],
             finished_at_least(2), check_failed_plan_drops_rest, 540),
    Scenario('duplicate_plan_id_runs_once', 'tree_nav2', 'obstacle_1',
             'the same plan_id sent twice (a planner resend) runs once',
             [('plan', 't6', [nav(2, -2)]), ('sleep', 0.5), ('plan', 't6', [nav(2, -2)])],
             finished_at_least(1), check_duplicate_runs_once, 180, settle_s=6.0),
    Scenario('two_plans_run_in_order', 'tree_nav2', 'obstacle_1',
             'two plans queued back to back both run, in order',
             [('plan', 't7a', [nav(2, -2)]), ('plan', 't7b', [nav(-2, 2)])],
             finished_at_least(2), check_two_in_order, 300),
    Scenario('unknown_skill_fails_fast', 'tree_nav2', 'obstacle_1',
             'an unknown skill name fails at once, with no retries',
             [('plan', 't8', [{'name': 'dance', 'object_id': '', 'x': 0.0, 'y': 0.0, 'z': 0.0}])],
             finished_at_least(1), check_unknown_skill, 90),
    Scenario('empty_plan_is_ignored', 'tree_nav2', 'obstacle_1',
             'an empty plan changes nothing, the next plan still runs',
             [('plan', 't10a', []), ('plan', 't10b', [nav(2, -2)])],
             finished_at_least(1), check_empty_plan_ignored, 180),
]


# ---------------------------------------------------------------- planner + tree + Nav2, canned planner
def check_canned_pick(tl, label='cube', object_id='cube_1'):
    f = common(tl)
    expect(f, logic.kinds(tl)[:4] == ['QUEUED', 'PLANNING', 'SENT', 'DONE'], f'planner said {logic.kinds(tl)}')
    plans = tl.plans()
    expect(f, len(plans) == 1, f'{len(plans)} plans published, wanted 1')
    if plans:
        names = [s['name'] for s in plans[0]['skills']]
        expect(f, names == ['navigate_to', 'pick'], f'plan was {names}')
        expect(f, plans[0]['skills'][-1]['object_id'] == object_id, f'picked {plans[0]["skills"][-1]["object_id"]}')
    expect(f, finished(tl)[-1:] == ['SUCCEEDED'], f'tree ended {finished(tl)[-1:]}')
    expect(f, held_values(tl)[-1:] == [object_id], f'hand ended as {held_values(tl)[-1:]}')
    return f + published_plans_valid(tl)


def check_canned_queue_order(tl):
    f = common(tl)
    k = logic.kinds(tl)
    expect(f, k.count('SENT') == 2 and k.count('DONE') == 2, f'planner said {k}')
    sent = [i for i, e in enumerate(tl.events) if e.kind == 'planner' and classify(e.text)[0] == 'SENT']
    done = [i for i, e in enumerate(tl.events) if e.kind == 'planner' and classify(e.text)[0] == 'DONE']
    if len(sent) == 2 and done:
        expect(f, sent[1] > done[0], 'the second plan was sent before the first one finished')
    plans = tl.plans()
    expect(f, [p['intent'] for p in plans] == ['pick up the cube', 'put the cube down'],
           f'plans were made for {[p["intent"] for p in plans]}')
    if len(plans) == 2:
        expect(f, plans[1]['skills'][-1]['name'] == 'place', 'second plan should end with place')
    expect(f, held_values(tl)[-1:] == [''], f'hand ended as {held_values(tl)[-1:]}, wanted empty')
    return f + published_plans_valid(tl)


def check_canned_queue_editing(tl):
    f = common(tl)
    lists = [parse_queue_line(t) for t in tl.texts('planner') if t.startswith('QUEUE:')]
    expect(f, len(lists) == 3, f'expected three list replies, got {lists}')
    if len(lists) == 3:
        expect(f, lists[0] == ['pick up the cup', 'pick up the ball', 'put the cube down'], f'first list {lists[0]}')
        expect(f, lists[1] == ['pick up the cup', 'put the cube down'], f'after remove 2: {lists[1]}')
        expect(f, lists[2] == [], f'after clear: {lists[2]}')
    removed = [t for t in tl.texts('planner') if t.startswith('REMOVED')]
    expect(f, removed == ['REMOVED "pick up the ball" (was #2)'], f'remove replies {removed}')
    expect(f, 'CLEARED: 2 removed' in tl.texts('planner'), 'no "CLEARED: 2 removed" reply')
    expect(f, planner_count(tl, 'SENT') == 1, f'{planner_count(tl, "SENT")} plans were sent, wanted only the first')
    expect(f, planner_count(tl, 'DONE') == 1, 'the first plan should finish normally')
    return f + published_plans_valid(tl)


def check_canned_queue_has_no_cap(tl):
    f = common(tl)
    expect(f, planner_count(tl, 'QUEUED') == 13, f'{planner_count(tl, "QUEUED")} QUEUED, wanted 13 (1 running + 12 waiting)')
    expect(f, planner_count(tl, 'REJECTED') == 0, f'{planner_count(tl, "REJECTED")} intents were rejected; there is no cap')
    lists = [parse_queue_line(t) for t in tl.texts('planner') if t.startswith('QUEUE:')]
    expect(f, lists and lists[0] == [f'pick up the cube {i}' for i in range(1, 13)],
           f'list showed {lists[0] if lists else None}, wanted the 12 waiting intents in order')
    expect(f, 'CLEARED: 12 removed' in tl.texts('planner'), 'clear should report "CLEARED: 12 removed"')
    return f


def _wait_planner(prefix, timeout=60):
    return ('wait_for', 'planner', prefix, timeout)


CANNED = [
    Scenario('canned_pick_the_cube', 'canned', 'obstacle_1',
             'planner (canned) -> tree -> Nav2 -> manipulation: "pick up the cube" ends with the cube in hand',
             [('intent', 'pick up the cube')], planner_at_least('DONE', 1), check_canned_pick, 240, settle_s=3.0),
    Scenario('canned_pick_the_cup', 'canned', 'obstacle_1',
             'the intent text selects the object: "pick up the cup" goes to cup_1',
             [('intent', 'pick up the cup')], planner_at_least('DONE', 1),
             lambda tl: check_canned_pick(tl, 'cup', 'cup_1'), 240, settle_s=3.0),
    Scenario('canned_queue_runs_in_order', 'canned', 'obstacle_1',
             'two intents typed together run one after the other; the second waits for the first',
             [('intent', 'pick up the cube'), ('intent', 'put the cube down')],
             planner_at_least('DONE', 2), check_canned_queue_order, 420, settle_s=3.0),
    Scenario('canned_queue_editing', 'canned', 'obstacle_1',
             'while a plan runs: list, remove 2, list, clear, list all act on the waiting intents only',
             [('intent', 'pick up the cube'), _wait_planner('SENT'),
              ('intent', 'pick up the cup'), ('intent', 'pick up the ball'), ('intent', 'put the cube down'),
              ('sleep', 0.5),
              ('command', 'list'), _wait_planner('QUEUE:', 10),
              ('command', 'remove 2'), _wait_planner('REMOVED', 10),
              ('command', 'list'), _wait_planner('QUEUE:', 10),
              ('command', 'clear'), _wait_planner('CLEARED', 10),
              ('command', 'list'), _wait_planner('QUEUE:', 10)],
             planner_at_least('DONE', 1), check_canned_queue_editing, 300, settle_s=3.0),
    Scenario('canned_queue_has_no_cap', 'canned', 'obstacle_1',
             'one plan running, 12 more intents: all 12 wait, in order, none is rejected',
             [('intent', 'pick up the cube'), _wait_planner('SENT')]
             + [('intent', f'pick up the cube {i}') for i in range(1, 13)]
             + [('sleep', 1.0), ('command', 'list'), _wait_planner('QUEUE:', 10), ('command', 'clear')],
             lambda tl: tl.count('planner', 'CLEARED') >= 1, check_canned_queue_has_no_cap, 180, settle_s=0.5),
]


# ---------------------------------------------------------------- planner + tree + Nav2, real LLM
def check_llm_go_to_xy(tl):
    f = common(tl)
    plans = tl.plans()
    expect(f, len(plans) == 1, f'{len(plans)} plans published, wanted 1')
    if plans:
        goal = goal_of(plans[0])
        expect(f, goal is not None and near((goal[0], goal[1], 0), (2.0, -3.0), 0.3), f'plan goal was {goal}, wanted (2, -3)')
    expect(f, finished(tl)[-1:] == ['SUCCEEDED'], f'tree ended {finished(tl)[-1:]}')
    expect(f, near(tl.last_pose, (2.0, -3.0), REACH_TOL_M), f'robot ended at {tl.last_pose}, not near (2, -3)')
    return f + published_plans_valid(tl)


def check_llm_pick_refuse_place(tl):
    f = common(tl)
    plans = tl.plans()
    expect(f, len(plans) == 2, f'{len(plans)} plans published (pick the cup, put it down), wanted 2')
    if plans:
        expect(f, any(s['name'] == 'pick' and s['object_id'] == 'cup_1' for s in plans[0]['skills']),
               f'first plan should pick cup_1: {plans[0]["skills"]}')
    if len(plans) == 2:
        expect(f, any(s['name'] == 'place' and s['object_id'] == 'cup_1' for s in plans[1]['skills']),
               f'second plan should place cup_1: {plans[1]["skills"]}')
    no_plan = [t for t in tl.texts('planner') if t.startswith('NO_PLAN')]
    expect(f, len(no_plan) == 1, f'{len(no_plan)} NO_PLAN lines, wanted exactly 1 (the refused pick of the cube)')
    expect(f, no_plan and ('hold' in no_plan[0].lower() or 'hand' in no_plan[0].lower()),
           f'the refusal should mention the hand: {no_plan}')
    expect(f, 'cup_1' in held_values(tl) and held_values(tl)[-1:] == [''], f'hand history {held_values(tl)}')
    return f + published_plans_valid(tl)


def check_llm_infeasible(tl):
    f = common(tl)
    expect(f, planner_count(tl, 'NO_PLAN') == 3, f'{planner_count(tl, "NO_PLAN")} NO_PLAN lines, wanted 3')
    expect(f, not tl.plans(), f'plans were published for impossible requests: {tl.plans()}')
    expect(f, run_starts(tl) == 0, 'the tree started something')
    return f


def check_llm_replan(tl):
    f = common(tl)
    k = logic.kinds(tl)
    expect(f, k.count('RETRYING') == 1 and k.count('REPLANNING') == 1,
           f'wanted exactly one RETRYING and one REPLANNING, planner said {k}')
    expect(f, any(t.startswith('FAILED 1/') for t in finished(tl)), f'the first plan should fail: {finished(tl)}')
    expect(f, len(tl.plans()) <= 2, f'{len(tl.plans())} plans for one intent (original + one replan at most)')
    expect(f, k[-1:] and k[-1] in ('DONE', 'GAVE_UP', 'NO_PLAN'), f'the story did not end cleanly: {k}')
    return f + published_plans_valid(tl)


def check_llm_two_in_order(tl):
    f = common(tl)
    plans = tl.plans()
    expect(f, len(plans) == 2, f'{len(plans)} plans published, wanted 2')
    if len(plans) == 2:
        g1, g2 = goal_of(plans[0]), goal_of(plans[1])
        expect(f, g1 and near((g1[0], g1[1], 0), (2.0, -3.0), 0.3), f'first goal {g1}')
        expect(f, g2 and near((g2[0], g2[1], 0), (-2.0, 2.0), 0.3), f'second goal {g2}')
    expect(f, finished(tl) == ['SUCCEEDED', 'SUCCEEDED'], f'results {finished(tl)}')
    return f + published_plans_valid(tl)


LLM = [
    Scenario('llm_go_to_xy', 'llm', 'obstacle_1',
             '"go to x=2 y=-3": the model plans one navigate_to there and the robot arrives',
             [('intent', 'go to x=2 y=-3')], planner_at_least('DONE', 1), check_llm_go_to_xy, 300),
    Scenario('llm_pick_refuse_put_down', 'llm', 'obstacle_1',
             'pick the cup (ok), pick the cube while holding the cup (refused, mentions the hand), put the cup down (ok)',
             [('intent', 'can you pick up the cup'), _wait_planner('DONE', 240),
              ('intent', 'pick up the cube'), _wait_planner('NO_PLAN', 90),
              ('intent', 'put the cup down'), _wait_planner('DONE', 240)],
             planner_at_least('DONE', 2), check_llm_pick_refuse_place, 700, settle_s=3.0),
    Scenario('llm_impossible_requests', 'llm', 'obstacle_1',
             'a nonsense request, an off-map goal and an unknown object are all refused, nothing runs',
             [('intent', 'fly to the moon'), _wait_planner('NO_PLAN', 90),
              ('intent', 'go to x=50 y=50'), _wait_planner('NO_PLAN', 90),
              ('intent', 'pick up the banana'), _wait_planner('NO_PLAN', 90)],
             planner_at_least('NO_PLAN', 3), check_llm_infeasible, 330, settle_s=1.0),
    Scenario('llm_replan_after_failure', 'llm', 'unreachable',
             'the model sends the robot into a sealed box: the tree fails it, the model gets ONE replan',
             [('intent', 'go to x=4 y=0')],
             lambda tl: any(planner_count(tl, k) for k in ('DONE', 'GAVE_UP')) or
             (planner_count(tl, 'REPLANNING') and planner_count(tl, 'NO_PLAN')),
             check_llm_replan, 900, settle_s=2.0),
    Scenario('llm_two_intents_in_order', 'llm', 'obstacle_1',
             'two typed intents are planned one at a time, in order, each from where the robot is then',
             [('intent', 'go to x=2 y=-3'), ('intent', 'go to x=-2 y=2')],
             planner_at_least('DONE', 2), check_llm_two_in_order, 600),
]

ALL = TREE_NAV2 + CANNED + LLM
BY_NAME = {s.name: s for s in ALL}
LAYERS = ('tree_nav2', 'canned', 'llm')


def select(names=None, include_llm=False):
    """Scenarios to run: the named ones, or every non-LLM scenario (plus LLM ones with include_llm)."""
    if names:
        unknown = [n for n in names if n not in BY_NAME]
        if unknown:
            raise KeyError(f'unknown scenario(s): {", ".join(unknown)}; known: {", ".join(BY_NAME)}')
        return [BY_NAME[n] for n in names]
    return [s for s in ALL if include_llm or s.layer != 'llm']
