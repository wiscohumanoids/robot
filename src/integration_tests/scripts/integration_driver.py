#!/usr/bin/env python3
"""Plays ONE integration scenario against a RUNNING stack (see tools/run_integration_tests.sh,
which starts the stack, the fake map and Nav2 around it, one fresh container per scenario).

    integration_driver.py --list [--llm]
    integration_driver.py --scenario nav_then_pick --out /out/nav_then_pick.json [--stack-log FILE]

Exit 0 = passed, 1 = failed. The scenario definitions and every check live in
integration_tests/scenarios.py (ROS-free, unit-tested); this file only plays them.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integration_tests import scenarios  # noqa: E402

STARTUP_S = 40.0


def log_problems(path):
    """Tracebacks in the stack log mean a node crashed, whatever the scenario looked like."""
    if not path or not os.path.isfile(path):
        return []
    with open(path, errors='replace') as f:
        text = f.read()
    problems = []
    if 'Traceback' in text:
        problems.append('a Python traceback is in the stack log')
    if 'process has died' in text:
        problems.append('a node process died (see the stack log)')
    return problems


def play(env, scenario):
    """Runs the scenario's actions, then waits for done(). Returns a list of problems."""
    problems = []
    for action in scenario.actions:
        kind = action[0]
        if kind == 'plan':
            env.publish_plan(action[1], action[2])
        elif kind == 'intent':
            env.say(action[1])
        elif kind == 'command':
            env.command(action[1])
        elif kind == 'sleep':
            env.pump(action[1])
        elif kind == 'wait_for':
            _, ev_kind, prefix, timeout_s = action
            if not env.wait_for(ev_kind, prefix, timeout_s):
                problems.append(f'timed out after {timeout_s:.0f} s waiting for {ev_kind} "{prefix}..."')
                return problems
        else:
            problems.append(f'unknown action {action!r}')
            return problems
        env.pump(0.05)
    if not env.wait_until(lambda: scenario.done(env.tl), scenario.timeout_s):
        problems.append(f'timed out after {scenario.timeout_s:.0f} s before the scenario finished')
    env.pump(scenario.settle_s)
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--llm', action='store_true', help='with --list: include the LLM scenarios')
    ap.add_argument('--scenario')
    ap.add_argument('--out')
    ap.add_argument('--stack-log')
    args, ros_args = ap.parse_known_args()

    if args.list:
        for s in scenarios.select(include_llm=args.llm):
            print(f'{s.name} {s.layer} {s.map}')
        return 0
    if not args.scenario:
        ap.error('--scenario is required')
    scenario = scenarios.select([args.scenario])[0]

    import rclpy
    from harness import Env
    rclpy.init(args=ros_args)
    env = Env()
    started = time.monotonic()
    failures = []
    try:
        # the harness itself subscribes to /skill_sequence, so the tree is the SECOND subscriber
        if scenario.layer == 'tree_nav2':
            who, ready = 'behavior_tree', env.wait_until(lambda: env._plan_pub.get_subscription_count() >= 2, STARTUP_S)
        else:
            who, ready = 'task_planner', env.wait_for_subscriber(env._intent_pub, STARTUP_S)
        if not ready:
            failures.append(f'{who} never started listening')
        elif not env.wait_until(lambda: env.tl.last('task') is not None and env.tl.last('held') is not None, STARTUP_S):
            failures.append('no /task_status or /manipulation/held_object heard (is the tree running?)')
        else:
            env.pump(1.0)
            failures += play(env, scenario)
            failures += scenario.check(env.tl)
            failures += log_problems(args.stack_log)
    finally:
        seconds = time.monotonic() - started
        passed = not failures
        payload = {'name': scenario.name, 'layer': scenario.layer, 'map': scenario.map,
                   'description': scenario.description, 'passed': passed, 'failures': failures,
                   'seconds': round(seconds, 1), 'timeline': env.timeline_dump()}
        if args.out:
            env.save(args.out, payload)
        print(f'{"PASS" if passed else "FAIL"}  {scenario.name}  ({seconds:.0f}s)')
        for f in failures:
            print(f'   - {f}')
        if not passed:
            print('   timeline (last events):')
            for t, kind, text in env.timeline_dump(60):
                print(f'     {t:7.1f}s {kind:8s} {text}')
        env.destroy_node()
        rclpy.try_shutdown()
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
