"""Check the live ROS graph against config/interface_contract.yaml.

For every contract topic up to --phase:
  - exactly one publishing node, and it is the named owner
  - the message type matches
  - the publish rate is within --tolerance of rate_hz (when rate_hz is set)
For every TF edge: it is being published, on /tf or /tf_static as declared,
and only the contract owners publish TF at all.
Topics that are published but not in the contract are reported as warnings.

Exit code 0 = all checks passed, 1 = at least one failure.

    ros2 run hsim_bringup contract_check --phase 1
"""

import argparse
import fnmatch
import os
import sys
import time
from collections import defaultdict

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rosidl_runtime_py.utilities import get_message
from tf2_msgs.msg import TFMessage

ALWAYS_IGNORED = {'/rosout', '/parameter_events', '/tf', '/tf_static'}
# Shared by convention (many nodes publish here); listed under shared_topics in the contract.
SELF = 'contract_check'


def publishers(node: Node, topic: str):
    """One entry per publisher endpoint, so two copies of the same node both count."""
    infos = [i for i in node.get_publishers_info_by_topic(topic) if i.node_name != SELF]
    return sorted(i.node_name for i in infos), sorted({i.topic_type for i in infos})


def spin_for(node: Node, seconds: float):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.05)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--contract', default=os.path.join(
        get_package_share_directory('hsim_bringup'), 'config', 'interface_contract.yaml'))
    ap.add_argument('--phase', type=int, default=1)
    ap.add_argument('--window', type=float, default=4.0, help='seconds to measure rates')
    ap.add_argument('--tolerance', type=float, default=0.3, help='allowed relative rate error')
    args = ap.parse_args(argv)

    contract = yaml.safe_load(open(args.contract))
    topics = {k: v for k, v in contract['topics'].items() if v['phase'] <= args.phase}
    edges = [e for e in contract.get('tf_edges', []) if e['phase'] <= args.phase]

    rclpy.init()
    node = Node(SELF)
    spin_for(node, 2.0)  # DDS discovery

    counts = defaultdict(int)
    fast_qos = QoSProfile(depth=200, reliability=ReliabilityPolicy.BEST_EFFORT)
    for name, spec in topics.items():
        if spec.get('rate_hz'):
            node.create_subscription(
                get_message(spec['type']), name,
                lambda _m, n=name: counts.__setitem__(n, counts[n] + 1), fast_qos)

    seen_edges = {}  # (parent, child) -> 'static' | 'dynamic'

    def on_tf(msg, kind):
        for t in msg.transforms:
            seen_edges[(t.header.frame_id, t.child_frame_id)] = kind

    node.create_subscription(TFMessage, '/tf', lambda m: on_tf(m, 'dynamic'), fast_qos)
    node.create_subscription(TFMessage, '/tf_static', lambda m: on_tf(m, 'static'), QoSProfile(
        depth=100, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))

    spin_for(node, 0.5)  # let subscriptions match before counting
    counts.clear()
    spin_for(node, args.window)

    failures, warnings, rows = [], [], []

    for name, spec in topics.items():
        owners, types = publishers(node, name)
        if not owners:
            if spec.get('required', True):
                failures.append(f'{name}: no publisher (owner should be {spec["owner"]})')
                rows.append(('FAIL', name, '-', 'no publisher'))
            else:
                rows.append(('skip', name, '-', 'not running (optional)'))
            continue
        problems = []
        if len(owners) > 1:
            problems.append(f'{len(owners)} publishers {owners} (duplicate or extra nodes)')
        else:
            allowed = spec['owner'] if isinstance(spec['owner'], list) else [spec['owner']]
            if owners[0] not in allowed:
                problems.append(f'owner is "{owners[0]}", contract says {allowed}')
        if types != [spec['type']]:
            problems.append(f'type {types} != {spec["type"]}')
        rate_txt = 'event'
        if spec.get('rate_hz'):
            measured = counts[name] / args.window
            rate_txt = f'{measured:.1f}/{spec["rate_hz"]} Hz'
            if abs(measured - spec['rate_hz']) > args.tolerance * spec['rate_hz']:
                problems.append(f'rate {measured:.1f} Hz, expected {spec["rate_hz"]}')
        failures += [f'{name}: {p}' for p in problems]
        rows.append(('FAIL' if problems else 'ok', name, ','.join(owners), '; '.join(problems) or rate_txt))

    # any owner listed in the contract (any phase) may publish TF; only listed owners may
    tf_owners = {e['owner'] for e in contract.get('tf_edges', [])}
    for topic in ('/tf', '/tf_static'):
        owners, _ = publishers(node, topic)
        extra = sorted({o for o in owners if o not in tf_owners})
        if extra:
            failures.append(f'{topic}: published by non-owners {extra}')
    for e in edges:
        key = (e['parent'], e['child'])
        want = None if e['static'] is None else ('static' if e['static'] else 'dynamic')
        got = seen_edges.get(key)
        label = f'TF {e["parent"]} -> {e["child"]}'
        if got is None:
            failures.append(f'{label}: not published (owner "{e["owner"]}")')
            rows.append(('FAIL', label, e['owner'], 'not published'))
        elif want is not None and got != want:
            failures.append(f'{label}: on /tf{"_static" if got == "static" else ""}, contract says {want}')
            rows.append(('FAIL', label, e['owner'], f'{got}, expected {want}'))
        else:
            rows.append(('ok', label, e['owner'], got))

    all_contract = set(contract['topics'])
    for name, _types in node.get_topic_names_and_types():
        if name in ALWAYS_IGNORED or name in all_contract or any(
                fnmatch.fnmatch(name, pat) for pat in contract.get('shared_topics', [])):
            continue
        owners, _ = publishers(node, name)
        if owners:
            warnings.append(f'{name}: published by {owners} but not in the contract')

    width = max(len(r[1]) for r in rows)
    print(f'\nInterface contract check (phase <= {args.phase}, {args.window:.0f}s window)\n')
    for status, name, owner, info in rows:
        print(f'  {status:4s}  {name:{width}s}  {owner:16s}  {info}')
    for w in warnings:
        print(f'  warn  {w}')
    for f in failures:
        if not any(f.startswith(r[1]) for r in rows if r[0] == 'FAIL'):
            print(f'  FAIL  {f}')
    n_ok = sum(r[0] == 'ok' for r in rows)
    print(f'\n{"PASS" if not failures else "FAIL"}: {n_ok} ok, {len(failures)} failed, '
          f'{sum(r[0] == "skip" for r in rows)} skipped, {len(warnings)} warnings')

    node.destroy_node()
    rclpy.try_shutdown()
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
