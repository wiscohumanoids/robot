#!/usr/bin/env python3
"""Live interface-contract check against a RUNNING stack.

    ros2 launch bringup full_stack.launch.py sim:=false &      # or sim:=true
    ros2 run bringup check_contract.py --no-sim                # sim:=false
    ros2 run bringup check_contract.py                         # sim:=true

Checks every entry of humanoid_interfaces/config/interface_contract.yaml:
  * topics : at most ONE publisher (the single-owner rule); if there is one, its
             type matches; `rate` topics reach min_rate_hz, `latched` topics
             have a publisher, `event` topics may be silent.
  * actions: an action server exists (its status topic has a publisher).
  * TF     : every contract edge is being published, and no contract child frame
             has two different parents.
Exits 0 if everything passes, 1 otherwise, printing one line per check.

--no-sim skips entries marked sim_only (they need MuJoCo/ros2_control running).
The complementary static check (no ROS needed) is tests/ in the repo root.
"""
import argparse
import os
import sys
import time

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosidl_runtime_py.utilities import get_message
from tf2_msgs.msg import TFMessage

def load_contract(path=None):
    if path is None:
        path = os.path.join(
            get_package_share_directory('humanoid_interfaces'), 'config', 'interface_contract.yaml')
    with open(path) as f:
        return yaml.safe_load(f)


class Checker(Node):

    def __init__(self, contract, skip_sim):
        super().__init__('contract_checker')
        self.contract = contract
        self.skip_sim = skip_sim
        self.counts = {}
        self.tf_edges = {}   # child -> set(parents)
        best_effort = QoSProfile(
            depth=10, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.BEST_EFFORT)
        for t in self._active_topics():
            if t['check'] == 'rate':
                self.counts[t['name']] = 0
                self.create_subscription(
                    get_message(t['type']), t['name'],
                    lambda _msg, name=t['name']: self._count(name), best_effort)
        self.create_subscription(TFMessage, '/tf', self._on_tf, 100)
        self.create_subscription(
            TFMessage, '/tf_static', self._on_tf,
            QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                       reliability=ReliabilityPolicy.RELIABLE))

    def _active_topics(self):
        return [t for t in self.contract['topics'] if not (self.skip_sim and t.get('sim_only'))]

    def _count(self, name):
        self.counts[name] += 1

    def _on_tf(self, msg):
        for tf in msg.transforms:
            self.tf_edges.setdefault(tf.child_frame_id, set()).add(tf.header.frame_id)

    def evaluate(self, window_s):
        results = []   # (ok, text)

        def add(ok, text):
            results.append((ok, text))

        for t in self._active_topics():
            name = t['name']
            pubs = self.get_publishers_info_by_topic(name)
            owners = sorted({p.node_name for p in pubs})
            if len(pubs) > 1:
                hint = (' (the same node name appears twice: an earlier launch is still running. '
                        'Run `jobs`, then `pkill -INT -f full_stack.launch.py` and relaunch)'
                        if len(owners) == 1 else '')
                add(False, f'{name}: {len(pubs)} publishers {owners}, must be exactly one '
                           f'(contract owner: {t["owner"]}){hint}')
                continue
            if len(pubs) == 1 and pubs[0].topic_type != t['type']:
                add(False, f'{name}: publisher {owners[0]} uses {pubs[0].topic_type}, '
                           f'contract says {t["type"]}')
                continue
            if t['check'] == 'event':
                add(True, f'{name}: {len(pubs)} publisher(s) (event topic)')
            elif not pubs:
                add(False, f'{name}: NO publisher (owner {t["owner"]} not running?)')
            elif t['check'] == 'latched':
                add(True, f'{name}: published by {owners[0]}')
            else:
                rate = self.counts.get(name, 0) / window_s
                ok = rate >= t['min_rate_hz']
                add(ok, f'{name}: {rate:.1f} Hz measured, need >= {t["min_rate_hz"]} '
                        f'(nominal {t["rate_hz"]}) from {owners[0]}')

        for a in self.contract.get('actions', []):
            status_topic = f'/{a["name"]}/_action/status'
            has_server = len(self.get_publishers_info_by_topic(status_topic)) >= 1
            add(has_server, f'action {a["name"]}: server {"present" if has_server else "MISSING"} '
                            f'(owner {a["server"]})')

        for edge in self.contract['tf']:
            if self.skip_sim and edge.get('sim_only'):
                continue
            seen = edge['parent'] in self.tf_edges.get(edge['child'], set())
            add(seen, f'tf {edge["parent"]} -> {edge["child"]}: '
                      f'{"seen" if seen else "NOT published"} (owner {edge["owner"]})')
        contract_children = {e['child'] for e in self.contract['tf']}
        for child, parents in sorted(self.tf_edges.items()):
            if child in contract_children and len(parents) > 1:
                add(False, f'tf frame {child} has multiple parents {sorted(parents)}, '
                           f'one publisher per edge')
        return results


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--no-sim', action='store_true', help='skip sim_only entries (stack launched with sim:=false)')
    ap.add_argument('--warmup', type=float, default=4.0, help='seconds to wait for discovery')
    ap.add_argument('--window', type=float, default=6.0, help='seconds over which rates are measured')
    ap.add_argument('--contract', default=None, help='path to interface_contract.yaml (default: installed copy)')
    args, ros_args = ap.parse_known_args()

    rclpy.init(args=ros_args)
    node = Checker(load_contract(args.contract), args.no_sim)
    try:
        end = time.monotonic() + args.warmup
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
        for name in node.counts:
            node.counts[name] = 0
        start = time.monotonic()
        while time.monotonic() - start < args.window:
            rclpy.spin_once(node, timeout_sec=0.05)
        results = node.evaluate(time.monotonic() - start)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

    failed = 0
    for ok, text in results:
        print(('PASS  ' if ok else 'FAIL  ') + text)
        failed += 0 if ok else 1
    print(f'\n{len(results) - failed}/{len(results)} contract checks passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
