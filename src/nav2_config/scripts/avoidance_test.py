#!/usr/bin/env python3
"""Live check that Nav2 drives AROUND the obstacles in a test map.

Run after full_stack (slam:=external navigation:=external), test_map.launch.py and
nav2.launch.py are up. Sends one NavigateToPose goal, records /odom while it runs, and
passes only if the goal succeeded, the robot ended near the goal, it never came closer
than --clearance metres (centre to nearest occupied cell) to the obstacles, and the path
actually bent around them (otherwise the test proves nothing).

    ros2 run nav2_config avoidance_test.py --map zigzag [--goal 4.5 0.0] [--save run.json]

A map whose metadata says expect = no_path (the sealed goal) passes only if Nav2 reports failure
(ABORTED) within --max-fail-time seconds without the robot wandering off. How FAST it gives up is
the behavior tree's business (stock Nav2 spins and backs up for about 25 s first), so the limits
are generous: the test catches a hang or a runaway, not a slow give-up.
--goal-yaw-deg also sets the goal's heading and requires the robot to end facing within
--yaw-tol radians of it (Nav2's own goal checker allows 0.5).

--save writes the robot's path, Nav2's planned path and the obstacle boxes as JSON,
for replaying the run in a browser.
"""
import argparse
import json
import math
import os
import sys
import time

import rclpy
from ament_index_python.packages import get_package_share_directory
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry, Path
from rclpy.action import ActionClient
from rclpy.node import Node


def map_base(map_name):
    return os.path.join(get_package_share_directory('nav2_config'), 'maps', map_name)


def load_meta(map_name):
    """title, goal, note and obstacle rectangles written by scripts/make_test_maps.py."""
    with open(map_base(map_name) + '.json') as f:
        return json.load(f)


def load_occupied(map_name):
    """Centres (x, y) in metres of every occupied cell of a maps/<name>.pgm (P5)."""
    base = map_base(map_name)
    meta = {}
    for line in open(base + '.yaml'):
        if ':' in line:
            k, v = line.split(':', 1)
            meta[k.strip()] = v.strip()
    res = float(meta['resolution'])
    ox, oy = [float(t) for t in meta['origin'].strip('[]').split(',')[:2]]
    with open(base + '.pgm', 'rb') as f:
        assert f.readline().strip() == b'P5'
        w, h = [int(t) for t in f.readline().split()]
        f.readline()
        data = f.read()
    return [(ox + (i % w + 0.5) * res, oy + (h - 1 - i // w + 0.5) * res)
            for i, v in enumerate(data) if v < 128]


class Recorder(Node):
    def __init__(self):
        super().__init__('avoidance_test')
        self.path = []        # (x, y)
        self.track = []       # (t, x, y, yaw), for --save
        self.plan = []        # first global plan of the run, (x, y); Nav2 re-plans every second
        self._t0 = None
        self.create_subscription(Odometry, '/odom', self._on_odom, 10)
        self.create_subscription(Path, '/plan', self._on_plan, 10)

    def _on_odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.path.append((p.x, p.y))
        now = self.get_clock().now().nanoseconds * 1e-9
        self._t0 = self._t0 if self._t0 is not None else now
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.track.append((now - self._t0, p.x, p.y, yaw))

    def _on_plan(self, msg):
        if self.plan or not msg.poses:
            return
        self.plan = [(ps.pose.position.x, ps.pose.position.y) for ps in msg.poses]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', default='obstacle_1', help='obstacle_1, obstacle_2, obstacle_3 or zigzag')
    ap.add_argument('--goal', nargs=2, type=float, help="override the map's default goal")
    ap.add_argument('--clearance', type=float, default=0.30, help='min centre-to-obstacle distance, m')
    ap.add_argument('--timeout', type=float, default=120.0)
    ap.add_argument('--goal-yaw-deg', type=float, help='goal heading, degrees; also checked at the end')
    ap.add_argument('--yaw-tol', type=float, default=0.5, help='allowed final heading error, rad')
    ap.add_argument('--max-fail-time', type=float, default=60.0, help='no-path maps: give up within this, s')
    ap.add_argument('--max-travel', type=float, default=0.6, help='no-path maps: max distance wandered, m')
    ap.add_argument('--save', help='write the run (paths and obstacles) to this JSON file')
    args = ap.parse_args()

    meta = load_meta(args.map)
    args.goal = args.goal or meta['goal']
    occupied = load_occupied(args.map)
    rclpy.init()
    node = Recorder()
    client = ActionClient(node, NavigateToPose, 'navigate_to_pose')
    if not client.wait_for_server(timeout_sec=20.0):
        print('FAIL: navigate_to_pose server not available')
        return 1
    while rclpy.ok() and not node.path:
        rclpy.spin_once(node, timeout_sec=0.2)
    node.path.clear()
    node.track.clear()
    node.plan = []
    node._t0 = None

    goal = NavigateToPose.Goal()
    goal.pose.header.frame_id = 'map'
    goal.pose.pose.position.x, goal.pose.pose.position.y = args.goal
    goal_yaw = None if args.goal_yaw_deg is None else math.radians(args.goal_yaw_deg)
    goal.pose.pose.orientation.z = math.sin((goal_yaw or 0.0) / 2.0)
    goal.pose.pose.orientation.w = math.cos((goal_yaw or 0.0) / 2.0)
    send = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, send, timeout_sec=10.0)
    handle = send.result()
    if handle is None or not handle.accepted:
        print('FAIL: goal rejected')
        return 1
    result = handle.get_result_async()
    t0 = time.monotonic()
    while rclpy.ok() and not result.done() and time.monotonic() - t0 < args.timeout:
        rclpy.spin_once(node, timeout_sec=0.1)
    if not result.done():
        print('FAIL: timed out')
        return 1
    elapsed = time.monotonic() - t0
    status = result.result().status   # 4 = SUCCEEDED, 5 = CANCELED, 6 = ABORTED
    status_name = {4: 'SUCCEEDED', 5: 'CANCELED', 6: 'ABORTED'}.get(status, str(status))
    node.destroy_node()
    rclpy.shutdown()

    path = node.path
    end = path[-1]
    to_goal = math.hypot(end[0] - args.goal[0], end[1] - args.goal[1])
    clearance = min(math.hypot(px - ox, py - oy) for px, py in path[::5] for ox, oy in occupied)
    detour = max(abs(p[1]) for p in path)
    travelled = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:]))
    yaw_err = None
    if goal_yaw is not None and node.track:
        yaw_err = abs(math.atan2(math.sin(node.track[-1][3] - goal_yaw), math.cos(node.track[-1][3] - goal_yaw)))

    if meta.get('expect') == 'no_path':
        print(f'map={args.map}  status={status_name}  time={elapsed:.1f}s  samples={len(path)}')
        print(f'  gave up after {elapsed:.1f} s (limit {args.max_fail_time}) | robot travelled {travelled:.2f} m (limit {args.max_travel})')
        ok = status == 6 and elapsed <= args.max_fail_time and travelled <= args.max_travel
    else:
        print(f'map={args.map}  status={status_name}  time={elapsed:.1f}s  samples={len(path)}')
        print(f'  ended {to_goal:.2f} m from goal | closest approach to an obstacle {clearance:.2f} m '
              f'(need >= {args.clearance}) | widest detour from straight line {detour:.2f} m')
        ok = status == 4 and to_goal <= 0.4 and clearance >= args.clearance and detour > 0.3
        if yaw_err is not None:
            print(f'  goal heading {args.goal_yaw_deg:.0f} deg | ended {math.degrees(yaw_err):.0f} deg off '
                  f'(allowed {math.degrees(args.yaw_tol):.0f})')
            ok = ok and yaw_err <= args.yaw_tol
    if args.save:
        stride = max(1, len(node.track) // 400)
        with open(args.save, 'w') as f:
            title = meta['title'] + ('' if goal_yaw is None else f', heading {args.goal_yaw_deg:.0f}\u00b0')
            json.dump({'map': args.map, 'goal': args.goal, 'goal_yaw': goal_yaw, 'ok': ok, 'status': status,
                       'expect': meta.get('expect'), 'elapsed': elapsed, 'yaw_err': yaw_err,
                       'clearance': clearance, 'to_goal': to_goal,
                       'title': title, 'note': meta['note'], 'boxes': meta['rects'],
                       'plan': [[round(x, 3), round(y, 3)] for x, y in node.plan[::2]],
                       'track': [[round(v, 3) for v in r] for r in node.track[::stride]]}, f)
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
