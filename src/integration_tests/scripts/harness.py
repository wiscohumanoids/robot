"""ROS side of the test harness (runs in the Docker container only; imports rclpy).

One node, `integration_harness`, that sends inputs (/user_intent, /planner_command,
/skill_sequence) and records everything it hears into a logic.Timeline. With a FakeTree it
also plays the behavior tree (publishes /task_status), for the stress tests that run with
behavior_tree:=false. No decisions are made here: see integration_tests/logic.py.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rclpy                                              # noqa: E402
from geometry_msgs.msg import PoseStamped                 # noqa: E402
from rclpy.node import Node                               # noqa: E402
from std_msgs.msg import String                           # noqa: E402

from humanoid_interfaces.msg import Skill, SkillSequence  # noqa: E402

from integration_tests import logic                       # noqa: E402


def yaw_of(q):
    import math
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Env(Node):

    def __init__(self, fake=None):
        super().__init__('integration_harness')
        self.t0 = time.monotonic()
        self.tl = logic.Timeline()
        self.mark = 0                     # timeline index just after the last thing we sent
        self.fake = fake
        self._last_fake_text = None
        self._last_fake_pub = 0.0

        self._intent_pub = self.create_publisher(String, '/user_intent', 50)
        self._command_pub = self.create_publisher(String, '/planner_command', 50)
        self._plan_pub = self.create_publisher(SkillSequence, '/skill_sequence', 50)
        self.create_subscription(String, '/planner_status', self._on_planner, 500)
        self.create_subscription(String, '/task_status', self._on_task, 500)
        self.create_subscription(SkillSequence, '/skill_sequence', self._on_plan, 500)
        self.create_subscription(PoseStamped, '/robot_pose', self._on_pose, 10)
        self.create_subscription(String, '/manipulation/held_object', self._on_held, 10)
        if fake is not None:
            self._status_pub = self.create_publisher(String, '/task_status', 50)
            self.create_timer(0.05, self._tick_fake)

    # ------------------------------------------------------------ time and spinning
    def now(self):
        return time.monotonic() - self.t0

    def pump(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.02)

    def wait_until(self, predicate, timeout_s):
        end = time.monotonic() + timeout_s
        while rclpy.ok():
            if predicate():
                return True
            if time.monotonic() >= end:
                return False
            rclpy.spin_once(self, timeout_sec=0.05)
        return False

    # ------------------------------------------------------------ recording
    def _on_planner(self, msg):
        self.tl.add(self.now(), 'planner', msg.data)

    def _on_task(self, msg):
        self.tl.add_task(self.now(), msg.data)

    def _on_held(self, msg):
        self.tl.add_held(self.now(), msg.data)

    def _on_pose(self, msg):
        p = msg.pose.position
        self.tl.last_pose = (p.x, p.y, yaw_of(msg.pose.orientation))

    def _on_plan(self, msg):
        skills = [{'name': s.name, 'object_id': s.object_id, 'x': s.target_pose.pose.position.x,
                   'y': s.target_pose.pose.position.y, 'z': s.target_pose.pose.position.z}
                  for s in msg.skills]
        data = {'plan_id': msg.plan_id, 'intent': msg.intent, 'skills': skills}
        self.tl.add(self.now(), 'plan', f'{msg.plan_id} {[s["name"] for s in skills]}', data)
        if self.fake is not None:
            self.fake.on_plan(msg.plan_id, skills, self.now())

    def _tick_fake(self):
        text = self.fake.status(self.now())
        now = time.monotonic()
        if text != self._last_fake_text or now - self._last_fake_pub >= 1.0:
            self._last_fake_text, self._last_fake_pub = text, now
            self._status_pub.publish(String(data=text))

    # ------------------------------------------------------------ sending
    def _sent(self, text):
        self.tl.add(self.now(), 'sent', text)
        self.mark = len(self.tl.events)

    def say(self, text):
        self._intent_pub.publish(String(data=text))
        self._sent(f'intent: {text[:80]}')

    def command(self, text):
        self._command_pub.publish(String(data=text))
        self._sent(f'command: {text}')

    def publish_plan(self, plan_id, skills):
        msg = SkillSequence()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.intent = f'scripted {plan_id}'
        msg.plan_id = plan_id
        for step in skills:
            skill = Skill()
            skill.name = step['name']
            skill.object_id = step['object_id']
            skill.target_pose.header.frame_id = 'map'
            skill.target_pose.pose.position.x = float(step['x'])
            skill.target_pose.pose.position.y = float(step['y'])
            skill.target_pose.pose.position.z = float(step['z'])
            skill.target_pose.pose.orientation.w = 1.0
            msg.skills.append(skill)
        self._plan_pub.publish(msg)
        self._sent(f'plan {plan_id}: {[s["name"] for s in skills]}')

    # ------------------------------------------------------------ waiting for replies
    def wait_for(self, kind, prefix, timeout_s):
        """Wait for an event of `kind` starting with `prefix`, seen after the last send."""
        return self.wait_until(lambda: self.tl.find(kind, prefix, self.mark) is not None, timeout_s)

    def wait_for_subscriber(self, publisher, timeout_s):
        return self.wait_until(lambda: publisher.get_subscription_count() > 0, timeout_s)

    def planner_alive(self):
        return 'task_planner' in self.get_node_names()

    def save(self, path, payload):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(payload, f, indent=1)

    def timeline_dump(self, limit=400):
        return [(round(e.t, 2), e.kind, e.text) for e in self.tl.events[-limit:]]
