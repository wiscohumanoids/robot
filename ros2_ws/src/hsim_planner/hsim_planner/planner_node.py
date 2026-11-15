"""LLM task planner (node name: `planner`), using Claude.

  /planner/command  (std_msgs/String in)   the person's request, e.g. "walk to the table"
  /world/summary    (std_msgs/String in)   the robot's object memory (JSON from world_model)
  /behavior/execute_plan (action client)   runs the plan on the behavior tree
  /planner/reply    (std_msgs/String out)  what the planner tells the person
  /planner/status   (std_msgs/String out)  JSON: state, plan, attempt, message (for tools / the monitor)

Flow: request -> Claude (structured JSON plan) -> sanity check against memory -> behavior tree.
If a step fails, Claude gets the failure reason and the updated memory and replans (max_replans).
"stop"/"cancel" cancels the running plan without calling Claude.
Needs ANTHROPIC_API_KEY (from the repo's .env, loaded into the container).
"""

import concurrent.futures as cf
import json
import os
import time

import anthropic
import rclpy
from hsim_interfaces.action import ExecutePlan
from hsim_interfaces.msg import Skill
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String

from hsim_planner.prompt import PLAN_SCHEMA, SYSTEM_PROMPT, check_plan, user_message

STOP_WORDS = {'stop', 'cancel', 'halt', 'abort', 'never mind', 'nevermind'}


def call_claude(client, model: str, effort: str, command: str, summary, history) -> dict:
    """One planning call. Returns the plan dict, or raises RuntimeError with a plain message."""
    response = client.beta.messages.create(
        model=model,
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=[{'role': 'user', 'content': user_message(command, summary, history)}],
        output_config={'effort': effort, 'format': {'type': 'json_schema', 'schema': PLAN_SCHEMA}},
        betas=['server-side-fallback-2026-07-01'],
        fallbacks='default',              # if one model declines, the API retries on another
    )
    if response.stop_reason == 'refusal':
        raise RuntimeError("I can't help with that request.")
    if response.stop_reason == 'max_tokens':
        raise RuntimeError('the plan was cut off (too long)')
    text = next((b.text for b in response.content if b.type == 'text'), None)
    if text is None:
        raise RuntimeError('the planner returned no plan')
    return json.loads(text)


class Planner(Node):
    def __init__(self, client=None):
        super().__init__('planner')
        self.declare_parameter('model', 'claude-opus-5-5')
        self.declare_parameter('effort', 'low')
        self.declare_parameter('max_replans', 2)
        self.declare_parameter('timeout_s', 90.0)
        self.model = self.get_parameter('model').value
        self.effort = self.get_parameter('effort').value
        self.max_replans = self.get_parameter('max_replans').value

        self.client = client
        if self.client is None and os.environ.get('ANTHROPIC_API_KEY'):
            self.client = anthropic.Anthropic(timeout=self.get_parameter('timeout_s').value, max_retries=2)
        self.pool = cf.ThreadPoolExecutor(max_workers=1)

        self.summary = None
        self.task = None                 # {'command', 'history', 'future', 'goal_handle', 'result_future', 'plan'}
        self.pub_reply = self.create_publisher(String, '/planner/reply', 10)
        self.pub_status = self.create_publisher(String, '/planner/status', 10)
        self.create_subscription(String, '/world/summary', lambda m: setattr(self, 'summary', json.loads(m.data)), 5)
        self.create_subscription(String, '/planner/command', self.on_command, 10)
        self.bt = ActionClient(self, ExecutePlan, '/behavior/execute_plan')
        self.create_timer(0.1, self.poll)
        state = f'model {self.model}, effort {self.effort}' if self.client else 'NO ANTHROPIC_API_KEY: will refuse requests'
        self.get_logger().info(f'planner ready ({state})')

    # ------------------------------------------------------------------ helpers
    def reply(self, text):
        self.get_logger().info(f'REPLY: {text}')
        self.pub_reply.publish(String(data=text))

    def status(self, state, **extra):
        self.pub_status.publish(String(data=json.dumps({'state': state, **extra})))

    # ------------------------------------------------------------------ flow
    def on_command(self, msg: String):
        command = msg.data.strip()
        if not command:
            return
        if command.lower().strip('.! ') in STOP_WORDS:
            self.cancel_running('stopped on request')
            self.reply('Stopping.')
            self.status('idle', message='stopped on request')
            return
        if self.client is None:
            self.reply('I can\'t plan: ANTHROPIC_API_KEY is not set (put it in the repo\'s .env and restart the container).')
            self.status('error', message='no API key')
            return
        self.cancel_running('replaced by a new request')
        self.get_logger().info(f'REQUEST: "{command}"')
        self.task = {'command': command, 'history': [], 'goal_handle': None, 'result_future': None, 'plan': None}
        self.plan_next()

    def plan_next(self):
        t = self.task
        self.status('planning', command=t['command'], attempt=len(t['history']) + 1)
        t['future'] = self.pool.submit(call_claude, self.client, self.model, self.effort,
                                       t['command'], self.summary, list(t['history']))
        t['t0'] = time.monotonic()

    def cancel_running(self, why):
        if self.task and self.task.get('goal_handle'):
            self.task['goal_handle'].cancel_goal_async()
        self.task = None

    def poll(self):
        t = self.task
        if not t:
            return
        fut = t.get('future')
        if fut is not None and fut.done():
            t['future'] = None
            try:
                plan = fut.result()
            except RuntimeError as e:
                return self.finish(str(e))
            except anthropic.APIConnectionError:
                return self.finish("I can't reach the planning service (network).")
            except anthropic.RateLimitError:
                return self.finish('The planning service is busy; try again in a moment.')
            except anthropic.APIStatusError as e:
                return self.finish(f'The planning service returned an error ({e.status_code}).')
            except Exception as e:     # malformed JSON etc.
                return self.finish(f'Planning failed: {type(e).__name__}')
            self.get_logger().info(f'PLAN ({time.monotonic() - t["t0"]:.1f}s): {plan["reasoning"]} | '
                                   + ' -> '.join(f'{s["name"]}({s["target"] or s["text"]})' for s in plan['skills']))
            if not plan['understood'] or not plan['skills']:
                return self.finish(plan['reply'] or "I'm not sure what you want me to do.")
            problem = check_plan(plan, self.summary)
            if problem:
                return self.retry(plan['skills'], f'(plan check, not executed) {problem}')
            t['plan'] = plan
            if plan['reply']:
                self.reply(plan['reply'])
            self.execute(plan)
        rf = t.get('result_future')
        if rf is not None and rf.done():
            t['result_future'] = None
            r = rf.result().result
            if r.success:
                return self.finish('Done.', success=True)
            self.retry(t['plan']['skills'], r.message)

    def retry(self, skills, failure):
        t = self.task
        t['history'].append({'skills': skills, 'failure': failure})
        if len(t['history']) > self.max_replans:
            return self.finish(f"I couldn't do it: {failure}")
        self.get_logger().info(f'replanning after: {failure}')
        self.plan_next()

    def execute(self, plan):
        t = self.task
        goal = ExecutePlan.Goal()
        goal.plan.task, goal.plan.reasoning = t['command'], plan['reasoning']
        for s in plan['skills']:
            goal.plan.skills.append(Skill(name=s['name'], target=s['target'], hints=list(s['hints']), text=s['text']))
        if not self.bt.wait_for_server(timeout_sec=2.0):
            return self.finish('My behavior system is not running.')
        self.status('executing', command=t['command'], plan=plan['skills'], attempt=len(t['history']) + 1)

        def on_goal(f):
            gh = f.result()
            if self.task is not t:
                return
            if not gh.accepted:
                return self.finish('The behavior system rejected the plan.')
            t['goal_handle'] = gh
            t['result_future'] = gh.get_result_async()
        self.bt.send_goal_async(goal).add_done_callback(on_goal)

    def finish(self, message, success=False):
        self.reply(message)
        self.status('done' if success else 'failed', message=message)
        self.task = None


def main():
    rclpy.init()
    node = Planner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
