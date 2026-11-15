"""Manipulation skill node (ROS name: manipulation). Action server /skills/manipulate.

The behavior tree calls it once the robot stands at the counter (after navigate_to). For the goal's
instruction it runs the VLA policy (services/vla_server, GR00T, reached over an SSH tunnel) in a loop:
  head-camera image + arm/gripper joint state + instruction -> chunk of absolute joint targets (16 steps),
  executed at 20 Hz on /arm/joint_targets. The next chunk is requested `prefetch_steps` before the current one's
  execution horizon (8) ends, with a fresh observation; when it arrives the steps that already passed are skipped,
  so the arms never pause for the server (a late chunk holds the last target).
until the success check passes (if goal.object and goal.target are set and perception sees both),
the time limit, or a cancel. Legs/waist stay with hsim_locomotion; this node only writes arm + gripper joints.

Inputs : /camera/head/color/image_raw (sensor_msgs/Image, rgb8 or bgr8), /joint_states, /perception/objects
Output : /arm/joint_targets (sensor_msgs/JointState, positions = targets)
Gripper: policy uses 0 = closed .. 1 = open; mapped to the Dex1-1 finger joint range (gripper_closed..gripper_open).
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from vision_msgs.msg import Detection3DArray

from hsim_interfaces.action import Manipulate

from .vla_client import VlaClient, resize_nearest

ARM = ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw')


class ManipulationNode(Node):
    def __init__(self):
        super().__init__('manipulation')
        p = self.declare_parameter
        p('vla_url', 'http://host.docker.internal:8200')
        p('image_topic', '/camera/head/color/image_raw')
        p('joint_states_topic', '/joint_states')
        p('targets_topic', '/arm/joint_targets')
        p('objects_topic', '/perception/objects')
        p('control_hz', 20.0)
        p('max_duration_s', 40.0)
        p('image_width', 320)                       # training images (F1 ego camera) are 320x240
        p('image_height', 240)
        p('waist_state', 'zeros')                   # training data has the waist welded at zero
        p('gripper_joints', ['left_dex1_finger_joint_1', 'left_dex1_finger_joint_2',
                             'right_dex1_finger_joint_1', 'right_dex1_finger_joint_2'])
        p('gripper_open', 0.0245)                   # Dex1-1 finger joint: open / closed (F1's model)
        p('gripper_closed', -0.020)
        p('prefetch_steps', 6)                      # request the next chunk this many steps early (>= server latency / 50 ms)
        p('success_xy_scale', 0.75)                 # object centre within this x target half-size of the target centre
        g = lambda n: self.get_parameter(n).value
        self.client = VlaClient(g('vla_url'))
        self.dt = 1.0 / g('control_hz')
        self.arm_joints = {s: [f'{s}_{j}_joint' for j in ARM] for s in ('left', 'right')}
        self.waist_joints = ['waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint']
        gj = list(g('gripper_joints'))
        self.grip_joints = {'left': [j for j in gj if j.startswith('left')], 'right': [j for j in gj if j.startswith('right')]}
        self.g_open, self.g_closed = g('gripper_open'), g('gripper_closed')
        self.lock = threading.Lock()
        self.image, self.q, self.objects = None, {}, []
        cb = ReentrantCallbackGroup()
        self.create_subscription(Image, g('image_topic'), self.on_image, qos_profile_sensor_data, callback_group=cb)
        self.create_subscription(JointState, g('joint_states_topic'), self.on_joints, 20, callback_group=cb)
        self.create_subscription(Detection3DArray, g('objects_topic'), self.on_objects, 5, callback_group=cb)
        self.pub = self.create_publisher(JointState, g('targets_topic'), 10)
        self.busy = False
        self.pool = ThreadPoolExecutor(max_workers=1)
        ActionServer(self, Manipulate, '/skills/manipulate', execute_callback=self.execute,
                     goal_callback=self.on_goal, cancel_callback=lambda _: CancelResponse.ACCEPT, callback_group=cb)
        self.get_logger().info(f"manipulation skill ready: VLA {g('vla_url')}, image {g('image_topic')} -> {g('targets_topic')}")

    # ------------------------------------------------------------------ inputs
    def on_image(self, msg):
        if msg.encoding not in ('rgb8', 'bgr8'):
            return
        a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
        if msg.encoding == 'bgr8':
            a = a[..., ::-1]
        with self.lock:
            self.image = a.copy()

    def on_joints(self, msg):
        with self.lock:
            for n, x in zip(msg.name, msg.position):
                self.q[n] = x

    def on_objects(self, msg):
        objs = []
        for d in msg.detections:
            label = d.results[0].hypothesis.class_id if d.results else d.id
            c, s = d.bbox.center.position, d.bbox.size
            objs.append((label.lower(), np.array([c.x, c.y, c.z]), np.array([s.x, s.y, s.z])))
        with self.lock:
            self.objects = objs

    # ------------------------------------------------------------------ helpers
    def grip_frac(self, side):
        j = self.grip_joints[side][0]
        return float(np.clip((self.q.get(j, self.g_open) - self.g_closed) / (self.g_open - self.g_closed), 0.0, 1.0))

    def state(self):
        with self.lock:
            missing = [j for s in ('left', 'right') for j in self.arm_joints[s] if j not in self.q]
            if self.image is None or missing:
                return None, None
            st = {'left_arm': [self.q[j] for j in self.arm_joints['left']],
                  'right_arm': [self.q[j] for j in self.arm_joints['right']],
                  'left_gripper': [self.grip_frac('left')], 'right_gripper': [self.grip_frac('right')],
                  'waist': [0.0, 0.0, 0.0] if self.get_parameter('waist_state').value == 'zeros'
                  else [self.q.get(j, 0.0) for j in self.waist_joints]}
            img = resize_nearest(self.image, self.get_parameter('image_width').value, self.get_parameter('image_height').value)
            return img, st

    def publish(self, chunk, i):
        names, pos = [], []
        for side in ('left', 'right'):
            names += self.arm_joints[side]
            pos += [float(x) for x in chunk[f'{side}_arm'][i]]
            g = float(np.clip(chunk[f'{side}_gripper'][i][0], 0.0, 1.0))
            for j in self.grip_joints[side]:
                names.append(j)
                pos.append(self.g_closed + g * (self.g_open - self.g_closed))
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name, msg.position = names, pos                 # rclpy uses array.array for position: assign, don't extend
        self.pub.publish(msg)

    def check_success(self, obj, tgt):
        """Object seen resting on/over the target (perception). None if it can't be checked."""
        if not obj or not tgt:
            return None
        with self.lock:
            objs = list(self.objects)
        o = [x for x in objs if obj.lower() in x[0]]
        t = [x for x in objs if tgt.lower() in x[0]]
        if not o or not t:
            return None
        (_, po, _), (_, pt, st) = o[0], t[0]
        on_xy = np.linalg.norm(po[:2] - pt[:2]) < self.get_parameter('success_xy_scale').value * max(st[0], st[1]) / 2
        above = po[2] > pt[2]
        return bool(on_xy and above)

    # ------------------------------------------------------------------ action
    def on_goal(self, goal):
        if self.busy:
            self.get_logger().warn('manipulate: already running a goal; rejecting')
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def execute(self, handle):
        self.busy = True
        try:
            return self._run(handle)
        finally:
            self.busy = False

    def _run(self, handle):
        goal = handle.request
        res, fb = Manipulate.Result(), Manipulate.Feedback()
        limit = goal.max_duration_s if goal.max_duration_s > 0 else self.get_parameter('max_duration_s').value
        try:
            exec_h = int(self.client.health()['execution_horizon'])
        except Exception as e:
            res.success, res.message = False, f"my manipulation policy server isn't reachable ({e})"
            handle.abort()
            return res
        # Clock-driven control on the ROS clock (sim time in the stack, like the 20 Hz training data): step n happens at
        # t0 + n*dt. Each tick publishes the target for the current time slot (chunk index = slots since the chunk's
        # observation), so actions keep their timing even if this loop runs late (late slots are skipped, not stretched).
        now = lambda: self.get_clock().now().nanoseconds * 1e-9
        t_start, published, chunk, chunk_step, last_lat = time.time(), 0, None, 0, 0.0
        t0, last_slot, skipped = None, -1, 0
        pending, req_step = None, 0                       # prefetch of the next chunk (future, slot it was asked at)
        prefetch = max(0, min(int(self.get_parameter('prefetch_steps').value), exec_h - 1))
        fb.phase = 'waiting_for_inputs'

        def finish(ok_override=None):
            ok = self.check_success(goal.object, goal.target) if ok_override is None else ok_override
            res.steps, res.verified = published, ok is not None
            res.success = bool(ok) if ok is not None else published > 0
            rate = f'{published / (now() - t0):.1f} Hz' if t0 and published and now() > t0 else 'n/a'
            timing = f' [{published} targets at {rate} (ROS time), {skipped} late slots skipped]'
            if ok:
                res.message = 'done'
            elif ok is not None:
                res.message = f"I tried for {limit:.0f} s but the {goal.object or 'object'} isn't on the {goal.target or 'target'}"
            else:
                res.message = f'ran for {limit:.0f} s (result not checked: no perception)'
            res.message += timing
            (handle.succeed if res.success else handle.abort)()
            return res

        while rclpy.ok():
            if handle.is_cancel_requested:
                handle.canceled()
                res.success, res.message, res.steps = False, 'cancelled', published
                return res
            if time.time() - t_start > limit:
                return finish()
            if chunk is None:                                  # first chunk: synchronous, starts the clock
                img, st = self.state()
                if img is None:
                    fb.phase = 'waiting_for_inputs'
                    handle.publish_feedback(fb)
                    time.sleep(0.1)
                    continue
                try:
                    chunk, last_lat = self.client.act(goal.instruction, img, st)
                except Exception as e:
                    res.success, res.message, res.steps = False, f'policy server error: {e}', published
                    handle.abort()
                    return res
                t0, chunk_step = now(), 0
            slot = int((now() - t0) / self.dt)
            if slot == last_slot:                              # sleep to the next 50 ms boundary (ROS time) in one go:
                time.sleep(min(self.dt, max(0.0005, t0 + (slot + 1) * self.dt - now())))   # few wake-ups = less GIL contention
                continue
            skipped += max(0, slot - last_slot - 1)
            last_slot = slot
            # a prefetched chunk arrived: switch to it (its index 0 is the slot its observation was taken)
            if pending is not None and pending.done():
                try:
                    new_chunk, last_lat = pending.result()
                except Exception as e:
                    res.success, res.message, res.steps = False, f'policy server error: {e}', published
                    handle.abort()
                    return res
                pending = None
                if slot - req_step < len(new_chunk['left_arm']):
                    chunk, chunk_step = new_chunk, req_step
            ci = slot - chunk_step
            # ask for the next chunk `prefetch` slots before this one's execution horizon ends (fresh observation)
            if pending is None and ci >= exec_h - prefetch:
                img, st = self.state()
                if img is not None:
                    pending, req_step = self.pool.submit(self.client.act, goal.instruction, img, st), slot
            self.publish(chunk, min(ci, len(chunk['left_arm']) - 1))   # chunk ran out (late server): hold its last target
            published += 1
            fb.step, fb.phase, fb.last_latency_ms = published, 'running', last_lat
            if published % 10 == 0:
                handle.publish_feedback(fb)
                if self.check_success(goal.object, goal.target):
                    return finish(True)
        res.success, res.message = False, 'shutting down'
        handle.abort()
        return res


def main():
    rclpy.init()
    node = ManipulationNode()
    ex = MultiThreadedExecutor(num_threads=2)            # action loop + subscriptions; more threads = more GIL contention
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
