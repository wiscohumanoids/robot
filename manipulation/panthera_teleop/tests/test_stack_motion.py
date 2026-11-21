"""Exercise the preflight and physical feedback gates without connecting an arm."""

import sys
import unittest
from importlib.util import find_spec
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import stack_motion


@unittest.skipUnless(find_spec('mujoco'), 'Stack preflight requires MuJoCo')
class StackMotionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = stack_motion.build_plan(
            [0, 1.2, 1.4, 0, 0, 0, 1], [-2.4, -.1, -.1, -1.6, -1.7, -2.5, 0],
            [2.4, 3.2, 4, 1.6, 1.7, 2.5, 2],
            [[.298, .086, .09925], [.330, .086, .09925]],
            [[.32, -.02, .09925], [.352, -.02, .09925]],
            gripper_open_rad=1., gripper_closed_rad=0., table_z_m=.05,
        )

    def run_with_gripper(self, obstruction, phases):
        state = self.plan.initial.copy()
        now = 0.

        def sleep(duration):
            nonlocal now
            now += duration

        def command(target):
            # A block between the jaws prevents the encoder reaching empty-closed.
            state[:] = target
            state[-1] = max(obstruction, state[-1])

        with patch.object(stack_motion, 'time', SimpleNamespace(monotonic=lambda: now, sleep=sleep)):
            stack_motion.execute_plan(self.plan, read=lambda: state.copy(), command=command,
                                      check_active=lambda: None, on_phase=phases.append)
        return state

    def test_preview_path_covers_every_commanded_waypoint(self):
        samples = sum(len(points) for points in self.plan.waypoints.values())
        self.assertEqual(self.plan.tcp_path.shape, (1 + samples, 3))
        self.assertTrue(np.isfinite(self.plan.tcp_path).all())
        release_index = -len(self.plan.waypoints['Retreat']) - 1
        self.assertGreater(self.plan.tcp_path[-1, 2] - self.plan.tcp_path[release_index, 2], .08)

    def test_complete_path(self):
        phases = []
        state = self.run_with_gripper(.40, phases)
        self.assertEqual(tuple(phases), self.plan.phases)
        self.assertEqual(phases[-1], 'Retreat')
        self.assertAlmostEqual(state[-1], self.plan.open_rad)

    def test_empty_grasp_stops_before_lift(self):
        phases = []
        with self.assertRaisesRegex(RuntimeError, 'Close jaws'):
            self.run_with_gripper(0., phases)
        self.assertEqual(phases[-1], 'Close jaws')

    def test_moved_start_rejected_before_command(self):
        with self.assertRaisesRegex(RuntimeError, 'moved since preview'):
            stack_motion.execute_plan(self.plan, read=lambda: self.plan.initial + .04,
                                      command=lambda _: self.fail('unexpected motion'),
                                      check_active=lambda: None, on_phase=lambda _: None)

    def test_closed_encoder_noise_is_never_sent_as_target(self):
        initial = self.plan.initial.copy()
        initial[-1] = -.018
        plan = replace(self.plan, initial=initial)

        def first_command(target):
            self.assertGreaterEqual(target[-1], plan.gripper_limits[0])
            raise RuntimeError('observed first target')

        with self.assertRaisesRegex(RuntimeError, 'observed first target'):
            stack_motion.execute_plan(plan, read=lambda: initial.copy(), command=first_command,
                                      check_active=lambda: None, on_phase=lambda _: None)

    def test_stop_between_measurement_and_write(self):
        checks = 0

        def lease():
            nonlocal checks
            checks += 1
            if checks == 2:
                raise RuntimeError('operator stopped')

        with self.assertRaisesRegex(RuntimeError, 'operator stopped'):
            stack_motion.execute_plan(self.plan, read=lambda: self.plan.initial.copy(),
                                      command=lambda _: self.fail('unexpected motion'),
                                      check_active=lease, on_phase=lambda _: None)
