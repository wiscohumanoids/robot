"""Hardware interlocks exercised with a fake SDK and timestamped camera receipts."""
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from stack_hardware import COMMAND_TIMEOUT_S, StackHardware


class HardwareTests(unittest.TestCase):
    def setUp(self):
        self.hw = StackHardware('left', 'wrist_port2')
        self.robot = self.hw._robot = Mock()
        self.hw.lower = np.zeros(7)
        self.hw.upper = np.ones(7)
        self.hw.contract = dict(max_tracking_error_rad=.1, max_joint_speed_rad_s=.3,
                                torque_limits_nm=[1] * 6, gripper_speed_rad_s=.6,
                                gripper_torque_limit_nm=.5, max_camera_age_s=.2)
        self.hw.read = Mock(return_value=dict(position_rad=[0.] * 6))
        self.addCleanup(self.hw.close)

    def test_tracking_error_or_limit_violation_stops_before_joint_command(self):
        with self.assertRaisesRegex(RuntimeError, 'tracking'):
            self.hw.command(np.full(6, .11), .3)
        self.robot.Joint_Pos_Vel.assert_not_called()
        self.robot.set_stop.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, 'stopped'):
            self.hw.command(np.zeros(6), .3)

    def test_partial_command_failure_latches_stop(self):
        self.robot.gripper_control.return_value = False
        with self.assertRaisesRegex(RuntimeError, 'gripper'):
            self.hw.command(np.full(6, .05), .3)
        self.robot.Joint_Pos_Vel.assert_called_once()
        self.robot.set_stop.assert_called_once()

    def test_motor_timeout_cannot_be_cleared_by_a_late_command(self):
        self.hw._last_command = time.monotonic() - COMMAND_TIMEOUT_S - .1
        with self.assertRaisesRegex(RuntimeError, 'timed out'):
            self.hw.command(np.zeros(6), .3)
        self.robot.Joint_Pos_Vel.assert_not_called()

    def test_watchdog_stops_an_abandoned_stream(self):
        self.hw._last_command = time.monotonic() - COMMAND_TIMEOUT_S
        self.hw._watch_commands()
        self.robot.set_stop.assert_called_once()

    def test_pair_requires_fresh_frames_within_one_camera_period(self):
        self.hw._frame_period_ns = 50_000_000
        now = time.monotonic_ns()
        frames = {name: dict(monotonic_ns=now) for name in ('overhead', 'wrist')}
        self.hw._cameras = SimpleNamespace(lock=threading.Lock(), errors={}, latest=frames, close=Mock())
        self.assertEqual(set(self.hw.frames()), {'overhead', 'wrist'})
        frames['wrist']['monotonic_ns'] -= 90_000_000
        with self.assertRaisesRegex(RuntimeError, 'frame period'):
            self.hw.frames()
        frames['wrist']['monotonic_ns'] -= 200_000_000
        with self.assertRaisesRegex(RuntimeError, 'stale'):
            self.hw.frames()
