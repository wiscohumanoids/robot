import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'app'))
from reset_grippers import ResetGrippers, command_gripper


class GripperResetTests(unittest.TestCase):
    def setUp(self):
        self.settings = json.loads((ROOT/'config/recording.json').read_text())['neutral_return']['gripper_open']
        self.reset = ResetGrippers(self.settings, {
            'left': dict(lower=0, upper=2), 'right': dict(lower=0, upper=1.8)}, 10)
        self.arms = {s: dict(gripper=dict(position=0)) for s in ('left', 'right')}

    def test_opens_to_fraction_of_each_travel_then_stays_passive_when_closed(self):
        commands = self.reset.tick(10, self.arms, {})
        self.assertAlmostEqual(commands['left']['position_rad'], 1.2)
        self.assertAlmostEqual(commands['right']['position_rad'], 1.08)
        robot = Mock()
        command_gripper(robot, self.arms['left'], commands['left'])
        robot.gripper_control.assert_called_once_with(1.2, .8, .5)
        self.assertEqual(self.arms['left']['gripper']['command'], commands['left'])
        self.arms['left']['gripper']['position'] = 1.19
        commands = self.reset.tick(12.5, self.arms, {})
        self.assertNotIn('left', commands)
        self.assertIn('right', commands)
        command_gripper(robot, self.arms['left'], commands.get('left'))
        robot.gripper_control_MIT.assert_called_once_with(0, 0, 0, 0, 0)
        self.arms['left']['gripper']['position'] = 0
        self.assertNotIn('left', self.reset.tick(13, self.arms, {}))

    def test_handle_release_cancels_opening_independently(self):
        commands = self.reset.tick(10.5, self.arms, dict(left=True, right=False))
        self.assertNotIn('left', commands)
        self.assertIn('right', commands)
        self.assertEqual(self.reset.status['left'], 'released')

    def test_reset_from_full_open_moves_back_to_partial_target(self):
        self.arms['left']['gripper']['position'] = 2.0
        commands = self.reset.tick(10.5, self.arms, {})
        self.assertAlmostEqual(commands['left']['position_rad'], 1.2)
        self.assertEqual(self.reset.status['left'], 'opening')

    def test_fraction_is_relative_to_nonzero_closed_limit(self):
        reset = ResetGrippers(self.settings, dict(left=dict(lower=.2, upper=2.2)), 0)
        self.assertAlmostEqual(reset.targets['left'], 1.4)
        for fraction in (-.1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                ResetGrippers(dict(self.settings, travel_fraction=fraction),
                              dict(left=dict(lower=0, upper=2)), 0)

    def test_obstruction_times_out_to_passive_without_repeated_attempts(self):
        self.assertEqual(self.reset.tick(15, self.arms, {}), {})
        self.assertEqual(self.reset.status, dict(left='timeout', right='timeout'))
        self.assertEqual(self.reset.tick(16, self.arms, {}), {})

    def test_invalid_limits_and_rejected_commands(self):
        with self.assertRaises(ValueError):
            ResetGrippers(self.settings, dict(left=dict(lower=0, upper=float('nan'))), 0)
        robot = Mock()
        robot.gripper_control.return_value = False
        command = self.reset.tick(10, self.arms, {})['left']
        with self.assertRaisesRegex(RuntimeError, 'rejected'):
            command_gripper(robot, self.arms['left'], command)


if __name__ == '__main__':
    unittest.main()
