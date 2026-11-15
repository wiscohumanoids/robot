import unittest

import numpy as np

from sim.stack_task import CUBE_EDGE, TABLE_CUBE_Z, StackReward, stack_metrics, task_potential


class StackMetricsTest(unittest.TestCase):
    def setUp(self):
        z = TABLE_CUBE_Z
        self.flat = np.array([[0.32, -0.10, z], [0.40, 0.0, z], [0.49, 0.11, z]])
        self.two = np.array([[0.40, 0.0, z], [0.40, 0.0, z + CUBE_EDGE], [0.50, 0.10, z]])
        self.three = np.array([
            [0.40, 0.0, z],
            [0.40, 0.0, z + CUBE_EDGE],
            [0.40, 0.0, z + 2 * CUBE_EDGE],
        ])

    def test_stack_levels(self):
        self.assertFalse(stack_metrics(self.flat)["two_stack"])
        self.assertTrue(stack_metrics(self.two)["two_stack"])
        self.assertFalse(stack_metrics(self.two)["three_stack"])
        self.assertTrue(stack_metrics(self.three)["three_stack"])

    def test_airborne_chain_is_not_success(self):
        airborne = self.three.copy()
        airborne[:, 2] += 0.15
        self.assertFalse(stack_metrics(airborne)["three_stack"])

    def test_success_requires_hold_and_release(self):
        reward = StackReward(success_hold_steps=3)
        ee = np.array([0.40, 0.0, 0.20])
        reward.reset(self.flat, ee)
        for _ in range(4):
            _value, success, _metrics = reward.step(self.three, ee, grasped=True)
            self.assertFalse(success)
        for expected in (False, False, True):
            _value, success, _metrics = reward.step(self.three, ee, grasped=False)
            self.assertEqual(success, expected)

    def test_potential_shaping_uses_learner_discount(self):
        far = np.array([0.8, 0.0, 0.4])
        near = self.flat[0].copy()
        reward = StackReward(
            gamma=0.9, action_delta_coef=0.0, action_acceleration_coef=0.0
        )
        initial = reward.reset(self.flat, far)["potential"]
        near_potential, _ = task_potential(self.flat, near, False)
        first, _success, _metrics = reward.step(self.flat, near, False)
        second, _success, _metrics = reward.step(self.flat, far, False)
        self.assertAlmostEqual(first, 0.9 * near_potential - initial - 0.01)
        self.assertAlmostEqual(second, 0.9 * initial - near_potential - 0.01)
        # A progress/reversal cycle is no longer profitable under the same
        # discounted return optimized by PPO.
        self.assertLess(first + 0.9 * second, 0.0)


if __name__ == "__main__":
    unittest.main()
