"""Task metrics and shaped reward for stacking the three Panthera cubes.

The reward is mostly a potential difference.  This is important for this task:
paying a dense reward for merely hovering near a cube teaches the robot to
hover forever, while a potential difference pays only for making progress.
Sparse event bonuses are added for the first two-cube stack and for a stable
three-cube stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np


TABLE_CUBE_Z = 0.0725
CUBE_EDGE = 0.045


def ordered_two_stack_metrics(positions: np.ndarray) -> dict:
    """Red (column 0) released on green (column 1), with green on the table."""
    positions = np.asarray(positions, dtype=np.float64)
    if positions.shape != (2, 3):
        raise ValueError(f"expected two xyz cube positions, got {positions.shape}")
    upper, lower = positions
    xy = float(np.linalg.norm(upper[:2] - lower[:2]))
    dz = float(upper[2] - lower[2])
    success = bool(xy <= .012 and .032 < dz < .060
                   and abs(lower[2] - TABLE_CUBE_Z) < .01)
    return dict(two_stack=success, positions=positions.tolist(),
                horizontal_alignment_m=xy, vertical_gap_m=dz)


def _ordered_pair_score(lower: np.ndarray, upper: np.ndarray) -> float:
    """Soft score for ``upper`` being supported by ``lower``."""
    xy = np.linalg.norm(upper[:2] - lower[:2])
    dz = upper[2] - lower[2]
    return float(np.exp(-((xy / 0.035) ** 2) - (((dz - CUBE_EDGE) / 0.014) ** 2)))


def stack_metrics(positions: np.ndarray) -> dict[str, float | int | bool | list]:
    """Geometry metrics shared by training, evaluation, and tests.

    Cube color/order is deliberately irrelevant.  A full stack also requires
    the bottom cube to be on the table, so lifting all three cubes cannot count
    as success.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if positions.shape != (3, 3):
        raise ValueError(f"expected three xyz cube positions, got {positions.shape}")

    pair_scores = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            if i != j:
                pair_scores[i, j] = _ordered_pair_score(positions[i], positions[j])

    best_chain = 0.0
    for bottom, middle, top in permutations(range(3)):
        base_score = np.exp(-(((positions[bottom, 2] - TABLE_CUBE_Z) / 0.018) ** 2))
        chain = base_score * np.sqrt(pair_scores[bottom, middle] * pair_scores[middle, top])
        best_chain = max(best_chain, float(chain))

    order = np.argsort(positions[:, 2])
    sorted_pos = positions[order]
    gaps = np.diff(sorted_pos[:, 2])
    adjacent_xy = np.linalg.norm(np.diff(sorted_pos[:, :2], axis=0), axis=1)
    bottom_on_table = abs(float(sorted_pos[0, 2]) - TABLE_CUBE_Z) < 0.022
    full_stack = bool(
        bottom_on_table
        and np.all(adjacent_xy < 0.040)
        and np.all((gaps > 0.032) & (gaps < 0.060))
    )

    pair_stack = False
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            dz = positions[j, 2] - positions[i, 2]
            xy = np.linalg.norm(positions[j, :2] - positions[i, :2])
            if xy < 0.040 and 0.032 < dz < 0.060:
                pair_stack = True

    return {
        "positions": positions.tolist(),
        "pair_score": float(pair_scores.max()),
        "chain_score": best_chain,
        "two_stack": pair_stack,
        "three_stack": full_stack,
        "max_height_m": float(positions[:, 2].max()),
        "lifted_cubes": int(np.sum(positions[:, 2] > TABLE_CUBE_Z + 0.025)),
        "horizontal_spread_m": float(np.max(adjacent_xy)),
        "vertical_gaps_m": gaps.tolist(),
    }


def task_potential(positions: np.ndarray, ee_pos: np.ndarray, grasped: bool) -> tuple[float, dict]:
    """Dense progress potential; callers reward its temporal difference."""
    positions = np.asarray(positions, dtype=np.float64)
    metrics = stack_metrics(positions)
    nearest = float(np.min(np.linalg.norm(positions - np.asarray(ee_pos), axis=1)))
    reach_score = float(np.exp(-((nearest / 0.10) ** 2)))
    lift_progress = float(
        np.clip((positions[:, 2] - TABLE_CUBE_Z) / (2.0 * CUBE_EDGE), 0.0, 1.0).sum()
    )
    potential = (
        0.20 * reach_score
        + 0.80 * float(grasped)
        + 0.75 * lift_progress
        + 4.0 * float(metrics["pair_score"])
        + 12.0 * float(metrics["chain_score"])
    )
    metrics.update({
        "nearest_cube_m": nearest,
        "reach_score": reach_score,
        "lift_progress": lift_progress,
        "grasped": bool(grasped),
        "potential": potential,
    })
    return potential, metrics


@dataclass
class StackReward:
    """Episode-local reward state and stable-success detector."""

    success_hold_steps: int = 5
    gamma: float = 0.99
    action_delta_coef: float = 0.01
    action_acceleration_coef: float = 0.02
    previous_potential: float = 0.0
    two_stack_seen: bool = False
    full_stack_seen: bool = False
    stable_steps: int = 0

    def reset(self, positions: np.ndarray, ee_pos: np.ndarray, grasped: bool = False) -> dict:
        self.previous_potential, metrics = task_potential(positions, ee_pos, grasped)
        self.two_stack_seen = bool(metrics["two_stack"])
        self.full_stack_seen = bool(metrics["three_stack"])
        self.stable_steps = int(metrics["three_stack"])
        return metrics

    def step(
        self,
        positions: np.ndarray,
        ee_pos: np.ndarray,
        grasped: bool,
        action_delta: np.ndarray | None = None,
        action_acceleration: np.ndarray | None = None,
    ) -> tuple[float, bool, dict]:
        potential, metrics = task_potential(positions, ee_pos, grasped)
        # Potential-based shaping must include the same discount used by the
        # learner.  Using ``potential - previous_potential`` with gamma < 1
        # makes a forward/backward cycle profitable because the negative half
        # arrives later.  That is a direct incentive for oscillation.
        reward = self.gamma * potential - self.previous_potential - 0.01
        self.previous_potential = potential

        if action_delta is not None:
            reward -= self.action_delta_coef * float(np.square(action_delta).mean())
        if action_acceleration is not None:
            reward -= self.action_acceleration_coef * float(
                np.square(action_acceleration).mean()
            )
        if metrics["two_stack"] and not self.two_stack_seen:
            reward += 5.0
            self.two_stack_seen = True
        if metrics["three_stack"] and not self.full_stack_seen:
            reward += 10.0
            self.full_stack_seen = True

        # The top cube must be released.  Otherwise the policy can terminate
        # while merely holding the last cube in the correct pose.
        stable = bool(metrics["three_stack"] and not grasped)
        self.stable_steps = self.stable_steps + 1 if stable else 0
        success = self.stable_steps >= self.success_hold_steps
        if success:
            reward += 40.0
        metrics["stable_stack_steps"] = self.stable_steps
        metrics["success"] = success
        metrics["reward"] = reward
        return reward, success, metrics
