"""What the skills need from a robot, real or simulated."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from core.camera import Camera


@dataclass
class Observation:
    """One camera image with the calibration it was taken with."""
    camera: Camera
    rgb: np.ndarray                                  # (H, W, 3) uint8
    time: float
    extra: dict = field(default_factory=dict)        # sim only: 'segmentation' (H, W) body ids, 'body_names'


@dataclass
class MoveResult:
    """How a move ended."""
    ok: bool
    position_error: float                            # metres from the target at the end
    reason: str = ''


class Robot(Protocol):
    """An arm with a parallel gripper and calibrated cameras, in its base frame."""

    table_z: float

    def camera_names(self) -> list[str]:
        """Cameras `observe` accepts."""

    def observe(self, camera: str) -> Observation:
        """A fresh image from `camera`, with its current pose (a wrist camera moves with the arm)."""

    def ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Gripper TCP position (3,) and rotation (3, 3)."""

    def reachable(self, position: np.ndarray, rotation: np.ndarray) -> bool:
        """Whether IK reaches the TCP pose (from the arm's current joints)."""

    def move_linear(self, position: np.ndarray, rotation: np.ndarray, speed: float | None = None) -> MoveResult:
        """Move the TCP along a straight line to the pose and stop there."""

    def set_gripper(self, opening: float) -> None:
        """Command the jaws (0 closed, 1 open) and wait for them to settle."""

    def gripper_width(self) -> float:
        """Distance between the jaws, metres."""

    def home(self) -> MoveResult:
        """Back to the start pose, jaws open."""


def go_home(robot: Robot, start: np.ndarray, rotation: np.ndarray) -> MoveResult:
    """Jaws open, straight up as high as the arm reaches here (at most the start height), then to the start pose.

    The pitched gripper's reach shrinks with height: far out over the table the start height itself is out of
    reach, and a lift straight to it would fail and leave the arm over the table, in the overhead camera's view.
    """
    robot.set_gripper(1.0)
    pos, _ = robot.ee_pose()
    up = max(pos[2], float(start[2]))
    while up > pos[2] + 1e-3 and not robot.reachable(np.array([pos[0], pos[1], up]), rotation):
        up -= 0.01
    if up > pos[2] + 1e-3:
        r = robot.move_linear(np.array([pos[0], pos[1], up]), rotation)
        if not r.ok:
            return r
    return robot.move_linear(np.asarray(start, float), rotation)
