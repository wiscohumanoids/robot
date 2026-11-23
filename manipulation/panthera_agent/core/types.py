"""Small types shared by the skills and their mixins."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.world import Found


@dataclass
class Result:
    """What a skill reports: whether it worked, a message for the LLM, images, and details for the log."""
    ok: bool
    message: str
    images: list[tuple[str, np.ndarray]] = field(default_factory=list)      # (caption, rgb)
    details: dict = field(default_factory=dict)


@dataclass
class Held:
    """The object in the jaws, and where it sits relative to them."""
    obj: Found
    offset: np.ndarray            # TCP - object centre, in the gripper frame
    yaw_offset: float             # object yaw - gripper yaw at the grasp
