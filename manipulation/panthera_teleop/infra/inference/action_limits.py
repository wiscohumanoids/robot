"""Clamp gripper predictions to valid physical command targets."""
import numpy as np

GRIPPER_ENCODER_TOLERANCE_RAD = 0.03
GRIPPER_INDICES = (6, 13)


def clamp_gripper_actions(actions, lower=(0., 0.), upper=(2., 2.)):
    """Clamp finite gripper targets and record corrections without modifying input.

    Joint predictions and nonfinite values are left unchanged for validation.
    """
    bounded = np.asarray(actions, dtype=float).copy()
    corrections = []
    for index, lo, hi in zip(GRIPPER_INDICES, lower, upper):
        values = bounded[:, index].copy()
        changed = np.isfinite(values) & ((values < lo) | (values > hi))
        bounded[changed, index] = np.clip(values[changed], lo, hi)
        for action in np.flatnonzero(changed):
            corrections.append(dict(action=int(action) + 1, index=index,
                                    raw_target_rad=float(values[action]),
                                    command_target_rad=float(bounded[action, index])))
    return bounded, corrections
