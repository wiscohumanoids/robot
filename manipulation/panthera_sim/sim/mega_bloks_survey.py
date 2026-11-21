"""Physical wrist camera poses for the Mega Bloks work area."""

from math import cos, radians, sin

import numpy as np

from .mega_bloks_motion import PickPlaceMotion, _move_duration, _quintic_progress, _slerp


def _survey_pose(motion, angle):
    """Aim the camera across the table along a tilted ellipse in world meters."""
    c, s = cos(angle), sin(angle)
    xyz = np.array([
        .214 + .037 * s - .014 * c,
        .164 + .076 * c - .026 * s,
        .330 - .005 * c - .009 * s,
    ])
    _, quat = motion._gripper_pose(xyz, radians(70 - 20 * c + 3 * s), 49 - 4 * c)
    return xyz, quat


def position_observer(motion: PickPlaceMotion) -> bool:
    """Park the right wrist camera on the opposite side of the work area.

    Create the motion with the right arm selected. Once it settles, switch to
    the left arm for the survey and transfer. The right arm holds its pose.
    """
    if motion.sim.active_arm != 'right':
        raise ValueError('The observer pose requires the right arm')
    position = (.20, -.19, .28)
    _, quat = motion._gripper_pose(position, radians(158), 25)
    return motion.move('right_observer', position, quat, 1.)


def survey(motion: PickPlaceMotion, capture, *, phase_prefix="survey", ready_at_entry=None):
    """Capture four moving views along a continuous three-quarter ellipse.

    Each capture runs after the physics tick so the image and camera pose
    describe the same instant. A calibration probe can finish at the entry
    pose if ready_at_entry() has already acquired its camera mount.
    """
    if motion.sim.active_arm != 'left':
        raise ValueError('The survey is calibrated for the left wrist')
    start, quat = _survey_pose(motion, 0.)
    entry = f"{phase_prefix}_entry"
    approach_ticks = 0
    if ready_at_entry is not None:
        if not motion.move(entry, start, quat, 1.):
            return None
        if ready_at_entry():
            return []
    else:
        # Blend the tail of the approach into the beginning of the sweep.
        # At entry both translation and rotation already follow the ellipse.
        approach_ticks = max(1, int(np.ceil(30 * _move_duration(
            motion._target, start, motion._quat, quat))))

    ticks = 144  # 4.8 seconds at the motion controller's 30 Hz
    overlap = min(30, approach_ticks)
    arc_start = approach_ticks - overlap
    indices = np.arange(1, arc_start + ticks + 1)
    progress = np.clip((indices - arc_start) / ticks, 0., 1.)
    angles = radians(270) * _quintic_progress(progress)
    capture_ticks = angles.searchsorted(np.deg2rad([17., 90., 177., 250.]))
    samples = []
    for index, angle in enumerate(angles):
        xyz, rotation = _survey_pose(motion, angle)
        if index + 1 < approach_ticks:
            u = _quintic_progress((index + 1) / approach_ticks)
            xyz = motion._target + u * (xyz - motion._target)
            rotation = _slerp(motion._quat, rotation, u)
        view = min(int(capture_ticks.searchsorted(index)), len(capture_ticks) - 1)
        phase = entry if index < arc_start else f"{phase_prefix}_{view}"
        samples.append((phase, xyz, rotation))
    captures = []

    def after_tick(index):
        if index in capture_ticks:
            captures.append(capture(samples[index][0]))

    if not motion.follow_path(phase_prefix, samples, 1., after_tick=after_tick):
        return None
    return captures


def plan_survey_transfer(motion: PickPlaceMotion, source_xyz, dest_xyz, yaws):
    """Keep the observed jaw axis when reachable and allow small skews for a stud row."""
    axes = tuple(yaws)
    yaw = motion.choose_transfer_yaw(source_xyz, dest_xyz, axes)
    if yaw is not None or len(axes) != 1:
        return yaw
    # Across a three-stud row, a 20 degree skew still leaves room for the
    # open fingers. Try smaller shifts first to keep better pad overlap.
    axis, = axes
    for degrees in (5, 10, 15, 20):
        angle = radians(degrees)
        yaw = motion.choose_transfer_yaw(source_xyz, dest_xyz, (axis + angle, axis - angle))
        if yaw is not None:
            return yaw
    return None
