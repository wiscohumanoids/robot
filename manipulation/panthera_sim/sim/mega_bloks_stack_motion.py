"""Carry a studded block to a matching stud pattern and release it into place."""

from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np

from .mega_bloks import PITCH_M
from .mega_bloks_motion import (
    PickPlaceMotion, _DEFAULT_PITCH_DEG, _UP, _move_duration, _quintic_progress, _slerp,
)
from .mega_bloks_stack import STACK_CENTER_GAP_M
from .panthera_env import GRIPPER_OPEN

# Open just above the seated height so gravity completes the last part of
# insertion without pushing the lower block against the bench.
_RELEASE_GAP_M = .002
_INSERTION_APPROACH_M = .025


@dataclass(frozen=True)
class _StackPlan:
    yaw: float
    pitch: int
    body_center: np.ndarray
    release: np.ndarray
    preinsert: np.ndarray
    above: np.ndarray
    quat: np.ndarray


def _rotation(quat):
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quat)
    return matrix.reshape(3, 3)


def _row_heading(studs):
    points = np.asarray(studs, dtype=float)
    if (points.ndim != 2 or not 2 <= len(points) <= 4 or points.shape[1] < 2
            or not np.isfinite(points).all()):
        raise ValueError("Placement requires a row of two to four finite stud positions")
    xy = points[:, :2] - points[:, :2].mean(axis=0)
    _, _, axes = np.linalg.svd(xy, full_matrices=False)
    offsets = np.sort(xy @ axes[0])
    if np.min(np.diff(offsets)) < .01 or np.max(np.abs(xy @ axes[1])) > .008:
        raise ValueError("Placement requires distinct, collinear studs")
    return math.atan2(axes[0, 1], axes[0, 0])


def _placement_target(source_studs, target_center, target_studs):
    """Find the release height and shortest rotation between two unordered stud rows."""
    delta = _row_heading(target_studs) - _row_heading(source_studs)
    center = np.asarray(target_center, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("Placement requires a finite 3D body center")
    desired = center + _RELEASE_GAP_M * _UP
    return desired, math.remainder(delta, math.pi)


def _seated_center(base_center, base_studs):
    return np.r_[np.mean(base_studs, axis=0)[:2], base_center[2] + STACK_CENTER_GAP_M]


class StackMotion(PickPlaceMotion):
    """Pick a studded row and align its sockets to a grid placement."""

    # The jaws have their own closing/opening intervals at contact.
    _settling_ticks = 0

    def __init__(self, sim, on_step=None):
        super().__init__(sim, on_step)
        self._grip_before_close = None
        self._lift_z = None
        self._planned_turn = None

    def _approach_and_descend(self, hover, goal, quat, *, placing, capture, refine):
        """Round the hover corner, refine in flight, then descend vertically to contact."""
        start, initial_quat = self._target, self._quat
        travel = max(1, math.ceil(30 * _move_duration(start, hover, initial_quat, quat)))
        # Start lowering in the last eight traverse ticks. The two quintic
        # displacements overlap smoothly while the block is clear of contact.
        descent_start = travel - min(8, travel // 2)
        if placing:
            opening = 0.
            approach_ticks, insertion_ticks = 33, 36
            precontact = goal + _INSERTION_APPROACH_M * _UP
            # Ease into the slower final 25 mm before the first descent ends.
            insertion_start = descent_start + approach_ticks - 6
            midpoint = max(1, travel // 2)
            phases = {1: 'transfer_mid', midpoint + 1: 'transfer_above',
                      travel + 1: 'preinsert', insertion_start + 1: 'insert'}
            views = {midpoint: 'transfer_mid', round(.75 * travel): 'transfer_above'}
            decision = travel + 5
        else:
            opening = 1.
            approach_ticks, insertion_ticks = 30, 0
            precontact = goal
            insertion_start = descent_start + approach_ticks
            phases = {1: 'approach', travel + 1: 'descend'}
            views = {max(1, travel - 16): 'preshape'}
            decision = travel
        end = insertion_start + insertion_ticks
        correction_ticks = insertion_start - decision
        phase = phases[1]
        actual_goal, actual_quat = goal, quat
        for tick in range(1, end + 1):
            along = _quintic_progress(min(tick / travel, 1.))
            down = _quintic_progress(np.clip((tick - descent_start) / approach_ticks, 0., 1.))
            xyz = start + along * (hover - start) + down * (precontact - hover)
            rotation = _slerp(initial_quat, quat, along)
            if placing and tick > insertion_start:
                u = _quintic_progress((tick - insertion_start) / insertion_ticks)
                xyz = xyz + u * (goal - precontact)
            if tick > decision:
                u = _quintic_progress(min((tick - decision) / correction_ticks, 1.))
                xyz = xyz + u * (actual_goal - goal)
                rotation = _slerp(quat, actual_quat, u)
            phase = phases.get(tick, phase)
            self._tick(phase, xyz, rotation, opening)
            if tick == decision or (placing and tick == insertion_start):
                if not self._check_target(phase, xyz, rotation, opening):
                    return False
            if capture is not None and tick in views:
                capture(views[tick])
            if tick == decision and refine is not None:
                update = refine()
                if update is None:
                    return False
                actual_goal, actual_quat = update
                above = actual_goal + _INSERTION_APPROACH_M * _UP
                q = self._reachable_waypoint(above, actual_quat, self.sim.q)
                if q is None or self._reachable_waypoint(actual_goal, actual_quat, q) is None:
                    return False

        return self._settle(phase, actual_goal, actual_quat, opening)

    def pick(self, center_xyz, yaw_rad, *, refine=None, observe_approach=None):
        """Sample on approach and blend a refined grasp into the vertical descent."""
        self._last_yaw = None
        self._grip_before_close = None
        pitch = self._approach_pitch(yaw_rad)
        goal, quat = self._gripper_pose(center_xyz, yaw_rad, pitch)

        def update():
            nonlocal yaw_rad
            result = refine(center_xyz, yaw_rad)
            if result is None:
                return None
            center, yaw_rad = result
            return self._gripper_pose(center, yaw_rad, pitch)

        capture = (lambda _: observe_approach()) if observe_approach is not None else None
        if not self._approach_and_descend(goal + self.clearance * _UP, goal, quat, placing=False,
                                          capture=capture, refine=update if refine else None):
            return False
        self._grip_before_close = self.sim.ee_pose()
        goal, quat = self._target, self._quat
        lift = goal + self.clearance * _UP
        if self._lift_z is not None:
            lift[2] = max(lift[2], self._lift_z)
        return self._close_and_lift(goal, quat, lift, yaw_rad, pitch,
                                    close_duration=.5, lift_duration=None)

    def _stack_plan(self, desired, delta, yaw, pitch, local_center=None):
        release, quat = self._gripper_pose(desired, yaw + delta, pitch)
        if local_center is not None:
            release = desired - _rotation(quat) @ local_center
        preinsert = release + _INSERTION_APPROACH_M * _UP
        above = release + self.clearance * _UP
        return _StackPlan(yaw + delta, pitch, desired, release, preinsert, above, quat)

    def choose_stack_yaw(self, source_center, source_studs, base_center, base_studs,
                         candidates_rad):
        """Choose a pickup heading for stacking directly on the observed base."""
        target = _seated_center(base_center, base_studs)
        return self.choose_placement_yaw(source_center, source_studs, target, base_studs, candidates_rad)

    def choose_placement_yaw(self, source_center, source_studs, target_center, target_studs,
                             candidates_rad):
        """Plan the pickup and aligned release at a specified seated body center."""
        desired, delta = _placement_target(source_studs, target_center, target_studs)
        # A row mates in either direction. At a right-angle crossing,
        # clockwise and counterclockwise releases can have different IK.
        turns = [delta]
        if abs(delta) > math.pi / 4:
            turns.append(delta - math.copysign(math.pi, delta))
        headings = []
        for candidate in candidates_rad:
            for yaw in (candidate, candidate + math.pi):
                yaw = math.remainder(yaw, 2 * math.pi)
                if all(abs(math.remainder(yaw - prior, 2 * math.pi)) > 1e-6 for prior in headings):
                    headings.append(yaw)
        self._planned_yaw = None
        self._lift_z = None
        self._planned_turn = None
        current = self.sim.q
        for pitch in (_DEFAULT_PITCH_DEG, 55, 45, 35):
            best = None
            for yaw in headings:
                pickup, pickup_quat = self._gripper_pose(source_center, yaw, pitch)
                hover = pickup + self.clearance * _UP
                q_hover = self._reachable_waypoint(hover, pickup_quat, current)
                if q_hover is None or self._reachable_waypoint(pickup, pickup_quat, q_hover) is None:
                    continue
                for turn in turns:
                    plan = self._stack_plan(desired, turn, yaw, pitch)
                    q = q_hover
                    cost = np.linalg.norm(q - current)
                    route = self._transfer_route(hover, plan.above, pickup_quat, plan.quat)
                    for _, waypoint, quat in (*route, ("preinsert", plan.preinsert, plan.quat),
                                              ("lower", plan.release, plan.quat)):
                        next_q = self._reachable_waypoint(waypoint, quat, q)
                        if next_q is None:
                            break
                        cost += .25 * np.linalg.norm(next_q - q)
                        q = next_q
                    else:
                        if best is None or cost < best[0]:
                            best = (cost, yaw, plan.above[2], turn)
            if best is not None:
                self._planned_yaw = best[1]
                self._lift_z = best[2]
                self._planned_turn = best[3]
                self._planned_pitch_deg = pitch
                return self._planned_yaw
        return None

    def stack(self, source_center, source_studs, base_center, base_studs, *,
              observe=None, refine=None):
        """Seat the held block on the two observed studs of the base."""
        target = _seated_center(base_center, base_studs)
        return self.place_aligned(source_center, source_studs, target, base_studs,
                                  observe=observe, refine=refine)

    def place_aligned(self, source_center, source_studs, target_center, target_studs, *,
                      observe=None, refine=None):
        """Release at the target body center, aligned to the target stud row.

        Call observe(phase) at lift and both transfer views. An optional
        refine(plan) early in the descent returns an updated world-frame
        release grip-site position. Measure the final seating separately.
        """
        if self._last_yaw is None or self._grip_before_close is None:
            return False
        desired, delta = _placement_target(source_studs, target_center, target_studs)
        if self._planned_turn is not None:
            # A refined row can cross the +/-90-degree boundary between views.
            delta = self._planned_turn + math.remainder(delta - self._planned_turn, math.pi)
        held_opening = self.finger_positions
        grip, quat = self._grip_before_close
        local_center = _rotation(quat).T @ (np.asarray(source_center) - grip)
        plan = self._stack_plan(desired, delta, self._last_yaw, self._last_pitch_deg, local_center)
        route = self._transfer_route(self._target, plan.above, self._quat, plan.quat)
        q = self.sim.q
        for _, waypoint, quat in (*route, ("preinsert", plan.preinsert, plan.quat),
                                  ("lower", plan.release, plan.quat)):
            q = self._reachable_waypoint(waypoint, quat, q)
            if q is None:
                return False
        if observe is not None:
            observe('lift')

        def correction():
            release = refine(plan)
            if release is None:
                return None
            release = np.asarray(release, dtype=float)
            if (release.shape != (3,) or not np.isfinite(release).all()
                    or np.linalg.norm(release - plan.release) > .03):
                return None
            return release, plan.quat

        # Pickup already lifted to the planned transport height. Carry any
        # residual height correction through the traverse, then turn downward.
        above = route[-1][1]
        if not self._approach_and_descend(above, plan.release, plan.quat, placing=True,
                                          capture=observe, refine=correction if refine else None):
            return False
        release = self._target
        # Leave the pads just outside the body until their tips rise past any
        # exposed neighboring studs. Opening fully at socket height can push a
        # lightly supported crossing sideways.
        opening = (PITCH_M / 2 + .005) / GRIPPER_OPEN
        if not self.move("release", release, plan.quat, opening, duration=.7):
            return False
        # Neighboring studs can stop a pad after it has cleared the block.
        # Compare each encoder with its actual pinching position in flight,
        # and require room across the jaws for the full one-stud body width.
        released_opening = self.finger_positions
        if np.min(released_opening - held_opening) < .001 or released_opening.sum() < PITCH_M + .002:
            self.events.append({'phase': 'release_clearance', 'time_s': float(self.sim.data.time),
                                'finger_positions_m': released_opening.tolist(),
                                'held_positions_m': held_opening.tolist(), 'ok': False})
            return False
        self._last_yaw = None
        self._planned_turn = None
        if released_opening.min() < opening * GRIPPER_OPEN - .001:
            # The outer pad faces are pressing on neighboring studs. Back off
            # into the measured gap before rising so friction cannot lift them.
            opening = (held_opening.max() + released_opening.min()) / (2 * GRIPPER_OPEN)
            if not self.move("unload_pads", release, plan.quat, opening, duration=.35):
                return False
        if not self.move("withdraw", release + _INSERTION_APPROACH_M * _UP,
                         plan.quat, opening, duration=.45):
            return False
        retreated = self.move("retreat", release + self.clearance * _UP, plan.quat, 1., duration=.8)
        return retreated and bool(np.all(self.finger_positions > GRIPPER_OPEN - .004))
