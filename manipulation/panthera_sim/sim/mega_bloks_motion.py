"""Cartesian pick and place primitives for the left Panthera arm."""

from __future__ import annotations

import math

import numpy as np

from .panthera_env import GRIPPER_OPEN, PantheraSim, mat_to_quat

_UP = np.array([0., 0., 1.])
_DEFAULT_PITCH_DEG = 65
_RELEASE_CLEARANCE_M = .012
_LINEAR_SPEED_M_S = .22
_ANGULAR_SPEED_RAD_S = 1.8
_LINEAR_JERK_M_S3 = 8.
_ANGULAR_JERK_RAD_S3 = 48.


def _quintic_progress(t):
    """Map normalized time to path progress with zero endpoint speed and acceleration."""
    return t**3 * (10 + t * (-15 + 6 * t))


def _slerp(start, end, u):
    """Interpolate along the shorter arc between unit quaternions."""
    dot = float(np.dot(start, end))
    if dot < 0:
        end, dot = -end, -dot
    if dot > .9995:
        mixed = (1 - u) * start + u * end
        return mixed / np.linalg.norm(mixed)
    theta = math.acos(dot)
    return (math.sin((1 - u) * theta) * start + math.sin(u * theta) * end) / math.sin(theta)


def _move_duration(start, target, initial_quat, quat):
    angle = 2 * math.acos(float(np.clip(abs(np.dot(quat, initial_quat)), 0., 1.)))
    distance = np.linalg.norm(target - start)
    # The quintic has peak normalized speed 1.875 and jerk 60.
    return max(.25, 1.875 * distance / _LINEAR_SPEED_M_S,
               1.875 * angle / _ANGULAR_SPEED_RAD_S,
               (60 * distance / _LINEAR_JERK_M_S3) ** (1 / 3),
               (60 * angle / _ANGULAR_JERK_RAD_S3) ** (1 / 3))


class PickPlaceMotion:
    """Drive the gripper from world-frame targets and arm feedback."""

    _settling_ticks = 9

    def __init__(self, sim: PantheraSim, on_step=None):
        """Call on_step(motion, phase) after each 30 Hz control tick."""
        self.sim = sim
        self.on_step = on_step
        self.events: list[dict] = []
        self.clearance = .10
        self._opening = 1.
        self._last_yaw: float | None = None
        self._last_pitch_deg = _DEFAULT_PITCH_DEG
        self._planned_yaw: float | None = None
        self._planned_pitch_deg = _DEFAULT_PITCH_DEG
        self._ticks = 0
        self._target, self._quat = sim.ee_pose()
        self._qctrl = sim.q

    @staticmethod
    def _gripper_pose(center_xyz, yaw_rad, pitch_deg):
        """Convert a block body's center and jaw heading into a grip-site pose."""
        closing = np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0.])
        forward = np.cross(closing, _UP)
        pitch = math.radians(pitch_deg)
        reach = math.cos(pitch) * forward - math.sin(pitch) * _UP
        rotation = np.column_stack((reach, closing, np.cross(reach, closing)))
        # The grip site is ahead of the pad centers. Aim slightly above the
        # body midpoint to keep the tips off the table during closure.
        target = np.asarray(center_xyz, dtype=float) + .018 * reach + .006 * _UP
        return target, mat_to_quat(rotation)

    def choose_yaw(self, center_xyz, candidates_rad) -> float | None:
        """Choose a reachable jaw heading from the caller's equivalent axes."""
        return self._choose_yaw(center_xyz, candidates_rad, dest_xyz=None)

    def choose_transfer_yaw(self, center_xyz, dest_xyz, candidates_rad) -> float | None:
        """Plan one jaw heading and pitch for pickup, transfer and release."""
        return self._choose_yaw(center_xyz, candidates_rad, dest_xyz)

    def _choose_yaw(self, center_xyz, candidates_rad, dest_xyz) -> float | None:
        headings = []
        for candidate in candidates_rad:
            for yaw in (candidate, candidate + math.pi):
                yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
                if not any(abs(yaw - previous) < 1e-6 for previous in headings):
                    headings.append(yaw)

        current_q = self.sim.q
        self._planned_yaw = None
        # Prefer a steep approach to clear nearby studs. Shallow pitches can
        # reach farther across the table because the wrist stays nearer the base.
        for pitch_deg in (_DEFAULT_PITCH_DEG, 55, 45, 35):
            best = None
            for yaw in headings:
                goal, quat = self._gripper_pose(center_xyz, yaw, pitch_deg)
                above = goal + self.clearance * _UP
                q_above = self._reachable_waypoint(above, quat, current_q)
                if q_above is None:
                    continue
                q_goal = self._reachable_waypoint(goal, quat, q_above)
                if q_goal is None:
                    continue
                cost = np.linalg.norm(q_above - current_q) + .25 * np.linalg.norm(q_goal - q_above)
                if dest_xyz is not None:
                    release, above_dest, _ = self._placement_poses(dest_xyz, yaw, pitch_deg)
                    # Seed each waypoint from the previous solve so the
                    # transfer and descent use the same IK branch.
                    q = q_above
                    feasible = True
                    route = self._transfer_route(above, above_dest, quat, quat)
                    for _, waypoint, rotation in (*route, ('lower', release, quat)):
                        next_q = self._reachable_waypoint(waypoint, rotation, q)
                        if next_q is None:
                            feasible = False
                            break
                        cost += .25 * np.linalg.norm(next_q - q)
                        q = next_q
                    if not feasible:
                        continue
                if best is None or cost < best[0]:
                    best = (cost, yaw)
            if best is not None:
                self._planned_yaw = best[1]
                self._planned_pitch_deg = pitch_deg
                return self._planned_yaw
        return None

    def _reachable_waypoint(self, target, quat, q_init):
        q, pos_error, rot_error = self.sim.ik(
            target, quat, q_init=q_init, max_joint_step=None, posture_gain=0.,
        )
        return q if pos_error <= .004 and rot_error <= .06 else None

    def _placement_poses(self, center_xyz, yaw_rad, pitch_deg):
        goal, quat = self._gripper_pose(center_xyz, yaw_rad, pitch_deg)
        # A tilted or low-sitting block can touch down before its nominal
        # body center reaches the bench pose. Open above that contact point.
        return goal + _RELEASE_CLEARANCE_M * _UP, goal + self.clearance * _UP, quat

    @staticmethod
    def _transfer_route(start, destination, start_quat, end_quat):
        high = max(start[2], destination[2])
        route = []
        if start[2] < high - .002:
            start = start.copy()
            start[2] = high
            route.append(('transfer_raise', start, start_quat))
        above = destination.copy()
        above[2] = high
        route.append(('transfer_mid', (start + above) / 2, _slerp(start_quat, end_quat, .5)))
        route.append(('transfer_above', above, end_quat))
        return route

    def nominal_grip_pose(self, center_xyz, yaw_rad):
        """Grip-site position and quaternion for the planned grasp at this block center."""
        return self._gripper_pose(center_xyz, yaw_rad, self._approach_pitch(yaw_rad))

    def _approach_pitch(self, yaw_rad) -> int:
        if self._planned_yaw is not None and abs(math.remainder(
                yaw_rad - self._planned_yaw, 2 * math.pi)) < .2:
            return self._planned_pitch_deg
        return _DEFAULT_PITCH_DEG

    def _tick(self, name, target, quat, opening):
        # Near the ends of a sweep, consecutive targets can be less than
        # 0.1 mm apart. Resolve those increments so the joints ease into rest.
        q, _, _ = self.sim.ik(
            target, quat, q_init=self._qctrl, iters=35,
            max_joint_step=.08, posture_gain=0., pos_tol=1e-6, rot_tol=1e-5,
        )
        self._qctrl = q
        self.sim.set_arm_ctrl(q)
        self.sim.set_gripper(opening)
        previous_steps = round(self._ticks / (30 * self.sim.dt))
        self._ticks += 1
        steps = round(self._ticks / (30 * self.sim.dt)) - previous_steps
        self.sim.step(steps)
        if self.on_step is not None:
            self.on_step(self, name)

    def _move_samples(self, name, target, quat, opening, duration, midpoint=None):
        start, initial_quat, initial_opening = self._target, self._quat, self._opening
        if duration is None:
            duration = _move_duration(start, target, initial_quat, quat)
        ticks = max(1, math.ceil(duration * 30))
        if midpoint is not None:
            ticks += ticks % 2
        for tick in range(1, ticks + 1):
            u = _quintic_progress(tick / ticks)
            phase = midpoint if midpoint is not None and tick <= ticks // 2 else name
            yield (phase, start + u * (target - start), _slerp(initial_quat, quat, u),
                   initial_opening + u * (opening - initial_opening))

    def move(self, name, target_xyz, quat, opening, *, duration=None) -> bool:
        """Interpolate to a world-frame grip-site pose and check arm tracking."""
        target = np.asarray(target_xyz, dtype=float)
        quat = np.asarray(quat, dtype=float)
        quat = quat / np.linalg.norm(quat)
        for sample in self._move_samples(name, target, quat, opening, duration):
            self._tick(*sample)
        return self._settle(name, target, quat, opening)

    def follow_path(self, name, samples, opening, *, after_tick=None):
        """Follow (phase, position, unit quaternion) samples at 30 Hz, then settle.

        Keep the gripper at a fixed opening. after_tick(index) runs after the
        indexed sample has been applied, starting at zero.
        """
        target = None
        for index, (phase, target, quat) in enumerate(samples):
            target = np.asarray(target, dtype=float)
            quat = np.asarray(quat, dtype=float)
            self._tick(phase, target, quat, opening)
            if after_tick is not None:
                after_tick(index)
        if target is None:
            raise ValueError("A motion path needs at least one pose")
        return self._settle(name, target, quat, opening)

    def _settle(self, name, target, quat, opening):
        for _ in range(self._settling_ticks):
            self._tick(name, target, quat, opening)
        return self._check_target(name, target, quat, opening)

    def _check_target(self, name, target, quat, opening):
        self._target, self._quat, self._opening = target.copy(), quat, opening
        actual_pos, actual_quat = self.sim.ee_pose()
        pos_error = float(np.linalg.norm(actual_pos - target))
        rot_error = 2 * math.acos(float(np.clip(abs(np.dot(actual_quat, quat)), 0., 1.)))
        ok = pos_error <= .012 and rot_error <= .12
        self.events.append({
            'phase': name, 'time_s': float(self.sim.data.time),
            'target_xyz_m': target.tolist(), 'actual_xyz_m': actual_pos.tolist(),
            'position_error_m': pos_error, 'rotation_error_rad': rot_error,
            'opening_command': opening, 'finger_positions_m': self.finger_positions.tolist(),
            'ok': ok,
        })
        if not ok:
            self._target, self._quat = actual_pos, actual_quat
            self._qctrl = self.sim.q
        return ok

    @property
    def finger_positions(self) -> np.ndarray:
        """Per-finger separation from the closed position, in meters."""
        return np.abs(self.sim.data.qpos[self.sim.finger_qadr])

    def approach(self, center_xyz, yaw_rad) -> bool:
        """Move the open gripper above a caller-provided block center."""
        goal, quat = self._gripper_pose(center_xyz, yaw_rad, self._approach_pitch(yaw_rad))
        return self.move('approach', goal + self.clearance * _UP, quat, 1.)

    def pick(self, center_xyz, yaw_rad, *, refine=None) -> bool:
        """Approach, close and lift. Refine the grasp from the settled hover if supplied."""
        self._last_yaw = None
        pitch_deg = self._approach_pitch(yaw_rad)
        goal, quat = self._gripper_pose(center_xyz, yaw_rad, pitch_deg)
        above = goal + self.clearance * _UP
        if (np.linalg.norm(self._target - above) > .002
                or abs(np.dot(self._quat, quat)) < .999):
            if not self.approach(center_xyz, yaw_rad):
                return False
        if refine is not None:
            update = refine(center_xyz, yaw_rad)
            if update is None:
                return False
            center_xyz, yaw_rad = update
            goal, quat = self._gripper_pose(center_xyz, yaw_rad, pitch_deg)
            above = goal + self.clearance * _UP
            q = self._reachable_waypoint(above, quat, self.sim.q)
            if q is None or self._reachable_waypoint(goal, quat, q) is None:
                return False
            if (np.linalg.norm(self._target - above) > .002
                    or abs(np.dot(self._quat, quat)) < .999):
                if not self.move('preshape', above, quat, 1.):
                    return False
        if not self.move('descend', goal, quat, 1., duration=.8):
            return False
        return self._close_and_lift(goal, quat, above, yaw_rad, pitch_deg)

    def _close_and_lift(self, goal, quat, above, yaw_rad, pitch_deg, *,
                        close_duration=.4, lift_duration=.8):
        if not self.move('close', goal, quat, 0., duration=close_duration):
            return False
        if not self.move('lift', above, quat, 0., duration=lift_duration):
            return False
        holding = bool(np.all(self.finger_positions > .004))
        self.events.append({
            'phase': 'grip_check', 'time_s': float(self.sim.data.time),
            'finger_positions_m': self.finger_positions.tolist(), 'ok': holding,
        })
        self._last_yaw = yaw_rad if holding else None
        self._last_pitch_deg = pitch_deg
        return holding

    def _carry(self, route, observe):
        if observe is not None:
            observe('lift')
        if route[0][0] == 'transfer_raise':
            phase, target, quat = route[0]
            if not self.move(phase, target, quat, 0.):
                return False
        phase, target, quat = route[-1]
        midpoint = route[-2][0]
        # The midpoint is a camera trigger on the horizontal leg. Keep the
        # time law continuous through it, even when inference takes wall time.
        samples = list(self._move_samples(phase, target, quat, 0., None, midpoint))
        for tick, sample in enumerate(samples, 1):
            self._tick(*sample)
            if observe is not None and tick == len(samples) // 2:
                observe(midpoint)
        if not self._settle(phase, target, quat, 0.):
            return False
        if observe is not None:
            observe(phase)
        return True

    def transfer_above(self, center_xyz, yaw_rad=None, *, observe=None) -> bool:
        """Carry at the higher hover height and observe during transport.

        The optional observe(phase) callback runs at the lift, midpoint and
        destination hover. A failed waypoint ends the transfer with jaws closed.
        """
        if yaw_rad is None:
            yaw_rad = self._last_yaw
        if yaw_rad is None or self._last_yaw is None:
            return False
        _, above, quat = self._placement_poses(center_xyz, yaw_rad, self._last_pitch_deg)
        route = self._transfer_route(self._target, above, self._quat, quat)
        q = self.sim.q
        for _, waypoint, quat in route:
            q = self._reachable_waypoint(waypoint, quat, q)
            if q is None:
                return False
        return self._carry(route, observe)

    def place(self, center_xyz, yaw_rad=None, *, refine=None) -> bool:
        """Lower and release. Refine the release pose from the settled hover if supplied."""
        if yaw_rad is None:
            yaw_rad = self._last_yaw
        if yaw_rad is None:
            return False
        goal, above, quat = self._placement_poses(center_xyz, yaw_rad, self._last_pitch_deg)
        # Retain the higher transport height when a transfer already ended
        # over this destination, including when dropping onto a lower surface.
        if (np.linalg.norm(self._target[:2] - above[:2]) > .002
                or self._target[2] < above[2] - .002
                or abs(np.dot(self._quat, quat)) < .999):
            if not self.move('transfer', above, quat, 0.):
                return False
        if refine is not None:
            update = refine(center_xyz, yaw_rad)
            if update is None:
                return False
            center_xyz, yaw_rad = update
            goal, above, quat = self._placement_poses(center_xyz, yaw_rad, self._last_pitch_deg)
            above[2] = max(above[2], self._target[2])
            q = self._reachable_waypoint(above, quat, self.sim.q)
            if q is None or self._reachable_waypoint(goal, quat, q) is None:
                return False
            if (np.linalg.norm(self._target - above) > .002
                    or abs(np.dot(self._quat, quat)) < .999):
                if not self.move('preplace', above, quat, 0.):
                    return False
        if not (self.move('lower', goal, quat, 0., duration=.8)
                and self.move('release', goal, quat, 1., duration=.4)):
            return False
        self._last_yaw = None
        retreated = self.move('retreat', above, quat, 1., duration=.8)
        return retreated and bool(np.all(self.finger_positions > GRIPPER_OPEN - .004))
