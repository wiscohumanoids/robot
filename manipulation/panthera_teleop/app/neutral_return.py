"""Nonblocking, measured-start joint return used only after an episode decision."""
import numpy as np
from collections import deque


class NeutralReturn:
    def __init__(self, settings, limits, requested_at):
        self.settings = settings
        self.limits = limits
        self.requested_at = requested_at
        self.started = None
        self.initial = None
        self.targets = {side: np.asarray(q, dtype=float) for side, q in settings['position_rad'].items()}
        self.duration = None
        self.last_targets = None
        self.arrival_samples = deque()
        self.diagnostics = {}
        for side, goal in self.targets.items():
            lower, upper = limits[side]
            if goal.shape != (6,) or not np.all(np.isfinite(goal)) or np.any(goal < lower) or np.any(goal > upper):
                raise ValueError(f'{side}: neutral pose is outside joint limits')

    def tick(self, now, arms):
        if now-self.requested_at < self.settings['release_delay_seconds']:
            return None, False, 'countdown'
        measured = {side: np.asarray(a['position_rad']) for side, a in arms.items()}
        if self.started is None:
            self.initial = measured
            largest = max(float(np.max(np.abs(self.targets[s]-q))) for s, q in measured.items())
            self.duration = max(self.settings.get('minimum_duration_seconds', 1.2),
                                1.875*largest/self.settings['max_joint_speed_rad_s'],
                                np.sqrt((10/np.sqrt(3))*largest /
                                        self.settings.get('max_joint_acceleration_rad_s2', 2.0)))
            self.started = now
            self.last_targets = self.initial
        # Compare against the command from the previous tick, not a future target.
        for side, q in measured.items():
            if np.max(np.abs(q-self.last_targets[side])) > self.settings['tracking_limit_rad']:
                raise RuntimeError(f'{side}: neutral-return tracking deviation; return stopped')
        progress = min(1.0, max(0.0, (now-self.started)/self.duration))
        blend = 10*progress**3-15*progress**4+6*progress**5
        commanded = {s: self.initial[s]+blend*(target-self.initial[s]) for s, target in self.targets.items()}
        self.last_targets = commanded
        window = self.settings.get('arrival_window_seconds', 0.25)
        within = all(np.max(np.abs(measured[s]-q)) < self.settings['arrival_tolerance_rad']
                     for s, q in self.targets.items())
        if progress >= 1 and within:
            self.arrival_samples.append((now, measured))
            # Retain the sample at or immediately before the beginning of the window.
            while len(self.arrival_samples) > 1 and self.arrival_samples[1][0] <= now-window:
                self.arrival_samples.popleft()
        else:
            self.arrival_samples.clear()
        span = now-self.arrival_samples[0][0] if self.arrival_samples else 0
        movement = {s: float(np.max(np.ptp(np.asarray([qs[s] for _, qs in self.arrival_samples]), axis=0)))
                    if self.arrival_samples else None for s in measured}
        self.diagnostics = dict(
            max_joint_error_rad={s: float(np.max(np.abs(q-self.targets[s]))) for s, q in measured.items()},
            reported_max_speed_rad_s={s: max(abs(v) for v in a['velocity_rad_s']) for s, a in arms.items()},
            encoder_motion_window_rad=movement, arrival_window_observed_seconds=span,
            planned_duration_seconds=self.duration)
        complete = within and span >= window and all(
            value is not None and value <= self.settings.get('arrival_stability_rad', 0.004)
            for value in movement.values())
        if not complete and now-self.started > self.duration+self.settings['settle_timeout_seconds']:
            detail = '; '.join(f"{s}: joint error {self.diagnostics['max_joint_error_rad'][s]:.4f} rad, "
                               f"encoder motion {movement[s]} rad" for s in measured)
            raise RuntimeError('Neutral pose did not settle within timeout: '+detail)
        return commanded, complete, 'moving'
