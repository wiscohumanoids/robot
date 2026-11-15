"""Hold the arrived arm pose until an explicit handoff signal is observed."""
import numpy as np
from collections import deque


class NeutralHold:
    def __init__(self, targets, settings, now):
        self.targets = {s: np.asarray(q, dtype=float).copy() for s, q in targets.items()}
        self.settings = settings
        self.started = now
        self.ready = {s: False for s in targets}
        self.baselines = {}
        self.history = {s: deque() for s in targets}
        self.pull_since = {s: None for s in targets}
        self.diagnostics = {}

    def tick(self, now, arms):
        for side, arm in arms.items():
            if self.ready[side]:
                continue
            if np.max(np.abs(np.asarray(arm['position_rad'])-self.targets[side])) > self.settings['hold_tracking_limit_rad']:
                raise RuntimeError(f'{side}: neutral-hold tracking deviation')
            torque = np.asarray(arm['measured_torque_nm'], dtype=float)
            if torque.shape != (6,) or not np.all(np.isfinite(torque)):
                raise RuntimeError(f'{side}: invalid hold torque feedback')
            if side not in self.baselines:
                self.history[side].append(torque.copy())
                if now-self.started < self.settings['baseline_seconds']:
                    continue
                self.baselines[side] = np.median(np.asarray(self.history[side]), axis=0)
                self.history[side].clear()
            delta = np.abs(torque-self.baselines[side])
            pulling = np.any(delta >= np.asarray(self.settings['pull_torque_threshold_nm']))
            self.pull_since[side] = (self.pull_since[side] if self.pull_since[side] is not None else now) if pulling else None
            self.diagnostics[side] = dict(torque_baseline_nm=self.baselines[side].tolist(),
                                         torque_change_nm=delta.tolist())
            if self.pull_since[side] is not None and now-self.pull_since[side] >= self.settings['pull_confirm_seconds']:
                self.ready[side] = True
        return dict(self.ready)
