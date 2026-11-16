"""Independent takeover by handle load or passive gripper motion."""
import numpy as np

from neutral_hold import NeutralHold


class TakeoverHold(NeutralHold):
    def __init__(self, targets, settings, recording_settings, now):
        super().__init__(targets, settings, now)
        self.motion_settings = {key: recording_settings[key] for key in
            ('motion_gripper_rad', 'motion_velocity_rad_s', 'motion_confirm_seconds')}
        self.gripper_baselines = {}
        self.gripper_history = {side: [] for side in targets}
        self.gripper_since = {side: None for side in targets}

    def tick(self, now, arms):
        previous = dict(self.ready)
        super().tick(now, arms)
        for side, arm in arms.items():
            if previous[side]:
                continue
            gripper = arm['gripper']
            position, velocity = float(gripper['position']), float(gripper['velocity'])
            if not np.isfinite([position, velocity]).all():
                raise RuntimeError(f'{side}: invalid takeover gripper feedback')
            if side not in self.gripper_baselines:
                self.gripper_history[side].append(position)
                if now-self.started < self.settings['baseline_seconds']:
                    continue
                self.gripper_baselines[side] = float(np.median(self.gripper_history[side]))
                self.gripper_history[side].clear()
            if self.ready[side]:
                self.diagnostics[side]['trigger'] = 'handle_pull'
                continue
            change = abs(position-self.gripper_baselines[side])
            moving = (change > self.motion_settings['motion_gripper_rad'] or
                      abs(velocity) > self.motion_settings['motion_velocity_rad_s'])
            self.gripper_since[side] = ((now if self.gripper_since[side] is None else
                                        self.gripper_since[side]) if moving else None)
            self.diagnostics[side].update(gripper_baseline_rad=self.gripper_baselines[side],
                gripper_position_rad=position, gripper_change_rad=change,
                gripper_velocity_rad_s=velocity)
            if (self.gripper_since[side] is not None and
                    now-self.gripper_since[side] >= self.motion_settings['motion_confirm_seconds']):
                self.ready[side] = True
                self.diagnostics[side]['trigger'] = 'gripper_motion'
        return dict(self.ready)
