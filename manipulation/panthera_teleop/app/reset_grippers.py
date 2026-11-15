"""Open grippers during a return, then release them to passive control."""
import math


class ResetGrippers:
    def __init__(self, settings, limits, now):
        self.settings = settings
        self.started = now
        fraction = float(settings.get('travel_fraction', 1.0))
        if not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError('Gripper opening fraction must be between 0 and 1')
        if any(not all(math.isfinite(float(bounds[k])) for k in ('lower', 'upper'))
               or bounds['upper'] <= bounds['lower'] for bounds in limits.values()):
            raise ValueError('Invalid gripper open limits')
        self.targets = {side: float(bounds['lower']+fraction*(bounds['upper']-bounds['lower']))
                        for side, bounds in limits.items()}
        self.status = {side: 'opening' for side in limits}

    def tick(self, now, arms, released):
        commands = {}
        for side, target in self.targets.items():
            if self.status[side] != 'opening':
                continue
            if released.get(side, False):
                self.status[side] = 'released'
            elif abs(arms[side]['gripper']['position']-target) <= self.settings['arrival_tolerance_rad']:
                self.status[side] = 'open'
            elif now-self.started >= self.settings['timeout_seconds']:
                self.status[side] = 'timeout'
            else:
                commands[side] = dict(mode='position_velocity', position_rad=target,
                                      speed_limit_rad_s=self.settings['speed_rad_s'],
                                      torque_limit_nm=self.settings['torque_limit_nm'])
        return commands


def command_gripper(robot, arm, command=None):
    if command is None:
        accepted = robot.gripper_control_MIT(0, 0, 0, 0, 0)
        arm['gripper']['command'] = [0.0]*5
        arm['gripper']['control_mode'] = 'passive'
    else:
        accepted = robot.gripper_control(command['position_rad'], command['speed_limit_rad_s'],
                                         command['torque_limit_nm'])
        arm['gripper']['command'] = dict(command)
        arm['gripper']['control_mode'] = 'reset_open'
    if not accepted:
        raise RuntimeError('Gripper command rejected')
