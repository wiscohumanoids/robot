"""The supervisor of practice: who demonstrates, corrects, judges and resets.

HIL-SERL's human does four jobs: gives 20-30 demonstrations, takes over when the policy is about to fail, labels
success frames for the reward classifier, and resets the scene. Here each job is filled by the cheapest thing that
can do it well:

| job          | SimSupervisor (sim)                          | AgentSupervisor (rig, or sim without privileges)    |
|--------------|----------------------------------------------|-----------------------------------------------------|
| demonstrate  | privileged expert (true poses)               | a person on the teleop device (panthera_teleop), or |
|              |                                              | the scripted search's successes                     |
| correct      | expert takes over when the error is large    | watchdog: back off and retry when the object stops  |
|              |                                              | going down (an abort, not a steer: Claude is far    |
|              |                                              | too slow to steer at 10 Hz); a person can take over |
| judge        | the simulator's state                        | the geometric check (depth from the arm's pose),   |
|              |                                              | Claude's look at the last frames (learning/label.py)|
|              |                                              | for labels, never as the RL reward itself           |
| reset        | restore the snapshot                         | the agent itself: Claude + pick/place put the scene |
|              |                                              | back (an instruction to the agent loop)             |

Claude labels and filters episodes; the RL reward is the trained classifier or the geometric check. VLM judges used
directly as rewards get exploited (they fire on "near the object" or the wrong object), per the research notes.
"""
from __future__ import annotations

import numpy as np


class SimSupervisor:
    """The sim's supervisor: privileged."""

    name = 'sim'

    def __init__(self, xy_limit: float = 0.004, yaw_limit_deg: float = 6.0):
        self.xy_limit, self.yaw_limit = xy_limit, np.deg2rad(yaw_limit_deg)

    def demo(self, env) -> np.ndarray:
        return env.expert()

    def intervene(self, env, action) -> np.ndarray | None:
        """The expert's action when the policy has drifted too far (HIL-SERL: take over before it fails), else
        None. Kept short: a long intervention that leads to success makes the value function over-optimistic."""
        xy, dyaw, depth = env.errors()
        if depth < 0.002 and (np.linalg.norm(xy) > self.xy_limit or abs(dyaw) > self.yaw_limit):
            return env.expert()
        return None

    def judge(self, env) -> bool:
        return bool(env.success())

    def reset(self, env) -> dict:
        return env.reset()


class AgentSupervisor:
    """On the rig (or the sim without privileged information): watchdog corrections, geometric judging, the
    agent's own resets, and Claude's labels afterwards."""

    name = 'agent'

    def __init__(self, skills=None, planner=None, reset_instruction: str | None = None, stall_steps: int = 6,
                 teleop=None):
        self.skills, self.planner = skills, planner
        self.reset_instruction = reset_instruction
        self.stall_steps = stall_steps
        self.teleop = teleop             # callable () -> action or None: a person's correction, when connected
        self._z = []

    def demo(self, env) -> np.ndarray:
        if self.teleop is not None:
            a = self.teleop()
            if a is not None:
                return np.asarray(a, float)
        return env.scripted_search()

    def intervene(self, env, action) -> np.ndarray | None:
        if self.teleop is not None:
            a = self.teleop()
            if a is not None:
                return np.asarray(a, float)
        z = float(env.observe()['state'][2])
        self._z.append(z)
        if len(self._z) > self.stall_steps and self._z[-self.stall_steps] - z < 0.0005 and action[2] < 0:
            self._z.clear()
            return np.array([0.0, 0.0, 1.0, 0.0, 0.0])       # stuck pressing: back off (the policy then retries)
        return None

    def judge(self, env) -> bool:
        return bool(env.success())

    def reset(self, env) -> dict:
        """Put the scene back with the agent itself, then hand over again."""
        if self.reset_instruction and self.planner is not None and self.skills is not None:
            import pathlib
            import time
            from llm.loop import AgentRun
            AgentRun(self.skills, self.planner, self.skills.cfg,
                     pathlib.Path('runs') / f'reset_{time.strftime("%Y%m%d_%H%M%S")}').run(self.reset_instruction)
        self._z.clear()
        return env.reset()
