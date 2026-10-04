"""Test doubles shared by the behavior_tree tests (no ROS)."""
from behavior_tree.skills import Outcome


class FakeCtx:
    """Records calls; returns the scripted outcomes in order (the last one repeats)."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes) or [Outcome(True)]
        self.calls = []

    def _next(self):
        return self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]

    def navigate(self, pose, timeout_s):
        self.calls.append(('navigate', pose, timeout_s))
        return self._next()

    def manipulate(self, task, object_id, pose, timeout_s):
        self.calls.append(('manipulate', task, object_id, pose, timeout_s))
        return self._next()
