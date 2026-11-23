"""A person's corrections during practice on the rig, from the terminal's keyboard (Linux / macOS).

Hold a key and the policy's action is replaced while it is held (HIL-SERL's intervention): W/S move away from /
toward the base, A/D left / right, R/F up / down, Q/E turn. Space: done. The terminal must have focus; the
keys are read without Enter. The viewer has the same keys for the sim.
"""
from __future__ import annotations

import select
import sys
import termios
import time
import tty

import numpy as np

KEYS = {'w': (0, 1), 's': (0, -1), 'a': (1, 1), 'd': (1, -1), 'r': (2, 1), 'f': (2, -1), 'q': (3, 1), 'e': (3, -1)}


class KeyboardTeleop:
    """`teleop()` -> an action while a key was pressed in the last `hold` seconds, else None."""

    def __init__(self, hold: float = 0.25):
        self.hold, self.last, self.action = hold, 0.0, None
        self._old = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    def close(self) -> None:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old)

    def __call__(self):
        while select.select([sys.stdin], [], [], 0)[0]:
            k = sys.stdin.read(1).lower()
            a = np.zeros(5)
            if k == ' ':
                a[4] = 1.0
            elif k in KEYS:
                i, v = KEYS[k]
                a[i] = v
            else:
                continue
            self.action, self.last = a, time.monotonic()
        return self.action if time.monotonic() - self.last < self.hold else None
