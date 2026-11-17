"""Message timestamps from the board's clock. No ROS.

Stamping each sample with its arrival time adds the USB adapter's and the host's jitter (several ms) to every
message. The board stamps every HI91 frame with its own millisecond clock (`system_time_ms`, uptime, wrapping at
24 h). As HiPNUC's ROS driver does, the stamp is the board time carried onto the host clock along the frames that
arrived quickest: offset = min over a recent window of (arrival - board time), so stamps keep the board's spacing,
follow slow drift between the clocks, and are never later than the arrival.
"""

from collections import deque

WRAP_S = 86400.0          # system_time_ms wraps at 24 h


class DeviceClock:
    """Maps board time (ms) to host time (s)."""

    def __init__(self, window_s=5.0, reset_jump_s=1.0):
        """`window_s`: how far back the quickest arrival is looked for; a board-time jump larger than
        `reset_jump_s` (board reboot) starts over.
        """
        self.window_s, self.reset_jump_s = window_s, reset_jump_s
        self.offsets = deque()                # (arrival, arrival - board time)
        self.last_board = None
        self.wraps = 0
        self.resets = 0

    def _unwrap(self, board_s):
        """Board time continued across the 24 h wrap."""
        t = board_s + self.wraps * WRAP_S
        if self.last_board is not None and t < self.last_board - WRAP_S / 2:
            self.wraps += 1
            t += WRAP_S
        return t

    def stamp(self, board_ms, arrival_s):
        """Host-clock stamp (s) for a frame with board time `board_ms` that arrived at `arrival_s`."""
        t = self._unwrap(board_ms / 1000.0)
        if self.last_board is not None and not (0.0 <= t - self.last_board <= self.reset_jump_s):
            self.offsets.clear()              # reboot or a long gap: the old offset no longer holds
            self.wraps = 0
            t = board_ms / 1000.0
            self.resets += 1
        self.last_board = t
        self.offsets.append((arrival_s, arrival_s - t))
        while self.offsets and arrival_s - self.offsets[0][0] > self.window_s:
            self.offsets.popleft()
        offset = min(o for _, o in self.offsets)
        return min(t + offset, arrival_s)
