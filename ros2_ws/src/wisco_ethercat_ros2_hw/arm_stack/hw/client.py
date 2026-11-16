"""The API every procedure talks to. The sim backend and the ROS 2 client both
implement it, so `arm <stage>` runs the same code with or without --sim.

Modes (spec: Hardware assumptions):
  'P'  CSP: the drive closes the position loop; we send q target and a torque
       feedforward (0x60B2) of gravity + friction. Stiff and safe.
  'T'  CST: the host law tau = Kp(q_d - q) + Kd(qd_d - qd) + g(q) + f(qd) + tau_ff
       runs in the RT loop. Compliant; needed for gravity comp and teaching.
Switching is bumpless: T starts from the measured torque, P from the actual
position.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


@dataclass
class JointState:
    t: float                      # monotonic seconds
    q: np.ndarray                 # rad, calibrated
    qd: np.ndarray                # rad/s
    tau: np.ndarray               # Nm, measured (from 0x6077 via Kt)
    raw_counts: np.ndarray        # 0x6064
    raw_permille: np.ndarray      # 0x6077
    status: np.ndarray            # 0x6041
    mode: str = 'P'
    fallback: bool = False        # watchdog fired: gravity comp + damping
    # torque the RT loop sent this cycle (Mode T: 0x6071 target; Mode P: 0x60B2 offset)
    tau_cmd: Optional[np.ndarray] = None


@dataclass
class Command:
    mode: str = 'P'
    q: Optional[np.ndarray] = None
    qd: Optional[np.ndarray] = None
    kp: Optional[np.ndarray] = None
    kd: Optional[np.ndarray] = None
    tau_ff: Optional[np.ndarray] = None
    gravity_ff: bool = True       # add model gravity (+ friction) in the RT loop
    torque_clamp: Optional[np.ndarray] = None   # Nm, per joint (0x6072 equivalent)
    # Stage 3 only: let Mode P targets pass the soft limits to find the hard
    # stops (the torque clamp still applies). Logged as an event.
    ignore_soft_limits: bool = False
    # scales the friction model in the feedforward (mocap-repeat tunes it)
    friction_ff_gain: float = 1.0
    # (kg, x, y, z): point mass at the flange, in the tool0 frame. The RT loop
    # keeps the last one sent and adds its gravity, watchdog fallback included;
    # None = leave it as it is, mass 0 = clear it.
    payload: Optional[Sequence[float]] = None


@dataclass
class Payload:
    """A mass hung from a link, for the lever (stage 4) and stiffness tests."""
    link: str
    mass_kg: float
    offset_m: Sequence[float] = field(default_factory=lambda: (0.0, 0.0, 0.0))


class ArmClient(abc.ABC):
    """Synchronous view of the RT loop. `step` advances by one or more cycles on
    the sim and sleeps/streams on hardware."""

    n: int
    dt: float
    names: Sequence[str]

    @abc.abstractmethod
    def state(self) -> JointState: ...

    @abc.abstractmethod
    def send(self, cmd: Command) -> None: ...

    @abc.abstractmethod
    def step(self, seconds: float = 0.0) -> JointState:
        """Let the loop run for `seconds` (at least one cycle) with the last command."""

    def hold(self, seconds: float, mode: str = 'P') -> JointState:
        """Hold the current position, re-sending every cycle so the watchdog stays fed."""
        s = self.state()
        q = s.q.copy()
        for _ in range(max(1, int(round(seconds / self.cmd_dt)))):
            self.send(Command(mode=mode, q=q))
            s = self.step(self.cmd_dt)
        return s

    # Optional capabilities; the sim implements all of them.
    def set_payload(self, payload: Optional[Payload]) -> None:
        raise NotImplementedError('payloads are mounted by hand on hardware')

    def place_in_jig(self, q_jig: np.ndarray) -> None:
        raise NotImplementedError('put the arm in the zeroing jig by hand on hardware')

    def mocap(self):
        """Flange marker-cluster pose source, or None."""
        return None

    def hand_guide(self, prompt: str, hint_q: Optional[np.ndarray] = None) -> JointState:
        """Hold the arm in gravity comp (Mode T, no stiffness) while the operator
        moves it, and return the state when they confirm. `hint_q` is where the
        procedure would like the arm (sim uses it; hardware prints it)."""
        raise NotImplementedError

    def drive_gains(self):
        """The drives' CSP gains as (kp Nm/rad, kd Nm s/rad) arrays, or None if unknown."""
        return None

    def set_drive_gains(self, kp_scale: np.ndarray, kd_scale: np.ndarray) -> None:
        """Scale the drives' own CSP gains (SDO). Hardware support is unconfirmed."""
        raise NotImplementedError('CSP gains are not settable over SDO on this drive (unconfirmed)')

    def time_since_power_on(self) -> float:
        return float('nan')

    def close(self) -> None:
        pass
