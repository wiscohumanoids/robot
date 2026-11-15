"""Dial-indicator cube: ISO 9283 repeatability without mocap (spec: Validation
and repeatability, "three dial indicators on a cube fixture").

A ground cube is bolted to the flange. Three dial indicators on a stand, with
orthogonal plunger axes a_1..a_3 (rows of A, in base_link), touch three faces
of the cube with the arm at one ISO pose. The arm cycles from home back to that
pose; each indicator reads its face's displacement along its axis, so for a
translation dp of the cube

    r = A dp        and        dp = A^-1 r

(A is orthonormal for a square stand, so A^-1 = A^T; a skewed stand still
inverts). A face is flat, so motion across an indicator's axis does not change
its reading, and small rotations of the cube (repeatability-sized, under 1e-4
rad) move a contact point by microns: the cube measures translation only. One
pose per fixture setup: the stand is set and the indicators zeroed (at mid
travel) at a pose, all cycles are run there, then the stand moves to the next.

Sources:
  SimIndicatorCube       true cube-centre position on the three axes, with
                         1 um resolution (and sub-um noise).
  OperatorIndicatorCube  hardware: the operator types the three readings (mm)
                         each cycle while the arm holds the pose; or a reader
                         hook for digital indicators (SerialIndicatorReader,
                         e.g. Mitutoyo USB input tools, stubbed).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from .client import Command

RESOLUTION_M = 1e-6


def readings_to_position(readings: Sequence[float], axes: np.ndarray) -> np.ndarray:
    """Cube displacement (base frame, m) from the three readings (m): solves A dp = r."""
    return np.linalg.solve(np.asarray(axes, float), np.asarray(readings, float))


def position_to_readings(dp: Sequence[float], axes: np.ndarray) -> np.ndarray:
    """Readings (m) for a cube displacement dp (base frame, m)."""
    return np.asarray(axes, float) @ np.asarray(dp, float)


def quantize(r: np.ndarray, resolution: float) -> np.ndarray:
    return np.round(np.asarray(r, float) / resolution) * resolution


@dataclass
class CubeConfig:
    axes: np.ndarray = field(default_factory=lambda: np.eye(3))   # rows: indicator axes in base_link
    cube_size_m: float = 0.04
    center_in_flange_m: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.025]))
    resolution_m: float = RESOLUTION_M
    range_m: float = 0.0127
    serial_port: Optional[str] = None
    serial_baud: int = 9600

    @staticmethod
    def from_config(raw: dict) -> 'CubeConfig':
        c = raw.get('indicator_cube') or {}
        d = CubeConfig()
        axes = np.array(c.get('axes_in_base', d.axes), float)
        axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)
        if abs(np.linalg.det(axes)) < 0.5:
            raise ValueError('indicator_cube.axes_in_base: the three axes must be close to orthogonal')
        ser = c.get('serial') or {}
        return CubeConfig(axes=axes, cube_size_m=float(c.get('cube_size_m', d.cube_size_m)),
                          center_in_flange_m=np.array(c.get('center_in_flange_m', d.center_in_flange_m), float),
                          resolution_m=float(c.get('resolution_m', d.resolution_m)),
                          range_m=float(c.get('range_m', d.range_m)),
                          serial_port=ser.get('port'), serial_baud=int(ser.get('baud', 9600)))


class IndicatorCube:
    """Position source for mocap_repeat.iso_run: one pose per fixture setup."""

    one_pose_per_setup = True
    name = 'indicator cube'

    def __init__(self, cfg: CubeConfig):
        self.cfg = cfg
        self.pose = None

    def setup(self, ctx, k: int, q) -> None:
        """The arm holds ISO pose k (q); set the stand and zero the indicators."""
        raise NotImplementedError

    def read(self, client, q_hold) -> np.ndarray:
        """The three readings, m, with the arm holding q_hold."""
        raise NotImplementedError

    def position(self, client, q_hold) -> np.ndarray:
        r = np.asarray(self.read(client, q_hold), float)
        half = 0.5 * self.cfg.range_m
        if np.any(np.abs(r) > half):
            raise RuntimeError('indicator reading %s mm is past half the %.1f mm travel: the cube is off '
                               'the plungers' % (np.round(r * 1e3, 3).tolist(), self.cfg.range_m * 1e3))
        return readings_to_position(r, self.cfg.axes)

    def static_noise_3sigma_m(self) -> float:
        """Reading noise; a digital indicator's is its last digit."""
        return self.cfg.resolution_m


class SimIndicatorCube(IndicatorCube):
    """Sim: the true cube centre (link angles, so compliance and backlash show)
    on the three axes, zeroed at setup, 1 um resolution."""

    name = 'indicator cube (sim)'

    def __init__(self, sim, cfg: Optional[CubeConfig] = None, noise_m: float = 0.3e-6, seed: int = 4321):
        super().__init__(cfg or CubeConfig())
        self.sim = sim
        self.noise_m = noise_m
        self.rng = np.random.default_rng(seed)
        self.zero = np.zeros(3)
        frame = 'flange' if sim.truth.model.existFrame('flange') else sim.truth.tool_frame
        self.frame = frame

    def true_center(self) -> np.ndarray:
        M = self.sim.truth.fk(self.sim.true_joint_angles(), self.frame)
        return M.act(self.cfg.center_in_flange_m)

    def setup(self, ctx, k, q) -> None:
        self.pose = k
        self.zero = position_to_readings(self.true_center(), self.cfg.axes)

    def read(self, client, q_hold) -> np.ndarray:
        r = position_to_readings(self.true_center(), self.cfg.axes) - self.zero
        if self.noise_m:
            r = r + self.rng.normal(0, self.noise_m, 3)
        return quantize(r, self.cfg.resolution_m)

    def static_noise_3sigma_m(self, samples: int = 200) -> float:
        r = np.array([self.read(None, None) for _ in range(samples)])
        return float(3 * np.max(r.std(axis=0)))


def hold_while(client, q_hold, fn: Callable):
    """Run fn() (an input() prompt, a serial read) while a thread keeps sending
    the Mode P hold, so the 50 ms watchdog stays fed."""
    done = threading.Event()
    q_hold = np.asarray(q_hold, float)

    def feed():
        while not done.is_set():
            client.send(Command(mode='P', q=q_hold))
            time.sleep(0.01)
    t = threading.Thread(target=feed, daemon=True)
    t.start()
    try:
        return fn()
    finally:
        done.set()
        t.join()


def parse_readings_mm(text: str) -> np.ndarray:
    vals = [float(v) for v in text.replace(',', ' ').split()]
    if len(vals) != 3:
        raise ValueError('want three readings, got %d' % len(vals))
    return np.array(vals) * 1e-3


class SerialIndicatorReader:
    """Hook for digital indicators on a USB/serial interface (e.g. Mitutoyo USB
    input tool / U-WAVE, one request returns the three channels). STUB: the
    request and the line format depend on the interface; fill in `request`
    and `parse` once it is on the bench. Returns readings in mm."""

    def __init__(self, port: str, baud: int = 9600, timeout_s: float = 1.0):
        self.port, self.baud, self.timeout_s = port, baud, timeout_s
        self._ser = None

    def open(self):
        try:
            import serial  # pyserial, optional
        except ImportError as e:
            raise RuntimeError('digital indicators need pyserial (pip install pyserial)') from e
        self._ser = serial.Serial(self.port, self.baud, timeout=self.timeout_s)

    def request(self) -> bytes:
        raise NotImplementedError('SerialIndicatorReader.request: send the interface\'s read command '
                                  '(depends on the Mitutoyo interface; not on the bench yet)')

    def parse(self, line: bytes) -> np.ndarray:
        raise NotImplementedError('SerialIndicatorReader.parse: three readings in mm from one reply')

    def __call__(self) -> np.ndarray:
        if self._ser is None:
            self.open()
        self._ser.write(self.request())
        return self.parse(self._ser.readline())


class OperatorIndicatorCube(IndicatorCube):
    """Hardware: the operator sets the stand per pose and types the three
    readings (mm, `a b c`) each cycle; or `reader()` -> 3 readings in mm."""

    def __init__(self, cfg: CubeConfig, reader: Optional[Callable[[], Sequence[float]]] = None,
                 ask: Callable[[str], str] = input):
        super().__init__(cfg)
        self.reader = reader
        self.ask = ask
        self.cycle = 0

    def setup(self, ctx, k, q) -> None:
        self.pose = k
        self.cycle = 0
        hold_while(ctx.client, q, lambda: self.ask(
            'ISO pose %d: set the indicator stand so the three plungers touch the cube faces at mid travel '
            '(axes %s in base_link), zero all three, then press Enter '
            % (k + 1, np.round(self.cfg.axes, 3).tolist())))

    def read(self, client, q_hold) -> np.ndarray:
        self.cycle += 1
        if self.reader is not None:
            return np.asarray(hold_while(client, q_hold, self.reader), float) * 1e-3

        def typed():
            while True:
                text = self.ask('pose %d cycle %d: readings x y z (mm) ' % (self.pose + 1, self.cycle))
                try:
                    return parse_readings_mm(text)
                except ValueError as e:
                    print('  %s; type three numbers, e.g. 0.003 -0.001 0.000' % e)
        return hold_while(client, q_hold, typed)


def for_context(ctx) -> IndicatorCube:
    """The cube source for this stage context: sim truth, a serial reader, or typed readings."""
    cfg = CubeConfig.from_config(ctx.arm.raw)
    if ctx.sim:
        return SimIndicatorCube(ctx.client, cfg)
    reader = SerialIndicatorReader(cfg.serial_port, cfg.serial_baud) if cfg.serial_port else None
    return OperatorIndicatorCube(cfg, reader)
