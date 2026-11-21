"""The real Panthera arm as a Robot (R2). NOT YET RUN ON HARDWARE: written against the SDK calls
../panthera_teleop already uses; check every step slowly with someone at the stop button.

- Arm and gripper: the vendor SDK (`Panthera_lib.Panthera`), set up by panthera_teleop (its `launch-teleop.py`
  `prepare()` writes the per-arm configs). Joint targets go out with `Joint_Pos_Vel` under the recording's speed and
  torque limits, the gripper with `gripper_control`.
- Kinematics: the same damped least-squares IK as the sim, on a MuJoCo copy of the arm built from the SDK's URDF
  (../panthera_sim/sim/panthera), kept in sync with the measured joints. At start-up its forward kinematics are
  checked against the SDK's own (`forward_kinematics`).
- Cameras: OpenCV on the USB paths in panthera_teleop's recording.json; each camera's intrinsics and pose come from
  the R0 calibration files. The wrist camera's pose is the flange's (forward kinematics) times its hand-eye transform.
- Safety: every commanded TCP must be inside the config's `safety` box; a move that does not settle within
  `settle_timeout_s`, or ends further than `max_tracking_error` from its target, stops the arm and fails.

Runs on the rig's Linux computer (the SDK is a Linux-only wheel), in panthera_teleop's environment with this
folder's requirements added.
"""
from __future__ import annotations

import importlib.util
import json
import math
import pathlib
import sys
import time

import cv2
import numpy as np

from core.camera import Camera
from core.geometry import gripper_rotation
from robots.base import MoveResult, Observation, go_home
from robots.sim_robot import SIM_ROOT

sys.path.insert(0, str(SIM_ROOT))
from sim.panthera_env import PantheraSim, mat_to_quat  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
TELEOP = ROOT.parent / 'panthera_teleop'
START_POSITION = np.array([0.20, 0.0, 0.30])
FK_AGREEMENT_M = 0.005


def _teleop_setup():
    """panthera_teleop's launcher module (device resolution, SDK path)."""
    spec = importlib.util.spec_from_file_location('teleop_setup', TELEOP / 'app' / 'launch-teleop.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class RealRobot:
    """One real Panthera arm with its wrist camera and the centre camera."""

    def __init__(self, config: dict, calibrating: bool = False):
        """`calibrating`: open the cameras with intrinsics only (tools/calibrate_extrinsics.py finds the poses)."""
        self.cfg = config
        self.calibrating = calibrating
        self.table_z = float(config['table']['z'])
        m, g = config['motion'], config['gripper']
        self.speed, self.hz, self.max_error = float(m['speed']), float(m['control_hz']), float(m['max_tracking_error'])
        self.joint_speed = np.full(6, float(m['max_joint_speed_rad_s']))
        self.torque_limits = list(m['torque_limits_nm'])
        self.settle_timeout = float(m['settle_timeout_s'])
        self.grip = g
        self.close_s = float(config['grasp']['close_s'])
        self.start_rotation = gripper_rotation(0.0, np.deg2rad(config['grasp']['pitch']))
        self.cams = self._open_cameras(config['cameras'])
        setup = _teleop_setup()
        sys.path.insert(0, str(setup.SDK / 'scripts'))
        from Panthera_lib import Panthera
        configs = setup.prepare()
        self.arm = Panthera(str(configs[config['arm']]))
        setup.wait_for_joint_feedback(self.arm, config['arm'])
        self.kin = PantheraSim(SIM_ROOT / 'sim' / 'panthera' / 'scene.xml')
        self._check_kinematics()

    # ---------- set-up ----------
    def _open_cameras(self, cams: dict) -> dict:
        devices = json.loads((TELEOP / 'config' / 'recording.json').read_text())['cameras']
        out = {}
        for name, c in cams.items():
            path = ROOT / c['calibration']
            if not path.exists():
                raise RuntimeError(f'{name}: no calibration at {path} (R0: tools/calibrate_intrinsics.py, '
                                   'tools/calibrate_extrinsics.py)')
            cal = json.loads(path.read_text())
            if not self.calibrating and 'T_base_cam' not in cal and 'T_flange_cam' not in cal:
                raise RuntimeError(f'{name}: {path} has intrinsics but no pose (R0: tools/calibrate_extrinsics.py)')
            cap = cv2.VideoCapture(devices[c['device_key']], cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(c['width']))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(c['height']))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                raise RuntimeError(f'{name}: cannot open {devices[c["device_key"]]}')
            out[name] = (cap, cal)
        return out

    def _sync(self) -> np.ndarray:
        """Measured joints into the kinematic model; returns them."""
        q = np.asarray(self.arm.get_current_pos(), float)[:6]
        self.kin.data.qpos[self.kin.arm_qadr] = q
        import mujoco
        mujoco.mj_forward(self.kin.model, self.kin.data)
        return q

    def _check_kinematics(self) -> None:
        """The MuJoCo model's TCP must agree with the SDK's forward kinematics at the current joints."""
        q = self._sync()
        p_sdk = np.asarray(self.arm.forward_kinematics(q)['position'], float).ravel()[:3]   # as hand_guide.py reads it
        link6 = self.kin.data.xpos[self.kin._link6]
        tcp = self.kin.ee_pos()
        if min(np.linalg.norm(link6 - p_sdk), np.linalg.norm(tcp - p_sdk)) > FK_AGREEMENT_M:
            raise RuntimeError(f'kinematics disagree: SDK {p_sdk}, model flange {link6} / TCP {self.kin.ee_pos()}; '
                               'check the URDF and joint zeros before moving')

    # ---------- cameras ----------
    def camera_names(self) -> list[str]:
        """The configured cameras."""
        return list(self.cams)

    def _camera(self, name: str) -> Camera:
        cap, cal = self.cams[name]
        cam = Camera.from_dict({'T_base_cam': np.eye(4).tolist(), **cal})
        if 'T_flange_cam' in cal:                        # wrist: rides the flange
            cam.T_base_cam = self.flange_pose() @ np.array(cal['T_flange_cam'], float)
        return cam

    def observe(self, camera: str) -> Observation:
        """The newest frame (the buffered ones are dropped) with the camera's pose now."""
        cap, _ = self.cams[camera]
        for _ in range(3):
            cap.grab()
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f'{camera}: no frame')
        return Observation(self._camera(camera), cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), time.time())

    # ---------- motion ----------
    def ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """TCP from the measured joints."""
        self._sync()
        pos, quat = self.kin.ee_pose()
        import mujoco
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, quat)
        return pos, R.reshape(3, 3)

    def _safe(self, p: np.ndarray) -> str:
        s = self.cfg['safety']
        if not (s['x'][0] <= p[0] <= s['x'][1] and s['y'][0] <= p[1] <= s['y'][1]):
            return f'({p[0]:.3f}, {p[1]:.3f}) is outside the safety box'
        if not (self.table_z + s['z_floor'] <= p[2] <= s['z_max']):
            return f'z {p[2]:.3f} is outside the safety box'
        return ''

    def reachable(self, position, rotation, tolerance: float = 2e-3) -> bool:
        """Inside the safety box, and IK from the measured joints reaches the pose."""
        if self._safe(np.asarray(position, float)):
            return False
        _, pe, re = self.kin.ik(np.asarray(position, float), mat_to_quat(rotation), q_init=self._sync(), iters=200,
                                max_joint_step=None)
        return pe < tolerance and re < 0.02

    def _send(self, q: np.ndarray) -> None:
        if not self.arm.Joint_Pos_Vel(q, self.joint_speed, max_tqu=self.torque_limits, iswait=False):
            self.stop()
            raise RuntimeError('joint command rejected')

    def move_linear(self, position, rotation, speed=None) -> MoveResult:
        """Straight line at `speed`, IK each tick from the last solution; then wait for the joints to settle."""
        target = np.asarray(position, float)
        why = self._safe(target)
        if why:
            return MoveResult(False, float('nan'), why)
        start, R0 = self.ee_pose()
        q = self._sync()
        quat, q0 = mat_to_quat(rotation), mat_to_quat(R0)
        if np.dot(quat, q0) < 0:
            quat = -quat
        n = max(1, math.ceil(max(0.4, np.linalg.norm(target - start) / (speed or self.speed)) * self.hz))
        for i in range(1, n + 1):
            u = i / n
            u = u * u * (3 - 2 * u)
            qq = q0 * (1 - u) + quat * u
            p = start * (1 - u) + target * u
            q, pe, _ = self.kin.ik(p, qq / np.linalg.norm(qq), q_init=q, iters=35, max_joint_step=0.08,
                                   posture_gain=0.0)
            if pe > 0.01:
                self.stop()
                return MoveResult(False, pe, f'IK cannot follow the line ({pe * 1000:.0f} mm)')
            t0 = time.monotonic()
            self._send(q)
            time.sleep(max(0.0, 1.0 / self.hz - (time.monotonic() - t0)))
        deadline = time.monotonic() + self.settle_timeout
        while time.monotonic() < deadline:
            if np.max(np.abs(self._sync() - q)) < 0.01:
                break
            time.sleep(0.02)
        err = float(np.linalg.norm(self.ee_pose()[0] - target))
        if err > self.max_error:
            self.stop()
            return MoveResult(False, err, f'ended {err * 1000:.0f} mm from the target')
        return MoveResult(True, err)

    def set_gripper(self, opening: float) -> None:
        """Jaws to `opening` (0 closed, 1 open) under the torque limit, then wait."""
        g = self.grip
        pos = g['closed_rad'] + float(np.clip(opening, 0, 1)) * (g['open_rad'] - g['closed_rad'])
        if not self.arm.gripper_control(pos, g['speed_rad_s'], g['torque_limit_nm']):
            raise RuntimeError('gripper command rejected')
        time.sleep(self.close_s)

    def gripper_width(self) -> float:
        """Jaw gap from the gripper motor's position (linear between the calibrated ends)."""
        g = self.grip
        pos = float(self.arm.get_current_state_gripper().position)
        return max(0.0, (pos - g['closed_rad']) / (g['open_rad'] - g['closed_rad']) * g['open_width'])

    def gripper_effort(self) -> float:
        """The gripper motor's torque (N m): a second grasp check next to the width."""
        return float(self.arm.get_current_state_gripper().torque)

    def home(self) -> MoveResult:
        """Up as far as reachable, then to the start pose, jaws open."""
        return go_home(self, START_POSITION, self.start_rotation)

    def flange_pose(self) -> np.ndarray:
        """4x4 pose of link6 (the wrist camera's mount) from the measured joints."""
        self._sync()
        T = np.eye(4)
        T[:3, :3] = self.kin.data.xmat[self.kin._link6].reshape(3, 3)
        T[:3, 3] = self.kin.data.xpos[self.kin._link6]
        return T

    def stop(self) -> None:
        """Stop the arm (the SDK's stop)."""
        try:
            self.arm.set_stop()
        except Exception as e:                           # stopping must never raise past here
            print(f'stop failed: {e}', file=sys.stderr)
