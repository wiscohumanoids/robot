"""Localize two resting 2×1 Mega Bloks from calibrated physical RGB cameras."""

import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

SIM_ROOT = Path(__file__).resolve().parents[2] / 'panthera_sim'
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

from perception.pickup import DinoPickupPerception
from sim.mega_bloks import BODY_HEIGHT_M, STUD_HEIGHT_M
from tools.dino_pick_camera import CameraCalibration
from tools.dino_stack_track import BlockMap, StackFrame


def _array(value, shape, name):
    data = np.asarray(value, dtype=float)
    if data.shape != shape or not np.isfinite(data).all():
        raise ValueError(f'{name} must contain finite values with shape {shape}')
    return data


class _Camera:
    def __init__(self, config):
        size = _array(config['resolution'], (2,), 'resolution')
        if np.any(size < 1) or not np.array_equal(size, size.astype(int)):
            raise ValueError('Camera resolution must contain positive integers')
        self.width, self.height = size.astype(int)
        self.matrix = _array(config['camera_matrix'], (3, 3), 'camera_matrix')
        fx, fy = self.matrix[0, 0], self.matrix[1, 1]
        cx, cy = self.matrix[:2, 2]
        if (min(fx, fy) <= 0 or not (0 <= cx < self.width and 0 <= cy < self.height)
                or not np.allclose(self.matrix, [[fx, 0, cx], [0, fy, cy], [0, 0, 1]])):
            raise ValueError('Expected OpenCV camera intrinsics for the capture resolution')
        self.distortion = np.asarray(config['distortion'], dtype=float)
        if (self.distortion.shape not in ((4,), (5,), (8,), (12,), (14,))
                or not np.isfinite(self.distortion).all()):
            raise ValueError('Expected a vector of OpenCV lens distortion coefficients')
        pose = _array(config['world_from_camera'], (4, 4), 'world_from_camera')
        rotation = pose[:3, :3]
        if not (np.allclose(pose[3], [0, 0, 0, 1]) and
                np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) and
                np.isclose(np.linalg.det(rotation), 1, atol=1e-5)):
            raise ValueError('world_from_camera must be a rigid camera-to-world transform')
        self.camera = CameraCalibration(
            pose[:3, 3], rotation[:, 0], -rotation[:, 1], rotation[:, 2], fx, np.array([cx, cy])
        )

    def _rectify(self, pixels):
        points = np.asarray(pixels, dtype=float).reshape(-1, 1, 2)
        return cv2.undistortPoints(points, self.matrix, self.distortion,
                                   P=self.camera.matrix).reshape(-1, 2)

    def observe(self, rgb, detector):
        if rgb.shape != (self.height, self.width, 3) or rgb.dtype != np.uint8:
            raise ValueError(f'Expected a {self.width}×{self.height} uint8 RGB camera frame')
        raw, rectified = [], []
        for block in detector.detect(rgb):
            x0, y0, x1, y1 = block.bbox_xyxy
            if (not block.color or not 2 <= len(block.studs_xy) <= 8 or x0 <= 0 or y0 <= 0
                    or x1 >= self.width or y1 >= self.height):
                continue
            corners = self._rectify([[x0, y0], [x0, y1], [x1, y0], [x1, y1]])
            studs = self._rectify(block.studs_xy)
            if not np.isfinite(corners).all() or not np.isfinite(studs).all():
                continue
            raw.append(block)
            rectified.append(replace(block, xy=tuple(studs.mean(axis=0)),
                                     bbox_xyxy=(*corners.min(axis=0), *corners.max(axis=0)),
                                     studs_xy=tuple(map(tuple, studs))))
        return raw, StackFrame(self.camera, tuple(rectified))


class StackVision:
    """Measure both blocks in the calibrated IK world from a stationary survey pose."""

    def __init__(self, config):
        self.table_z = float(_array(config['table_z_m'], (), 'table_z_m'))
        auto = config.get('auto_calibrate', False)
        if not isinstance(auto, bool):
            raise ValueError('auto_calibrate must be true or false')
        self.calibrator = None
        self.verified = False
        if not auto:
            self.overhead = _Camera(config['overhead'])
            self.wrist = _Camera(config['wrist'])
            self.reference = _array(config['wrist']['reference_q_rad'], (6,), 'wrist reference_q_rad')
        self.detector = DinoPickupPerception(config['checkpoint'],
                                             landmark_checkpoint=config.get('landmark_checkpoint'))
        if auto:
            from stack_autocal import StackAutoCalibration
            self.calibrator = StackAutoCalibration(config, self.detector)

    def calibrate(self, overhead_rgb, wrist_rgb, q, gripper, *, finish=False):
        """Collect a fit sample, or verify the existing mounts on a held-out frame pair."""
        if self.calibrator is None:
            raise ValueError('This connection uses fixed camera calibration')
        self.verified = False
        measured = _array(q, (6,), 'arm joint feedback')
        opening = float(_array(gripper, (), 'gripper encoder'))
        if not finish:
            return self.calibrator.sample(overhead_rgb, wrist_rgb, measured, opening)
        report = self.calibrator.verify(overhead_rgb, wrist_rgb, measured, opening)
        if report['ready']:
            cameras = self.calibrator.camera_configs(measured)
            # The dashboard projects its path into the raw overhead camera image.
            # Validate the exported intrinsics and optical-frame convention before Run.
            for name in ('overhead', 'wrist'):
                _Camera(cameras[name])
            report = dict(report, cameras=cameras)
            self.verified = True
        return report

    def close(self):
        if self.calibrator is not None:
            self.calibrator.close()

    def locate(self, overhead_rgb, wrist_rgb, wrist_q_rad):
        """Return left-to-right detections with raw overhead boxes and metric body/stud positions."""
        measured = _array(wrist_q_rad, (6,), 'wrist joint feedback')
        if self.calibrator is None:
            if np.max(np.abs(measured - self.reference)) > .01:
                raise ValueError('Wrist arm is outside its calibrated survey pose (0.01 rad)')
            top_camera, wrist_camera = self.overhead, self.wrist
        else:
            if not self.verified:
                raise ValueError('Finish and verify both camera mounts before locating blocks')
            cameras = self.calibrator.camera_configs(measured)
            top_camera, wrist_camera = _Camera(cameras['overhead']), _Camera(cameras['wrist'])
        raw, overhead = top_camera.observe(overhead_rgb, self.detector)
        _, wrist = wrist_camera.observe(wrist_rgb, self.detector)
        if len(raw) != 2 or len(wrist.detections) != 2:
            raise ValueError('Both cameras must see two complete colored two-stud rows')
        poses = BlockMap(overhead).observe(overhead, [wrist])
        if (set(poses) != {0, 1} or poses[0].detection is poses[1].detection
                or any(pose.studs_xyz.shape != (2, 3) or not np.isfinite(pose.studs_xyz).all()
                       for pose in poses.values())):
            raise ValueError('Could not match both stud rows unambiguously across the two cameras')
        results = []
        for index, block in enumerate(raw):
            pose = poses[index]
            center = pose.xyz - [0., 0., (BODY_HEIGHT_M + STUD_HEIGHT_M) / 2]
            if abs(center[2] - (self.table_z + BODY_HEIGHT_M / 2)) > .008:
                raise ValueError('Measured block height disagrees with a block resting on the calibrated table')
            results.append(dict(color=block.color, bbox_xyxy=list(map(int, block.bbox_xyxy)),
                                center_xyz=center.tolist(), studs_xyz=pose.studs_xyz.tolist()))
        return sorted(results, key=lambda block: sum(block['bbox_xyxy'][::2]))
