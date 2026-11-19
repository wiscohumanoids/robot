"""A calibrated camera: intrinsics, lens distortion and its pose in the robot's base frame.

Conventions (OpenCV's): the camera looks down its +z, x right and y down the image. `T_base_cam` maps camera
coordinates to the robot base frame (the sim's world frame). Pixels are (u, v) = (column, row), with (0, 0) the
centre of the top-left pixel.

MuJoCo cameras look down their -z with +y up the image, so a MuJoCo camera's frame is this one turned half a turn
about x (`from_mujoco`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# MuJoCo camera axes -> OpenCV camera axes (x stays, y and z flip)
_MJ_TO_CV = np.diag([1.0, -1.0, -1.0])


@dataclass
class Camera:
    """Pinhole camera with optional distortion (OpenCV `pinhole` or `fisheye` model) and pose."""
    name: str
    width: int
    height: int
    K: np.ndarray                                   # 3x3 intrinsics
    T_base_cam: np.ndarray                          # 4x4, camera -> base
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5))
    model: str = 'pinhole'                          # 'pinhole' (k1 k2 p1 p2 k3) or 'fisheye' (k1..k4)

    @staticmethod
    def from_mujoco(model, data, name: str, width: int, height: int) -> 'Camera':
        """The ideal camera MuJoCo renders: square pixels from fovy, pose from the current `data`."""
        import mujoco
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if cid < 0:
            raise KeyError(f'no camera named {name!r}')
        fy = 0.5 * height / np.tan(np.deg2rad(model.cam_fovy[cid]) / 2)
        K = np.array([[fy, 0, (width - 1) / 2], [0, fy, (height - 1) / 2], [0, 0, 1.0]])
        T = np.eye(4)
        T[:3, :3] = data.cam_xmat[cid].reshape(3, 3) @ _MJ_TO_CV
        T[:3, 3] = data.cam_xpos[cid]
        return Camera(name, width, height, K, T)

    @property
    def position(self) -> np.ndarray:
        """Optical centre in the base frame."""
        return self.T_base_cam[:3, 3].copy()

    def undistort(self, uv: np.ndarray) -> np.ndarray:
        """(N, 2) pixels -> (N, 2) normalised image coordinates (x/z, y/z), distortion removed."""
        uv = np.asarray(uv, float).reshape(-1, 1, 2)
        if self.model == 'fisheye':
            out = cv2.fisheye.undistortPoints(uv, self.K, self.dist[:4])
        elif np.any(self.dist):
            out = cv2.undistortPoints(uv, self.K, self.dist)
        else:
            Kinv = np.linalg.inv(self.K)
            h = np.concatenate([uv.reshape(-1, 2), np.ones((uv.shape[0], 1))], axis=1) @ Kinv.T
            return h[:, :2] / h[:, 2:3]
        return out.reshape(-1, 2)

    def rays(self, uv: np.ndarray) -> np.ndarray:
        """(N, 2) pixels -> (N, 3) unit ray directions in the base frame (from `position`)."""
        xy = self.undistort(uv)
        d = np.concatenate([xy, np.ones((xy.shape[0], 1))], axis=1) @ self.T_base_cam[:3, :3].T
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def project(self, points: np.ndarray) -> np.ndarray:
        """(N, 3) base-frame points -> (N, 2) pixels (NaN behind the camera)."""
        p = np.asarray(points, float).reshape(-1, 3)
        T_cam_base = np.linalg.inv(self.T_base_cam)
        pc = p @ T_cam_base[:3, :3].T + T_cam_base[:3, 3]
        rvec = np.zeros(3)
        tvec = np.zeros(3)
        if self.model == 'fisheye':
            uv = cv2.fisheye.projectPoints(pc.reshape(-1, 1, 3), rvec, tvec, self.K, self.dist[:4])[0]
        else:
            uv = cv2.projectPoints(pc, rvec, tvec, self.K, self.dist)[0]
        uv = uv.reshape(-1, 2)
        uv[pc[:, 2] <= 0] = np.nan
        return uv

    def to_dict(self) -> dict:
        """Plain-data form, for calibration files and logs."""
        return dict(name=self.name, width=self.width, height=self.height, K=self.K.tolist(),
                    dist=np.asarray(self.dist).tolist(), model=self.model, T_base_cam=self.T_base_cam.tolist())

    @staticmethod
    def from_dict(d: dict) -> 'Camera':
        """Inverse of `to_dict`."""
        return Camera(d['name'], int(d['width']), int(d['height']), np.array(d['K'], float),
                      np.array(d['T_base_cam'], float), np.array(d.get('dist', np.zeros(5)), float),
                      d.get('model', 'pinhole'))


def pose_error(translation_m: float, rotation_deg: float, rng: np.random.Generator) -> np.ndarray:
    """A random rigid transform (4x4) moving by `translation_m` and turning by `rotation_deg` about a random axis.

    Composed onto a camera's pose (`T_base_cam @ E`) it stands in for extrinsic / hand-eye calibration error (S3):
    the image comes from the true pose, the estimates are made with the wrong one.
    """
    axis = rng.normal(size=3)
    shift = rng.normal(size=3)
    E = np.eye(4)
    E[:3, :3] = cv2.Rodrigues(axis / np.linalg.norm(axis) * np.deg2rad(rotation_deg))[0]
    E[:3, 3] = shift / np.linalg.norm(shift) * translation_m
    return E
