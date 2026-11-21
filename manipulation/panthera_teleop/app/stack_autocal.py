"""Fit overhead and wrist mounts to observed Panthera CAD features and tabletop studs."""

from collections import deque
from dataclasses import replace
from itertools import combinations, permutations
from pathlib import Path
import sys

import cv2
import mujoco
import numpy as np

SIM_ROOT = Path(__file__).resolve().parents[2] / 'panthera_sim'
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

from sim.mega_bloks import BODY_HEIGHT_M, STUD_HEIGHT_M
from tools.dino_camera_fit import fit_camera
from tools.dino_passive_camera import _residuals, _transform, nominal_mount
from tools.dino_pick_refine import _coherent_studs, _region_fits_footprint
from tools.dino_pick_track import _clipped
from tools.dino_robot_features import RobotFeatures


class _Lens:
    """The CAD fitter uses centered square pixels. Preserve physical K/D for raw-frame projection."""

    def __init__(self, config, nominal_fovy):
        resolution = np.asarray(config['resolution'])
        if (resolution.shape != (2,) or not np.issubdtype(resolution.dtype, np.integer)
                or np.min(resolution) < 1):
            raise ValueError('Camera resolution must be positive integer [width, height]')
        width, height = self.size = tuple(map(int, resolution))
        self.center = (resolution - 1.) / 2
        self.map = self.matrix = self.distortion = None
        if 'camera_matrix' in config or 'distortion' in config:
            self.matrix = np.asarray(config.get('camera_matrix'), dtype=float)
            self.distortion = np.asarray(config.get('distortion'), dtype=float)
            k, d = self.matrix, self.distortion
            if (k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0
                    or not (0 <= k[0, 2] < width and 0 <= k[1, 2] < height)
                    or not np.allclose(k, [[k[0, 0], 0, k[0, 2]], [0, k[1, 1], k[1, 2]], [0, 0, 1]])
                    or d.shape not in ((4,), (5,), (8,), (12,), (14,)) or not np.isfinite(d).all()):
                raise ValueError('Invalid measured camera_matrix or OpenCV distortion coefficients')
            self.focal = float(k[0, 0])
            cx, cy = self.center
            target = np.array([[self.focal, 0, cx], [0, self.focal, cy], [0, 0, 1]])
            self.map = cv2.initUndistortRectifyMap(k, d, None, target, self.size, cv2.CV_32FC1)
        else:
            fovy = float(config.get('fovy_deg', nominal_fovy))
            if not 0 < fovy < 180:
                raise ValueError('fovy_deg must be the vertical field of view between 0 and 180 degrees')
            self.focal = height / (2 * np.tan(np.deg2rad(fovy) / 2))

    def image(self, rgb):
        w, h = self.size
        if rgb.shape != (h, w, 3) or rgb.dtype != np.uint8:
            raise ValueError(f'Expected a {w}×{h} uint8 RGB camera frame')
        return cv2.remap(rgb, *self.map, cv2.INTER_LINEAR) if self.map is not None else rgb

    def config(self, camera):
        pose = np.eye(4)
        pose[:3, 3] = camera.position
        pose[:3, :3] = np.column_stack((camera.right, -camera.up, camera.forward))
        k = camera.matrix if self.map is None else self.matrix
        d = [camera.k1, 0., 0., 0.] if self.map is None else self.distortion.tolist()
        return dict(resolution=list(self.size), camera_matrix=k.tolist(), distortion=d,
                    world_from_camera=pose.tolist())


def _stud_observations(overhead, top_blocks, wrist, wrist_blocks, height, shape):
    """Associate two-stud rows on the measured plane, allowing extra detector peaks."""
    points, pixels = [], []
    for top in top_blocks:
        side = [b for b in wrist_blocks if b.color == top.color]
        if (not top.color or sum(b.color == top.color for b in top_blocks) != 1 or len(side) != 1
                or not 2 <= len(top.studs_xy) <= 8 or not 2 <= len(side[0].studs_xy) <= 8
                or _clipped(side[0], shape)):
            continue
        candidates = []
        for pair in combinations(top.studs_xy, 2):
            try:
                xyz = np.array([overhead.on_plane(p, height) for p in pair])
                expected = np.array([wrist.project(p) for p in xyz])
            except ValueError:
                continue
            if not _coherent_studs(xyz[:, :2]) or not _region_fits_footprint(overhead, top, xyz[:, :2], height):
                continue
            for measured in permutations(side[0].studs_xy, 2):
                residual = np.asarray(measured) - expected
                shift = residual.mean(axis=0)
                shape_error = np.linalg.norm(residual - shift, axis=1).max()
                scale = min(shape[:2]) / 512
                if shape_error <= 4 * scale and np.linalg.norm(shift) <= 55 * scale:
                    candidates.append((shape_error + .25 * np.linalg.norm(shift), xyz, measured))
        candidates.sort(key=lambda c: c[0])
        if candidates and (len(candidates) == 1 or candidates[1][0] - candidates[0][0] > 1.5):
            _, world, observed = candidates[0]
            points.extend(world)
            pixels.extend(observed)
    return np.asarray(points).reshape(-1, 3), np.asarray(pixels).reshape(-1, 2)


def _evidence(mount, local, uv, studs, shape):
    errors = np.full(len(uv), np.inf) if mount is None else _residuals(mount, local, uv)
    cad_count = len(uv) - studs
    # SIFT can report several descriptors at one pixel. Count each CAD
    # corner once, and require coverage in both image dimensions.
    _, unique = np.unique(np.rint(uv[:cad_count] / 2), axis=0, return_index=True)
    inliers = unique[errors[unique] <= 3.5]
    spread = 0.
    if len(inliers) >= 6:
        points = uv[inliers]
        spread = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)[1] / np.sqrt(len(inliers))
    cad_ok = len(inliers) >= max(7, .35 * len(unique)) and spread >= min(shape[:2]) * .025
    stud_ok = studs >= 4 and max(errors[cad_count:], default=np.inf) <= 4.
    good = np.r_[errors[inliers], errors[cad_count:][errors[cad_count:] <= 4.]]
    metrics = dict(robot_matches=cad_count, stud_matches=studs, inliers=len(good),
                   rms_px=round(float(np.sqrt(np.mean(good**2))), 2) if good.size else None)
    return bool(cad_ok or stud_ok), metrics


def _distinct(first, second):
    p, rotation = first
    q, other = second
    return (np.linalg.norm(p - q) >= .008 or
            np.trace(rotation.T @ other) <= 1 + 2 * np.cos(np.deg2rad(4)))


class StackAutoCalibration:
    def __init__(self, config, detector):
        self.detector = detector
        self.stud_z = float(config['table_z_m']) + BODY_HEIGHT_M + STUD_HEIGHT_M / 2
        self.open_rad = float(config['gripper_open_rad'])
        self.closed_rad = float(config['gripper_closed_rad'])
        arm = config['arm']
        if (arm not in ('left', 'right') or not np.isfinite([self.stud_z, self.open_rad, self.closed_rad]).all()
                or self.open_rad == self.closed_rad):
            raise ValueError('Invalid selected arm, table height, or gripper encoder endpoints')
        self.lenses = {name: _Lens(config[name], fovy) for name, fovy in [('overhead', 80.), ('wrist', 70.)]}
        self.features = RobotFeatures()
        model = self.features.model
        base = model.body(f'{arm}_base')
        prefix = '' if arm == 'left' else 'right_'
        self.joints = [model.joint(prefix + f'joint{i}').qposadr.item() for i in range(1, 7)]
        self.fingers = [model.joint(prefix + side + '_finger_joint').qposadr.item() for side in ('L', 'R')]
        self.site = prefix + 'grip_site'
        # Use the same base axes as the real-arm IK planner. Geometry belonging
        # to the unobserved partner arm must never supply a 3-D image landmark.
        origin = base.pos.copy()
        base.pos[:] = 0
        model.geom_group[model.body_rootid[model.geom_bodyid] != base.id] = 5
        self.features.visibility.geomgroup[5] = 0
        self.nominal = {name: nominal_mount(name, lens.focal, lens.center)
                        for name, lens in self.lenses.items()}
        top = self.nominal['overhead']
        self.nominal['overhead'] = replace(top, position=top.position - origin)
        self.mounts = {}
        self.banks = {name: deque(maxlen=24) for name in self.lenses}
        self.poses = deque(maxlen=24)
        self.ready = self.verified = False

    def close(self):
        self.features.close()

    def _pose(self, q_rad, gripper_rad=None):
        joints = np.asarray(q_rad, dtype=float)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError('Expected six measured joint angles for the calibration exposure')
        data = self.features.data
        data.qpos[self.joints] = joints
        if gripper_rad is not None:
            if not np.isfinite(gripper_rad):
                raise ValueError('Invalid measured gripper encoder')
            travel = np.clip((gripper_rad - self.closed_rad) / (self.open_rad - self.closed_rad), 0., 1.)
            data.qpos[self.fingers] = np.array([.04, -.04]) * travel
        mujoco.mj_kinematics(self.features.model, data)
        site = data.site(self.site)
        return site.xpos.copy(), site.xmat.reshape(3, 3).copy()

    def _initial(self, name, parent):
        prior = self.mounts.get(name, self.nominal[name])
        return _transform(prior, *parent)

    def _fit(self, name, xyz, uv, studs, shape):
        if not len(xyz):
            return
        self.banks[name].append((xyz, uv, studs))
        xyz = np.concatenate([view[0] for view in self.banks[name]])
        uv = np.concatenate([view[1] for view in self.banks[name]])
        nominal = self.nominal[name]
        result = fit_camera(xyz, uv, shape, nominal.focal, fit_distortion=self.lenses[name].map is None)
        if result is None:
            return
        camera, _ = result
        if (np.linalg.norm(camera.position - nominal.position) > .12 or
                camera.forward @ nominal.forward < np.cos(np.deg2rad(25)) or
                camera.up @ nominal.up < np.cos(np.deg2rad(25))):
            return
        previous = self.mounts.get(name)
        if previous is not None:
            # Compare on identical pooled observations, limiting the influence
            # of false feature matches as in the simulation rollout.
            def cost(mount):
                return np.mean(np.minimum(_residuals(mount, xyz, uv), 5.) ** 2)
            if cost(camera) >= cost(previous):
                return
        self.mounts[name] = camera

    def _capture(self, overhead_rgb, wrist_rgb, q_rad, gripper_rad, training):
        top_image, wrist_image = (self.lenses[name].image(img)
                                 for name, img in [('overhead', overhead_rgb), ('wrist', wrist_rgb)])
        grip = self._pose(q_rad, gripper_rad)
        if any(not _distinct(grip, saved) for saved in self.poses):
            return False, 'Choose a different wrist pose (at least 8 mm or 4° from the training views).', {}
        encoders = self.features.data.qpos.copy()
        diagnostics = {}
        evidence = {}
        for name, rgb, parent in [('overhead', top_image, (np.zeros(3), np.eye(3))), ('wrist', wrist_image, grip)]:
            camera = self._initial(name, parent)
            world, uv = self.features.match(rgb, replace(camera, k1=0.), encoders)
            studs = 0
            if name == 'wrist' and 'overhead' in self.mounts and self.detector is not None:
                tops = [b for b in self.detector.detect(top_image) if not _clipped(b, top_image.shape)]
                blocks, pixels = _stud_observations(self.mounts['overhead'], tops, camera,
                                                     self.detector.detect(wrist_image), self.stud_z, rgb.shape)
                studs = len(blocks)
                world = np.concatenate([world, blocks])
                uv = np.concatenate([uv, pixels])
            local = (world - parent[0]) @ parent[1]
            if training:
                self._fit(name, local, uv, studs, rgb.shape)
            mount = self.mounts.get(name)
            evidence[name], diagnostics[name] = _evidence(mount, local, uv, studs, rgb.shape)
            usable = sum(_evidence(mount, *view, rgb.shape)[0] for view in self.banks[name])
            diagnostics[name]['matching_views'] = usable
        if training and any(d['robot_matches'] + d['stud_matches'] >= 2 for d in diagnostics.values()):
            self.poses.append(grip)
        missing = [name for name, ok in evidence.items() if not ok]
        if missing:
            return False, f"Insufficient matching landmarks in {', '.join(missing)} view. Reposition the arm and capture again.", diagnostics
        if any(d['matching_views'] < 2 for d in diagnostics.values()):
            return False, 'Each mount needs matching landmarks at two distinct poses. Capture another view.', diagnostics
        return True, ('Both mounts fitted. Finish at a separate survey pose.' if training else
                      'Both camera mounts passed held-out reprojection.'), diagnostics

    def sample(self, overhead_rgb, wrist_rgb, q_rad, gripper_rad):
        self.verified = False
        self.ready, message, metrics = self._capture(overhead_rgb, wrist_rgb, q_rad, gripper_rad, True)
        return dict(ready=self.ready, message=message, samples=len(self.poses), metrics=metrics)

    def verify(self, overhead_rgb, wrist_rgb, q_rad, gripper_rad):
        self.verified = False
        if not self.ready:
            raise ValueError('Capture distinct views until both camera mounts are fitted')
        self.verified, message, metrics = self._capture(overhead_rgb, wrist_rgb, q_rad, gripper_rad, False)
        return dict(ready=self.verified, message=message, samples=len(self.poses), metrics=metrics)

    def camera_configs(self, q_rad):
        if not self.verified:
            raise ValueError('Validate both fitted camera mounts at a held-out pose before using them')
        wrist = _transform(self.mounts['wrist'], *self._pose(q_rad))
        return dict(overhead=self.lenses['overhead'].config(self.mounts['overhead']),
                    wrist=dict(self.lenses['wrist'].config(wrist), reference_q_rad=list(q_rad)))
