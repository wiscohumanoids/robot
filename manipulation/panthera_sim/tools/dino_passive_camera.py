"""Refine camera mounts from robot landmarks and DINO studs during a rollout."""

from collections import defaultdict, deque
from dataclasses import dataclass, replace

import cv2
import numpy as np

from tools.dino_camera_fit import fit_camera
from tools.dino_pick_camera import CameraCalibration, prepare_rgb
from tools.dino_pick_targets import Miss
from tools.dino_robot_features import RobotFeatures


def _transform(camera, origin, rotation):
    return replace(camera, position=rotation @ camera.position + origin,
                   right=rotation @ camera.right, up=rotation @ camera.up,
                   forward=rotation @ camera.forward)


def _residuals(camera, points, pixels):
    offset = points - camera.position
    depth = offset @ camera.forward
    xy = np.column_stack((offset @ camera.right, -offset @ camera.up)) / np.maximum(depth[:, None], .001)
    predicted = camera.center + camera.focal * xy * (1 + camera.k1 * (xy * xy).sum(1)[:, None])
    return np.where(depth > .001, np.linalg.norm(predicted - pixels, axis=1), np.inf)


def nominal_mount(name, focal, center):
    """Nominal overhead mount in bench axes or wrist mount in grip-site axes."""
    position, pitch = ([.02, 0., .42], 45.) if name == 'overhead' else ([-.12, 0., .10], 20.)
    angle = np.deg2rad(pitch)
    return CameraCalibration(np.array(position), np.array([0., -1., 0.]),
                             np.array([np.sin(angle), 0., np.cos(angle)]),
                             np.array([np.cos(angle), 0., -np.sin(angle)]), focal, center)


@dataclass
class _MountUpdate:
    robot_matches: int
    stud_matches: int
    mount: CameraCalibration | None = None
    history: dict | None = None


class PassiveCameras:
    """Pool raw pixel correspondences in each camera's rigid mount frame.

    A supplied single-thread executor owns the CAD renderer and calibration
    fits. Keep it running until close() releases the renderer.
    """

    def __init__(self, sim, renderer, detector=None, *, flat_bench=True, executor=None):
        self.executor = executor
        self.pending = {}
        self.sim = sim
        self.renderer = renderer
        self.features = RobotFeatures(sim)
        self.blocks = None
        if detector is not None:
            from tools.dino_block_points import BlockSamples

            self.blocks = BlockSamples(detector, flat_bench)
        self.mounts = {}
        self.maps = {}
        self.samples = defaultdict(lambda: deque(maxlen=24))
        self.report = {}
        self.last_time = {}

    def close(self):
        try:
            self.flush()
        finally:
            if self.executor is None:
                self.features.close()
            else:
                # Release the CAD renderer on the same worker that created its GL context.
                self.executor.submit(self.features.close).result()

    def poll(self):
        """Publish any completed camera fits without waiting for the worker."""
        for name, future in tuple(self.pending.items()):
            if future.done():
                self.flush(name)

    def flush(self, name=None):
        """Wait for queued fits and apply them before a calibration-dependent decision."""
        for key in tuple(self.pending) if name is None else (name,):
            future = self.pending.pop(key, None)
            if future is not None:
                self._publish(key, future.result())

    def _parent(self, name):
        if name == "overhead":
            return np.zeros(3), np.eye(3)
        site_name = {"left_wrist": "grip_site", "right_wrist": "right_grip_site"}[name]
        site = self.sim.data.site(site_name)
        return site.xpos.copy(), site.xmat.reshape(3, 3).copy()

    def _nominal(self, name):
        height, width = self.renderer.height, self.renderer.width
        fovy = self.sim.model.camera(name).fovy.item()
        focal = height / (2 * np.tan(np.deg2rad(fovy) / 2))
        return nominal_mount(name, focal, (np.array([width, height]) - 1) / 2)

    def _raw(self, name):
        prepare_rgb(self.sim, self.renderer, name)
        return self.renderer.render()

    def _rectify(self, name, rgb):
        if name in self.maps:
            return cv2.remap(rgb, *self.maps[name], cv2.INTER_LINEAR)
        return rgb

    def image(self, name):
        return self._rectify(name, self._raw(name))

    def sample(self, name, phase="", rgb=None, *, wait=False):
        """Capture a calibration exposure, allowing one outstanding fit per camera."""
        self.poll()
        if wait:
            self.flush(name)
        now = float(self.sim.data.time)
        if self.last_time.get(name) == now or name in self.pending:
            return
        self.last_time[name] = now
        if rgb is None:
            rgb = self._raw(name)
        overhead = self.mounts.get("overhead")
        top_rgb = None
        if (self.blocks and name != "overhead" and overhead is not None
                and (phase.startswith("survey") or phase in ("approach", "right_observer", "preshape"))):
            top_rgb = self._raw("overhead").copy()
        args = (name, phase, now, rgb.copy(), self.features.encoder_positions(),
                *self._parent(name), self._nominal(name), self.mounts.get(name), top_rgb, overhead)
        if self.executor is None:
            self._publish(name, self._estimate(*args))
        else:
            self.pending[name] = self.executor.submit(self._estimate, *args)
            if wait:
                self.flush(name)

    def _estimate(self, name, phase, now, rgb, encoder_positions, origin, rotation,
                  nominal, previous, top_rgb, overhead):
        # All live camera and arm state was copied at exposure. Only this
        # serial worker uses the CAD renderer and correspondence histories.
        prior = previous or nominal
        camera = _transform(prior, origin, rotation)
        points, pixels = self.features.match(rgb, replace(camera, k1=0.), encoder_positions)
        cad_count = len(points)
        block_count = 0
        bundle = None
        if top_rgb is not None:
            block_pts, observed, bundle = self.blocks.collect(
                name, rgb, origin, rotation, prior, top_rgb, overhead, phase)
            block_count = len(block_pts)
            points = np.concatenate((points, block_pts))
            pixels = np.concatenate((pixels, observed))
        update = _MountUpdate(cad_count, block_count)
        bank = self.samples[name]
        if len(points):
            bank.append(((points - origin) @ rotation, pixels.copy()))
        elif not bank or bundle is None:
            return update
        xyz, uv = (np.concatenate(arrays) for arrays in zip(*bank))
        fit = fit_camera(xyz, uv, rgb.shape, prior.focal)
        candidates = [] if fit is None else [fit]
        if bundle is not None:
            candidates.append((bundle.mount, bundle.rmse_px))

        def cost(candidate):
            # False matches outside the consensus should not dominate the comparison.
            return float(np.mean(np.minimum(_residuals(candidate, xyz, uv), 5.) ** 2))

        previous_cost = cost(previous) if previous is not None else None
        for candidate, rms in sorted(candidates, key=lambda pair: cost(pair[0])):
            if (np.linalg.norm(candidate.position - nominal.position) > .12
                    or candidate.forward @ nominal.forward < np.cos(np.deg2rad(25))
                    or candidate.up @ nominal.up < np.cos(np.deg2rad(25))):
                continue
            candidate_cost = cost(candidate)
            if previous_cost is not None and candidate_cost >= previous_cost:
                continue
            update.mount = candidate
            update.history = dict(time_s=now, phase=phase, pooled_frames=len(bank),
                                  pooled_points=len(xyz), rms_px=float(rms), k1=float(candidate.k1),
                                  estimator="stud_bundle" if bundle and candidate is bundle.mount
                                  else "landmarks", previous_cost_px2=previous_cost,
                                  accepted_cost_px2=candidate_cost)
            break
        return update

    def _publish(self, name, update):
        status = self.report.setdefault(name, dict(frames_examined=0, updates=0, accepted=False,
                                                   robot_matches=0, stud_matches=0, history=[]))
        status["frames_examined"] += 1
        status["robot_matches"] += update.robot_matches
        status["stud_matches"] += update.stud_matches
        if update.mount is None:
            return
        candidate = update.mount
        previous = self.mounts.get(name)
        self.mounts[name] = candidate
        if previous is None or previous.k1 != candidate.k1:
            if candidate.k1:
                self.maps[name] = cv2.initUndistortRectifyMap(
                    candidate.matrix, np.array([candidate.k1, 0., 0., 0.]), None,
                    candidate.matrix, (self.renderer.width, self.renderer.height), cv2.CV_32FC1)
            else:
                self.maps.pop(name, None)
        status.update(accepted=True, updates=status["updates"] + 1, rms_px=update.history["rms_px"],
                      k1=float(candidate.k1), mount_position_m=candidate.position.tolist(),
                      mount_forward=candidate.forward.tolist(), mount_up=candidate.up.tolist())
        status["history"].append(update.history)

    def read(self, name):
        self.poll()
        if name not in self.mounts:
            self.flush(name)
        raw = self._raw(name)
        if (name not in self.mounts
                or self.sim.data.time - self.last_time.get(name, -np.inf) >= .65):
            self.sample(name, rgb=raw, wait=name not in self.mounts)
        if name not in self.mounts:
            raise Miss(f"camera_calibration_unobserved_{name}")
        camera = _transform(self.mounts[name], *self._parent(name))
        return self._rectify(name, raw), replace(camera, k1=0.)
