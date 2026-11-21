"""Fit physical camera mounts from JPEGs and encoders in the selected arm's base frame."""

from contextlib import closing
import os
from pathlib import Path
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')

import cv2
import mujoco
import numpy as np
import pytest

from perception.pickup import BlockDetection
from sim.mega_bloks_motion import PickPlaceMotion
from sim.mega_bloks_stack import build_stack_scene
from sim.mega_bloks_survey import _survey_pose
from sim.panthera_env import GRIPPER_OPEN, PantheraSim
from tools.dino_pick_camera import read_camera

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'panthera_teleop/app'))
from stack_autocal import StackAutoCalibration


def _change_mount(model, name, offset, pitch):
    camera = model.camera(name)
    camera.pos[:] += offset
    delta = np.empty(4)
    mujoco.mju_axisAngle2Quat(delta, np.array([1., 0., 0.]), np.deg2rad(pitch))
    mujoco.mju_mulQuat(camera.quat, camera.quat.copy(), delta)


class LabeledBlocks:
    """Use visible stud centroids to exercise ray geometry independently of detector weights."""
    def __init__(self):
        self.detections = {}

    def register(self, rgb, renderer, sim, blocks):
        renderer.enable_segmentation_rendering()
        labels = renderer.render()
        renderer.disable_segmentation_rendering()
        geom_ids = labels[:, :, 0]
        is_geom = labels[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM
        result = []
        for block in blocks:
            studs = []
            for name in block['stud_geoms']:
                pixels = (geom_ids == sim.model.geom(name).id) & is_geom
                if np.count_nonzero(pixels) >= 5:
                    y, x = np.nonzero(pixels)
                    studs.append((float(x.mean()), float(y.mean())))
            if studs:
                block_geoms = [sim.model.geom(name).id for name in block['stud_geoms'] + block['shell_geoms']]
                y, x = np.nonzero(np.isin(geom_ids, block_geoms) & is_geom)
                box = (x.min(), y.min(), x.max() + 1, y.max() + 1)
                result.append(BlockDetection(tuple(np.mean(studs, axis=0)), block['color'],
                                             1., box, tuple(studs), len(x)))
        self.detections[rgb.tobytes()] = result

    def detect(self, rgb):
        return self.detections.get(rgb.tobytes(), [])


def _capture(sim, renderer, blocks, detector):
    origin = sim.data.body('left_base').xpos.copy()
    frames, truth = {}, {}
    for role, name in [('overhead', 'overhead'), ('wrist', 'left_wrist')]:
        rgb, camera = read_camera(sim, renderer, name)
        encoded = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 85])[1]
        frames[role] = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        detector.register(frames[role], renderer, sim, blocks)
        pose = np.eye(4)
        pose[:3, :3] = np.column_stack((camera.right, -camera.up, camera.forward))
        pose[:3, 3] = camera.position - origin
        truth[role] = pose
    gripper = float(sim.data.qpos[sim.finger_qadr[0]] / GRIPPER_OPEN)
    args = (frames['overhead'], frames['wrist'], sim.q.copy(), gripper)
    return args, truth


def test_estimates_selected_arm_mounts_and_checks_fresh_survey(tmp_path):
    scene, blocks = build_stack_scene(tmp_path / 'bench.xml', seed=0)
    sim = PantheraSim(scene)
    sim.model.body('right_base').pos[0] += 2.
    _change_mount(sim.model, 'overhead', [.010, -.015, .012], -3.)
    _change_mount(sim.model, 'left_wrist', [.006, 0., .004], 3.)
    sim.reset(randomize=False)
    sim.step(250)
    motion = PickPlaceMotion(sim)
    detector = LabeledBlocks()
    config = dict(arm='left', gripper_open_rad=1., gripper_closed_rad=0., table_z_m=0.,
                  overhead={'resolution': [640, 480], 'fovy_deg': 80.},
                  wrist={'resolution': [640, 480], 'fovy_deg': 70.})
    with (mujoco.Renderer(sim.model, height=480, width=640) as renderer,
          closing(StackAutoCalibration(config, detector)) as estimator):
        captures = {}
        for degrees in (0, 17, 45, 90, 135, 177, 215, 250):
            xyz, quat = _survey_pose(motion, np.deg2rad(degrees))
            assert motion.move('survey', xyz, quat, 1.)
            captures[degrees] = _capture(sim, renderer, blocks, detector)

        with pytest.raises(ValueError, match='held-out'):
            estimator.camera_configs(captures[0][0][2])
        for degrees in (0, 45, 135, 215, 17, 177):
            report = estimator.sample(*captures[degrees][0])
        assert report['ready'], report
        assert not estimator.verify(*captures[177][0])['ready']

        args, truth = captures[250]
        report = estimator.verify(*args)
        assert report['ready'], report
        fitted = estimator.camera_configs(args[2])
        for role in ('overhead', 'wrist'):
            measured = np.asarray(fitted[role]['world_from_camera'])
            expected = truth[role]
            assert np.linalg.norm(measured[:3, 3] - expected[:3, 3]) < .008
            rot = measured[:3, :3].T @ expected[:3, :3]
            assert np.trace(rot) > 1 + 2 * np.cos(np.deg2rad(1.2))

        blank = np.zeros_like(args[0])
        assert not estimator.verify(blank, blank, *args[2:])['ready']
        with pytest.raises(ValueError, match='held-out'):
            estimator.camera_configs(args[2])


def test_right_arm_uses_its_own_base_frame_and_excludes_left_arm_landmarks():
    from sim.mega_bloks import BASE_SCENE

    bench = PantheraSim(BASE_SCENE)
    bench.select_arm('right')
    q = np.array([.62, 1.31, .91, -.29, .33, -.4])
    bench.data.qpos[bench.arm_qadr] = q
    mujoco.mj_forward(bench.model, bench.data)
    base = bench.data.body('right_base')
    base_rotation = base.xmat.reshape(3, 3)
    tip = bench.data.site('right_grip_site')
    config = dict(arm='right', table_z_m=0., gripper_open_rad=1., gripper_closed_rad=0.,
                  overhead={'resolution': [640, 480]}, wrist={'resolution': [640, 480]})
    with closing(StackAutoCalibration(config, None)) as estimator:
        p, rotation = estimator._pose(q, .75)
        np.testing.assert_allclose(p, base_rotation.T @ (tip.xpos - base.xpos), atol=1e-12)
        np.testing.assert_allclose(rotation, base_rotation.T @ tip.xmat.reshape(3, 3), atol=1e-12)
        # The CAD renderer has no encoder feedback for the other arm.
        model = estimator.features.model
        visible = estimator.features.visibility.geomgroup[model.geom_group].astype(bool)
        roots = model.body_rootid[model.geom_bodyid]
        assert visible[roots == model.body('right_base').id].any()
        assert not visible[roots == model.body('left_base').id].any()
        np.testing.assert_allclose(estimator.nominal['overhead'].position,
                                   [.02, .29, .37], atol=1e-12)
