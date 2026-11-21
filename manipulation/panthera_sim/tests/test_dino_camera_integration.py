"""Recover changed camera mounts from robot appearance and motion on a plain table."""

from contextlib import closing
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np

from sim.mega_bloks import build_mega_bloks_scene
from sim.mega_bloks_motion import PickPlaceMotion
from sim.mega_bloks_survey import position_observer, survey
from sim.panthera_env import PantheraSim
from tools.dino_passive_camera import PassiveCameras
from tools.dino_pick_camera import read_camera


def _change_mount(model, name, offset, pitch):
    index = model.camera(name).id
    model.cam_pos[index] += offset
    delta = np.empty(4)
    mujoco.mju_axisAngle2Quat(delta, np.array([1., 0., 0.]), np.deg2rad(pitch))
    mujoco.mju_mulQuat(model.cam_quat[index], model.cam_quat[index].copy(), delta)


def _compare_at_work_height(camera, truth):
    assert np.linalg.norm(camera.position - truth.position) < .008
    assert camera.forward @ truth.forward > np.cos(np.deg2rad(1.2))
    for point in ([.25, .025, .09], [.31, -.035, .105], [.34, .045, .15]):
        assert np.linalg.norm(camera.project(point) - truth.project(point)) < 3.


def _plain_bench(tmp_path):
    scene, _ = build_mega_bloks_scene(tmp_path / "bench.xml", count=2, seed=0)
    sim = PantheraSim(scene)
    table = sim.model.geom("table_top")
    table.matid[:] = -1
    table.rgba[:] = [.54, .54, .54, 1.]
    sim.reset(randomize=False)
    sim.step(250)
    return sim


def _observe_survey(sim, sample):
    last_sample = -float("inf")

    def on_step(motion, _phase):
        nonlocal last_sample
        if sim.data.time - last_sample >= .45:
            mujoco.mj_forward(sim.model, sim.data)
            sample()
            last_sample = sim.data.time

    sim.select_arm("right")
    assert position_observer(PickPlaceMotion(sim, on_step=on_step))
    sim.select_arm("left")
    assert survey(PickPlaceMotion(sim, on_step=on_step), lambda _: None) is not None


def test_estimated_mount_follows_inactive_wrist_after_feature_loss(tmp_path, monkeypatch):
    sim = _plain_bench(tmp_path)
    _change_mount(sim.model, "overhead", [.024, -.018, .022], -6.)
    _change_mount(sim.model, "right_wrist", [.018, 0., .012], 5.)
    mujoco.mj_forward(sim.model, sim.data)
    with (mujoco.Renderer(sim.model, height=512, width=512) as renderer,
          closing(PassiveCameras(sim, renderer)) as reader):

        def sample():
            for name in ("overhead", "right_wrist"):
                reader.sample(name)

        _observe_survey(sim, sample)
        for name in ("overhead", "right_wrist"):
            _, estimate = reader.read(name)
            _, truth = read_camera(sim, renderer, name)
            _compare_at_work_height(estimate, truth)

        status = reader.report["right_wrist"]
        assert status["updates"] >= 2
        assert status["history"][-1]["pooled_frames"] >= 2
        updates_before = status["updates"]
        previous_position = estimate.position.copy()
        sim.select_arm("right")
        sim.data.qpos[sim.arm_qadr[0]] += .20
        sim.select_arm("left")
        mujoco.mj_forward(sim.model, sim.data)

        rgb, truth = read_camera(sim, renderer, "right_wrist")
        monkeypatch.setattr(renderer, "render", lambda: np.zeros_like(rgb))
        reader.sample("right_wrist")
        assert status["updates"] == updates_before
        _, carried = reader.read("right_wrist")
        assert np.linalg.norm(carried.position - previous_position) > .025
        _compare_at_work_height(carried, truth)


def test_rectified_rgb_and_rays_agree_with_undistorted_render(tmp_path, monkeypatch):
    sim = _plain_bench(tmp_path)
    k1 = .18
    with (mujoco.Renderer(sim.model, height=512, width=512) as renderer,
          closing(PassiveCameras(sim, renderer)) as reader):
        _, truth = read_camera(sim, renderer, "overhead")
        y, x = np.indices((renderer.height, renderer.width), dtype=np.float32)
        warped_pixels = np.stack((x, y), axis=-1)
        sample = cv2.undistortPoints(warped_pixels.reshape(-1, 1, 2), truth.matrix,
                                     np.array([k1, 0., 0., 0.]), P=truth.matrix)
        render = renderer.render

        def warped_render():
            return cv2.remap(render(), sample.reshape(*x.shape, 2), None, cv2.INTER_LINEAR)

        monkeypatch.setattr(renderer, "render", warped_render)
        _observe_survey(sim, lambda: reader.sample("overhead"))
        rectified, camera = reader.read("overhead")
        assert abs(reader.report["overhead"]["k1"] - k1) < .04
        _compare_at_work_height(camera, truth)
        # Compare aligned intensity edges independently of the fitted correspondences.
        reference = render()
        raw = warped_render()
        crop = np.s_[32:-32, 32:-32]
        reference_crop = reference[crop].astype(float)
        rectified_error = np.mean(np.abs(rectified[crop] - reference_crop))
        raw_error = np.mean(np.abs(raw[crop] - reference_crop))
        assert rectified_error < .55 * raw_error


def test_async_calibration_uses_exposure_encoders_and_bounds_backlog(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, get_ident

    sim = _plain_bench(tmp_path)
    ready, release = Event(), Event()
    main_thread = get_ident()
    with (mujoco.Renderer(sim.model, height=512, width=512) as renderer,
          ThreadPoolExecutor(1) as executor,
          closing(PassiveCameras(sim, renderer, executor=executor)) as reader):
        rgb, _ = read_camera(sim, renderer, "left_wrist")
        exposure = rgb.copy()
        encoders = sim.data.qpos[reader.features.joints].copy()
        checked = []
        match = reader.features.match

        def delayed_match(frame, camera, positions):
            ready.set()
            assert release.wait(10)
            assert get_ident() != main_thread
            np.testing.assert_array_equal(frame, exposure)
            np.testing.assert_array_equal(positions, encoders)
            checked.append(True)
            return match(frame, camera, positions)

        monkeypatch.setattr(reader.features, "match", delayed_match)
        try:
            reader.sample("left_wrist", rgb=rgb)
            assert ready.wait(10)
            rgb[:] = 0
            sim.data.qpos[sim.arm_qadr[0]] += .1
            sim.step(10)
            reader.sample("left_wrist")
            assert "left_wrist" not in reader.report
        finally:
            release.set()
        reader.flush()
        assert checked == [True]
        assert reader.report["left_wrist"]["frames_examined"] == 1
