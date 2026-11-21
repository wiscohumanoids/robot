"""Check wrist camera coverage and arm clearance on the two-platform bench."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from sim.mega_bloks_motion import PickPlaceMotion
from sim.mega_bloks_platforms import build_mega_bloks_platform_scene
from sim.mega_bloks_survey import plan_survey_transfer, position_observer, survey
from sim.panthera_env import PantheraSim
from tools.dino_pick_camera import read_camera


def test_right_wrist_sees_both_blocks_during_left_survey(tmp_path):
    scene, blocks, _ = build_mega_bloks_platform_scene(tmp_path / "scene.xml", seed=0, count=2)
    sim = PantheraSim(scene)
    sim.reset(randomize=False)
    sim.step(400)
    body_ids = [sim.model.body(block["body_name"]).id for block in blocks]
    initial_centers = sim.data.xpos[body_ids].copy()
    right_geoms = {
        index for index, body in enumerate(sim.model.geom_bodyid)
        if sim.model.body(body).name.startswith("right_")
    }

    left_phases = []
    left_positions = []
    left_commands = []

    def check_contacts(_motion, _phase):
        if sim.active_arm == "left":
            left_phases.append(_phase)
            left_positions.append(sim.ee_pos())
            left_commands.append(sim.data.ctrl[sim.arm_act].copy())
        for contact in sim.data.contact[:sim.data.ncon]:
            assert (contact.geom1 in right_geoms) == (contact.geom2 in right_geoms)

    sim.select_arm("right")
    assert position_observer(PickPlaceMotion(sim, on_step=check_contacts))
    sim.select_arm("left")

    with mujoco.Renderer(sim.model, height=512, width=512) as renderer:
        _, parked = read_camera(sim, renderer, "right_wrist")

        def capture(phase):
            # The left wrist must still be moving when the frame is taken.
            assert np.linalg.norm(left_positions[-1] - left_positions[-2]) > .0005
            rgb, right = read_camera(sim, renderer, "right_wrist")
            _, left = read_camera(sim, renderer, "left_wrist")
            np.testing.assert_allclose(right.position, parked.position, atol=.001)
            assert np.linalg.norm(right.position - left.position) > .30
            for block, center in zip(blocks, initial_centers):
                right_ray = center - right.position
                left_ray = center - left.position
                cosine = (right_ray @ left_ray) / (
                    np.linalg.norm(right_ray) * np.linalg.norm(left_ray)
                )
                assert cosine < np.cos(np.deg2rad(25))
                x, y = np.rint(right.project(center)).astype(int)
                assert 20 < x < 492 and 20 < y < 492
                pixels = rgb[y - 8:y + 9, x - 8:x + 9].astype(float)
                if block["color"] == "blue":
                    visible = pixels[..., 2] > pixels[..., 0] + 40
                else:
                    assert block["color"] == "orange"
                    visible = pixels[..., 0] > pixels[..., 1] + 40
                assert np.count_nonzero(visible) > 80
            return phase, left.position

        observations = survey(PickPlaceMotion(sim, on_step=check_contacts), capture)

    assert [phase for phase, _ in observations] == [f"survey_{i}" for i in range(4)]
    entry = left_phases.index("survey_0")
    assert np.linalg.norm(left_positions[entry] - left_positions[entry - 1]) > .0005
    camera_positions = np.stack([position for _, position in observations])
    assert np.linalg.norm(np.diff(camera_positions, axis=0), axis=1).min() > .02
    # Catch a rate-clipped turn or a dead band that bunches tiny IK updates
    # into a single tick. Commands are equally spaced at 30 Hz.
    joint_jerk = np.diff(left_commands, n=3, axis=0) * 30**3
    assert np.max(np.abs(joint_jerk)) < 30.  # rad/s³ per joint
    np.testing.assert_allclose(sim.data.xpos[body_ids], initial_centers, atol=.001)


def test_transfer_uses_small_jaw_skew_for_unreachable_stud_axis(tmp_path):
    scene, _, _ = build_mega_bloks_platform_scene(tmp_path / "scene.xml", seed=40)
    sim = PantheraSim(scene)
    sim.reset(randomize=False)
    sim.step(400)
    motion = PickPlaceMotion(sim)
    assert survey(motion, lambda _: None) is not None

    source = [.417, .205, .103]
    destination = [.229, .090, .139]
    axis = np.deg2rad(171)
    assert motion.choose_transfer_yaw(source, destination, [axis]) is None
    yaw = plan_survey_transfer(motion, source, destination, [axis])
    assert yaw is not None
    skew = np.rad2deg((yaw - axis + np.pi / 2) % np.pi - np.pi / 2)
    assert 0 < abs(skew) <= 20
    assert motion.pick(source, yaw)
    assert motion.transfer_above(destination, yaw)
    assert motion.place(destination, yaw)
