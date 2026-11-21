import mujoco
import numpy as np
import pytest

from sim.mega_bloks_platforms import build_mega_bloks_platform_scene, nudge_platform_block
from sim.panthera_env import PantheraSim
from tools.dino_platform_score import PlatformObserver


def _trial(tmp_path, seed, count=1):
    path, blocks, supports = build_mega_bloks_platform_scene(
        tmp_path / "platforms.xml", seed=seed, count=count,
    )
    sim = PantheraSim(path)
    sim.reset(randomize=False)
    sim.step(250)
    mujoco.mj_forward(sim.model, sim.data)
    return sim, blocks, supports


def _move_block(sim, block, position):
    joint = sim.model.joint(block["body_name"])
    adr = int(joint.qposadr[0])
    sim.data.qpos[adr:adr + 3] = position
    mujoco.mj_forward(sim.model, sim.data)


@pytest.mark.parametrize(("seed", "direction"), [(0, "uphill"), (1, "downhill")])
def test_scores_requested_surface_and_click_using_true_geometry(tmp_path, seed, direction):
    sim, blocks, supports = _trial(tmp_path, seed)
    block, = blocks
    observer = PlatformObserver(sim.model, sim.data, blocks, supports)
    observer.capture("start")
    source = sim.data.geom(block["body_geom"]).xpos.copy()
    goal_name = max(supports, key=lambda name: np.linalg.norm(
        sim.data.geom(name).xpos[:2] - source[:2],
    ))
    goal = sim.data.geom(goal_name).xpos.copy()
    goal[2] += sim.model.geom(goal_name).size[2]
    camera_pos = np.array([.34, .12, .42])
    source_ray = (camera_pos, source - camera_pos)
    destination_ray = (camera_pos, goal - camera_pos)
    _move_block(sim, block, source + [0, 0, .09])
    observer.capture("lift")
    _move_block(sim, block, goal + [0, 0, .02])
    observer.capture("final")

    result = observer.score(
        source_xy_m=source[:2] + .08, destination_xy_m=goal[:2] + .01,
        source_surface_z_m=source[2] - .016, destination_surface_z_m=goal[2] - .008,
        source_pixel_ray=source_ray, destination_ray=destination_ray,
    )
    assert result["success"]
    assert result["direction"] == direction
    assert result["intended_body"] == block["body_name"]
    assert result["source_height_error_m"] == pytest.approx(.004, abs=.0003)
    assert result["destination_height_error_m"] == pytest.approx(.008)
    assert result["destination_projection_error_m"] == pytest.approx(np.sqrt(2) * .01)
    assert result["pickup_height_m"] == pytest.approx(.09)
    assert result["placement_error_m"] == pytest.approx(0)
    assert result["destination_gap_m"] == pytest.approx(0, abs=.0001)


def test_does_not_credit_same_support_or_a_block_hovering_above_destination(tmp_path):
    sim, blocks, supports = _trial(tmp_path, seed=2)
    block, = blocks
    observer = PlatformObserver(sim.model, sim.data, blocks, supports)
    observer.capture("start")
    source = sim.data.geom(block["body_geom"]).xpos.copy()
    goal_name = max(supports, key=lambda name: np.linalg.norm(
        sim.data.geom(name).xpos[:2] - source[:2],
    ))
    goal = sim.data.geom(goal_name).xpos.copy()
    goal[2] += sim.model.geom(goal_name).size[2]
    _move_block(sim, block, source + [0, 0, .09])
    observer.capture("lift")
    _move_block(sim, block, goal + [0, 0, .035])
    observer.capture("final")

    result = observer.score(source_xy_m=source[:2], destination_xy_m=goal[:2])
    assert result["pickup_success"]
    assert not result["on_destination"]
    assert result["destination_gap_m"] == pytest.approx(.015, abs=.0001)
    assert not result["success"]
    same_support = observer.score(source_xy_m=source[:2], destination_xy_m=source[:2])
    assert same_support["direction"] == "level"
    assert not same_support["success"]


def test_does_not_credit_lift_of_additional_block(tmp_path):
    sim, blocks, supports = _trial(tmp_path, seed=2, count=2)
    observer = PlatformObserver(sim.model, sim.data, blocks, supports)
    observer.capture("start")
    initial = [sim.data.geom(block["body_geom"]).xpos.copy() for block in blocks]
    for block, position in zip(blocks, initial):
        _move_block(sim, block, position + [0, 0, .08])
    observer.capture("lift")
    observer.capture("final")

    result = observer.score(source_xy_m=initial[0][:2], destination_xy_m=initial[1][:2])
    assert result["wrong_object"]
    assert not result["pickup_success"]
    assert not result["success"]


@pytest.mark.parametrize("seed", [0, 1, 6])
def test_disturbance_moves_block_on_its_original_pad(tmp_path, seed):
    sim, blocks, supports = _trial(tmp_path, seed)
    block, = blocks
    observer = PlatformObserver(sim.model, sim.data, blocks, supports)
    observer.capture("start")
    geom = sim.data.geom(block["body_geom"])
    before = geom.xpos.copy()
    rotation = geom.xmat.copy()

    delta = nudge_platform_block(sim)

    assert np.linalg.norm(delta) == pytest.approx(.020)
    np.testing.assert_allclose(geom.xpos, before + [*delta, 0.])
    np.testing.assert_allclose(geom.xmat, rotation)
    observer.capture("lift")
    sim.step(250)
    mujoco.mj_forward(sim.model, sim.data)
    observer.capture("final")
    score = observer.score(source_xy_m=before[:2], destination_xy_m=geom.xpos[:2])
    assert score["direction"] == "level"
    assert score["on_destination"]
    assert geom.xpos[:2] == pytest.approx(before[:2] + delta, abs=.0001)


def test_surface_finishes_are_reproducible_and_separate_from_collision(tmp_path):
    first, _, supports = build_mega_bloks_platform_scene(tmp_path / "first" / "scene.xml", seed=5)
    second, _, _ = build_mega_bloks_platform_scene(tmp_path / "second" / "scene.xml", seed=5)
    model = mujoco.MjModel.from_xml_path(str(first))
    repeated = mujoco.MjModel.from_xml_path(str(second))

    for name in supports:
        support = model.geom(name)
        assert int(support.type[0]) == mujoco.mjtGeom.mjGEOM_BOX
        assert int(support.contype[0]) != 0
        top = support.pos + [0, 0, support.size[2]]
        on_top = np.all(np.isclose(model.geom_pos, top + [0, 0, .0001]), axis=1)
        plane = model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE
        finish, = np.flatnonzero(on_top & plane)
        assert model.geom_contype[finish] == model.geom_conaffinity[finish] == 0
        assert model.geom_matid[finish] == repeated.geom_matid[finish] == -1
        np.testing.assert_array_equal(model.geom_rgba[finish], repeated.geom_rgba[finish])

    assert not list(tmp_path.rglob("*.png"))
