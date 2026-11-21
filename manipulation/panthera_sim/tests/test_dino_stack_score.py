import mujoco
import numpy as np
import pytest

from sim.mega_bloks_stack import STACK_CENTER_GAP_M, build_stack_scene
from sim.panthera_env import PantheraSim
from tools.dino_stack_score import StackObserver


@pytest.fixture
def trial(tmp_path):
    path, blocks = build_stack_scene(tmp_path / "stack.xml", seed=0)
    sim = PantheraSim(path)
    sim.reset(randomize=False)
    sim.step(250)
    mujoco.mj_forward(sim.model, sim.data)
    observer = StackObserver(sim.model, sim.data, blocks)
    observer.capture("start")
    names = [block["body_name"] for block in blocks]
    rays = [(sim.data.body(name).xpos + [0, 0, .5], [0, 0, -1]) for name in names]
    return sim, observer, names, rays


def _put(sim, name, pose):
    sim.data.joint(name).qpos[:] = pose
    sim.data.joint(name).qvel[:] = 0.
    mujoco.mj_forward(sim.model, sim.data)


def _lift(sim, source, base, both):
    for name in ([source, base] if both else [source]):
        pose = sim.data.joint(name).qpos.copy()
        pose[2] += .08
        _put(sim, name, pose)


@pytest.mark.parametrize("fault", [None, "both", "hover", "base_shift", "hold_break", "short"])
def test_stack_evaluates_the_clicked_pair_and_a_settled_hold(trial, fault):
    sim, observer, (source, base), (source_ray, destination_ray) = trial
    initial_base_pose = sim.data.joint(base).qpos.copy()
    _lift(sim, source, base, both=fault == "both")
    observer.capture("lift")
    base_pose = initial_base_pose.copy()
    if fault == "base_shift":
        base_pose[1] += .025
    _put(sim, base, base_pose)
    top_pose = base_pose.copy()
    top_pose[2] += STACK_CENTER_GAP_M
    if fault == "hover":
        top_pose[2] += .015
    _put(sim, source, top_pose)
    if fault != "hover":
        sim.step(125)
        mujoco.mj_forward(sim.model, sim.data)
    observer.capture("release")

    if fault == "hold_break":
        _put(sim, source, top_pose + [.003, 0, .015, 0, 0, 0, 0])
        observer.capture("hold")
        _put(sim, source, top_pose)
    if fault == "hover":
        sim.data.time += .3
    elif fault != "short":
        sim.step(200)
        mujoco.mj_forward(sim.model, sim.data)
    observer.capture("final")

    score = observer.score(source_pixel_ray=source_ray, destination_ray=destination_ray)
    assert score["intended_body"] == source
    assert score["destination_body"] == base
    assert score["pickup_success"] == (fault != "both")
    assert score["stack_contact"] == (fault not in ("hover", "hold_break"))
    assert score["stable_hold"] == (fault not in ("hover", "hold_break", "short"))
    assert score["success"] == (fault is None)


def test_stack_rejects_two_clicks_on_the_same_block(trial):
    sim, observer, (source, base), (source_ray, _) = trial
    _lift(sim, source, base, both=False)
    observer.capture("lift")
    observer.capture("release")
    observer.capture("final")
    score = observer.score(source_pixel_ray=source_ray, destination_ray=source_ray)
    assert score["pickup_success"]
    assert not score["distinct_selections"]
    assert not score["success"]


@pytest.mark.parametrize("fault", [None, "off_grid", "missing", "hold_break", "wrong_lift", "no_lift"])
def test_build_scores_each_grid_cell_and_its_support(tmp_path, fault):
    from sim.mega_bloks import PITCH_M

    path, blocks = build_stack_scene(tmp_path / "build.xml", count=4, seed=7)
    sim = PantheraSim(path)
    sim.reset(randomize=False)
    sim.step(250)
    mujoco.mj_forward(sim.model, sim.data)
    names = [b["body_name"] for b in blocks]
    rays = [(sim.data.body(name).xpos + [0, 0, .5], [0, 0, -1]) for name in names]
    observer = StackObserver(sim.model, sim.data, blocks)
    observer.capture("start")
    observer.capture("lift")
    anchor = sim.data.joint(names[0]).qpos.copy()
    row = sim.data.body(names[0]).xmat.reshape(3, 3)[:2, 0].copy()
    offsets = [(2 * PITCH_M, 0., 0), (0., 0., 1), (0., 0., 2)]
    for name, (along, _, level) in zip(names[1:], offsets):
        if fault != "no_lift" or name != names[-1]:
            observer.capture_build_start()
            lifted_name = names[0] if fault == "wrong_lift" and name == names[-1] else name
            before = sim.data.joint(lifted_name).qpos.copy()
            _put(sim, lifted_name, before + [0, 0, .08, 0, 0, 0, 0])
            observer.capture_build_lift()
            _put(sim, lifted_name, before)
        pose = anchor.copy()
        pose[:2] += (along + (.02 if fault == "off_grid" and level == 0 else 0.)) * row
        pose[2] += level * STACK_CENTER_GAP_M
        _put(sim, name, pose)
    sim.step(250)
    mujoco.mj_forward(sim.model, sim.data)
    observer.capture("release")
    if fault == "hold_break":
        pose = sim.data.joint(names[-1]).qpos.copy()
        pose[2] += .02
        _put(sim, names[-1], pose)
        observer.capture("hold")
    sim.step(250)
    mujoco.mj_forward(sim.model, sim.data)
    observer.capture("final")
    placements = [(ray, *offset) for ray, offset in zip(rays[1:], offsets)]
    if fault == "missing":
        placements.pop()
    score = observer.score_build(anchor_ray=rays[0], placements=placements)
    assert score["success"] == (fault is None)
    if fault in ("wrong_lift", "no_lift"):
        assert score["placement_success"]
        assert not score["pickup_success"]
        assert not score["pickups"][-1]["success"]
    if fault is None:
        assert score["pickup_success"]
        assert [b["support"] for b in score["blocks"]] == ["bench", "bench", names[0], names[2]]


@pytest.mark.parametrize("layout", ["bridge", "bridge_missing_pier", "one_stud", "wrong_turn", "wrong_length"])
def test_blueprint_scores_rotated_socket_contacts(tmp_path, layout):
    from sim.mega_bloks import PITCH_M

    bridge = layout.startswith("bridge")
    lengths = [2, 2, 4] if bridge else [2, 3]
    path, blocks = build_stack_scene(tmp_path / "blueprint.xml", lengths=lengths)
    sim = PantheraSim(path)
    sim.reset(randomize=False)
    sim.step(250)
    mujoco.mj_forward(sim.model, sim.data)
    names = [b["body_name"] for b in blocks]
    rays = [(sim.data.body(name).xpos + [0, 0, .5], [0, 0, -1]) for name in names]
    observer = StackObserver(sim.model, sim.data, blocks)
    observer.capture("start")
    observer.capture("lift")
    anchor = sim.data.joint(names[0]).qpos.copy()
    yaw = 2 * np.arctan2(anchor[6], anchor[3])
    basis = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    placements = ([(rays[1], (0, 3, 0), 0, 2), (rays[2], (0, 0, 1), 1, 4)]
                  if bridge else [(rays[1], (0, -1, 1), 1, 3)])
    for index, (_, (x, y, level), turn, length) in enumerate(placements, start=1):
        name = names[index]
        observer.capture_build_start()
        pose = sim.data.joint(name).qpos.copy()
        pose[2] += .08
        _put(sim, name, pose)
        observer.capture_build_lift()
        pose = anchor.copy()
        direction = np.array([0, 1] if turn else [1, 0])
        center = np.array([x - .5, y]) + direction * (length - 1) / 2
        pose[:2] += basis @ center * PITCH_M
        pose[2] += level * STACK_CENTER_GAP_M
        theta = yaw + turn * np.pi / 2
        pose[3:] = [np.cos(theta / 2), 0, 0, np.sin(theta / 2)]
        _put(sim, name, pose)
    sim.step(250)
    mujoco.mj_forward(sim.model, sim.data)
    observer.capture("release")
    if layout == "bridge_missing_pier":
        pose = sim.data.joint(names[1]).qpos.copy()
        pose[1] += .10
        _put(sim, names[1], pose)
        observer.capture("hold")
    sim.step(250)
    mujoco.mj_forward(sim.model, sim.data)
    observer.capture("final")
    if layout == "wrong_turn":
        placements[-1] = (rays[1], (-1, 0, 1), 0, 3)
    elif layout == "wrong_length":
        placements[-1] = (rays[1], (0, -1, 1), 1, 4)
    score = observer.score_blueprint(anchor_ray=rays[0], placements=placements)
    assert score["success"] == (layout in ("bridge", "one_stud")), score
    if bridge:
        assert score["blocks"][-1]["supports"] == names[:2]
        assert score["blocks"][-1]["supported"] == (layout == "bridge")
