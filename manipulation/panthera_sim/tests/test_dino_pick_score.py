"""Check source attribution and tabletop outcomes from three pose snapshots."""

import sys
from pathlib import Path

import mujoco
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.dino_pick_score import PickObserver


@pytest.fixture
def scene():
    model = mujoco.MjModel.from_xml_string("""<mujoco>
      <worldbody>
        <geom name="table_top" type="box" pos="0 0 -.025" size=".5 .5 .025"/>
        <body name="first" pos="0 0 .02">
          <freejoint/><geom name="first_box" type="box" size=".02 .02 .02"/>
        </body>
        <body name="second" pos=".06 0 .02">
          <freejoint/><geom name="second_box" type="box" size=".02 .02 .02"/>
        </body>
      </worldbody>
    </mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    observer = PickObserver(model, data, [
        {"body_name": "first", "body_geom": "first_box"},
        {"body_name": "second", "body_geom": "second_box"},
    ])
    observer.capture("start")

    def set_position(body, xyz, quaternion=(1, 0, 0, 0)):
        address = model.jnt_qposadr[model.body(body).jntadr[0]]
        data.qpos[address:address + 7] = (*xyz, *quaternion)
        mujoco.mj_forward(model, data)

    return observer, set_position


def test_scores_a_completed_pick_and_place(scene):
    observer, set_position = scene
    with pytest.raises(RuntimeError, match="before scoring"):
        observer.score(source_xy_m=(0, 0), destination_xy_m=(.2, .1))
    set_position("first", (0, 0, .12))
    observer.capture("lift")
    set_position("first", (.21, .1, .02))
    observer.capture("final")
    set_position("first", (.8, .1, .02))

    result = observer.score(source_xy_m=(.002, 0), destination_xy_m=(.2, .1))
    assert result["success"]
    assert result["intended_body"] == "first"
    assert not result["wrong_object"]
    assert result["pickup_height_m"] == pytest.approx(.10)
    assert result["placement_error_m"] == pytest.approx(.01)
    assert result["final_height_error_m"] == pytest.approx(0)
    assert result["table_gap_m"] == pytest.approx(0)


@pytest.mark.parametrize("lifted", [("second",), ("first", "second")])
def test_flags_extra_or_wrong_object(scene, lifted):
    observer, set_position = scene
    for body in lifted:
        set_position(body, (0, 0, .12))
    observer.capture("lift")
    set_position("first", (.2, .1, .02))
    observer.capture("final")

    result = observer.score(source_xy_m=(0, 0), destination_xy_m=(.2, .1))
    assert result["lifted_bodies"] == list(lifted)
    assert result["wrong_object"]
    assert not result["pickup_success"]
    assert not result["success"]


@pytest.mark.parametrize("final_position", [(.2, .1, .08), (.51, .1, .02)])
def test_requires_placement_on_the_same_table(scene, final_position):
    observer, set_position = scene
    set_position("first", (0, 0, .12))
    observer.capture("lift")
    set_position("first", final_position)
    observer.capture("final")

    result = observer.score(source_xy_m=(0, 0), destination_xy_m=final_position[:2])
    assert result["placement_error_m"] == 0
    assert result["pickup_success"]
    assert not result["on_table"]
    assert not result["placement_success"]


def test_rejects_tipped_placement_even_when_block_touches_the_table(scene):
    observer, set_position = scene
    set_position("first", (0, 0, .12))
    observer.capture("lift")
    # At a 45-degree roll, a cube's lowest corner is half a face diagonal from its center.
    rolled_height = .02 * 2 ** .5
    set_position("first", (.2, .1, rolled_height), (.9238795325, .3826834324, 0, 0))
    observer.capture("final")

    result = observer.score(source_xy_m=(0, 0), destination_xy_m=(.2, .1))
    assert result["on_table"]
    assert result["table_gap_m"] == pytest.approx(0, abs=1e-9)
    assert result["final_height_error_m"] == pytest.approx(rolled_height - .02)
    assert not result["placement_success"]


@pytest.mark.parametrize(("source", "status"), [((.03, 0), "ambiguous"), ((.16, 0), "out_of_range")])
def test_does_not_attribute_an_uncertain_source(scene, source, status):
    observer, _ = scene
    observer.capture("lift")
    observer.capture("final")

    result = observer.score(source_xy_m=source, destination_xy_m=(.2, .1))
    assert result["source_match"] == status
    assert result["intended_body"] is None
    assert result["wrong_object"] is None
    assert not result["success"]
