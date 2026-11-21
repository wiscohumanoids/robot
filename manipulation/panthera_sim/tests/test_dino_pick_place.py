"""Check camera association and the commanded pick and drop heights."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perception.pickup import BlockDetection
from tools.dino_pick_place import BODY_CENTER_Z, Miss, create_run_directory, run_policy, yaw_candidates


class Camera:
    """One millimeter per pixel on a horizontal plane."""
    def on_plane(self, xy, z):
        return np.r_[np.array(xy) / 1000, z]

    def project(self, xyz):
        return np.array(xyz[:2]) * 1000


def block(x):
    return BlockDetection((x, 120), "red", .98, (x-16, 104, x+17, 137), ((x, 120),), 900)


@pytest.mark.parametrize(("wrist_x", "failure"), [
    ([306], None), ([], "wrist_target_lost"), ([340], "wrist_target_lost"),
    ([306, 311], "wrist_target_ambiguous"),
])
def test_wrist_association_controls_descent(tmp_path, wrist_x, failure):
    frames = iter([[block(300)], [block(x) for x in wrist_x]])
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    actions = {}
    args = SimpleNamespace(source_pixel=(300, 120), destination_pixel=(420, 120),
                           interactive=False, overhead_only=False)
    def capture(name):
        def move(pose, yaw):
            actions[name] = pose.copy()
            return True
        return move

    motion = SimpleNamespace(choose_transfer_yaw=lambda *_: 0., approach=lambda *_: True,
                             pick=capture("pick"), place=capture("place"))
    detector = SimpleNamespace(detect=lambda rgb, **kwargs: next(frames))

    def execute():
        run_policy(lambda name: (image, Camera()), lambda side: motion, detector, args, tmp_path, {}, lambda: None)

    if failure:
        with pytest.raises(Miss, match=failure):
            execute()
        assert actions == {}
    else:
        execute()
        np.testing.assert_allclose(actions["pick"], [.303, .12, BODY_CENTER_Z])
        np.testing.assert_allclose(actions["place"], [.42, .12, BODY_CENTER_Z])


def test_run_directory_preserves_previous_attempt(tmp_path):
    path = tmp_path / "pick"
    first = create_run_directory(path)
    (first / "summary.json").write_text("first")
    assert create_run_directory(path) == tmp_path / "pick_2"
    assert create_run_directory(path) == tmp_path / "pick_3"
    assert (first / "summary.json").read_text() == "first"


def test_square_block_closes_across_a_side():
    angle = np.deg2rad(23)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    studs = [rotation @ [x, y] for x in (-16, 16) for y in (-16, 16)]
    axes = yaw_candidates(Camera(), SimpleNamespace(studs_xy=studs))
    assert len(axes) == 2
    np.testing.assert_allclose(np.sin(2 * (np.asarray(axes) - angle)), 0, atol=1e-10)


def test_interactive_retries_and_uses_arm_that_reaches_both_poses(tmp_path):
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    detector = SimpleNamespace(detect=lambda rgb: [block(300)])
    args = SimpleNamespace(source_pixel=None, destination_pixel=None,
                           interactive=True, overhead_only=True)
    moved = []

    def motion_for(side):
        def move(*args):
            moved.append(side)
            return True
        return SimpleNamespace(choose_yaw=lambda *_: 0.,
                               choose_transfer_yaw=lambda *_: 0. if side == "right" else None,
                               approach=move, pick=move, place=move)

    def choose(rgb, boxes, boundary, source_ok, destination_ok):
        assert source_ok((0, 0))
        source, goal = (300, 120), (420, 120)
        assert source_ok(source) is None
        assert destination_ok(source, source)
        assert destination_ok(source, goal) is None
        assert not moved
        return source, goal

    record = {}
    run_policy(lambda _: (image, Camera()), motion_for, detector, args, tmp_path,
               record, lambda: None, choose)
    assert record["arm"] == "right"
    assert [item["reason"] for item in record["rejected_clicks"]] == [
        "source_outside_detection", "destination_at_source",
    ]
    assert moved == ["right", "right", "right"]
