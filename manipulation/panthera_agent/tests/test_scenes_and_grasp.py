"""The agent's scenes load and lay out; grasps follow an object's shape (no DINO needed)."""
import numpy as np
import pytest

from core.build import config_for_scene, load_config
from core.grasp import fingers_clear, grasp_options, lowest_tcp
from core.world import Found, ObjectType
from robots.sim_robot import SimRobot
from scenes.catalog import SCENES


@pytest.mark.parametrize('name', sorted(SCENES))
def test_every_scene_loads_and_its_objects_rest_on_the_table(name):
    cfg = config_for_scene(load_config(), f'agent_{name}.xml')
    robot = SimRobot(cfg, seed=3)
    robot.settle(0.5)
    for body, (pos, _) in robot.truth_objects().items():
        if not body.startswith('bead'):
            assert abs(pos[2] - 0.05) < 0.003, (body, pos)            # body origins sit at the base


def _found(kind, footprint, height, outline, yaw=0.0, top=(0.40, 0.0), parts=None):
    t = ObjectType('thing', {'thing'}, height, footprint, np.deg2rad({'box': 90, 'long': 180}.get(kind, 360)), kind,
                   outline, parts, False, True)
    return Found(1, 'thing', 'thing', t, np.array([top[0], top[1], 0.05 + height]), yaw, footprint, 1.0, 'overhead',
                 1, True)


def _rect(L, W):
    return [[-L / 2, -W / 2], [L / 2, -W / 2], [L / 2, W / 2], [-L / 2, W / 2]]


def test_a_long_box_is_gripped_across_its_short_side_never_across_corners():
    cfg = load_config()
    o = _found('long', (0.075, 0.034), 0.03, _rect(0.075, 0.034))
    opts = grasp_options(o, cfg)
    assert opts
    for g in opts:
        assert abs(np.sin(2 * (g.yaw - o.yaw))) < 0.05                # faces, not corners
        assert abs(g.width - 0.034) < 0.003                           # the 75 mm side does not fit the jaws


def test_the_top_part_grips_just_below_the_top_and_flat_things_are_pinched_from_above():
    cfg = load_config()
    peg = _found('box', (0.024, 0.024), 0.06, _rect(0.024, 0.024))
    mid, top = grasp_options(peg, cfg)[0], grasp_options(peg, cfg, part='top')[0]
    assert top.z > mid.z + 0.01 and abs(top.z - (peg.top[2] - 0.014)) < 1e-6
    cloth = _found('flat', (0.2, 0.2), 0.004, _rect(0.2, 0.2))
    cloth.type.deformable = True
    g = grasp_options(cloth, cfg)[0]
    assert g.pitch is not None and g.z < 0.06


def test_open_fingers_must_clear_the_object_itself():
    cfg = load_config()
    # A 60 mm mug body with a handle sticking out 20 mm: jaws closing across the body, centred, clear; the same
    # grasp turned so a finger lands where the handle is does not.
    body = [[0.03 * np.cos(a), 0.03 * np.sin(a)] for a in np.linspace(0, 2 * np.pi, 24, endpoint=False)]
    handle = [[0.03, -0.006], [0.05, -0.006], [0.05, 0.006], [0.03, 0.006]]
    outline = np.array(body + handle) + [0.40, 0.0]
    assert fingers_clear(cfg, [0.40, 0.0], 0.08, 0.0, np.array(body) + [0.40, 0.0], 0.115)
    assert not fingers_clear(cfg, [0.40, 0.0], 0.08, np.pi / 2, outline, 0.115)
    assert 0.005 < lowest_tcp(cfg) < 0.03
