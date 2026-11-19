"""Ground-truth segmentation and the overlays."""
import numpy as np

from core.build import load_config
from perception.overlay import draw_instances, draw_table_grid
from perception.sim_truth import SimTruthSegmenter
from robots.sim_robot import SimRobot


def test_queries_match_colour_and_type():
    cfg = load_config()
    robot = SimRobot(cfg, seed=1)
    seg = SimTruthSegmenter(cfg['objects'])
    obs = robot.observe('overhead')
    assert [i.label for i in seg.segment(obs, 'red cube')] == ['cube_red']
    assert [i.label for i in seg.segment(obs, 'the green block')] == ['cube_green']
    assert sorted(i.label for i in seg.segment(obs, 'blocks')) == ['cube_blue', 'cube_green', 'cube_red']
    assert seg.segment(obs, 'yellow cube') == []
    assert seg.segment(obs, 'cup') == []


def test_overlays_draw_without_changing_the_input():
    cfg = load_config()
    robot = SimRobot(cfg, seed=1)
    obs = robot.observe('overhead')
    before = obs.rgb.copy()
    mask = np.zeros(obs.rgb.shape[:2], bool)
    mask[100:140, 200:260] = True
    a = draw_instances(obs.rgb, [(7, mask)])
    b = draw_table_grid(obs.rgb, obs.camera, 0.05, cfg['table']['x'], cfg['table']['y'])
    assert np.array_equal(obs.rgb, before)
    assert (a != before).any() and (b != before).any()
