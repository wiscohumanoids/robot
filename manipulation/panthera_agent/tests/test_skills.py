"""The skills in the sim: find, pick, place, with perfect masks and with the overhead camera miscalibrated."""
import numpy as np
import pytest

from core.build import load_config, make_segmenter, make_skills
from core.camera import pose_error
from core.tasks import TASKS
from robots.sim_robot import SimRobot


def stack_red_on_green(seed, camera_error=None):
    cfg = load_config()
    robot = SimRobot({**cfg, 'scene': 'scene_two_blocks.xml'}, seed=seed)
    if camera_error:
        robot.camera_errors['overhead'] = pose_error(*camera_error, np.random.default_rng(seed))
    s = make_skills(robot, make_segmenter('sim_truth', cfg), cfg)
    s.find('red cube')
    s.find('green cube')
    red, green = [f.number for f in sorted(s.world.objects.values(), key=lambda f: f.number)]
    assert s.pick(red).ok
    assert s.held is not None and robot.gripper_width() > 0.03
    assert s.place_on(green).ok
    return TASKS['stack_red_on_green'].check(robot.truth_objects()), s


@pytest.mark.parametrize('seed', [0, 5])
def test_stack_with_perfect_perception(seed):
    m, _ = stack_red_on_green(seed)
    assert m['success'] and m['alignment_mm'] < 2.0


def test_wrist_refine_and_overhead_correction_absorb_calibration_error():
    """Overhead pose off 5 mm / 1 deg (about 15 mm on the table): the pick is refined, the place uses the learned
    correction, and the stack still lands within the task's 12 mm."""
    m, s = stack_red_on_green(1, camera_error=(0.005, 1.0))
    refine = [e for e in s.trace if e['event'] == 'refine']
    assert refine and refine[0]['correction_mm'] > 5.0
    assert m['success'] and m['alignment_mm'] < 6.0


def test_find_reports_unknown_types_and_pick_needs_a_number():
    cfg = load_config()
    s = make_skills(SimRobot(cfg, seed=0), make_segmenter('sim_truth', cfg), cfg)
    assert not s.find('coffee mug').ok
    assert not s.pick(99).ok
    assert not s.place_on(1).ok                    # holding nothing


def test_unstack_finds_the_upper_block_and_sets_it_on_the_table():
    """Red starts on green: find measures it two levels up, pick takes it off, place_at puts it on the table."""
    cfg = load_config()
    task = TASKS['unstack']
    robot = SimRobot({**cfg, 'scene': task.scene}, seed=4)
    task.setup(robot)
    s = make_skills(robot, make_segmenter('sim_truth', cfg), cfg)
    s.find('red cube')
    red = next(iter(s.world.objects.values()))
    assert red.level == 2 and abs(red.top[2] - (0.05 + 2 * 0.045)) < 1e-6
    assert not task.check(robot.truth_objects())['success']
    assert s.pick(red.number).ok
    assert s.place_at(0.30, 0.15).ok
    truth = robot.truth_objects()
    assert task.check(truth)['success']
    assert np.linalg.norm(truth['cube_red'][0][:2] - [0.30, 0.15]) < 0.01


def test_a_pick_that_would_hit_a_neighbour_is_refused_not_attempted():
    """Seed 9: after red goes on green, blue stands right beside the tower. Every reachable approach to blue puts
    the open gripper into the tower, so pick refuses and names it, and the tower stays up (before the clearance
    check, this pick knocked it over)."""
    cfg = load_config()
    robot = SimRobot(cfg, seed=9)
    s = make_skills(robot, make_segmenter('sim_truth', cfg), cfg)
    s.find('red cube')
    s.find('green cube')
    assert s.pick(1).ok and s.place_on(2).ok
    s.home()
    s.find('blue cube')
    blue = [f for f in s.world.objects.values() if f.query == 'blue cube'][-1]
    r = s.pick(blue.number)
    assert not r.ok and 'would hit' in r.message
    assert TASKS['stack_red_on_green'].check(robot.truth_objects())['success']     # the tower is untouched
