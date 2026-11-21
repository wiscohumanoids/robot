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


class TowerMasks:
    """Ground truth merged the way DINO + SAM 2.1 and SAM 3 segment a tower: one mask over a block and every block
    under it (the top face and all the side faces the camera sees)."""

    name = 'tower_masks'

    def __init__(self, robot, cfg):
        self.robot, self.truth = robot, make_segmenter('sim_truth', cfg)

    def segment(self, obs, query):
        poses = self.robot.truth_objects()
        every = {i.label: i.mask for i in self.truth.segment(obs, 'cube')}
        out = []
        for inst in self.truth.segment(obs, query):
            p = poses[inst.label][0]
            under = [n for n, (q, _) in poses.items() if n != inst.label and np.linalg.norm(q[:2] - p[:2]) < 0.02
                     and q[2] < p[2] and n in every]
            if any(n for n, (q, _) in poses.items() if n != inst.label and np.linalg.norm(q[:2] - p[:2]) < 0.02
                   and q[2] > p[2]):
                continue                                   # covered by the block on top: part of its mask
            out.append(type(inst)(np.logical_or.reduce([inst.mask] + [every[n] for n in under]), inst.label, 1.0))
        return out


def test_a_tower_the_agent_built_keeps_its_height_with_tower_masks():
    """Tower masks make a 2-high stack look 3 high from its size (the viewer's three-cube run said "3 high" after
    one place). The agent remembers the stacks it built: 2 then 3 high, on the true top face within a few mm."""
    cfg = load_config()
    task = TASKS['stack_three']
    robot = SimRobot({**cfg, 'scene': task.scene}, seed=2)
    s = make_skills(robot, TowerMasks(robot, cfg), cfg)
    s.find('red cube')
    s.find('green cube')
    by = {f.query: f.number for f in s.world.objects.values()}
    assert s.pick(by['red cube']).ok and s.place_on(by['green cube']).ok and s.home().ok
    for levels, top in ((2, 'cube_red'), (3, 'cube_blue')):
        if levels == 3:
            s.find('blue cube')
            blue = [f for f in s.world.objects.values() if f.query == 'blue cube'][-1]
            tower = [f for f in s.world.objects.values() if f.level == 2][-1]
            assert s.pick(blue.number).ok and s.place_on(tower.number).ok and s.home().ok
        r = s.find('cube')
        tower = max(s.world.objects.values(), key=lambda f: f.top[2])
        assert tower.level == levels and tower.level_known and 'a stack I built' in r.message, r.message
        true_top = robot.truth_objects()[top][0] + np.array([0, 0, 0.0225])
        assert np.linalg.norm(tower.top - true_top) < 0.005
    assert task.check(robot.truth_objects())['success']


def test_a_built_stack_that_lost_its_top_block_is_measured_again():
    """Red goes on green, then falls off: find measures green one high, says so, and forgets the stack."""
    cfg = load_config()
    robot = SimRobot({**cfg, 'scene': 'scene_two_blocks.xml'}, seed=0)
    s = make_skills(robot, TowerMasks(robot, cfg), cfg)
    s.find('red cube')
    s.find('green cube')
    red, green = sorted(s.world.objects)
    assert s.pick(red).ok and s.place_on(green).ok and s.home().ok
    pos, quat = robot.sim.object_poses()
    names = robot.sim.object_names
    pos[names.index('cube_red')] += np.array([0.0, 0.12, -0.045])      # knocked off onto the table beside it
    robot.sim.set_object_poses(pos, quat)
    robot.settle()
    r = s.find('green cube')
    g = [f for f in s.world.objects.values() if f.query == 'green cube'][-1]
    assert g.level == 1 and 'did the top block fall' in r.message and not s.world.stacks
