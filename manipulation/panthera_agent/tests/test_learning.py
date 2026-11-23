"""Learning objects on first sight with local DINO (downloads DINOv3 ViT-S/16 once), and the local-skill learning
loop's pieces: the training environment, episodes, imitation, the success classifier."""
import tempfile

import numpy as np
import pytest

from core.build import config_for_scene, load_config, make_segmenter, make_skills
from robots.sim_robot import SimRobot

torch = pytest.importorskip('torch')
pytest.importorskip('timm')


def _choose(skills, body):
    """The oracle: the numbered points lying on a sim body."""
    obs, _, P, _ = skills.proposals['overhead']
    m = obs.extra['segmentation'] == obs.extra['body_names'].index(body)
    return [c.number for c in P.candidates
            if m[max(0, int(c.uv[1]) - 4):int(c.uv[1]) + 5, max(0, int(c.uv[0]) - 4):int(c.uv[0]) + 5].mean() > 0.3]


@pytest.fixture(scope='module')
def learned():
    from perception.memory import ObjectMemory
    cfg = config_for_scene(load_config(), 'agent_objects.xml')
    mem = ObjectMemory(tempfile.mkdtemp())
    seg = make_segmenter('dino_local', cfg, mem)
    robot = SimRobot(cfg, seed=1)
    s = make_skills(robot, seg, cfg, mem)
    for body, name in (('cube_red', 'red cube'), ('can_yellow', 'yellow can'), ('box_blue', 'blue box')):
        r = s.propose_points()
        assert r.ok and s.proposals['overhead'][2].candidates
        pts = _choose(s, body)
        assert pts, f'no candidate point on {body}'
        assert s.learn(name, pts).ok
    return cfg, mem, seg


def test_learned_shapes_match_the_true_ones(learned):
    _, mem, _ = learned
    true = {'red cube': (0.045, 0.045), 'yellow can': (0.075, 0.044), 'blue box': (0.030, 0.075)}
    kinds = {'red cube': 'box', 'yellow can': 'round', 'blue box': 'long'}
    for name, (h, long_) in true.items():
        e = mem.lookup(name)[0]
        assert abs(e.shape.height - h) < 0.004, (name, e.shape.height)
        assert abs(e.shape.footprint[0] - long_) < 0.006, (name, e.shape.footprint)
        assert e.shape.kind == kinds[name]


def test_found_and_picked_in_a_new_layout_without_any_server(learned):
    cfg, mem, seg = learned
    robot = SimRobot(cfg, seed=4)
    s = make_skills(robot, seg, cfg, mem)
    assert s.find('yellow can').ok
    can = max(s.world.objects.values(), key=lambda o: (o.trusted, o.score))
    truth = robot.truth_objects()['can_yellow'][0]
    assert np.linalg.norm(can.top[:2] - truth[:2]) < 0.008
    r = s.pick(can.number)
    assert r.ok, r.message
    assert robot.truth_objects()['can_yellow'][0][2] > 0.08               # lifted
    assert s.place_at(0.30, -0.20 if truth[1] > 0 else 0.20).ok


def test_unknown_objects_are_reported_with_how_to_learn_them(learned):
    cfg, mem, seg = learned
    s = make_skills(SimRobot(cfg, seed=4), seg, cfg, mem)
    r = s.find('purple mug')
    assert not r.ok and 'propose_points' in r.message


def test_insert_env_expert_bc_and_classifier(tmp_path, monkeypatch):
    monkeypatch.setenv('PANTHERA_AGENT_DATA', str(tmp_path))
    import learning.policy as P
    monkeypatch.setattr(P, 'ROOT', tmp_path)
    from learning.envs import InsertEnv
    from learning.practice import Practice
    from learning.supervisor import SimSupervisor
    from learning.train import train_reward
    env = InsertEnv(load_config(), seed=0)
    pr = Practice(env, SimSupervisor(), log_dir=tmp_path / 'log')
    assert pr.demos(4) == 1.0                                              # the privileged expert inserts
    assert len(pr.store.entries()) == 4 and pr.store.views(pr.store.entries()[0]['id'])
    pol = pr.bc(60)
    a = pol.act(env.reset())
    assert a.shape == (5,) and np.all(np.abs(a) <= 1)
    _, rec = train_reward('insert', steps=40, log=lambda d: None)
    assert rec['positives'] >= 4
