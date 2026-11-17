"""The Python linkage reference (robots/wisco_v1/linkage.py) against the golden file the C++ controller is tested
with (test/data/linkage_parity.json, tools/make_linkage_golden.py), its motors -> joints solve, and the example
controller config against the robot's numbers. No ROS needed.
"""
import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip('numpy')
yaml = pytest.importorskip('yaml')

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parents[2]
sys.path.insert(0, str(REPO / 'robots' / 'wisco_v1'))
from linkage import Mechanism  # noqa: E402

GOLDEN = json.loads((PKG / 'test' / 'data' / 'linkage_parity.json').read_text())


@pytest.mark.parametrize('name', ['left_ankle', 'right_ankle', 'waist'])
def test_reference_reproduces_the_golden_file(name):
    """Joints -> motors and the Jacobian, as recorded (regenerate with tools/make_linkage_golden.py if this fails)."""
    m = Mechanism(name)
    for c in GOLDEN['mechanisms'][name]['cases']:
        assert m.motor_angles(c['q']) == pytest.approx(c['motor'], abs=1e-10)
        assert m.jacobian(c['q']) == pytest.approx(np.array(c['jacobian']), abs=1e-6)


@pytest.mark.parametrize('name', ['left_ankle', 'right_ankle', 'waist'])
def test_motors_to_joints_round_trip(name):
    """joint_angles inverts motor_angles over the whole range, from a cold start."""
    m = Mechanism(name)
    for c in GOLDEN['mechanisms'][name]['cases']:
        assert m.joint_angles(c['motor']) == pytest.approx(c['q'], abs=1e-10)


def test_example_config_matches_the_robot():
    """config/wisco_parallel_joints.yaml: gains from robot_numbers.yaml, joints and motors per mechanism."""
    cfg = yaml.safe_load((PKG / 'config' / 'wisco_parallel_joints.yaml').read_text())['parallel_joints']['ros__parameters']
    gains = yaml.safe_load((REPO / 'robots/wisco_v1/model/robot_numbers.yaml').read_text())['control']['gains']
    assert (REPO / cfg['linkage_file']).is_file()
    for mech in cfg['mechanisms']:
        joints = Mechanism(mech).joints
        keys = [j.removeprefix('left_').removeprefix('right_') for j in joints]
        assert cfg[mech]['kp'] == [gains[k]['kp'] for k in keys]
        assert cfg[mech]['kd'] == [gains[k]['kd'] for k in keys]
        assert len(cfg[mech]['motors']) == 2 and all(lo < hi for lo, hi in
                                                    zip(cfg[mech]['position_min'], cfg[mech]['position_max']))


def test_joint_target_config_covers_the_locomotion_joints():
    """config/wisco_joint_targets.yaml: every non-arm joint of the URDF once; the chained ones are exactly the
    two-motor mechanisms' joints; the manifest it names gives kp / kd for all of them."""
    import re
    cfg = yaml.safe_load((PKG / 'config' / 'wisco_joint_targets.yaml').read_text())['leg_targets']['ros__parameters']
    urdf = (REPO / 'robots/wisco_v1/robot.urdf').read_text()
    movable = re.findall(r'<joint name="([a-z_]+)" type="(?:revolute|continuous)"', urdf)
    arm = ('shoulder', 'elbow', 'forearm', 'wrist')
    assert sorted(cfg['joints']) == sorted(j for j in movable if not any(a in j for a in arm))
    linkage = json.loads((REPO / 'robots/wisco_v1/linkage.json').read_text())
    assert sorted(cfg['chained_joints']) == sorted(j for m in linkage.values() for j in m['joints'])
    manifest = yaml.safe_load((REPO / cfg['manifest_file']).read_text())
    gains = {j['name']: (j['kp'], j['kd']) for j in manifest['joints']}
    numbers = yaml.safe_load((REPO / 'robots/wisco_v1/model/robot_numbers.yaml').read_text())['control']['head_gains']
    for j, kp, kd in zip(cfg['gain_override_joints'], cfg['gain_override_kp'], cfg['gain_override_kd']):
        assert j not in gains and (kp, kd) == (numbers['kp'], numbers['kd'])     # only joints the policy lacks
        gains[j] = (kp, kd)
    assert all(j in gains and gains[j][0] > 0 for j in cfg['joints'])
