"""The ethercat_sim backend files: one hardware joint per motor, the robot computer's controller chain with the trained
gains, the bus cycle against the physics step; on the toy model of test_ros2_control_backend.py, with its two hinges
made a two-motor mechanism where the test needs one.
"""

import json
import xml.etree.ElementTree as ET

import pytest

mujoco = pytest.importorskip('mujoco')
yaml = pytest.importorskip('yaml')

from hsim_sim import ethercat_sim_backend as ecs  # noqa: E402
from test_ros2_control_backend import SCENE  # noqa: E402


@pytest.fixture
def scene(tmp_path):
    """The toy scene on disk."""
    path = tmp_path / 'toy.xml'
    path.write_text(SCENE)
    return str(path)


@pytest.fixture
def ethercat(tmp_path):
    """A profile `ethercat_sim` section making hip + knee one mechanism ("leg"), and the files it names."""
    linkage = tmp_path / 'linkage.json'
    linkage.write_text(json.dumps({'leg': {'joints': ['hip', 'knee']}}))
    parallel = tmp_path / 'parallel.yaml'
    parallel.write_text(yaml.safe_dump({'parallel_joints': {'ros__parameters': {
        'type': 'wisco_ethercat_ros2_hw/ParallelLinkageController', 'linkage_file': 'relative/linkage.json',
        'mechanisms': ['leg'], 'command_mode': 'pd_in_drive',
        'motor_command_interfaces': ['position', 'velocity', 'effort', 'kp', 'kd'],
        'leg': {'motors': ['leg_motor_a', 'leg_motor_b'], 'motor_directions': [1.0, -1.0],
                'kp': [1.0, 1.0], 'kd': [0.1, 0.1]}}}}))
    return {'parallel_joints': str(parallel), 'linkage_file': str(linkage), 'linkage_motor_peak_nm': {'leg': 34.0}}


def ros2_control(urdf_path):
    """{joint: {params}} and the hardware params of the URDF's ros2_control block."""
    rc = ET.parse(urdf_path).getroot().find('ros2_control')
    hw = {p.get('name'): p.text for p in rc.find('hardware').findall('param')}
    joints = {j.get('name'): {p.get('name'): p.text for p in j.findall('param')} for j in rc.findall('joint')}
    return rc, hw, joints


def test_a_drive_per_joint_without_mechanisms(scene, tmp_path):
    """No ethercat_sim section: every actuated joint is its own drive, peak torque = the actuator's force range."""
    files = ecs.build(scene, str(tmp_path / 'out'))
    rc, hw, joints = ros2_control(files['urdf'])
    assert rc.find('hardware/plugin').text == 'hsim_mujoco_plugins/VirtualRobotSystem'
    assert set(joints) == {'hip', 'knee'}
    assert float(joints['hip']['peak_effort_nm']) == 60.0 and float(joints['knee']['peak_effort_nm']) == 30.0
    assert hw['mujoco_model'] == files['scene'] and hw['free_joint'] == 'floating_base'
    assert hw['mode_of_operation'] == '5' and hw['node_name'] == 'virtual_robot' and 'linkage_file' not in hw
    assert {i.get('name') for i in rc.find('joint').findall('command_interface')} >= {'position', 'kp', 'kd'}
    ctrl = yaml.safe_load(open(files['controllers']))
    assert 'parallel_joints' not in ctrl and 'chained_joints' not in ctrl['leg_targets']['ros__parameters']
    assert files['controllers_to_spawn'][0] == 'leg_targets' and files['plugin_node'] == '/virtual_robot'


def test_controllers_carry_the_trained_gains(scene, tmp_path):
    """leg_targets writes position + kp + kd to the drives, the gains read from the model's position actuators."""
    ctrl = yaml.safe_load(open(ecs.build(scene, str(tmp_path / 'out'))['controllers']))
    p = ctrl['leg_targets']['ros__parameters']
    assert p['joints'] == ['hip', 'knee'] and p['gain_interfaces'] == ['kp', 'kd']
    assert p['kp'] == [80.0, 40.0] and p['kd'] == [4.0, 2.0]
    js = ctrl['joint_states']['ros__parameters']
    assert js['type'] == 'wisco_ethercat_ros2_hw/ChainedJointStateBroadcaster' and js['joints'] == ['hip', 'knee']
    assert ctrl['drive_states']['ros__parameters']['use_local_topics'] is True


def test_mechanism_motors_and_chain(scene, tmp_path, ethercat):
    """A mechanism's joints become its two motors on the bus; leg_targets reaches them through parallel_joints,
    whose gains come from the model and whose linkage file is the profile's absolute path.
    """
    files = ecs.build(scene, str(tmp_path / 'out'), ethercat=ethercat)
    _, hw, joints = ros2_control(files['urdf'])
    assert set(joints) == {'leg_motor_a', 'leg_motor_b'}
    assert joints['leg_motor_b'] == {'peak_effort_nm': '34.0', 'effort_nm_per_permille': str(34.0 / 3000.0),
                                     'mechanism': 'leg', 'crank': '1', 'direction': '-1.0'}
    assert hw['linkage_file'] == ethercat['linkage_file']
    ctrl = yaml.safe_load(open(files['controllers']))
    pj = ctrl['parallel_joints']['ros__parameters']
    assert pj['linkage_file'] == ethercat['linkage_file']
    assert pj['leg']['kp'] == [80.0, 40.0] and pj['leg']['kd'] == [4.0, 2.0]
    lt = ctrl['leg_targets']['ros__parameters']
    assert lt['chained_joints'] == ['hip', 'knee'] and lt['chain_prefix'] == 'parallel_joints'
    assert ctrl['joint_states']['ros__parameters']['chained_joints'] == ['hip', 'knee']
    assert files['controllers_to_spawn'][:2] == ['parallel_joints', 'leg_targets']


def test_bus_cycle_against_physics_step(scene, tmp_path):
    """A bus cycle of whole physics steps keeps the model's step, a shorter one becomes the step, others fail."""
    m = mujoco.MjModel.from_xml_string(SCENE)               # 5 ms physics step
    assert ecs.physics_timestep(m, 200) == 0.0 and ecs.physics_timestep(m, 100) == 0.0
    assert ecs.physics_timestep(m, 1000) == pytest.approx(0.001)
    with pytest.raises(ValueError, match='whole number'):
        ecs.physics_timestep(m, 150)
    _, hw, _ = ros2_control(ecs.build(scene, str(tmp_path / 'out'), update_rate=1000)['urdf'])
    assert float(hw['control_period']) == pytest.approx(0.001) and float(hw['physics_timestep']) == pytest.approx(0.001)


def test_mechanism_joints_must_be_actuated(scene, tmp_path, ethercat):
    """A mechanism naming joints the model does not drive is refused."""
    open(ethercat['linkage_file'], 'w').write(json.dumps({'leg': {'joints': ['hip', 'ankle']}}))
    with pytest.raises(ValueError, match='not actuated joints'):
        ecs.build(scene, str(tmp_path / 'out'), ethercat=ethercat)
