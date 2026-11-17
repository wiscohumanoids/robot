"""The ros2_control backend files: prefix removal, IMU sensors, URDF, controllers and standing targets, on a small
mjlab-style model (a floating base and two position-actuated hinges, every name prefixed "robot/").
"""

import xml.etree.ElementTree as ET

import pytest

mujoco = pytest.importorskip('mujoco')
yaml = pytest.importorskip('yaml')

from hsim_sim import ros2_control_backend as rcb  # noqa: E402

SCENE = """
<mujoco model="toy">
  <option timestep="0.005"/>
  <default><default class="robot/main"><geom size="0.05"/></default></default>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/>
    <body name="robot/pelvis" pos="0 0 1">
      <freejoint name="robot/floating_base"/>
      <geom class="robot/main" type="sphere"/>
      <site name="robot/imu_site"/>
      <body name="robot/thigh">
        <joint name="robot/hip" axis="0 1 0" range="-1 1"/>
        <geom class="robot/main" type="sphere"/>
        <body name="robot/shin">
          <joint name="robot/knee" axis="0 1 0"/>
          <geom class="robot/main" type="sphere"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="robot/knee" joint="robot/knee" kp="40" kv="2" forcerange="-30 30"/>
    <position name="robot/hip" joint="robot/hip" kp="80" kv="4" forcerange="-60 60"/>
  </actuator>
  <sensor><gyro name="robot/imu_ang_vel" site="robot/imu_site"/></sensor>
  <keyframe><key name="init_state" qpos="0 0 1 1 0 0 0 0.2 0.4" ctrl="0.4 0.2"/></keyframe>
</mujoco>
"""


@pytest.fixture
def files(tmp_path):
    """Backend files built from the toy scene."""
    scene = tmp_path / 'toy.xml'
    scene.write_text(SCENE)
    return rcb.build(str(scene), str(tmp_path / 'out'), update_rate=250)


def test_scene_names_and_imu(files):
    """Names lose the prefix (default classes keep it), the IMU triple sits on the gyro's site, gains are unchanged."""
    m = mujoco.MjModel.from_xml_path(files['scene'])
    assert m.joint('hip') and m.actuator('knee') and m.body('pelvis')
    for name in ('imu_quat', 'imu_gyro', 'imu_accel'):
        assert m.site(int(m.sensor(name).objid[0])).name == 'imu_site'
    assert m.actuator('hip').gainprm[0] == 80 and m.actuator('hip').biasprm[2] == -4
    assert 'robot/main' in open(files['scene']).read()


def test_urdf(files):
    """One position command per joint in joint order, the plugin settings, the IMU sensor and the floating base."""
    root = ET.parse(files['urdf']).getroot()
    rc = root.find('ros2_control')
    params = {p.get('name'): p.text for p in rc.find('hardware').findall('param')}
    assert params['mujoco_model'] == files['scene'] and params['initial_keyframe'] == 'init_state'
    assert params['odom_free_joint_name'] == 'floating_base'
    joints = rc.findall('joint')
    assert [j.get('name') for j in joints] == ['hip', 'knee']
    assert all([c.get('name') for c in j.findall('command_interface')] == ['position'] for j in joints)
    assert joints[1].find("state_interface[@name='position']/param").text == '0.4'   # keyframe qpos
    assert rc.find('sensor').get('name') == 'imu'
    kinds = {j.get('name'): j.get('type') for j in root.findall('joint')}
    assert kinds == {'hip': 'revolute', 'knee': 'continuous'}


def test_controllers_and_targets(files):
    """Controllers list the same joints (no YAML aliases); standing targets are the keyframe ctrl in joint order."""
    text = open(files['controllers']).read()
    assert '&id' not in text and '*id' not in text
    c = yaml.safe_load(text)
    assert c['controller_manager']['ros__parameters']['update_rate'] == 250
    assert c['joint_targets_controller']['ros__parameters']['joints'] == ['hip', 'knee']
    assert c['joint_targets_controller']['ros__parameters']['interface_name'] == 'position'
    j = yaml.safe_load(open(files['joints']))
    assert j == {'joints': ['hip', 'knee'], 'standing': [0.2, 0.4]}


def test_rejects_non_position_actuator(tmp_path):
    """A joint driven by a motor (torque) actuator would run another law: refused."""
    scene = tmp_path / 'motor.xml'
    scene.write_text(SCENE.replace('<position name="robot/knee" joint="robot/knee" kp="40" kv="2"',
                                   '<motor name="robot/knee" joint="robot/knee"'))
    with pytest.raises(ValueError, match='not a position actuator'):
        rcb.build(str(scene), str(tmp_path / 'out'))
