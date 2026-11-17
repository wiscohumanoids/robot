"""The ros2_control backend files: prefix removal, IMU sensors, URDF, the robot's controllers and plugins, on a small
mjlab-style model (a floating base and two position-actuated hinges, every name prefixed "robot/").
"""

import math
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


def test_controllers_and_plugins(files):
    """The robot's own controllers (JointTargetController per joint group, positions only, model limits; no YAML
    aliases), the broadcasters, and the ground-truth plugin."""
    text = open(files['controllers']).read()
    assert '&id' not in text and '*id' not in text
    c = yaml.safe_load(text)
    assert c['controller_manager']['ros__parameters']['update_rate'] == 250
    legs = c['leg_targets']['ros__parameters']
    assert legs['type'] == 'wisco_ethercat_ros2_hw/JointTargetController'
    assert legs['joints'] == ['hip', 'knee'] and legs['topic'] == '/locomotion/joint_targets'
    assert 'gain_interfaces' not in legs and legs['timeout_action'] == 'hold'    # gains live in the MuJoCo model
    assert legs['position_min'][0] == pytest.approx(-math.radians(1.0))       # hip range -1..1 (MJCF degrees)
    assert legs['position_max'][1] == 10.0                                    # knee: unlimited
    assert 'arm_targets' not in c                                                 # no arm joints in this model
    assert files['controllers_to_spawn'] == ['joint_state_broadcaster', 'imu_sensor_broadcaster', 'leg_targets']
    p = yaml.safe_load(open(files['plugins']))['/**']['ros__parameters']['mujoco_plugins']['ground_truth']
    assert p['type'] == 'hsim_mujoco_plugins/GroundTruthPlugin' and p['state_topic'] == '/sim/ground_truth/state'


def test_arm_joints_get_their_own_controller():
    """Arm joints (shoulder, elbow, forearm, wrist) go to arm_targets, the rest to leg_targets."""
    joints = [('left_hip_pitch', 0), ('left_shoulder_pitch', 1), ('torso_yaw', 2), ('right_wrist_roll', 3)]
    assert rcb.groups(joints) == {'leg_targets': ['left_hip_pitch', 'torso_yaw'],
                                  'arm_targets': ['left_shoulder_pitch', 'right_wrist_roll']}


def test_compiled_env_world_is_used_as_is(files, tmp_path):
    """An env's world_ros2_control.mjb (already converted) is loaded directly; an unconverted .mjb is refused."""
    m = mujoco.MjModel.from_xml_path(files['scene'])
    good = tmp_path / 'world_ros2_control.mjb'
    mujoco.mj_saveModel(m, str(good), None)
    out = rcb.build(str(good), str(tmp_path / 'env'))
    assert out['world'] == str(good) and 'scene' not in out
    assert ET.parse(out['urdf']).getroot().find('ros2_control/hardware/param[@name="mujoco_model"]').text == str(good)
    raw = tmp_path / 'raw.mjb'
    mujoco.mj_saveModel(mujoco.MjModel.from_xml_string(SCENE), str(raw), None)
    with pytest.raises(ValueError, match='not a plugin-ready world'):
        rcb.build(str(raw), str(tmp_path / 'raw'))


def test_floating_base_is_the_robots_not_a_loose_object(files):
    """A world with a loose object before the robot: odometry still follows the robot's base."""
    world = SCENE.replace('<worldbody>', '<worldbody><body name="cup" pos="1 0 1"><freejoint name="cup_free"/>'
                          '<geom type="sphere" size="0.02"/></body>')
    world = world.replace('qpos="0 0 1 1 0 0 0 0.2 0.4"', 'qpos="1 0 1 1 0 0 0 0 0 1 1 0 0 0 0.2 0.4"')
    m = mujoco.MjModel.from_xml_string(world)
    assert rcb.free_joint(m) == 'robot/floating_base'


def test_rejects_non_position_actuator(tmp_path):
    """A joint driven by a motor (torque) actuator would run another law: refused."""
    scene = tmp_path / 'motor.xml'
    scene.write_text(SCENE.replace('<position name="robot/knee" joint="robot/knee" kp="40" kv="2"',
                                   '<motor name="robot/knee" joint="robot/knee"'))
    with pytest.raises(ValueError, match='not a position actuator'):
        rcb.build(str(scene), str(tmp_path / 'out'))
