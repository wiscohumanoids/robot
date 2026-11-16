"""Tests for the joint-table config generation (no ROS graph needed)."""

import math
import os
import sys
import xml.etree.ElementTree as ET

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from wisco_ethercat_ros2_hw.hardware_config import (  # noqa: E402
    ConfigError, HardwareConfig, load_joint_profiles)

CONFIG = os.path.join(ROOT, 'config')
MODELS = os.path.join(CONFIG, 'actuator_models.yaml')
TEMPLATE = os.path.join(CONFIG, 'drive_pdo_template.yaml')
XACRO = os.path.join(ROOT, 'urdf', 'test_bench.urdf.xacro')
ARM01 = os.path.normpath(os.path.join(ROOT, '..', '..', '..', 'robots', 'arm01'))   # arm01_description

URDF = """<robot name="r">
  <link name="base"/><link name="l1"/><link name="l2"/><link name="l3"/><link name="l4"/>
  <joint name="a" type="revolute"><parent link="base"/><child link="l1"/>
    <limit lower="-1" upper="2" effort="10" velocity="20"/></joint>
  <joint name="b" type="revolute"><parent link="l1"/><child link="l2"/>
    <limit lower="-0.5" upper="0.5" effort="34" velocity="8"/></joint>
  <joint name="c" type="continuous"><parent link="l2"/><child link="l3"/></joint>
  <joint name="fixed" type="fixed"><parent link="l3"/><child link="l4"/></joint>
</robot>"""


def write(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data))
    return str(path)


def make(tmp_path, joints, control_mode=None, calibration=None, urdf=URDF):
    table = write(tmp_path, 'joints.yaml', {'control_mode': 'velocity', 'joints': joints})
    cal = write(tmp_path, 'cal.yaml', calibration) if calibration is not None else None
    return HardwareConfig(MODELS, table, urdf, calibration_file=cal,
                          control_mode=control_mode, template_file=TEMPLATE)


def channel(cfg, pdo, index):
    for p in cfg[pdo]:
        for ch in p['channels']:
            if ch['index'] == index:
                return ch
    raise KeyError(hex(index))


def test_mixed_model_order_keeps_bus_positions(tmp_path):
    # x4_36 before x4_10 on the ring: the old count-based chain regrouped these.
    hw = make(tmp_path, [
        {'name': 'b', 'bus_position': 1, 'model': 'x4_10'},
        {'name': 'a', 'bus_position': 0, 'model': 'x4_36'},
    ])
    assert [(j.name, j.bus_position, j.model_name) for j in hw.joints] == [
        ('a', 0, 'x4_36'), ('b', 1, 'x4_10')]
    assert hw.mock_joints == ['c']
    eff_a = channel(hw.slave_config(hw.joints[0]), 'tpdo', 0x6077)['factor']
    eff_b = channel(hw.slave_config(hw.joints[1]), 'tpdo', 0x6077)['factor']
    assert eff_a == pytest.approx(0.01159)   # x4_36: 1.9 Nm/A x 6.1 A / 1000
    assert eff_b == pytest.approx(0.00624)   # x4_10: 0.8 Nm/A x 7.8 A / 1000


@pytest.mark.parametrize('direction', [1, -1])
def test_factors_fold_in_direction_and_zero(tmp_path, direction):
    zero = 0.3
    hw = make(tmp_path, [{'name': 'a', 'bus_position': 0, 'model': 'x4_10',
                          'direction': direction}],
              calibration={'version': 1, 'zero_offsets_rad': {'a': zero}})
    cfg = hw.slave_config(hw.joints[0])
    counts = 50000
    pos = channel(cfg, 'tpdo', 0x6064)
    angle = pos['factor'] * counts + pos['offset']
    assert angle == pytest.approx(direction * counts * 2 * math.pi / 131072 - zero)

    # 0x607A: the driver writes back the joint position it just read, so the
    # command factor must invert the state factor and land on the same counts.
    cmd = channel(cfg, 'rpdo', 0x607a)
    assert cmd['factor'] * angle + cmd['offset'] == pytest.approx(counts)

    vel_cmd = channel(cfg, 'rpdo', 0x60ff)['factor']
    vel_state = channel(cfg, 'tpdo', 0x606c)['factor']
    assert vel_cmd * vel_state == pytest.approx(1.0)
    assert math.copysign(1, vel_state) == direction

    eff_cmd = channel(cfg, 'rpdo', 0x6071)['factor']
    eff_state = channel(cfg, 'tpdo', 0x6077)['factor']
    assert eff_cmd * eff_state == pytest.approx(1.0)
    assert eff_state == pytest.approx(direction * 0.00624)


def test_control_mode_selects_mode_and_interface(tmp_path):
    joints = [{'name': 'a', 'bus_position': 0, 'model': 'x4_10'}]
    vel = make(tmp_path, joints)
    cfg = vel.slave_config(vel.joints[0])
    assert vel.mode_of_operation == 9
    assert cfg['sdo'][0]['value'] == 9
    assert channel(cfg, 'rpdo', 0x60ff)['command_interface'] == 'velocity'
    assert 'command_interface' not in channel(cfg, 'rpdo', 0x6071)
    assert vel.controller_name() == 'velocity_controller'

    eff = make(tmp_path, joints, control_mode='effort')
    cfg = eff.slave_config(eff.joints[0])
    assert eff.mode_of_operation == 10
    assert channel(cfg, 'rpdo', 0x6060)['default'] == 10
    assert channel(cfg, 'rpdo', 0x6071)['command_interface'] == 'effort'
    assert 'command_interface' not in channel(cfg, 'rpdo', 0x60ff)
    assert eff.controller_name() == 'effort_controller'


def test_dc_sync_and_torque_cap(tmp_path):
    hw = make(tmp_path, [{'name': 'a', 'bus_position': 0, 'model': 'x4_10'},
                         {'name': 'b', 'bus_position': 1, 'model': 'x4_36',
                          'max_torque_permille': 100}])
    a = hw.slave_config(hw.joints[0])
    b = hw.slave_config(hw.joints[1])
    assert a['assign_activate'] == 0x0300
    assert channel(a, 'rpdo', 0x6072)['default'] == 480
    assert channel(b, 'rpdo', 0x6072)['default'] == 100


def test_arm_mode_maps_torque_offset_in_1603(tmp_path):
    # The RMD-X carries 0x60B2 only in the optional RxPDO 0x1603 = [0x60B0,
    # 0x60B1, 0x60B2] (firmware 26042402+); the default 0x1600 is unchanged.
    hw = make(tmp_path, [{'name': 'a', 'bus_position': 0, 'model': 'x4_10'}],
              control_mode='arm')
    cfg = hw.slave_config(hw.joints[0])
    pdos = {p['index']: [ch['index'] for ch in p['channels']] for p in cfg['rpdo']}
    assert pdos[0x1603] == [0x60b0, 0x60b1, 0x60b2]
    assert 0x60b2 not in pdos[0x1600]
    off = channel(cfg, 'rpdo', 0x60b2)
    assert off['command_interface'] == 'torque_offset'
    assert off['type'] == 'int16'
    assert off['factor'] == pytest.approx(1 / 0.00624)
    assert 'torque_offset' in hw.command_interfaces

    vel = make(tmp_path, [{'name': 'a', 'bus_position': 0, 'model': 'x4_10'}])
    assert 0x1603 not in [p['index'] for p in vel.slave_config(vel.joints[0])['rpdo']]


def test_controller_parameters_use_limits(tmp_path):
    hw = make(tmp_path, [
        {'name': 'a', 'bus_position': 0, 'model': 'x4_10'},
        {'name': 'b', 'bus_position': 1, 'model': 'x4_36', 'soft_limits_rad': [-0.2, 0.25],
         'max_velocity_rad_s': 1.0},
    ])
    params = hw.controller_parameters()
    assert params['joints'] == ['a', 'b']
    assert params['interface_name'] == 'velocity'
    assert params['position_min'] == [-1.0, -0.2]
    assert params['position_max'] == [2.0, 0.25]
    assert params['max_command'] == [2.0, 1.0]
    full = hw.controllers_yaml(update_rate=500, joint_state_rate=100)
    assert full['controller_manager']['ros__parameters']['update_rate'] == 500
    assert full['joint_state_broadcaster']['ros__parameters']['update_rate'] == 100
    assert full['controller_manager']['ros__parameters']['velocity_controller']['type'] == \
        'wisco_ethercat_ros2_hw/SafeCommandController'


@pytest.mark.parametrize('mode', ['arm', 'velocity', 'effort'])
def test_legacy_pdos_match_fixed_firmware_layout(tmp_path, mode):
    hw = make(tmp_path, [
        {'name': 'a', 'bus_position': 0, 'model': 'x6_60', 'pdo_profile': 'legacy'},
        {'name': 'b', 'bus_position': 1, 'model': 'x4_10'},
    ], control_mode=mode)
    cfg = hw.slave_config(hw.joints[0])
    assert [p['index'] for p in cfg['rpdo']] == [0x1600]
    assert [ch['index'] for ch in cfg['rpdo'][0]['channels']] == [
        0x6040, 0x607a, 0x60ff, 0x6071, 0x6072, 0x6060, 0x5ffe]
    assert [ch['index'] for ch in cfg['tpdo'][0]['channels']] == [
        0x6041, 0x6064, 0x606c, 0x6077, 0x603f, 0x6061, 0x5ffe]
    assert hw.arm_controller_parameters()['torque_offset_enabled'] == [False, True]
    assert 'assign_activate' not in cfg
    modern = hw.slave_config(hw.joints[1])
    assert modern['rpdo'][0]['channels'][-1]['index'] == 0x2ffd
    assert modern['tpdo'][0]['channels'][-1]['index'] == 0x2ffe


def test_arm01_mixed_firmware_interfaces(tmp_path):
    xacro = pytest.importorskip('xacro')
    description = os.path.join(ARM01, 'urdf', 'arm01.ros2_control.xacro')
    plain = xacro.process_file(description).toxml()
    hw = HardwareConfig(MODELS, os.path.join(ARM01, 'config', 'arm01.yaml'), plain,
                        template_file=TEMPLATE)
    for mock in (False, True):
        manifest = hw.write(str(tmp_path / str(mock)), use_mock_hardware=mock)
        urdf = ET.fromstring(xacro.process_file(
            description, mappings={'hardware_manifest': manifest}).toxml())
        joints = urdf.find('ros2_control').findall('joint')
        assert len(joints) == 7
        for joint in joints:
            names = [c.get('name') for c in joint.findall('command_interface')]
            legacy = joint.get('name') in ('shoulder_roll', 'shoulder_yaw')
            assert ('torque_offset' in names) == (not legacy)
            assert {'position', 'effort', 'mode_of_operation', 'max_torque'} <= set(names)


def test_commissioned_scales_are_applied_to_hardware(tmp_path):
    hw = make(tmp_path, [{'name': 'a', 'bus_position': 0, 'model': 'x4_10',
                          'counts_per_rev': 262144, 'rated_current_a': 7.8}],
              control_mode='arm', calibration={'kt_output_nm_per_a': {'a': 1.0}})
    hw.auto_state_transitions = False
    cfg = hw.slave_config(hw.joints[0])
    assert channel(cfg, 'tpdo', 0x6064)['factor'] == pytest.approx(2 * math.pi / 262144)
    assert channel(cfg, 'tpdo', 0x6077)['factor'] == pytest.approx(0.0078)
    assert cfg['auto_state_transitions'] is False
    params = hw.arm_controller_parameters()
    assert params['hold_on_startup'] and params['hold_on_timeout']
    assert params['gravity_enabled'] is False


def test_arm_controller_gets_weighed_link_masses(tmp_path):
    # the Python ArmModel applies link_masses_kg; the C++ gravity model must see them too
    hw = make(tmp_path, [{'name': 'a', 'bus_position': 0, 'model': 'x4_10'}])
    params = hw.arm_controller_parameters({'link_masses_kg': {'l1': 0.82, 'l2': None, 'l3': 1}})
    assert params['link_mass_names'] == ['l1', 'l3']
    assert params['link_mass_values'] == [0.82, 1.0]
    for table in ({'link_masses_kg': {'l1': None}}, {}, None):
        params = hw.arm_controller_parameters(table)
        assert 'link_mass_names' not in params and 'link_mass_values' not in params


def test_continuous_joint_limits_are_finite_in_parameters(tmp_path):
    hw = make(tmp_path, [{'name': 'c', 'bus_position': 0, 'model': 'x4_10'}])
    params = hw.controller_parameters()
    assert params['position_min'] == [-1e9]
    assert params['position_max'] == [1e9]


@pytest.mark.parametrize('joints, message', [
    ([{'name': 'nope', 'bus_position': 0, 'model': 'x4_10'}], 'not a movable joint'),
    ([{'name': 'fixed', 'bus_position': 0, 'model': 'x4_10'}], 'not a movable joint'),
    ([{'name': 'a', 'bus_position': 0, 'model': 'x9'}], 'unknown model'),
    ([{'name': 'a', 'bus_position': 0, 'model': 'x4_10', 'direction': 2}], 'direction'),
    ([{'name': 'a', 'bus_position': 0, 'model': 'x4_10', 'pdo_profile': 'typo'}], 'pdo_profile'),
    ([{'name': 'a', 'bus_position': 0, 'model': 'x4_10'},
      {'name': 'a', 'bus_position': 1, 'model': 'x4_10'}], 'listed twice'),
    ([{'name': 'a', 'bus_position': 0, 'model': 'x4_10'},
      {'name': 'b', 'bus_position': 0, 'model': 'x4_10'}], 'more than one joint'),
    ([{'name': 'a', 'bus_position': 1, 'model': 'x4_10'}], 'no gaps'),
    ([{'name': 'a', 'model': 'x4_10'}], 'missing'),
])
def test_validation_errors(tmp_path, joints, message):
    with pytest.raises(ConfigError, match=message):
        make(tmp_path, joints)


def test_bad_control_mode_and_calibration(tmp_path):
    joints = [{'name': 'a', 'bus_position': 0, 'model': 'x4_10'}]
    with pytest.raises(ConfigError, match='control_mode'):
        make(tmp_path, joints, control_mode='position')
    with pytest.raises(ConfigError, match='not in the URDF'):
        make(tmp_path, joints, calibration={'zero_offsets_rad': {'ghost': 0.1}})


def test_warnings(tmp_path):
    hw = make(tmp_path, [{'name': 'a', 'bus_position': 0, 'model': 'x4_10'}])
    text = '\n'.join(hw.warnings())
    assert 'no zero offset for a' in text
    # The datasheet scales (Kt x I_rated) sit below each model's peak torque, so
    # no model is flagged; a scale that implies more than peak still is.
    assert 'effort_nm_per_permille implies' not in text
    hw.models['x4_10'] = dict(hw.models['x4_10'], effort_nm_per_permille=0.015)
    assert any('model x4_10' in w for w in hw.warnings())


def test_shipped_tables_load(tmp_path):
    profiles = load_joint_profiles(MODELS, os.path.join(CONFIG, 'joints.yaml'))
    assert [p['bus_position'] for p in profiles.values()] == [0, 1, 2, 3]
    assert profiles['left_elbow']['k_stiction'] == pytest.approx(1.952)


def test_write_and_xacro_expansion(tmp_path):
    xacro = pytest.importorskip('xacro')
    plain = xacro.process_file(XACRO, mappings={'hardware_manifest': ''}).toxml()
    hw = HardwareConfig(MODELS, os.path.join(CONFIG, 'joints.yaml'), plain,
                        calibration_file=os.path.join(CONFIG, 'calibration.yaml'),
                        template_file=TEMPLATE)
    for mock in (False, True):
        out = tmp_path / ('mock' if mock else 'real')
        manifest = hw.write(str(out), use_mock_hardware=mock, control_frequency=250)
        urdf = ET.fromstring(xacro.process_file(
            XACRO, mappings={'hardware_manifest': manifest}).toxml())
        systems = {s.get('name'): s for s in urdf.findall('ros2_control')}
        ec = systems['EtherCATSystem']
        plugin = ec.find('hardware/plugin').text
        assert plugin == ('mock_components/GenericSystem' if mock
                          else 'ethercat_driver/EthercatDriver')
        names = [j.get('name') for j in ec.findall('joint')]
        assert names == ['left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw',
                         'left_elbow']
        for joint in ec.findall('joint'):
            assert [c.get('name') for c in joint.findall('command_interface')] == ['velocity']
        if not mock:
            params = {p.get('name'): p.text for p in ec.findall('hardware/param')}
            assert float(params['control_frequency']) == 250
            module = ec.findall('joint')[2].find('ec_module')
            mparams = {p.get('name'): p.text for p in module.findall('param')}
            assert mparams['position'] == '2'
            assert mparams['mode_of_operation'] == '9'
            assert mparams['slave_config'].endswith('drive_02_left_shoulder_yaw.yaml')
        rest = [j.get('name') for j in systems['MockSystem'].findall('joint')]
        assert len(rest) == 27 and 'left_elbow' not in rest
