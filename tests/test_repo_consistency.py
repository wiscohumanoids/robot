"""Cross-file consistency checks that used to be "verified once by hand":
the 23-joint order and nominal pose live in several files and must not drift.
Also proves the URDF actually expands (a broken xacro stops every launch file).
Runs anywhere; the xacro test needs `pip install xacro` and skips without it."""
import ast
import os

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'src')
XACRO = os.path.join(SRC, 'g1_description', 'g1_23dof.urdf.xacro')


def canonical():
    with open(os.path.join(SRC, 'humanoid_interfaces', 'config', 'canonical_joint_order.yaml')) as f:
        return yaml.safe_load(f)['joint_names']


def test_canonical_order_has_23_unique_joints():
    names = canonical()
    assert len(names) == 23 and len(set(names)) == 23


def test_controllers_yaml_matches_canonical_order():
    with open(os.path.join(SRC, 'bringup', 'config', 'controllers.yaml')) as f:
        params = yaml.safe_load(f)['joint_impedance_controller']['ros__parameters']
    assert params['joints'] == canonical()
    assert len(params['kp']) == len(params['kd']) == 23


def test_ethercat_slaves_match_canonical_order():
    with open(os.path.join(SRC, 'ethercat_bridge', 'config', 'ethercat_slaves.yaml')) as f:
        slaves = yaml.safe_load(f)['slaves']
    assert [s['joint'] for s in slaves] == canonical()


def test_cpp_default_joint_list_matches_canonical_order():
    import re
    text = open(os.path.join(SRC, 'lowlevel_control', 'src', 'joint_impedance_controller.cpp')).read()
    start = text.index('kDefaultJoints')
    block = text[start:text.index('};', start)]
    assert re.findall(r'"(\w+_joint)"', block) == canonical()


def _nominal_pose(path):
    tree = ast.parse(open(path).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', '') == 'NOMINAL_POSE':
            return ast.literal_eval(node.value)
    raise AssertionError(f'{path}: no NOMINAL_POSE')


def _expanded_urdf():
    xacro = pytest.importorskip('xacro')
    import xml.etree.ElementTree as ET
    return ET.fromstring(xacro.process_file(XACRO).toxml())


@pytest.mark.parametrize('plugin', [
    'mujoco_ros2_control/MujocoSystem', 'ethercat_bridge/EthercatHardwareInterface'])
def test_urdf_expands_for_both_hardware_plugins(plugin):
    xacro = pytest.importorskip('xacro')
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xacro.process_file(XACRO, mappings={'hardware_plugin': plugin}).toxml())
    assert root.find('ros2_control/hardware/plugin').text == plugin


def test_urdf_matches_canonical_order_and_has_single_base_link_root():
    root = _expanded_urdf()
    assert [j.get('name') for j in root.find('ros2_control').findall('joint')] == canonical()
    links = [l.get('name') for l in root.findall('link')]
    children = {j.find('child').get('link') for j in root.findall('joint')}
    assert [l for l in links if l not in children] == ['base_link']
    joints = {j.get('name'): j for j in root.findall('joint')}
    for name in canonical():
        assert joints[name].find('limit').get('effort'), f'{name} has no effort limit'


def test_nominal_poses_match_urdf_initial_values():
    root = _expanded_urdf()
    initial = [float(j.find("state_interface[@name='position']/param").text)
               for j in root.find('ros2_control').findall('joint')]
    for pkg, mod in (('locomotion_runner', 'locomotion_node'), ('manipulation_runner', 'manipulation_node')):
        path = os.path.join(SRC, pkg, pkg, mod + '.py')
        assert _nominal_pose(path) == initial, f'{pkg} NOMINAL_POSE drifted from the URDF initial_value'


def test_mjcf_has_every_canonical_joint():
    import xml.etree.ElementTree as ET
    mj = ET.parse(os.path.join(SRC, 'g1_description', 'g1_23dof_rev_1_0.xml')).getroot()
    names = {j.get('name') for j in mj.iter('joint')}
    assert set(canonical()) <= names


def test_package_manifests_are_rosdep_resolvable_shape():
    """`rosdep install` cannot resolve <buildtool_depend>ament_python</...>:
    ament_python is a build TYPE (declared in <export>), not a package. It broke
    the Docker build once; every ament_python package must declare it only there."""
    import glob
    import xml.etree.ElementTree as ET
    for path in glob.glob(os.path.join(SRC, '*', 'package.xml')):
        root = ET.parse(path).getroot()
        build_type = root.find('export/build_type').text
        buildtools = [e.text for e in root.findall('buildtool_depend')]
        assert 'ament_python' not in buildtools, f'{path}: remove <buildtool_depend>ament_python</...>'
        if build_type == 'ament_cmake':
            assert 'ament_cmake' in buildtools, f'{path}: ament_cmake package without buildtool_depend'


def test_shell_scripts_have_valid_syntax():
    import glob
    import subprocess
    scripts = glob.glob(os.path.join(ROOT, 'scripts', '*.sh')) + glob.glob(os.path.join(ROOT, 'docker', '*.sh'))
    assert scripts
    for path in scripts:
        result = subprocess.run(['bash', '-n', path], capture_output=True, text=True)
        assert result.returncode == 0, f'{path}: {result.stderr}'


def test_controller_declares_parameters_only_if_absent():
    """controller_manager pre-declares every parameter from controllers.yaml; an
    unconditional declare_parameter() throws ParameterAlreadyDeclaredException and
    takes down the whole controller_manager process (seen on first sim launch)."""
    text = open(os.path.join(SRC, 'lowlevel_control', 'src', 'joint_impedance_controller.cpp')).read()
    declared = text.count('->declare_parameter<')
    guarded = text.count('if (!node->has_parameter(')
    assert declared >= 1 and declared == guarded, (
        f'{declared} declare_parameter calls but {guarded} has_parameter guards')


def _urdf_effort_limits():
    root = _expanded_urdf()
    joints = {j.get('name'): j for j in root.findall('joint')}
    return [float(joints[n].find('limit').get('effort')) for n in canonical()]


def test_controller_effort_limits_equal_urdf_limits():
    """The MuJoCo bridge never clamps to the URDF limit (has_effort_limits is never
    set), so lowlevel_control must clamp, and its numbers must match the URDF."""
    import re
    with open(os.path.join(SRC, 'bringup', 'config', 'controllers.yaml')) as f:
        params = yaml.safe_load(f)['joint_impedance_controller']['ros__parameters']
    urdf = _urdf_effort_limits()
    assert params['effort_limits'] == urdf
    cpp = open(os.path.join(SRC, 'lowlevel_control', 'src', 'joint_impedance_controller.cpp')).read()
    start = cpp.index('kDefaultEffortLimits')
    block = cpp[start:cpp.index('};', start)]
    assert [float(x) for x in re.findall(r'(\d+\.\d+)', block)] == urdf
