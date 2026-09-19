"""Static interface-contract enforcement: runs anywhere, no ROS needed.

Catches, at PR time, the mistakes the live checker (bringup/check_contract.py)
would only catch after launching: a second publisher on a topic, a node that
publishes something the contract doesn't list, a stub missing its executable,
a launch switch that doesn't exist, docs that drifted from the YAML.
"""
import ast
import glob
import os
import re
import subprocess
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'src')
CONTRACT = os.path.join(SRC, 'humanoid_interfaces', 'config', 'interface_contract.yaml')

with open(CONTRACT) as _f:
    C = yaml.safe_load(_f)
NODES = {n['id']: n for n in C['nodes']}
STATUSES = {'real', 'stub', 'scaffold', 'external', 'missing'}
# Vendored third-party code: not held to the contract-literal scan.
SCAN_SKIP = {'mujoco_ros2_control'}


def _sources(package):
    base = os.path.join(SRC, package)
    for pattern in ('**/*.py', '**/*.cpp'):
        for path in glob.glob(os.path.join(base, pattern), recursive=True):
            if '/test/' not in path and '/scripts/' not in path:
                yield path


def _publisher_topics(package):
    """Literal topic names passed to create_publisher in a package's sources."""
    found = set()
    py = re.compile(r"create_publisher\(\s*\w+\s*,\s*'(/[^']+)'")
    cpp = re.compile(r'create_publisher<[\w:]+>\(\s*"(/[^"]+)"')
    for path in _sources(package):
        text = open(path).read()
        found |= set(py.findall(text)) | set(cpp.findall(text))
    return found


def test_schema_and_references():
    assert C['contract_version'] == 1
    layer_ids = {l['id'] for l in C['layers']}
    assert len(NODES) == len(C['nodes']), 'duplicate node ids'
    for n in C['nodes']:
        assert n['status'] in STATUSES, n['id']
        assert n['layer'] in layer_ids, n['id']
        assert n.get('summary') and n.get('area'), n['id']
    for t in C['topics']:
        assert t['owner'] in NODES, f"{t['name']}: unknown owner {t['owner']}"
        assert all(x in NODES for x in t['consumers']), t['name']
        assert t['check'] in ('rate', 'latched', 'event'), t['name']
        if t['check'] == 'rate':
            assert 0 < t['min_rate_hz'] <= t['rate_hz'], t['name']
    for a in C['actions']:
        assert a['server'] in NODES and all(x in NODES for x in a['clients']), a['name']
    for e in C['tf']:
        assert e['owner'] in NODES, e


def test_one_publisher_per_topic_and_per_tf_child():
    names = [t['name'] for t in C['topics']]
    assert len(names) == len(set(names)), 'a topic is listed twice'
    children = [e['child'] for e in C['tf']]
    assert len(children) == len(set(children)), 'a TF frame has two contract parents'


def test_owner_status_matches_reality():
    """A 'missing' node cannot be relied on to publish; a rate-checked topic
    must therefore not be owned by one (it would fail the live check forever)."""
    for t in C['topics']:
        if t['check'] in ('rate', 'latched'):
            assert NODES[t['owner']]['status'] != 'missing', t['name']


def test_python_packages_have_the_declared_executable():
    for n in C['nodes']:
        if not n.get('package') or not n.get('executable'):
            continue
        setup_py = os.path.join(SRC, n['package'], 'setup.py')
        assert os.path.isfile(setup_py), f"{n['id']}: package {n['package']} not in src/"
        assert f"'{n['executable']} = " in open(setup_py).read(), (
            f"{n['id']}: {n['package']} setup.py lacks console_script {n['executable']}")


def test_launch_switches_exist_in_full_stack():
    text = open(os.path.join(SRC, 'bringup', 'launch', 'full_stack.launch.py')).read()
    for n in C['nodes']:
        arg = n.get('launch_arg')
        if arg:
            assert f"'{arg}'" in text, f"{n['id']}: launch_arg {arg} not handled by full_stack.launch.py"


def test_every_literal_publisher_topic_is_in_the_contract_with_the_right_owner():
    owner_of = {t['name']: NODES[t['owner']].get('package') for t in C['topics']}
    for package in sorted(os.listdir(SRC)):
        if package in SCAN_SKIP or not os.path.isdir(os.path.join(SRC, package)):
            continue
        for topic in _publisher_topics(package):
            assert topic in owner_of, (
                f'{package} publishes {topic}, which is not in interface_contract.yaml '
                f'(add it, or you are creating an undeclared interface)')
            assert owner_of[topic] == package, (
                f'{package} publishes {topic}, but the contract says {owner_of[topic]} owns it '
                f'(two publishers on one topic)')


def test_every_owned_topic_is_published_by_its_owner_package():
    """The reverse direction: the contract must not promise a topic that the
    owner's code never publishes (excluding third-party/missing owners)."""
    for t in C['topics']:
        owner = NODES[t['owner']]
        if owner['status'] in ('missing', 'external') or not owner.get('package'):
            continue
        assert t['name'] in _publisher_topics(owner['package']), (
            f"{t['name']}: owner {owner['package']} has no create_publisher for it")


def test_tf_edges_published_in_code_match_the_contract():
    contract_edges = {(e['parent'], e['child']): NODES[e['owner']].get('package') for e in C['tf']}
    for package in sorted(os.listdir(SRC)):
        if package in SCAN_SKIP or not os.path.isdir(os.path.join(SRC, package)):
            continue
        for path in _sources(package):
            text = open(path).read()
            if 'sendTransform' not in text:
                continue
            # convention: the TransformStamped variable is named `tf` and its frame
            # ids are string literals, so the edge can be read statically
            parent = re.search(r"\btf\.header\.frame_id = '(\w+)'", text)
            child = re.search(r"\btf\.child_frame_id = '(\w+)'", text)
            assert parent and child, (
                f'{path}: sendTransform needs `tf.header.frame_id = \'..\'` and '
                f'`tf.child_frame_id = \'..\'` literals so the contract can be checked')
            edge = (parent.group(1), child.group(1))
            assert edge in contract_edges, f'{path} publishes TF {edge} not in the contract'
            assert contract_edges[edge] == package, f'{path}: TF {edge} owned by {contract_edges[edge]}'


def test_custom_message_types_exist():
    for t in C['topics']:
        pkg, kind, name = t['type'].split('/')
        if pkg == 'humanoid_interfaces':
            assert os.path.isfile(os.path.join(SRC, pkg, kind, name + '.' + kind)), t['type']
    for a in C['actions']:
        pkg, kind, name = a['type'].split('/')
        if pkg == 'humanoid_interfaces':
            assert os.path.isfile(os.path.join(SRC, pkg, kind, name + '.' + kind)), a['type']


def test_msg_files_are_registered_in_cmake():
    cmake = open(os.path.join(SRC, 'humanoid_interfaces', 'CMakeLists.txt')).read()
    for path in glob.glob(os.path.join(SRC, 'humanoid_interfaces', '*', '*.[ma]*')):
        kind = os.path.basename(os.path.dirname(path))
        if kind in ('msg', 'action'):
            assert f'"{kind}/{os.path.basename(path)}"' in cmake, f'{path} not in CMakeLists.txt'


def test_every_python_source_parses():
    for path in glob.glob(os.path.join(SRC, '**', '*.py'), recursive=True):
        ast.parse(open(path).read(), path)


def test_generated_docs_are_current():
    result = subprocess.run(
        [sys.executable, os.path.join(ROOT, 'tools', 'gen_docs.py'), '--check'],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
