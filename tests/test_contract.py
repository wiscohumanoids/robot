"""Static interface-contract checks: run anywhere, no ROS needed.

The live check is `ros2 run hsim_bringup contract_check --phase N` (needs a running stack).
These tests catch the cheaper mistakes before anything is launched:
  - the contract YAML is well formed
  - every owner in the contract is a real node (in code or a launch file)
  - every topic the stack's own code publishes (literal name) is in the contract
  - every custom interface type exists on disk and is registered in CMake (and the reverse)

    python -m pytest tests -q
"""
import fnmatch
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'ros2_ws' / 'src'
CONTRACT_PATH = SRC / 'hsim_bringup' / 'config' / 'interface_contract.yaml'
CONTRACT = yaml.safe_load(CONTRACT_PATH.read_text())
INTERFACES = SRC / 'hsim_interfaces'

# Packages whose publishers are outside the stack contract: the ros2_control hardware side (its controllers
# talk to ros2_control's own topics) and the robot description packages.
OUTSIDE_CONTRACT = ('wisco_ethercat_ros2_hw', 'arm01_description')

# Contract owners that are people / CLIs rather than nodes started by our launch files.
NON_NODE_OWNERS = {'operator'}

TYPE_RE = re.compile(r'^[a-z0-9_]+/(msg|srv|action)/[A-Z][A-Za-z0-9]*$')
PY_PUB = re.compile(r'create_publisher\s*\(\s*[\w.]+\s*,\s*["\']([^"\']+)["\']')
CPP_PUB = re.compile(r'create_publisher\s*<[^>]*>\s*\(\s*["\']([^"\']+)["\']')


def _owners(spec):
    """Owners of a contract entry as a list."""
    o = spec['owner']
    return o if isinstance(o, list) else [o]


def _stack_sources():
    """Source files of the robot stack that the contract covers (not tests or packages outside it)."""
    for p in list(SRC.rglob('*.py')) + list(SRC.rglob('*.cpp')) + list((ROOT / 'tools').rglob('*.py')):
        rel = p.relative_to(ROOT).as_posix()
        if any(f'/src/{pkg}/' in f'/{rel}' for pkg in OUTSIDE_CONTRACT) or '/test/' in rel:
            continue
        yield p


def test_contract_schema():
    assert isinstance(CONTRACT.get('version'), int)
    for name, spec in CONTRACT['topics'].items():
        assert name.startswith('/'), name
        assert TYPE_RE.match(spec['type']), f'{name}: bad type {spec["type"]!r}'
        assert _owners(spec) and all(isinstance(o, str) and o for o in _owners(spec)), name
        assert spec.get('rate_hz') is None or spec['rate_hz'] > 0, name
        assert isinstance(spec['phase'], int), name
        assert set(spec) <= {'type', 'owner', 'rate_hz', 'phase', 'required'}, f'{name}: unknown keys'
    children = [e['child'] for e in CONTRACT['tf_edges']]
    assert len(children) == len(set(children)), 'a TF child frame has more than one parent/owner'
    for e in CONTRACT['tf_edges']:
        assert set(e) == {'parent', 'child', 'owner', 'static', 'phase'}, e
    for key, kind in (('actions', 'action'), ('services', 'srv')):
        for name, spec in CONTRACT.get(key, {}).items():
            assert name.startswith('/'), name
            assert set(spec) == {'type', 'server', 'clients', 'phase'}, f'{name}: keys {sorted(spec)}'
            assert TYPE_RE.match(spec['type']) and f'/{kind}/' in spec['type'], f'{name}: bad type {spec["type"]!r}'


def test_every_owner_is_a_real_node():
    """A stub and the real node share a ROS node name; that name must exist somewhere."""
    text = '\n'.join(p.read_text(errors='ignore') for p in
                     list(SRC.rglob('*.py')) + list(SRC.rglob('*.cpp')) + list((ROOT / 'tools').rglob('*.py')))
    owners = {o for s in CONTRACT['topics'].values() for o in _owners(s)} | {e['owner'] for e in CONTRACT['tf_edges']}
    missing = []
    for o in sorted(owners - NON_NODE_OWNERS):
        pats = [f"name='{o}'", f'name="{o}"', f"('{o}'", f'("{o}"', f"'{o}':", f'"{o}":']
        if not any(p in text for p in pats):
            missing.append(o)
    assert not missing, f'contract owners with no node of that name in code or launch files: {missing}'


def test_every_literal_publisher_is_in_the_contract():
    topics = set(CONTRACT['topics'])
    shared = CONTRACT.get('shared_topics', [])
    extra = {}
    for p in _stack_sources():
        t = p.read_text(errors='ignore')
        for m in list(PY_PUB.finditer(t)) + list(CPP_PUB.finditer(t)):
            name = m.group(1)
            if name.startswith('/') and name not in topics and not any(fnmatch.fnmatch(name, s) for s in shared):
                extra.setdefault(name, []).append(p.relative_to(ROOT).as_posix())
    assert not extra, f'published in code but missing from {CONTRACT_PATH.name}: {extra}'


def _registered_interfaces():
    """Interface files registered in hsim_interfaces' CMake."""
    cmake = (INTERFACES / 'CMakeLists.txt').read_text()
    return set(re.findall(r'"((?:msg|srv|action)/[A-Za-z0-9]+\.(?:msg|srv|action))"', cmake))


def test_interface_files_match_cmake():
    """Every .msg/.srv/.action is registered in CMake, and every registered one exists on disk."""
    on_disk = {p.relative_to(INTERFACES).as_posix() for kind in ('msg', 'srv', 'action')
               for p in (INTERFACES / kind).glob(f'*.{kind}')}
    registered = _registered_interfaces()
    assert registered - on_disk == set(), f'registered in CMake but missing on disk: {registered - on_disk}'
    assert on_disk - registered == set(), f'on disk but not registered in CMake: {on_disk - registered}'


def test_custom_types_in_the_contract_exist():
    registered = _registered_interfaces()
    entries = {**CONTRACT['topics'], **CONTRACT.get('actions', {}), **CONTRACT.get('services', {})}
    for name, spec in entries.items():
        pkg, kind, typ = spec['type'].split('/')
        if pkg == 'hsim_interfaces':
            assert f'{kind}/{typ}.{kind}' in registered, f'{name}: {spec["type"]} is not defined'
