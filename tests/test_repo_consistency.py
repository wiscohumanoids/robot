"""Cross-file consistency checks (no ROS needed). They catch the silent breakage that moving or
renaming files causes: a robot profile pointing at a world that no longer exists, a manifest naming a
joint the robot does not have, a hard-coded container path that went stale.

    python -m pytest tests -q
"""
import ast
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROFILES = sorted((ROOT / 'ros2_ws/src/hsim_bringup/config/robots').glob('*.yaml'))
MANIFESTS = sorted((ROOT / 'models').glob('**/manifest.yaml'))
CONTAINER_ROOT = '/hsim/'   # the repo root inside the ROS container (infra/docker/compose.yaml)


def _tracked(*patterns):
    """Tracked or new files matching the patterns."""
    out = subprocess.run(['git', 'ls-files', '--cached', '--others', '--exclude-standard', *patterns],
                         cwd=ROOT, capture_output=True, text=True).stdout.split('\n')
    return [ROOT / f for f in out if f and (ROOT / f).exists()]


def _host(path):
    """/hsim/x -> <repo>/x"""
    assert path.startswith(CONTAINER_ROOT), path
    return ROOT / path[len(CONTAINER_ROOT):]


def _world_joints(world):
    """Joint names in a MuJoCo world and the files it includes (mjlab prefixes them with 'robot/')."""
    names, todo = set(), [world]
    while todo:
        f = todo.pop()
        text = f.read_text()
        names |= {n.split('/')[-1] for n in re.findall(r'<joint\b[^>]*\bname="([^"]+)"', text)}
        todo += [f.parent / i for i in re.findall(r'<include\b[^>]*\bfile="([^"]+)"', text)]
    return names


@pytest.mark.parametrize('profile', PROFILES, ids=lambda p: p.stem)
def test_robot_profile_paths_and_joints(profile):
    p = yaml.safe_load(profile.read_text())
    assert p['name'] == profile.stem, 'profile name must match its file name (robot:=<name>)'
    world, policy = _host(p['world']), _host(p['policy'])
    assert world.is_file(), f'world not found: {p["world"]}'
    assert (policy / 'manifest.yaml').is_file() and (policy / 'policy.onnx').is_file(), f'policy incomplete: {p["policy"]}'
    man = yaml.safe_load((policy / 'manifest.yaml').read_text())
    missing = [j['name'] for j in man['joints'] if j['name'] not in _world_joints(world)]
    assert not missing, f'{profile.stem}: manifest joints not in the world: {missing}'


@pytest.mark.parametrize('manifest', MANIFESTS, ids=lambda p: p.parent.name)
def test_manifest_is_self_consistent(manifest):
    m = yaml.safe_load(manifest.read_text())
    assert (manifest.parent / 'policy.onnx').is_file(), 'manifest without policy.onnx'
    names = [j['name'] for j in m['joints']]
    assert len(names) == len(set(names)), 'duplicate joint names'
    for j in m['joints']:
        assert {'name', 'default_pos', 'kp', 'kd'} <= set(j), j
    if m.get('manifest_version', 1) >= 2:
        assert set(m['action']['joints']) <= set(names), 'action joints not in joints'
        held = m.get('held_joints', {}).get('names', [])
        assert set(held) <= set(names), 'held joints not in joints'
        assert not set(held) & set(m['action']['joints']), 'a joint is both policy-driven and held'


def test_container_paths_exist():
    """Every literal /hsim/<path> in code and config points at something in the repo (or a build output)."""
    generated = ('ros2_ws/install', 'ros2_ws/build', 'assets/envs')   # created by colcon / tools/env_build.py
    bad = []
    for f in _tracked('*.py', '*.sh', '*.yaml', '*.yml', '*.xml'):
        if f.is_relative_to(ROOT / 'manipulation') or f.is_relative_to(ROOT / 'hardware') or f.is_relative_to(ROOT / 'tests'):
            continue   # standalone components (not run inside the ROS container), and these tests
        for m in re.finditer(r'(?<![\w.])/hsim/([A-Za-z0-9_./-]+)', f.read_text(errors='ignore')):
            rel = m.group(1).rstrip('./')
            if rel.startswith(generated) or rel.startswith('.') or not rel:
                continue
            if not (ROOT / rel).exists():
                bad.append(f'{f.relative_to(ROOT)}: /hsim/{rel}')
    assert not bad, 'stale container paths:\n  ' + '\n  '.join(bad)


def test_every_python_file_parses():
    for f in _tracked('*.py'):
        ast.parse(f.read_text(errors='ignore'), str(f))


@pytest.mark.skipif(not shutil.which('bash'), reason='needs bash')
def test_shell_scripts_have_valid_syntax():
    bad = [f'{f.relative_to(ROOT)}: {r.stderr.strip()}' for f in _tracked('*.sh')
           if (r := subprocess.run(['bash', '-n', str(f)], capture_output=True, text=True)).returncode]
    assert not bad, '\n'.join(bad)
