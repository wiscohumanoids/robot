"""The C++/Python parity golden file (test/data/arm01_parity.json) is current.

test/test_gravity_parity.cpp checks the controller's C++ GravityModel against
this file; this test checks the Python ArmModel still reproduces it, so the two
models are tied to the same numbers. On a failure after an intended model or
URDF change, regenerate: python3 tools/make_parity_golden.py
"""

import importlib.util
import json
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GOLDEN = os.path.join(ROOT, 'test', 'data', 'arm01_parity.json')


def _generator():
    spec = importlib.util.spec_from_file_location('make_parity_golden',
                                                  os.path.join(ROOT, 'tools', 'make_parity_golden.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _close(a, b, path=''):
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), path
        for k in a:
            _close(a[k], b[k], path + '/' + str(k))
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _close(x, y, '%s[%d]' % (path, i))
    elif isinstance(a, float) or isinstance(b, float):
        assert a is not None and b is not None and abs(a - b) <= 1e-12 * max(1.0, abs(a)), (path, a, b)
    else:
        assert a == b, (path, a, b)


@pytest.fixture(scope='module')
def golden():
    with open(GOLDEN) as f:
        return json.load(f)


def test_python_model_reproduces_the_golden_file(golden):
    _close(_generator().make(golden['seed'], len(golden['cases']) - 1), golden)


def test_golden_exercises_every_calibration_layer(golden):
    cal = golden['calibration']
    li = cal['link_inertials']
    assert any(e['mass_kg'] is None for e in li.values())                 # mass from link_masses_kg
    assert any(None in e['com_m'] for e in li.values())                   # partial COM: ignored
    assert 'wrist_crank' in li                                            # not a driven joint's link
    assert set(golden['link_masses_kg']) - set(li)                         # weighed only
    assert len(cal['friction']) == len(golden['joints'])
    ov = cal['kinematic_overlay']
    assert any('drot' in o for o in ov.values()) and any('drot' not in o for o in ov.values())
    g, g0 = (np.array([c[k] for c in golden['cases']]) for k in ('gravity', 'gravity_nominal'))
    assert np.abs(g - g0).max() > 0.05
