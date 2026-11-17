"""mocap_check.summarize: rate, gaps, still-body noise and the pass rule."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('mocap_check', Path(__file__).parents[1] / 'scripts' / 'mocap_check.py')
mocap_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mocap_check)


def stream(n=2400, hz=240.0, noise_m=0.00005, gap_at=None, seed=0):
    """A still body at 240 Hz with Gaussian position noise; optionally a 100 ms gap."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / hz
    if gap_at is not None:
        t[gap_at:] += 0.1
    pos = np.array([0.5, 0.1, 0.3]) + rng.normal(0, noise_m, (n, 3))
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)) + rng.normal(0, 1e-5, (n, 4))
    return t, pos, quat


def test_clean_stream_passes():
    """A steady, quiet 240 Hz stream passes."""
    s = mocap_check.summarize(*stream(), rate=240)
    assert s['pass'] and abs(s['rate_hz'] - 240) < 1 and s['gaps'] == 0
    assert max(s['noise_3sigma_mm']) < 0.2 and s['orientation_3sigma_deg'] < 0.1


def test_gap_noise_and_rate_fail():
    """A gap, too much noise, a wrong rate or too few samples fail with the reason."""
    assert 'gap' in mocap_check.summarize(*stream(gap_at=1000), rate=240)['reason']
    assert 'noise' in mocap_check.summarize(*stream(noise_m=0.001), rate=240)['reason']
    assert 'rate' in mocap_check.summarize(*stream(hz=100.0), rate=240)['reason']
    assert not mocap_check.summarize([0.0], [[0, 0, 0]], [[0, 0, 0, 1]])['pass']
