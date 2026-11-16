from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from arm_stack.calib.stop_zero import verify_stop


@pytest.mark.parametrize('second_hit, expected', [
    (35.06, None),
    (36.30, 'did not repeat'),
    (-15.0, 'wrong side'),
])
def test_passive_stop_check_uses_independent_angles(monkeypatch, capsys, second_hit, expected):
    from arm_stack.config import load_arm
    joint = next(j for j in load_arm().joints if j.name == 'wrist_pitch')
    first_hit = -15.0 if expected == 'wrong side' else 35.05
    poses = iter([0.0, 10.0, first_hit, 0.0, second_hit])
    answers = iter(['upper', '0', '10', '', '', ''])
    monkeypatch.setattr('builtins.input', lambda prompt: next(answers))
    reader = SimpleNamespace(disable_for_verification=lambda arm: None,
                             capture=lambda arm, j: (next(poses) + 13) * joint.counts_per_rev / 360)
    reader.positions = lambda arm: nullcontext(reader)
    args = SimpleNamespace(arm='arm01', sim=False, verify_stop='wrist_pitch')
    if expected:
        with pytest.raises(ValueError, match=expected):
            verify_stop(args, reader)
    else:
        assert verify_stop(args, reader) == 0
        assert 'conditional CAD angle 35.04 deg' in capsys.readouterr().out


@pytest.mark.parametrize('side, endpoint', [('lower', -90), ('upper', 180)])
def test_j1_endpoints_relative_to_start(monkeypatch, capsys, side, endpoint):
    poses = iter([0, endpoint, 0, endpoint])
    answers = iter([side, '', '', '', ''])
    monkeypatch.setattr('builtins.input', lambda prompt: next(answers))
    reader = SimpleNamespace(disable_for_verification=lambda arm: None,
                             capture=lambda arm, j: 48000 + next(poses) * j.counts_per_rev / 360)
    reader.positions = lambda arm: nullcontext(reader)
    args = SimpleNamespace(arm='arm01', sim=False, verify_stop='shoulder_pitch')
    assert verify_stop(args, reader) == 0
    report = capsys.readouterr().out
    assert f'configured endpoint {endpoint:.2f} deg' in report
    assert f'hand-seated readings {endpoint:.2f}, {endpoint:.2f}' in report


def test_missing_torque_is_not_a_stop(monkeypatch):
    from arm_stack.calib import limits
    state = SimpleNamespace(q=np.zeros(1), qd=np.zeros(1), tau=np.zeros(1), t=0.0)
    client = SimpleNamespace(state=lambda: state, cmd_dt=0.01)
    ctx = SimpleNamespace(client=client, quick=False,
                          arm=SimpleNamespace(names=['axis'], joints=[SimpleNamespace(peak_torque_nm=5)]),
                          model=SimpleNamespace(n=1, lower=np.array([-1.0]), upper=np.array([1.0]),
                                                gravity=lambda q: np.zeros(1)))
    monkeypatch.setattr(limits, 'move_to', lambda *args, **kwargs: state)
    def stalled(*args, **kwargs):
        state.t += client.cmd_dt
        return state
    monkeypatch.setattr(limits, 'stream', stalled)
    assert limits.find_stop(ctx, 0, 1, np.zeros(1), np.ones(1)) is None


def test_verify_disables_only_enabled_drives_and_checks_status(monkeypatch):
    from arm_stack.calib.commission import EncoderReader
    from arm_stack.config import load_arm
    reader = EncoderReader()
    arm = load_arm()
    enabled = {1, 2}
    writes = []
    def command(*args):
        if args == ('master',):
            return 'Active: no\nSlaves: 7'
        _, _, pos, _, kind, index, sub, value = args
        assert (kind, index, sub, value) == ('uint16', '0x6040', '0', '0')
        writes.append(int(pos))
        enabled.remove(int(pos))
    monkeypatch.setattr(reader, 'command', command)
    monkeypatch.setattr(reader, 'upload', lambda pos, kind, index: 4 if pos in enabled else 0)
    reader.disable_for_verification(arm)
    assert writes == [1, 2]
