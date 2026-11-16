"""Physical-grasp and control/replay contract regressions."""
import json
from unittest.mock import patch

import mujoco
import numpy as np
import pytest

from sim.dynamics import CONTACT_DYNAMICS, LEGACY_DYNAMICS, provenance_dynamics, recorded_dynamics
from sim.panthera_env import PantheraSim
from tools.investigate_gripper import SCENARIOS, run, trajectory


def test_control_interpolation_preserves_endpoints_and_zero_step_updates():
    sim = PantheraSim()
    start = sim.data.ctrl[:6].copy()
    target = start.copy()
    target[0] += .08
    seen = []
    original = mujoco.mj_step

    def capture(model, data):
        seen.append(data.ctrl[:6].copy())
        original(model, data)

    sim.set_arm_ctrl(target)
    sim.step(0)
    np.testing.assert_array_equal(sim._applied_arm_ctrl, start)
    target[0] += .02  # Pending targets must not become the interpolation start.
    sim.set_arm_ctrl(target)
    with patch('sim.panthera_env.mujoco.mj_step', side_effect=capture):
        sim.step(17)
    np.testing.assert_allclose(seen, [start+(target-start)*j/17 for j in range(1,18)])
    np.testing.assert_array_equal(sim.data.ctrl[:6], target)
    np.testing.assert_array_equal(sim._applied_arm_ctrl, target)
    assert sim.data.time == pytest.approx(17*.002)
    sim.reset(randomize=False)
    np.testing.assert_array_equal(sim._applied_arm_ctrl, sim.data.ctrl[:6])
    sim.data.ctrl[:6] = target
    sim.sync_control_state()
    sim.step()
    np.testing.assert_array_equal(sim._applied_arm_ctrl, target)


def test_legacy_mode_holds_step_targets_without_interpolation():
    sim = PantheraSim(dynamics=LEGACY_DYNAMICS)
    assert sim.model.opt.impratio == 10
    target = sim.data.ctrl[:6].copy()
    target[0] += .05
    seen = []
    original = mujoco.mj_step

    def capture(model, data):
        seen.append(data.ctrl[:6].copy())
        original(model, data)

    sim.set_arm_ctrl(target)
    with patch('sim.panthera_env.mujoco.mj_step', side_effect=capture):
        sim.step(5)
    np.testing.assert_array_equal(seen, np.tile(target, (5,1)))


@pytest.mark.parametrize('scenario', ['seated55', 'seated30', 'shallow30', 'jump55'])
def test_physical_lift_and_shake_without_weld(scenario):
    case = SCENARIOS[scenario]
    initial, commands, err = trajectory(case, 2)
    result = run('production', case, initial, commands, err)
    assert result['retained'] and result['released']
    assert not any(p['last']['latched'] for p in result['phases'].values())
    assert result['phases']['close']['last']['grasped']
    assert not result['phases']['release']['last']['grasped']
    assert result['phases']['lift']['close_slip_mm'] < (1 if 'jump' in scenario else .4)
    assert result['phases']['recover']['close_slip_mm'] < 4
    assert not any(result['warning_counts'])


@pytest.mark.parametrize('scenario', ['heavy55', 'weak55'])
def test_insufficient_grip_cannot_hold_an_object(scenario):
    case = SCENARIOS[scenario]
    initial, commands, err = trajectory(case, 1)
    result = run('production', case, initial, commands, err)
    assert not result['retained']
    assert not result['phases']['recover']['last']['grasped']


def test_partial_opening_releases_without_legacy_command_threshold():
    case = SCENARIOS['partial_open55']
    initial, commands, err = trajectory(case, 1)
    result = run('production', case, initial, commands, err)
    assert result['released']
    assert result['phases']['release']['last']['normal'] == [0, 0]
    assert not result['phases']['release']['last']['grasped']


def test_contact_detector_requires_load_persistence_and_clears_on_loss():
    sim = PantheraSim()
    sim.set_gripper(0)
    forces = np.zeros((3,2))
    forces[0] = .1
    with patch.object(sim, 'pad_normal_forces', return_value=forces):
        sim.step(9)
        assert not sim.grasped
        sim.step()
        assert sim.grasp_flags().tolist() == [True, False, False]
        sim.set_gripper(1)
        sim.step()
        assert sim.grasped  # An open command alone is not physical release.
        forces[0,1] = 0
        sim.step()
        assert not sim.grasped
    assert not any(sim.data.eq_active[e] for e in sim._grasp_eq)


def test_dynamics_provenance_and_mixed_dataset_rejection(tmp_path):
    from teleop.episode import Episode, FIELDS
    episode = Episode()
    episode.add({key: 0. for key in FIELDS})
    episode.save(tmp_path, {}, 30, False)
    metadata = json.loads((tmp_path/'meta.json').read_text())
    assert recorded_dynamics(metadata) == CONTACT_DYNAMICS
    assert recorded_dynamics({}) == LEGACY_DYNAMICS
    assert provenance_dynamics({'sources': {'a': metadata}}) == CONTACT_DYNAMICS
    with pytest.raises(ValueError, match='mixes'):
        provenance_dynamics({'sources': {'a': metadata, 'b': {}}})
    with pytest.raises(ValueError, match='Unknown'):
        recorded_dynamics({'simulation_dynamics': 'future-unknown'})
