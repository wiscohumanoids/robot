"""Shared physics, dual IK isolation and smooth command contract regressions."""
import mujoco
import numpy as np
import pytest

from sim.bimanual_actuator_env import BimanualActuatorSim, CAMERAS
from sim.actuator_env import TABLE_Z
from tools.bimanual_actuator_ik import BimanualTrial, smoothstep
from tools.try_actuator_assembly import pin_entry_ready, pin_placement


@pytest.fixture(scope='module')
def sim():
    return BimanualActuatorSim(seed=17)


def test_two_arms_three_attached_camera_streams_and_units(sim):
    assert sim.model.nu == 14
    assert sim.model.ncam == 3
    assert sim.model.neq == 2  # Only finger couplings, no grasp constraints.
    assert set(sim.cameras) == set(CAMERAS)
    for side in ('left', 'right'):
        cid = sim.model.camera(side+'_wrist').id
        prefix = '' if side == 'left' else 'right_'
        assert sim.model.cam_bodyid[cid] == sim.model.body(prefix+'link6').id
    assert sim.state().shape == sim.action().shape == (14,)
    assert sim.state_names[6].endswith('gripper_open_m')


def test_gears_start_directly_on_table_without_pickup_stands(sim):
    sim.reset()
    for name in ('gear_1', 'gear_2', 'gear_3', 'gear_4'):
        assert mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, name+'_rack') == -1
        assert sim.data.xpos[sim.model.body(name).id, 2] == pytest.approx(TABLE_Z+.0002)


def test_randomized_reset_is_repeatable_and_does_not_erase_right_home(sim):
    sim.reset()
    qpos, ctrl = sim.data.qpos.copy(), sim.data.ctrl.copy()
    sim.step(40)
    sim.reset()
    np.testing.assert_array_equal(sim.data.qpos, qpos)
    np.testing.assert_array_equal(sim.data.ctrl, ctrl)
    np.testing.assert_array_equal(sim.data.qvel, 0.)
    for arm in sim.arms.values():
        np.testing.assert_array_equal(arm.q, arm._q_nominal)
    sim.seed += 1; sim.reset()
    assert not np.array_equal(qpos, sim.data.qpos)


@pytest.mark.parametrize('side', ['left', 'right'])
def test_ik_is_independent_and_reports_actual_returned_residual(sim, side):
    sim.reset()
    before = sim.data.qpos.copy()
    arm = sim.arms[side]
    target, quat = arm.ee_pose()
    target += [.005, -.003, -.005]
    q, pe, re = arm.ik(target, quat, iters=2, max_joint_step=.001)
    np.testing.assert_array_equal(sim.data.qpos, before)
    assert np.max(np.abs(q-arm.q)) <= .001+1e-12
    scratch = mujoco.MjData(sim.model); scratch.qpos[:] = before
    scratch.qpos[arm.arm_qadr] = q
    mujoco.mj_kinematics(sim.model, scratch)
    err = arm._ee_error(target, quat, scratch)
    assert pe == pytest.approx(np.linalg.norm(err[:3]), abs=1e-12)
    assert re == pytest.approx(np.linalg.norm(err[3:]), abs=1e-12)


def test_both_pending_commands_interpolate_and_zero_steps_preserve_them(sim):
    sim.reset()
    for side, arm in sim.arms.items():
        q = arm.q; q[0] += .02 if side == 'left' else -.02
        arm.set_arm_ctrl(q)
    action = sim.action().copy()
    sim.step(0)
    np.testing.assert_array_equal(sim.action(), action)
    sim.step(10)
    np.testing.assert_array_equal(sim.action(), action)
    for arm in sim.arms.values():
        np.testing.assert_array_equal(arm.applied, sim.data.ctrl[arm.act_ids])
    with pytest.raises(ValueError): sim.step(1.5)


def test_command_speed_acceleration_and_stationary_support_arm(sim, tmp_path):
    sim.reset()
    trial = BimanualTrial(sim, tmp_path)
    trial.select('left')
    p, q = trial.target.copy(), trial.quat.copy()
    support_target = trial.poses['right'][0].copy()
    initial = sim.action().copy()
    for _ in range(20): trial.tick(p+[.015, -.02, -.01], q, 1.)
    actions = np.vstack([initial, trial.actions])
    joints = actions[:, [0,1,2,3,4,5,7,8,9,10,11,12]]
    velocities = np.diff(joints, axis=0)/.05
    assert abs(velocities).max() <= .6+1e-10
    accelerations = np.diff(np.vstack([np.zeros(12), velocities]), axis=0)/.05
    assert abs(accelerations).max() <= 2.+1e-10
    # Supporting arm's Cartesian target is retained throughout.
    np.testing.assert_array_equal(trial.poses['right'][0], support_target)
    assert all(len(a) == 14 for a in trial.actions)


def test_quintic_endpoint_derivatives():
    assert smoothstep(0) == 0 and smoothstep(1) == 1
    u = np.linspace(0,1,101)
    assert np.all(np.diff(smoothstep(u)) >= 0)
    assert smoothstep(1e-5)/1e-5 < 1e-7
    assert (1-smoothstep(1-1e-5))/1e-5 < 1e-7


def test_shallow_pin_entry_allows_press_without_counting_it_as_seated(sim):
    sim.reset(randomize=False)
    for name in sim.object_names:
        sim.set_part_pose(name, *sim.assembled_pose(name))
    position, quat = sim.assembled_pose('pin_2')
    sim.set_part_pose('pin_2', position+[0,0,.0165], quat)
    mujoco.mj_forward(sim.model, sim.data)
    assert pin_entry_ready(sim, 'pin_2')
    assert not pin_placement(sim, 'pin_2')['success']
    for delta in ([.0004,0,.0165], [0,0,.023]):
        sim.set_part_pose('pin_2', position+delta, quat)
        mujoco.mj_forward(sim.model, sim.data)
        assert not pin_entry_ready(sim, 'pin_2')


def test_blocked_push_records_no_advance_and_overload_fails(sim, tmp_path):
    sim.reset(); trial = BimanualTrial(sim, tmp_path)
    trial.tool_load = lambda side: 21.
    positions = []
    trial.tick = lambda p, q, g: positions.append(np.asarray(p))
    end = np.array([.3, 0., .1]); direction = np.array([0.,0.,-1.])
    trial.target = end-direction*.004
    trial.push('blocked', end, direction, np.array([1.,0.,0.]), .004, duration=.1)
    assert trial.stages[-1]['progress'] == 0
    for pos in positions: np.testing.assert_array_equal(pos, end-direction*.004)
    trial.tool_load = lambda side: 36.
    with pytest.raises(RuntimeError, match='35 N'):
        trial.push('overload', end, direction, np.array([1.,0.,0.]), .004, duration=.1)
    assert not trial.contact_mode
