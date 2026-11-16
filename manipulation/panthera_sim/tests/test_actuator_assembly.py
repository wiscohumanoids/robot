"""CAD import, reset distribution and insertion collision regressions."""
import numpy as np
import mujoco
import pytest
from sim.actuator_env import ActuatorSim, NOMINAL, TABLE_Z, GEAR_RACK_HEIGHT, PIN_NOMINAL, PIN_GEARS


@pytest.fixture(scope='module')
def sim():
    return ActuatorSim()


@pytest.fixture(scope='module')
def pin_sim():
    return ActuatorSim(with_pins=True)


def test_reset_is_reproducible_and_bounded(sim):
    sim.seed = 23
    sim.reset()
    first = sim.data.qpos.copy()
    fixture = sim.model.body_pos[sim.model.body('small_carrier').id].copy()
    sim.step(10)
    sim.reset()
    np.testing.assert_array_equal(first, sim.data.qpos)
    np.testing.assert_array_equal(fixture, sim.model.body_pos[sim.model.body('small_carrier').id])
    for name, nominal in NOMINAL.items():
        pos = sim.data.xpos[sim.model.body(name).id]
        assert np.max(np.abs(pos[:2]-nominal)) <= .002
        height = GEAR_RACK_HEIGHT if name.startswith('gear_') else 0.
        assert pos[2] == pytest.approx(TABLE_Z+height+.0002)
    assert len(sim.object_names) == 5
    assert not any('stator' in n.lower() or 'rotor' in n.lower() for n in sim.object_names)


def test_as_assembled_transform_has_no_part_intersections(sim):
    sim.reset(randomize=False)
    for name in sim.object_names:
        sim.set_part_pose(name, *sim.assembled_pose(name))
    mujoco.mj_forward(sim.model, sim.data)
    for name in sim.object_names:
        assert all(c['distance_m'] > -.0001 for c in sim.part_contacts(name, include_robot=False))


def test_gear_requires_side_insertion(sim):
    sim.reset(randomize=False)
    sim.set_part_pose('large_carrier', *sim.assembled_pose('large_carrier'))
    target, quat = sim.assembled_pose('gear_1')
    sim.set_part_pose('gear_1', target+[0, 0, .008], quat)
    mujoco.mj_forward(sim.model, sim.data)
    assert min(c['distance_m'] for c in sim.part_contacts('gear_1', include_robot=False)) < -.001
    for d in np.linspace(.04, 0, 21):
        sim.set_part_pose('gear_1', target+[d, 0, 0], quat)
        mujoco.mj_forward(sim.model, sim.data)
        assert all(c['distance_m'] > -.0001 for c in sim.part_contacts('gear_1', include_robot=False))


def test_reset_clears_velocity_and_nominal_fixture(sim):
    sim.data.qvel[:] = 1
    sim.reset(randomize=False)
    assert np.max(np.abs(sim.data.qvel)) == 0
    np.testing.assert_allclose(sim.data.xpos[sim.model.body('small_carrier').id], [*NOMINAL['small_carrier'], .0502])


def test_collision_masks_keep_support_hulls_out_of_assembly(sim):
    def pair(a,b):
        a,b=sim.model.geom(a).id,sim.model.geom(b).id
        return bool((sim.model.geom_contype[a]&sim.model.geom_conaffinity[b]) or
                    (sim.model.geom_contype[b]&sim.model.geom_conaffinity[a]))
    assert pair('gear_1_col_000','large_carrier_col_000')
    assert pair('L_finger_pad','large_carrier_col_000')
    assert pair('gear_1_table_contact','table_top')
    assert not pair('gear_1_col_000','table_top')
    assert not pair('gear_1_table_contact','large_carrier_col_000')
    assert not pair('gear_1_table_contact','large_carrier_table_contact')
    assert not pair('L_finger_pad','large_carrier_table_contact')
    part=sim.manifest['parts']['large_carrier']
    np.testing.assert_allclose(np.diff(part['bounds'],axis=0)[0],[.05298,.05298,.01921],atol=1e-7)


@pytest.mark.parametrize('iters,max_joint_step', [(0, None), (1, None), (1, .001)])
def test_ik_reports_returned_pose_without_changing_live_state(sim, iters, max_joint_step):
    sim.reset(randomize=False)
    before = sim.data.qpos.copy()
    target, quat = sim.ee_pose()
    target += [.025, -.01, .015]
    q, pe, re = sim.ik(target, quat, iters=iters, max_joint_step=max_joint_step)
    np.testing.assert_array_equal(sim.data.qpos, before)
    scratch = mujoco.MjData(sim.model)
    scratch.qpos[:] = before
    scratch.qpos[sim.arm_qadr] = q
    mujoco.mj_kinematics(sim.model, scratch)
    error = sim._ee_error(target, quat, scratch)
    assert pe == pytest.approx(np.linalg.norm(error[:3]), abs=1e-12)
    assert re == pytest.approx(np.linalg.norm(error[3:]), abs=1e-12)


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_all_gear_grasps_have_reachable_clear_insertion_paths(sim, seed):
    from tools.try_actuator_assembly import gear_grasp
    from sim.panthera_env import mat_to_quat
    sim.seed = seed
    sim.reset()
    for name in sim.object_names:
        sim.set_part_pose(name, *sim.assembled_pose(name))
    mujoco.mj_forward(sim.model, sim.data)
    robot = {sim.model.body(n).id for n in
             ['link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'L_finger', 'R_finger']}
    for name in ['gear_1', 'gear_2', 'gear_3', 'gear_4']:
        target, part_quat = sim.assembled_pose(name)
        offset, rotation, outward = gear_grasp(sim, name)
        quat = mat_to_quat(rotation)
        q = sim._q_nominal
        for distance in np.linspace(.030, 0, 31):
            part_pos = target+outward*distance
            q, pe, re = sim.ik(part_pos+offset, quat, q_init=q,
                              max_joint_step=None, iters=150, posture_gain=0,
                              min_damping=.001, pos_tol=1e-6, rot_tol=1e-5)
            assert pe < .00001, (name, distance, pe)
            assert re < .0001, (name, distance, re)
            sim.data.qpos[sim.arm_qadr] = q
            sim.data.qpos[sim.finger_qadr] = [.0082, -.0082]
            sim.set_part_pose(name, part_pos, part_quat)
            mujoco.mj_forward(sim.model, sim.data)
            held = sim.model.body(name).id
            for contact in sim.data.contact:
                b1, b2 = sim.model.geom_bodyid[[contact.geom1, contact.geom2]]
                if (b1 in robot) != (b2 in robot):
                    other = b2 if b1 in robot else b1
                    if other != held:
                        assert contact.dist > -.0001, (name, sim.model.body(other).name)


def test_optional_pin_scene_preserves_original_reset(pin_sim, sim):
    sim.seed = pin_sim.seed = 13
    sim.reset()
    pin_sim.reset()
    np.testing.assert_array_equal(sim.data.qpos, pin_sim.data.qpos[:sim.model.nq])
    assert len(pin_sim.object_names) == 9
    first = pin_sim.data.qpos.copy()
    pin_sim.step(10)
    pin_sim.reset()
    np.testing.assert_array_equal(first, pin_sim.data.qpos)
    for name, nominal in PIN_NOMINAL.items():
        body = pin_sim.model.body(name).id
        assert np.max(np.abs(pin_sim.data.xpos[body, :2]-nominal)) <= .002
        assert pin_sim.model.body_mass[body] == pytest.approx(.006998694, rel=1e-6)
        pin, _ = pin_sim.assembled_pose(name)
        gear, _ = pin_sim.assembled_pose(PIN_GEARS[name])
        np.testing.assert_allclose(pin[:2], gear[:2], atol=1e-12)


def test_pin_depth_stops_follow_fixture_and_collide_physically(pin_sim):
    pin_sim.seed = 27
    pin_sim.reset()
    for name in PIN_NOMINAL:
        target, quat = pin_sim.assembled_pose(name)
        stop = pin_sim.model.geom(name+'_depth_stop').id
        np.testing.assert_allclose(pin_sim.data.geom_xpos[stop, :2], target[:2], atol=1e-9)
        assert pin_sim.data.geom_xpos[stop, 2]+pin_sim.model.geom_size[stop, 1] == pytest.approx(target[2])
        pin_sim.set_part_pose(name, target-[0, 0, .0005], quat)
        mujoco.mj_forward(pin_sim.model, pin_sim.data)
        contacts = pin_sim.part_contacts(name, include_robot=False)
        assert any(name+'_depth_stop' in (c['geom1'], c['geom2']) and
                   c['distance_m'] < -.0004 for c in contacts)
    assert pin_sim.model.neq == 1  # Finger coupling only; no pin or grasp welds.


def test_nominal_pin_insertion_paths_are_clear(pin_sim):
    pin_sim.reset(randomize=False)
    for name in pin_sim.object_names:
        if name not in PIN_NOMINAL:
            pin_sim.set_part_pose(name, *pin_sim.assembled_pose(name))
    mujoco.mj_forward(pin_sim.model, pin_sim.data)
    for name in PIN_NOMINAL:
        target, quat = pin_sim.assembled_pose(name)
        for distance in np.linspace(.025, 0, 51):
            pin_sim.set_part_pose(name, target+[0, 0, distance], quat)
            mujoco.mj_forward(pin_sim.model, pin_sim.data)
            assert all(c['distance_m'] > -.00005 for c in pin_sim.part_contacts(name, include_robot=False))
    for name in pin_sim.object_names:
        assert all(c['distance_m'] > -.0001 for c in pin_sim.part_contacts(name, include_robot=False))


def test_pin_target_tracks_actual_carrier_holes(pin_sim):
    from tools.try_actuator_assembly import pin_target_pose, pin_placement
    pin_sim.reset(randomize=False)
    pos, quat = pin_sim.assembled_pose('large_carrier')
    pin_sim.set_part_pose('large_carrier', pos+[.0004, -.0003, 0], quat)
    mujoco.mj_forward(pin_sim.model, pin_sim.data)
    for name in PIN_NOMINAL:
        target, quat = pin_target_pose(pin_sim, name)
        np.testing.assert_allclose(target-pin_sim.assembled_pose(name)[0], [.0004, -.0003, 0], atol=1e-12)
        pin_sim.set_part_pose(name, target+[0, 0, .004], quat)
        mujoco.mj_forward(pin_sim.model, pin_sim.data)
        assert not pin_placement(pin_sim, name)['success']


def test_pin_success_requires_full_depth_and_bore_engagement(pin_sim):
    from tools.try_actuator_assembly import pin_placement
    pin_sim.reset(randomize=False)
    for name in pin_sim.object_names:
        pin_sim.set_part_pose(name, *pin_sim.assembled_pose(name))
    mujoco.mj_forward(pin_sim.model, pin_sim.data)
    for name in PIN_NOMINAL:
        placed = pin_placement(pin_sim, name)
        assert placed['success'] and placed['through_gear'] and placed['through_carrier']
        target, quat = pin_sim.assembled_pose(name)
        for offset in ([.0003, 0, 0], [0, 0, .009], [0, 0, -.004]):
            pin_sim.set_part_pose(name, target+offset, quat)
            mujoco.mj_forward(pin_sim.model, pin_sim.data)
            assert not pin_placement(pin_sim, name)['success']
        pin_sim.set_part_pose(name, target, quat)
        mujoco.mj_forward(pin_sim.model, pin_sim.data)


def test_pin_press_poses_clear_hub_and_other_parts(pin_sim):
    from tools.try_actuator_assembly import gear_grasp
    from sim.panthera_env import mat_to_quat
    pin_sim.reset(randomize=False)
    for name in pin_sim.object_names:
        pin_sim.set_part_pose(name, *pin_sim.assembled_pose(name))
    mujoco.mj_forward(pin_sim.model, pin_sim.data)
    robot = {pin_sim.model.body(n).id for n in
             ['link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'L_finger', 'R_finger']}
    for name, gear in PIN_GEARS.items():
        target, _ = pin_sim.assembled_pose(name)
        _, rotation, outward = gear_grasp(pin_sim, gear)
        height = pin_sim.manifest['parts'][name]['bounds'][1][2]
        q = pin_sim._q_nominal
        for lift in [.015, .005, .0001]:
            q, pe, re = pin_sim.ik(target+outward*.005+[0, 0, height+lift],
                                  mat_to_quat(rotation), q_init=q, max_joint_step=None,
                                  iters=150, posture_gain=0, min_damping=.001,
                                  pos_tol=1e-6, rot_tol=1e-5)
            assert pe < .00001 and re < .0001
            pin_sim.data.qpos[pin_sim.arm_qadr] = q
            pin_sim.data.qpos[pin_sim.finger_qadr] = 0
            mujoco.mj_forward(pin_sim.model, pin_sim.data)
            for contact in pin_sim.data.contact:
                b1, b2 = pin_sim.model.geom_bodyid[[contact.geom1, contact.geom2]]
                if (b1 in robot) != (b2 in robot):
                    other = b2 if b1 in robot else b1
                    if other != pin_sim.model.body(name).id:
                        assert contact.dist > -.0001, (name, pin_sim.model.body(other).name)
