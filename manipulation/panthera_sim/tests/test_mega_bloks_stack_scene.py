"""Check that free blocks seat and stay on the stud contacts."""

import mujoco
import numpy as np

from sim.mega_bloks import BODY_HEIGHT_M, PITCH_M, TABLE_Z
from sim.mega_bloks_stack import STACK_CENTER_GAP_M, build_stack_scene


def test_spawn_is_seeded_and_keeps_blocks_separated(tmp_path):
    first, blocks = build_stack_scene(tmp_path / 'first.xml', seed=7)
    repeated, same_blocks = build_stack_scene(tmp_path / 'repeated.xml', seed=7)
    changed, _ = build_stack_scene(tmp_path / 'changed.xml', seed=8)
    assert blocks == same_blocks
    assert first.read_bytes() == repeated.read_bytes()
    assert first.read_bytes() != changed.read_bytes()
    assert len({block['color'] for block in blocks}) == 2

    model = mujoco.MjModel.from_xml_path(str(first))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
    mujoco.mj_step(model, data, nstep=250)
    upper, lower = [data.body(block['body_name']).xpos.copy() for block in blocks]
    assert np.allclose([upper[2], lower[2]], TABLE_Z + BODY_HEIGHT_M / 2, atol=.0002)
    assert 0 < upper[1] < lower[1]
    assert np.linalg.norm(upper[:2] - lower[:2]) > max(blocks[0]['dimensions_m'][:2])


def test_socket_roofs_rest_on_both_studs_after_drop(tmp_path):
    scene, (upper, lower) = build_stack_scene(tmp_path / 'scene.xml')
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
    mujoco.mj_step(model, data, nstep=300)
    top_qpos = data.joint(upper['body_name']).qpos
    bottom_qpos = data.joint(lower['body_name']).qpos
    # Set the initial drop pose above the base, then let gravity seat the studs.
    top_qpos[:] = bottom_qpos
    top_qpos[2] += STACK_CENTER_GAP_M + .015
    mujoco.mj_forward(model, data)
    mujoco.mj_step(model, data, nstep=1200)

    assert np.allclose(top_qpos[:2], bottom_qpos[:2], atol=.0002)
    assert np.isclose(top_qpos[2] - bottom_qpos[2], STACK_CENTER_GAP_M, atol=.0002)
    assert np.isclose(bottom_qpos[2], TABLE_Z + BODY_HEIGHT_M / 2, atol=.0002)
    assert np.linalg.norm(data.joint(upper['body_name']).qvel) < .001
    roof = model.geom(upper['body_geom']).id
    studs = {model.geom(name).id for name in lower['stud_geoms']}
    contacts = [set(contact.geom) for contact in data.contact]
    assert all({roof, stud} in contacts for stud in studs)


def test_long_beam_seats_across_two_walls(tmp_path):
    scene, (left, right, beam) = build_stack_scene(tmp_path / 'gate.xml', lengths=(2, 2, 4))
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
    height = TABLE_Z + BODY_HEIGHT_M / 2
    row_y = .075
    center_x = .20
    for wall, y in ((left, row_y), (right, row_y + 3 * PITCH_M)):
        data.joint(wall['body_name']).qpos[:] = [center_x, y, height, 1, 0, 0, 0]
    # The beam crosses both walls at their positive-X studs. Only its
    # outer sockets are supported, leaving the two middle sockets over the gate.
    beam_qpos = data.joint(beam['body_name']).qpos
    beam_qpos[:] = [center_x + PITCH_M / 2, row_y + 1.5 * PITCH_M,
                    height + STACK_CENTER_GAP_M + .004, np.sqrt(.5), 0, 0, np.sqrt(.5)]
    mujoco.mj_forward(model, data)
    mujoco.mj_step(model, data, nstep=1200)
    np.testing.assert_allclose(beam_qpos[:3],
                               [center_x + PITCH_M / 2, row_y + 1.5 * PITCH_M,
                                height + STACK_CENTER_GAP_M], atol=.0003)
    touching = [set(c.geom) for c in data.contact if c.efc_address >= 0]
    roof = model.geom(beam['body_geom']).id
    for wall in (left, right):
        assert {roof, model.geom(wall['stud_geoms'][1]).id} in touching
        assert {roof, model.geom(wall['stud_geoms'][0]).id} not in touching


def test_three_stud_row_can_balance_on_one_stud_at_right_angle(tmp_path):
    scene, (wall, cap) = build_stack_scene(tmp_path / 'cap.xml', lengths=(2, 3))
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
    lower = data.joint(wall['body_name']).qpos
    upper = data.joint(cap['body_name']).qpos
    height = TABLE_Z + BODY_HEIGHT_M / 2
    lower[:] = [.22, .10, height, 1, 0, 0, 0]
    upper[:] = [.22 + PITCH_M / 2, .10, height + STACK_CENTER_GAP_M + .004,
                np.sqrt(.5), 0, 0, np.sqrt(.5)]
    mujoco.mj_forward(model, data)
    mujoco.mj_step(model, data, nstep=1200)
    np.testing.assert_allclose(upper[:3], [.22 + PITCH_M / 2, .10, height + STACK_CENTER_GAP_M],
                               atol=.0005)
    settled = upper.copy()
    mujoco.mj_step(model, data, nstep=400)
    np.testing.assert_allclose(upper, settled, atol=.0002)
    roof = model.geom(cap['body_geom']).id
    stud = model.geom(wall['stud_geoms'][1]).id
    assert any(set(c.geom) == {roof, stud} for c in data.contact if c.efc_address >= 0)
