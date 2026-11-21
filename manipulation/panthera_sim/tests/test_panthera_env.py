"""Arm selection routes commands and observations in a shared MuJoCo scene."""

import numpy as np
import pytest

from sim.panthera_env import HERE, PantheraSim


@pytest.mark.parametrize("active, inactive", [("left", "right"), ("right", "left")])
def test_bimanual_selection(active, inactive):
    sim = PantheraSim(HERE / "panthera" / "scene_wood_bimanual.xml")
    assert sim.active_arm == "left"
    assert sim.available_arms == ("left", "right")
    sim.select_arm(inactive)
    inactive_pos = sim.ee_pos()
    inactive_ctrl = np.r_[sim.arm_act, sim.grip_act]
    home_ctrl = sim.data.ctrl.copy()

    sim.select_arm(active)
    pos, quat = sim.ee_pose()
    assert np.linalg.norm(pos - inactive_pos) > 0.5
    target = pos + [0.01, 0.01, 0.01]
    home_qpos = sim.data.qpos.copy()
    q, position_error, rotation_error = sim.ik(target, quat)
    assert position_error < 1e-4
    assert rotation_error < 1e-3
    np.testing.assert_array_equal(sim.data.qpos, home_qpos)

    sim.set_arm_ctrl(q)
    sim.set_gripper(0.0)
    sim.step(500)
    np.testing.assert_allclose(sim.ee_pos(), target, atol=0.003)
    np.testing.assert_allclose(sim.q, q, atol=0.005)
    prefix = "right_" if active == "right" else ""
    first_joint = sim.model.joint(prefix + "joint1").id
    assert sim.dq[0] == sim.data.qvel[sim.model.jnt_dofadr[first_joint]]
    np.testing.assert_allclose(sim.data.qpos[sim.finger_qadr], 0, atol=0.001)
    np.testing.assert_array_equal(sim.data.ctrl[sim.arm_act], q)
    np.testing.assert_array_equal(sim.data.ctrl[inactive_ctrl], home_ctrl[inactive_ctrl])
    assert sim.object_poses()[0].shape == (0, 3)
    assert not sim.grasped

    elapsed = sim.data.time
    sim.select_arm(inactive)
    assert sim.data.time == elapsed
    np.testing.assert_allclose(sim.ee_pos(), inactive_pos, atol=0.005)
    np.testing.assert_allclose(sim.data.qpos[sim.finger_qadr], [0.04, -0.04], atol=0.001)

    sim.select_arm(active)
    sim.reset()
    assert sim.active_arm == active
    np.testing.assert_array_equal(sim.data.ctrl, home_ctrl)
    np.testing.assert_array_equal(sim.data.qpos, home_qpos)


def test_single_arm_scene_rejects_right():
    sim = PantheraSim()
    with pytest.raises(ValueError, match="unavailable"):
        sim.select_arm("right")
    assert sim.active_arm == "left"
