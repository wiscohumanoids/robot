"""Manifest v1 / v2 / v3 contract tests (pure Python: no ROS needed).
    python -m pytest ros2_ws/src/hsim_locomotion/test -q"""
from pathlib import Path

import numpy as np
import pytest
import yaml

from hsim_locomotion.policy_contract import (TRACKING_OBS_LAYOUT, TRACKING_OBS_LAYOUT_NO_ESTIMATION, HeldTargets,
                                             MotionAlignment, load_contract, quat_to_mat)

ROOT = Path(__file__).resolve().parents[4]
V1 = ROOT / "models/locomotion/g1_velocity_all_joints"
LAYOUT = ['base_lin_vel', 'base_ang_vel', 'projected_gravity', 'joint_pos', 'joint_vel', 'actions', 'command']


def write_v2(tmp_path, n_policy=3, n_held=2):
    """Write a small v2 (arms-free) manifest with `n_policy` driven and `n_held` held joints into `tmp_path`."""
    names = [f"p{i}" for i in range(n_policy)] + [f"h{i}" for i in range(n_held)]
    joints = []
    for i, n in enumerate(names):
        j = {"name": n, "default_pos": 0.1 * i, "kp": 10.0 + i, "kd": 1.0}
        j.update({"action_scale": 0.5, "role": "policy"} if n.startswith("p") else {"role": "held"})
        joints.append(j)
    man = {"manifest_version": 2, "policy_id": "t", "timing": {"sim_dt": 0.005, "decimation": 4},
           "joints": joints, "observation": {"terms": LAYOUT},
           "action": {"type": "joint_position", "joints": list(reversed(names[:n_policy]))},
           "held_joints": {"names": names[n_policy:], "source_topic": "/arm/joint_targets"}}
    (tmp_path / "manifest.yaml").write_text(yaml.safe_dump(man))
    return tmp_path


@pytest.mark.skipif(not (V1 / "manifest.yaml").exists(), reason="all-joints policy folder not present")
def test_v1_manifest_unchanged_behaviour():
    c = load_contract(V1)
    assert c.version == 1 and len(c.obs_joint_names) == 29 and c.act_joint_names == c.obs_joint_names
    assert c.obs_dim == 99 and not c.held_joint_names
    man = yaml.safe_load((V1 / "manifest.yaml").read_text())
    exp = np.array([j["default_pos"] for j in man["joints"]]) + np.array([j["action_scale"] for j in man["joints"]])
    assert np.allclose(c.action_targets(np.ones(29)), exp)


def test_v2_subset(tmp_path):
    c = load_contract(write_v2(tmp_path))
    assert c.version == 2 and c.obs_joint_names == ["p0", "p1", "p2", "h0", "h1"]
    assert c.act_joint_names == ["p2", "p1", "p0"]            # action order from action.joints
    assert c.held_joint_names == ["h0", "h1"] and c.held_topic == "/arm/joint_targets"
    assert c.obs_dim == 9 + 2 * 5 + 3 + 3
    obs = c.observation(np.zeros(3), np.zeros(3), [0, 0, -1], c.default_pos + 1, np.zeros(5), np.zeros(3), [0.1, 0, 0])
    assert obs.shape == (c.obs_dim,) and np.allclose(obs[9:14], 1.0)
    assert np.allclose(c.action_targets([1, 0, 0]), [0.2 + 0.5, 0.1, 0.0])


def test_held_targets(tmp_path):
    c = load_contract(write_v2(tmp_path))
    h = HeldTargets(c)
    assert not h.received and np.allclose(h.targets, [0.3, 0.4])          # default pose
    assert h.update(["h1", "zz"], [1.5, 9.0]) == ["zz"]
    assert h.received and np.allclose(h.targets, [0.3, 1.5])               # h0 keeps its target


def test_v2_rejects_missing_action_scale(tmp_path):
    p = write_v2(tmp_path)
    man = yaml.safe_load((p / "manifest.yaml").read_text())
    man["action"]["joints"].append("h0")
    (p / "manifest.yaml").write_text(yaml.safe_dump(man))
    with pytest.raises(ValueError):
        load_contract(p)


def write_v3(tmp_path, n=4, terms=TRACKING_OBS_LAYOUT):
    """Write a small v3 (tracking) manifest with `n` joints into `tmp_path`."""
    joints = [{"name": f"j{i}", "default_pos": 0.1 * i, "kp": 20.0, "kd": 1.0, "action_scale": 0.5} for i in range(n)]
    man = {"manifest_version": 3, "kind": "tracking", "policy_id": "trk", "timing": {"sim_dt": 0.005, "decimation": 4},
           "tracking": {"anchor_body": "torso", "anchor_index": 2, "motion_steps": 100},
           "joints": joints, "observation": {"terms": list(terms)}, "action": {"type": "joint_position"}}
    (tmp_path / "manifest.yaml").write_text(yaml.safe_dump(man))
    return tmp_path


def yaw_quat(yaw):
    """(w, x, y, z) quaternion of a yaw."""
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def test_v3_tracking_contract(tmp_path):
    c = load_contract(write_v3(tmp_path))
    assert c.kind == "tracking" and c.anchor_index == 2 and c.motion_steps == 100
    assert c.act_joint_names == c.obs_joint_names            # tracking drives every joint
    assert c.obs_dim == 2 * 4 + 3 + 6 + 3 + 3 + 4 + 4 + 4
    obs = c.tracking_observation(np.ones(4), np.zeros(4), [1.0, 0.0, 0.8], np.eye(3), [0.0, 0.0, 0.8], np.eye(3),
                                 [0.1, 0, 0], [0, 0, 0.2], c.default_pos, np.zeros(4), np.zeros(4))
    assert obs.shape == (c.obs_dim,)
    assert np.allclose(obs[8:11], [1.0, 0.0, 0.0])          # clip anchor 1 m ahead of the robot's anchor
    assert np.allclose(obs[11:17], [1, 0, 0, 1, 0, 0])      # same orientation: first two columns of identity


def test_v3_anchor_terms_follow_the_robot_frame(tmp_path):
    c = load_contract(write_v3(tmp_path))
    R_robot = quat_to_mat(yaw_quat(np.pi / 2))               # robot faces +y
    obs = c.tracking_observation(np.zeros(4), np.zeros(4), [0.0, 2.0, 0.0], np.eye(3), [0.0, 0.0, 0.0], R_robot,
                                 np.zeros(3), np.zeros(3), c.default_pos, np.zeros(4), np.zeros(4))
    assert np.allclose(obs[8:11], [2.0, 0.0, 0.0])          # +2 m world y is straight ahead of the robot
    assert np.allclose(obs[11:17], R_robot.T[:, :2].reshape(-1))


def test_v3_without_state_estimation(tmp_path):
    c = load_contract(write_v3(tmp_path, terms=TRACKING_OBS_LAYOUT_NO_ESTIMATION))
    assert c.obs_dim == 2 * 4 + 6 + 3 + 4 + 4 + 4


def test_v3_rejects_velocity_layout(tmp_path):
    with pytest.raises(ValueError):
        load_contract(write_v3(tmp_path, terms=LAYOUT))


def test_motion_alignment_starts_the_clip_on_the_robot():
    m_pos, m_quat = np.array([3.0, -1.0, 0.9]), yaw_quat(0.3)
    r_pos, r_quat = np.array([0.5, 2.0, 0.85]), yaw_quat(-1.2)
    a = MotionAlignment(m_pos, m_quat, r_pos, r_quat)
    pos, R = a.apply(m_pos, m_quat)
    assert np.allclose(pos[:2], r_pos[:2]) and np.isclose(pos[2], m_pos[2])   # xy moved, height kept
    assert np.isclose(np.arctan2(R[1, 0], R[0, 0]), -1.2)                      # faces the robot's heading
