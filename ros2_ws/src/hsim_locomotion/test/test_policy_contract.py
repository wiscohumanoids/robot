"""Manifest v1 / v2 contract tests (pure Python: no ROS needed).
    python -m pytest ros2_ws/src/hsim_locomotion/test -q"""
from pathlib import Path

import numpy as np
import pytest
import yaml

from hsim_locomotion.policy_contract import HeldTargets, load_contract

ROOT = Path(__file__).resolve().parents[4]
V1 = ROOT / "models/locomotion/g1_velocity_flat_v1"
LAYOUT = ['base_lin_vel', 'base_ang_vel', 'projected_gravity', 'joint_pos', 'joint_vel', 'actions', 'command']


def write_v2(tmp_path, n_policy=3, n_held=2):
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


@pytest.mark.skipif(not (V1 / "manifest.yaml").exists(), reason="v1 policy folder not present")
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
