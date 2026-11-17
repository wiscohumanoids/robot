"""PACE fit on a small fixed-base model: the plan stays inside the limits, replay matches a plain MuJoCo run, and the
fit recovers known armature, friction, damping, encoder bias and delay from a simulated log.

    python -m pytest training/sim2real -q
"""
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
mujoco = pytest.importorskip("mujoco")
pytest.importorskip("yaml")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pace  # noqa: E402

SCENE = """
<mujoco>
  <compiler angle="radian"/>
  <option timestep="0.005"/>
  <worldbody>
    <body name="robot/pelvis" pos="0 0 1">
      <freejoint name="robot/floating_base"/>
      <geom type="box" size="0.1 0.1 0.1" mass="5"/>
      <body name="robot/thigh" pos="0 0 -0.15">
        <joint name="robot/hip" axis="0 1 0" range="-1 1" armature="0.02" frictionloss="0.2" damping="0.1"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.03" mass="2"/>
        <body name="robot/shin" pos="0 0 -0.3">
          <joint name="robot/knee" axis="0 1 0" range="0 2" armature="0.01" frictionloss="0.1" damping="0.05"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.025" mass="1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="robot/hip" joint="robot/hip" kp="60" kv="3"/>
    <position name="robot/knee" joint="robot/knee" kp="40" kv="2"/>
  </actuator>
  <keyframe><key name="init_state" qpos="0 0 1 1 0 0 0 0.2 0.5"/></keyframe>
</mujoco>
"""


@pytest.fixture
def scene(tmp_path):
    """The toy scene written to a file."""
    path = tmp_path / "toy.xml"
    path.write_text(SCENE)
    return path


def make_plan(scene, seconds=4.0):
    """Fixed-base model, joints and a plan with gains, as a log without measurements."""
    model = pace.fixed_base_model(scene)
    joints = pace.actuated_joints(model)
    p = pace.plan(model, joints, seconds=seconds, hold=pace.standing_pose(scene, joints))
    kp, kd = zip(*(pace.gains(model, a) for _, _, a in joints))
    return model, joints, {**p, "joints": [n for n, _, _ in joints], "kp": np.array(kp), "kd": np.array(kd)}


def test_fixed_base_and_plan(scene):
    """The base is removed, the hold is the keyframe, and the chirps stay inside the limits."""
    model, joints, p = make_plan(scene)
    assert model.nq == 2 and [n for n, _, _ in joints] == ["hip", "knee"]
    assert p["hold"].tolist() == pytest.approx([0.2, 0.5])
    for i, (_, j, _) in enumerate(joints):
        lo, hi = model.jnt_range[j]
        assert p["q_cmd"][:, i].min() >= lo and p["q_cmd"][:, i].max() <= hi
    assert len(p["segments"]) == 2 and p["segments"][1][1] > p["segments"][0][2]


def test_replay_matches_plain_simulation(scene):
    """With nominal values, the batched replay equals stepping the model by hand."""
    model, joints, p = make_plan(scene, seconds=1.0)
    log_k = pace.segment_log(p, 0)
    i, s, e = log_k["segment"]
    q = pace.replay([mujoco.MjModel.from_binary_path(pace._tmp_binary(model))], joints, log_k, i,
                    [pace.nominal(model, joints[i][1])])[0]
    m = mujoco.MjModel.from_binary_path(pace._tmp_binary(model))
    d = mujoco.MjData(m)
    lead = int(1.0 / p["dt"])
    cmd = p["q_cmd"][s - lead:e]
    d.qpos[:] = cmd[0]
    mujoco.mj_forward(m, d)
    out = []
    for row in cmd:
        d.ctrl[:] = row
        mujoco.mj_step(m, d)
        out.append(d.qpos[i])
    assert np.allclose(q, np.array(out)[lead:], atol=1e-9)


def test_delay_shifts_the_command():
    """A 10 ms delay at 5 ms steps shifts the command by two samples."""
    x = np.arange(10.0)
    assert pace.delayed(x, 0.01, 0.005) == pytest.approx([0, 0, 0, 1, 2, 3, 4, 5, 6, 7])


def test_fit_recovers_known_parameters(scene):
    """From a simulated log, the fit finds the parameters that made it."""
    model, joints, p = make_plan(scene)
    truth = {"knee": {"armature": 0.03, "frictionloss": 0.3, "damping": 0.15, "encoder_bias": 0.01,
                      "delay_s": 0.01}}
    log = {**p, "q_meas": pace.simulate_log(model, joints, p, truth)}
    params, before, after, _ = pace.fit_joint(model, joints, log, 1, generations=40, popsize=12)
    assert after < 0.05 * before
    assert params["encoder_bias"] == pytest.approx(0.01, abs=1e-3)
    assert params["delay_s"] == pytest.approx(0.01, abs=2e-3)
    assert params["armature"] == pytest.approx(0.03, rel=0.25)
