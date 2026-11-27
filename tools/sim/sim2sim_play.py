"""Run a trained mjlab locomotion policy in plain MuJoCo on the CPU (sim-to-sim check).

Loads the training world (assets/robots/g1/mjlab_1.6.0/g1_velocity_flat.xml) and a
policy folder (models/locomotion/<id>/ with policy.onnx + manifest.yaml), then:
  - every physics step (5 ms): MuJoCo steps; its position actuators are the PD loop
  - every 4 steps (50 Hz): build the observation exactly like mjlab (99 values for the
    all-joints (29-joint) policy, 85 for a legs+waist (arms-free) policy), run the ONNX policy,
    set targets = default_pos + action_scale * action for the policy's joints
  - manifest v2 (legs+waist policy): the held joints (arms) get their default pose, or
    with --arms random a scripted random arm motion like the one used in training
  The manifest / observation logic is shared with the ROS node
  (ros2_ws/src/hsim_locomotion/hsim_locomotion/policy_contract.py).

Viewer (macOS needs mjpython; tools/sim/play.sh handles it):
    tools/sim/play.sh [--policy models/locomotion/g1_velocity_all_joints]
  Keys:  Up/Down = forward/back speed   Left/Right = turn   , . = sidestep
         / = stop (zero command)        R = reset robot

Headless test (scripted commands, prints how well it tracked them):
    python tools/sim/sim2sim_play.py --headless
    python tools/sim/sim2sim_play.py --headless --script deadband   # small commands
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import sys

import mujoco
import numpy as np
import onnxruntime as ort

from viewer_camera import clear_view

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ros2_ws/src/hsim_locomotion"))
from hsim_locomotion.policy_contract import Contract, load_contract  # noqa: E402
DEFAULT_SCENE = ROOT / "assets/robots/g1/mjlab_1.6.0/g1_velocity_flat.xml"
DEFAULT_POLICY = ROOT / "models/locomotion/g1_velocity_all_joints"
PREFIX = "robot/"            # mjlab's exported scene prefixes every robot name
FALLEN_HEIGHT = 0.45         # pelvis below this = fell
CMD_LIMITS = np.array([[-1.0, 1.5], [-0.5, 0.5], [-0.7, 0.7]])  # vx, vy, wz (within training ranges)
CMD_STEP = np.array([0.1, 0.1, 0.1])

# Headless test: (seconds, (vx, vy, wz), label). Turn/sidestep commands are above the
# policy's learned deadband.
SCRIPT = [
    (2.0, (0.0, 0.0, 0.0), "stand"),
    (4.0, (0.5, 0.0, 0.0), "forward 0.5"),
    (4.0, (1.0, 0.0, 0.0), "forward 1.0"),
    (3.0, (0.0, 0.0, 1.0), "turn left 1.0"),
    (3.0, (0.0, 0.6, 0.0), "sidestep left 0.6"),
    (3.0, (-0.4, 0.0, 0.0), "backward 0.4"),
    (2.0, (0.0, 0.0, 0.0), "stand"),
]

# Small commands the all-joints policy ignored. "responds" = measured >= 50 % of command.
DEADBAND_SCRIPT = [
    (2.0, (0.0, 0.0, 0.0), "stand"),
    (4.0, (0.0, 0.0, 0.3), "turn 0.3"),
    (4.0, (0.0, 0.0, 0.5), "turn 0.5"),
    (4.0, (0.0, 0.0, 0.8), "turn 0.8"),
    (4.0, (0.0, 0.0, 1.0), "turn 1.0"),
    (4.0, (0.0, 0.0, -0.5), "turn right 0.5"),
    (4.0, (0.0, 0.2, 0.0), "sidestep 0.2"),
    (4.0, (0.0, 0.4, 0.0), "sidestep 0.4"),
    (4.0, (0.1, 0.0, 0.0), "forward 0.1"),
    (4.0, (0.2, 0.0, 0.0), "forward 0.2"),
    (4.0, (0.2, 0.0, 0.3), "forward 0.2 + turn 0.3"),
    (2.0, (0.0, 0.0, 0.0), "stand"),
]
SCRIPTS = {"standard": SCRIPT, "deadband": DEADBAND_SCRIPT}


@dataclass
class Policy:
    """A deployed policy folder: ONNX session and manifest contract."""
    session: ort.InferenceSession
    input_name: str
    c: Contract

    @property
    def joint_names(self):            # every joint the observation reads
        """Every joint the observation reads."""
        return self.c.obs_joint_names

    @property
    def decimation(self):
        """Physics steps per policy step."""
        return self.c.decimation

    @property
    def sim_dt(self):
        """Physics timestep."""
        return self.c.sim_dt

    @classmethod
    def load(cls, folder: Path) -> "Policy":
        """Load and check a policy folder (observation size must match the manifest)."""
        c = load_contract(folder)
        if c.kind != 'velocity':
            raise SystemExit(f'{folder} is a {c.kind} policy; this player runs walking policies '
                             '(play tracking policies with tools/sim/play_mjlab.py or in the ROS stack)')
        sess = ort.InferenceSession(str(folder / "policy.onnx"), providers=["CPUExecutionProvider"])
        n_in = sess.get_inputs()[0].shape[-1]
        if isinstance(n_in, int) and n_in != c.obs_dim:
            raise ValueError(f"policy expects {n_in} observations, manifest implies {c.obs_dim}")
        return cls(session=sess, input_name=sess.get_inputs()[0].name, c=c)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: obs[None].astype(np.float32)})[0][0]


class Robot:
    """Reads mjlab-equivalent observations from MuJoCo and applies policy actions."""

    def __init__(self, model: mujoco.MjModel, policy: Policy, arms: str = "default"):
        """Bind the policy to the scene (timesteps must match)."""
        self.m, self.p, self.arms = model, policy, arms
        if abs(model.opt.timestep - policy.sim_dt) > 1e-9:
            raise ValueError(f"scene timestep {model.opt.timestep} != policy sim_dt {policy.sim_dt}")

        # Map every policy joint to its qpos/qvel slot and its actuator, BY NAME
        # (the scene's actuator order differs from the policy's joint order).
        act_of_joint = {model.actuator_trnid[a, 0]: a for a in range(model.nu)}

        def lookup(name):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, PREFIX + name)
            if jid < 0 or jid not in act_of_joint:
                raise ValueError(f"joint/actuator for {name} not found in scene")
            return jid

        c = policy.c
        obs_j = [lookup(n) for n in c.obs_joint_names]
        self.qpos_idx = np.array([model.jnt_qposadr[j] for j in obs_j])
        self.qvel_idx = np.array([model.jnt_dofadr[j] for j in obs_j])
        self.act_idx = np.array([act_of_joint[lookup(n)] for n in c.act_joint_names])
        self.held_act_idx = np.array([act_of_joint[lookup(n)] for n in c.held_joint_names], dtype=int)
        held_j = [lookup(n) for n in c.held_joint_names]
        self.held_lo = np.array([model.jnt_range[j][0] for j in held_j]) if held_j else np.zeros(0)
        self.held_hi = np.array([model.jnt_range[j][1] for j in held_j]) if held_j else np.zeros(0)
        self.rng = np.random.default_rng(0)
        self._arm = None

        def sensor_slice(name):
            s = model.sensor(PREFIX + name)
            return slice(s.adr[0], s.adr[0] + s.dim[0])

        self.lin_vel = sensor_slice("imu_lin_vel")     # velocimeter at imu_in_pelvis (base frame)
        self.ang_vel = sensor_slice("imu_ang_vel")     # gyro at imu_in_pelvis
        self.pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PREFIX + "pelvis")
        self.key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init_state")
        self.last_action = np.zeros(len(policy.c.act_joint_names))

    def reset(self, d: mujoco.MjData):
        """Reset to the policy's keyframe and the held arms to their defaults."""
        mujoco.mj_resetDataKeyframe(self.m, d, self.key)
        mujoco.mj_forward(self.m, d)
        self.last_action[:] = 0.0
        c = self.p.c
        self._arm = {"start": c.held_default.copy(), "goal": c.held_default.copy(), "t": 0.0, "T": 1.0, "next": 1.0}
        if len(self.held_act_idx):
            d.ctrl[self.held_act_idx] = c.held_default

    def held_targets(self, dt: float) -> np.ndarray:
        """Held joints: default pose, or (--arms random) the training-style scripted arm motion."""
        c, a = self.p.c, self._arm
        if self.arms != "random":
            return c.held_default
        a["t"] += dt
        a["next"] -= dt
        s = min(a["t"] / a["T"], 1.0)
        s = s * s * (3 - 2 * s)
        cur = a["start"] + s * (a["goal"] - a["start"])
        if a["next"] <= 0:
            lo = c.held_default - 0.6 * 0.9 * (c.held_default - self.held_lo).clip(min=0)
            hi = c.held_default + 0.6 * 0.9 * (self.held_hi - c.held_default).clip(min=0)
            goal = self.rng.uniform(lo, hi) if self.rng.random() > 0.3 else c.held_default
            a.update(start=cur, goal=goal, t=0.0, T=self.rng.uniform(0.3, 1.5), next=self.rng.uniform(1.0, 3.0))
        return cur

    def observation(self, d: mujoco.MjData, command: np.ndarray) -> np.ndarray:
        """The observation vector, exactly as the ROS node builds it."""
        # projected gravity: world gravity direction (0,0,-1) rotated into the pelvis frame
        g_b = np.zeros(3)
        q_inv = np.zeros(4)
        mujoco.mju_negQuat(q_inv, d.xquat[self.pelvis])
        mujoco.mju_rotVecQuat(g_b, np.array([0.0, 0.0, -1.0]), q_inv)
        return self.p.c.observation(
            d.sensordata[self.lin_vel], d.sensordata[self.ang_vel], g_b,
            d.qpos[self.qpos_idx], d.qvel[self.qvel_idx],   # default joint velocity is 0
            self.last_action,                               # raw policy output, before scaling
            command,
        )

    def control_step(self, d: mujoco.MjData, command: np.ndarray) -> float:
        """Run the policy once and step physics `decimation` times. Returns inference ms."""
        obs = self.observation(d, command)
        t0 = time.perf_counter()
        action = self.p(obs)
        ms = (time.perf_counter() - t0) * 1e3
        self.last_action = action.astype(np.float64)
        d.ctrl[self.act_idx] = self.p.c.action_targets(action)
        if len(self.held_act_idx):
            d.ctrl[self.held_act_idx] = self.held_targets(self.p.sim_dt * self.p.decimation)
        for _ in range(self.p.decimation):
            mujoco.mj_step(self.m, d)
        return ms

    def fallen(self, d: mujoco.MjData) -> bool:
        """Whether the pelvis is below the fall height."""
        return d.xpos[self.pelvis][2] < FALLEN_HEIGHT


def run_headless(robot: Robot, d: mujoco.MjData, script=SCRIPT, deadband=False) -> int:
    """Run the command script without a window and print measured against commanded velocity."""
    robot.reset(d)
    period = robot.p.sim_dt * robot.p.decimation
    print(f"{'segment':20s} {'cmd vx vy wz':>18s} {'measured vx vy wz':>22s}  min height")
    ok, infer = True, []
    responded = []
    for seconds, cmd, label in script:
        cmd = np.array(cmd)
        steps = int(seconds / period)
        meas, heights = [], []
        for i in range(steps):
            infer.append(robot.control_step(d, cmd))
            heights.append(d.xpos[robot.pelvis][2])
            if i >= steps // 2:   # measure the second half, after the robot has adapted
                meas.append(np.r_[d.sensordata[robot.lin_vel][:2], d.sensordata[robot.ang_vel][2]])
            if robot.fallen(d):
                print(f"{label:20s} FELL after {i * period:.1f}s")
                return 1
        meas = np.mean(meas, axis=0)
        err = np.abs(meas - cmd)
        if deadband:
            active = np.abs(cmd) > 1e-9
            seg_ok = bool(np.all(meas[active] * np.sign(cmd[active]) >= 0.5 * np.abs(cmd[active]))) if active.any() else True
            if active.any():
                responded.append((label, seg_ok))
            verdict = "responds" if seg_ok else "IGNORED / under 50 %"
        else:
            seg_ok = bool(np.all(err < np.maximum(0.25, 0.35 * np.abs(cmd))))
            verdict = "ok" if seg_ok else "POOR TRACKING"
        ok &= seg_ok
        print(f"{label:24s} {cmd[0]:6.2f}{cmd[1]:6.2f}{cmd[2]:6.2f}   {meas[0]:7.2f}{meas[1]:7.2f}{meas[2]:7.2f}"
              f"   {min(heights):.2f} m  {verdict}")
    print(f"\npolicy inference on CPU: mean {np.mean(infer):.3f} ms, max {np.max(infer):.3f} ms")
    if deadband:
        n_ok = sum(r for _, r in responded)
        print(f"deadband: responds to {n_ok}/{len(responded)} small commands (>= 50 % of commanded)")
    print("PASS: walked the whole script without falling" + ("" if ok else " (but tracking was poor on some segments)"))
    return 0


def run_viewer(robot: Robot, d: mujoco.MjData):
    """Run in the MuJoCo viewer with keyboard commands."""
    import mujoco.viewer

    cmd = np.zeros(3)
    lock = threading.Lock()
    reset_requested = [False]
    KEYS = {265: (0, +1), 264: (0, -1), 263: (2, +1), 262: (2, -1), 44: (1, +1), 46: (1, -1)}

    def on_key(key):
        with lock:
            if key in KEYS:
                axis, sign = KEYS[key]
                cmd[axis] = np.clip(cmd[axis] + sign * CMD_STEP[axis], *CMD_LIMITS[axis])
            elif key == 47:          # '/'
                cmd[:] = 0.0
            elif key == 82:          # 'R'
                reset_requested[0] = True
            else:
                return
            print(f"\rcommand: vx={cmd[0]:+.1f}  vy={cmd[1]:+.1f}  wz={cmd[2]:+.1f}      ", end="", flush=True)

    robot.reset(d)
    period = robot.p.sim_dt * robot.p.decimation
    print(__doc__.split("Headless")[0])
    with mujoco.viewer.launch_passive(robot.m, d, key_callback=on_key) as viewer:
        viewer.cam.trackbodyid = robot.pelvis
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.azimuth, viewer.cam.distance, viewer.cam.elevation = clear_view(robot.m, d, robot.pelvis)
        next_t = time.perf_counter()
        while viewer.is_running():
            with lock:
                if reset_requested[0] or robot.fallen(d):
                    if robot.fallen(d):
                        print("\nfell, resetting")
                    robot.reset(d)
                    reset_requested[0] = False
                c = cmd.copy()
            robot.control_step(d, c)
            viewer.sync()
            next_t += period
            time.sleep(max(0.0, next_t - time.perf_counter()))


def main():
    """Play a deployed policy in plain MuJoCo (no ROS), headless or in the viewer."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", type=Path, default=DEFAULT_POLICY, help="folder with policy.onnx + manifest.yaml")
    ap.add_argument("--env", help="environment, e.g. test/table_room (built with tools/build/env_build.py)")
    ap.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    ap.add_argument("--headless", action="store_true", help="run the scripted test instead of the viewer")
    ap.add_argument("--script", choices=sorted(SCRIPTS), default="standard", help="headless command script")
    ap.add_argument("--arms", choices=["default", "random"], default="default",
                    help="arms-free (legs+waist) policies: held arm joints at default pose, or training-style random motion")
    args = ap.parse_args()

    policy = Policy.load(args.policy)
    scene = ROOT / "assets/envs" / args.env / "build/world.mjb" if args.env else args.scene
    model = (mujoco.MjModel.from_binary_path(str(scene)) if str(scene).endswith(".mjb")
             else mujoco.MjModel.from_xml_path(str(scene)))
    robot = Robot(model, policy, arms=args.arms)
    data = mujoco.MjData(model)
    print(f"policy {args.policy.name} (manifest v{policy.c.version}): drives {len(policy.c.act_joint_names)}"
          f"/{len(policy.joint_names)} joints, arms={args.arms if policy.c.held_joint_names else 'policy'}, "
          f"{1 / (policy.sim_dt * policy.decimation):.0f} Hz | world {args.env or args.scene.name}: dt {model.opt.timestep}s")
    if args.headless:
        raise SystemExit(run_headless(robot, data, SCRIPTS[args.script], deadband=args.script == "deadband"))
    run_viewer(robot, data)


if __name__ == "__main__":
    main()
