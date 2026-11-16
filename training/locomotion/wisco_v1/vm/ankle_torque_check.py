"""Per-motor ankle torque distribution for a Wisco checkpoint (fix 4 version of the coordinator's script).

Runs the play config (fixed curriculum level: WISCO_PLAY_LEVEL / WISCO_PLAY_LEVEL2, default 1 = full difficulty,
arms random, payload on) with the policy for 1000 steps and reports per-motor X4-36 torque statistics.
Motor torques are the ankle actuator's own clamped values (exact, every physics step's last value), so after
fix 4 nothing can exceed the 34 N*m peak; the interesting numbers are time above RATED (10.5) and the RMS.
Run from ~/humanoid-sim-stack/training/locomotion:
  MUJOCO_GL=egl .venv/bin/python ../../training/locomotion/wisco_v1/vm/ankle_torque_check.py <checkpoint.pt> [num_envs]
"""
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ck = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 256
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # training/locomotion/wisco_v1 (wisco_mjlab)
import wisco_mjlab  # noqa: E402
import mjlab.tasks  # noqa: E402,F401
from wisco_mjlab import linkage_penalty  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402

dev = "cuda:0" if torch.cuda.is_available() else "cpu"
task = "Wisco-V1-Velocity-Flat"
cfg = load_env_cfg(task, play=True); cfg.scene.num_envs = N
agent = load_rl_cfg(task)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=dev), clip_actions=agent.clip_actions)
runner = (load_runner_cls(task) or MjlabOnPolicyRunner)(env, asdict(agent), device=dev)
runner.load(ck, load_cfg={"actor": True}, strict=True, map_location=dev)
pol = runner.get_inference_policy(device=dev)
model = linkage_penalty._model(env.unwrapped)
print(f"motor torque source: {'ankle actuator (exact, clamped)' if model._ankle_act is not None else 'joint torque mapped through J (post-step)'}")
robot = env.unwrapped.scene["robot"]
acts = list(robot.actuator_names)
env.reset(); T = []; J = []
ema = torch.zeros(N, 4, device=dev); alpha = env.unwrapped.step_dt / 2.0; EMA = []
for i in range(1000):                       # 20 s with the play command sampler (random walking commands)
    with torch.no_grad():
        env.step(pol(env.get_observations()))
    t = model.motor_torques()
    ema += alpha * (t * t - ema)
    T.append(t.abs().cpu().numpy()); EMA.append(torch.sqrt(ema).cpu().numpy())
    J.append(robot.data.actuator_force[:, [acts.index("left_ankle_pitch"), acts.index("left_ankle_roll")]].abs().cpu().numpy())
T = np.concatenate(T); J = np.concatenate(J); E = np.concatenate(EMA[100:])
m = T.max(axis=1)                            # largest of the 4 motors per robot-step
print(f"samples (robot-steps): {len(m)}")
for thr, lab in ((10.5, "rated 10.5"), (20, "20"), (34, "PEAK 34"), (40, "40")):
    print(f"  steps with any ankle motor above {lab:>10s} N*m: {100*np.mean(m > thr):6.2f} %")
print(f"  motor-steps above rated 10.5 N*m: {100*np.mean(T > 10.5):6.2f} %   RMS per motor {np.sqrt(np.mean(T**2)):.1f} N*m")
print(f"  2 s moving RMS (thermal proxy): mean {E.mean():.1f}, p95 {np.percentile(E,95):.1f}, share of motor-time with sustained RMS > rated {100*np.mean(E > 10.5):.1f} %")
print(f"  per-motor torque percentiles  p50 {np.percentile(T,50):.1f}  p95 {np.percentile(T,95):.1f}  p99 {np.percentile(T,99):.1f}  p99.9 {np.percentile(T,99.9):.1f}  max {T.max():.1f} N*m")
print(f"  joint torque (left ankle pitch/roll) p99: {np.percentile(J[:,0],99):.1f} / {np.percentile(J[:,1],99):.1f} N*m (joint limits 86.6 / 35.2)")
