"""On the GPU machine: build the real mjlab environment and check it before any training.

Checks (prints PASS/FAIL per line, writes check_env_result.json next to this file):
  1. task builds; 15 actions (arms-free recipe: legs + waist); observation sizes; reward / event / termination terms present
  2. robot joints, default pose and ankle soft limits as configured
  3. mass randomization applied (total robot mass spread across envs)
  4. ankle penalty: torch motor torques == numpy model/ankle_motor.py on live states
  5. zero action for 5 s: no NaN; how many robots fall (pushes are on, so some may)
  6. speed: env steps per second at NUM_ENVS (default 4096)
  7. arms-free recipe at forced curriculum level 1: arm target range, hand payload, share of pure-turn commands

Run from training/locomotion:  MUJOCO_GL=egl .venv/bin/python ../../training/locomotion/wisco_v1/vm/check_env.py [num_envs]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # wisco_mjlab
sys.path.insert(0, str(HERE.parents[3] / "robots" / "wisco_v1" / "model"))
import wisco_mjlab  # noqa: E402  (registers the task)
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402
from wisco_mjlab import linkage_penalty  # noqa: E402

import ankle_motor  # noqa: E402
import robot_params as P  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
out, ok_all = {}, True


def report(name, ok, detail):
  """Record and print one check result."""
  global ok_all
  ok_all &= bool(ok)
  out[name] = {"pass": bool(ok), **detail}
  print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")


cfg = load_env_cfg(wisco_mjlab.TASK_ID)
cfg.scene.num_envs = N
t0 = time.time()
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
env = ManagerBasedRlEnv(cfg=cfg, device=DEVICE)
build_s = time.time() - t0
obs, _ = env.reset()
robot = env.scene["robot"]

# 1. task structure
act_dim = env.action_manager.total_action_dim
obs_shapes = {k: list(v.shape) for k, v in obs.items()} if isinstance(obs, dict) else {"obs": list(obs.shape)}
report("1_task_builds", act_dim == len(P.LOCOMOTION_ACTION_JOINTS) == 15, {"actions": act_dim, "obs": obs_shapes, "build_s": round(build_s, 1),
       "rewards": list(env.reward_manager.active_terms), "events": list(cfg.events),
       "terminations": list(cfg.terminations)})

# 2. joints, default pose, soft limits
params = P.joint_params()
names = list(robot.joint_names)
dq = robot.data.default_joint_pos[0].cpu().numpy()
pose_err = max(abs(dq[names.index(n)] - p["default"]) for n, p in params.items())
soft = robot.data.soft_joint_pos_limits[0].cpu().numpy()
ap = names.index("left_ankle_pitch")
report("2_joints_and_pose", set(names) == set(params) and pose_err < 1e-6,
       {"joints": len(names), "default_pose_max_err": float(pose_err),
        "left_ankle_pitch_soft_limits": soft[ap].round(3).tolist()})

# 3. mass randomization
try:
  bm = env.sim.model.body_mass
  bm = torch.as_tensor(bm)
  ids = torch.as_tensor(robot.indexing.body_ids, device=bm.device)
  tot = (bm[:, ids] if bm.dim() == 2 else bm[ids][None]).sum(dim=1).cpu().numpy()
  report("3_mass_randomized", tot.max() - tot.min() > 0.5,
         {"robot_mass_min": round(float(tot.min()), 2), "robot_mass_max": round(float(tot.max()), 2),
          "nominal": 40.55})
except Exception as e:  # field layout differs: report, don't stop
  report("3_mass_randomized", False, {"error": repr(e)})

# 4 + 5. zero action for 5 s; compare ankle torques
zero = torch.zeros((N, act_dim), device=env.device)
fell = torch.zeros(N, dtype=torch.bool, device=env.device)
nan = False
steps = int(5.0 / env.step_dt)
for _ in range(steps):
  obs, rew, term, trunc, extras = env.step(zero)
  fell |= term
  nan |= bool(torch.isnan(rew).any())
_m = linkage_penalty._model(env)                     # torch grid mapping of applied joint torques (not the actuator's own)
_q, _t = robot.data.joint_pos, robot.data.actuator_force
_out = []
for _s in ("left", "right"):
  _J = _m._jacobian(_s, _q[:, _m.joint_ids[_s][0]], _q[:, _m.joint_ids[_s][1]])
  _out.append(linkage_penalty.joint_to_motor(_J, _t[:, _m.act_ids[_s]]))
torch_tau = torch.cat(_out, dim=1)[:8].cpu().numpy()
q = robot.data.joint_pos[:8].cpu().numpy(); tau = robot.data.actuator_force[:8].cpu().numpy()
act_names = list(robot.actuator_names)
np_tau = []
for e in range(8):
  row = []
  for side in ("left", "right"):
    jp, jr = names.index(f"{side}_ankle_pitch"), names.index(f"{side}_ankle_roll")
    ap_, ar_ = act_names.index(f"{side}_ankle_pitch"), act_names.index(f"{side}_ankle_roll")
    row += list(ankle_motor.motor_torques(side, q[e, jp], q[e, jr], tau[e, ap_], tau[e, ar_]))
  np_tau.append(row)
diff = float(np.abs(np.array(np_tau) - torch_tau).max())
report("4_ankle_penalty_matches", diff < 1e-3, {"max_diff_nm": diff, "example_motor_torques_nm": torch_tau[0].round(2).tolist()})
report("5_zero_action_5s", not nan, {"nan": nan, "fraction_fell": round(float(fell.float().mean()), 3),
       "note": "pushes are on, so some falls are expected; the policy learns to recover"})

# 6. speed
sync = torch.cuda.synchronize if DEVICE != "cpu" else (lambda: None)
sync(); t0 = time.time()
for _ in range(100):
  env.step(zero)
sync(); dt = time.time() - t0
report("6_speed", True, {"num_envs": N, "env_steps_per_s": int(100 * N / dt),
       "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if DEVICE != "cpu" else None})

# 7. arms-free recipe at full curriculum difficulty (L = L2 = 1): arm targets, payload, pure-turn share, final stds
from wisco_mjlab import arms_free_terms as V
c = V._curr(env); c.L, c.L2 = 1.0, 1.0
V._apply(env, c, {**V.DEFAULT_CURRICULUM, **cfg.curriculum["level"].params})
st = env._wisco_arm_state
st.time_left[:] = 0
V.arm_targets_step(env, None, **env.event_manager.get_term_cfg("arm_targets").params)
V.hand_payload(env, None, **env.event_manager.get_term_cfg("hand_payload").params)
st.target = st.goal.clone()
cmd = env.command_manager.get_term("twist")
yaw_share = 0.0
K = max(5, -(-2000 // N))                     # >= 2000 samples so the share isn't sampling noise
for _ in range(K):
  cmd._resample_command(torch.arange(N, device=env.device))
  yaw_share += float(cmd.is_yaw_only_env.float().mean()) / K
hb = robot.indexing.body_ids[list(robot.body_names).index("left_hand")]
hand = env.sim.model.body_mass[:, hb] if env.sim.model.body_mass.dim() == 2 else None
report("7_v2_recipe", float((st.target - st.default).abs().max()) > 0.05 and yaw_share >= 0.15,
       {"arm_target_max_dev_rad": round(float((st.target - st.default).abs().max()), 3),
        "pure_yaw_share": round(yaw_share, 3),
        "left_hand_mass_kg_min_max": None if hand is None else [round(float(hand.min()), 3), round(float(hand.max()), 3)]})

(HERE / "check_env_result.json").write_text(json.dumps(out, indent=1, default=str))
print("ALL PASS" if ok_all else "SOME CHECKS FAILED")
env.close()
