"""Showcase: play an exported ONNX policy (tracking or velocity, G1 or Wisco) on an open floor and record it.

The scene is the task's own mjlab play scene (flat plane, no kitchen), so observations, physics, actuators and
the reference motion are exactly what the policy was trained with. Actions come from the ONNX file (the same
artifact that gets deployed), not from a training checkpoint.

  # tracking (G1): headless mp4 (VM, MUJOCO_GL=egl) or --viewer on the Mac
  python tools/play_mjlab.py --task Mjlab-Tracking-Flat-Unitree-G1 --policy run.onnx \
      --motion-file ~/hsim_data/motions/g1/dance1_subject2.npz --seconds 30 --video dance.mp4
  # tracking (Wisco)
  python tools/play_mjlab.py --task Wisco-V1-Tracking-Flat --policy run.onnx --motion-file wisco_dance.npz --viewer
  # velocity: fixed command vx,vy,wz
  python tools/play_mjlab.py --task Wisco-V1-Velocity-Flat --policy run.onnx --cmd 0.5,0,0 --seconds 20 --video walk.mp4
Run with the mjlab env: VM `cd training/locomotion && uv run python ../../tools/play_mjlab.py ...`,
Mac `.venv-mjlab/bin/python tools/play_mjlab.py ...` (CPU; add CUDA_VISIBLE_DEVICES= if no GPU).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]


def register_tasks():
  """mjlab's own tasks + ours (each optional, so this works whatever is present)."""
  import mjlab.tasks  # noqa: F401
  for path, mod in ((REPO / "training" / "locomotion" / "wisco_v1", "wisco_mjlab"),
                    (REPO / "training" / "tracking" / "wisco_v1", "wisco_tracking"),
                    (REPO / "training" / "locomotion", "hsim_tasks")):
    if path.exists():
      sys.path.insert(0, str(path))
      if mod == "wisco_tracking":
        sys.path.insert(0, str(REPO / "training" / "locomotion" / "wisco_v1"))
      try:
        __import__(mod)
      except Exception as e:  # a half-built task package shouldn't stop the others
        print(f"[showcase] skipped {mod}: {type(e).__name__}: {e}")


def build_env(task: str, motion_file: str | None, cmd: tuple[float, float, float] | None, device: str,
              render: bool, width: int, height: int):
  """The task's play environment, with a fixed command or motion file."""
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg
  from mjlab.tasks.tracking.mdp import MotionCommandCfg
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = 1
  cfg.terminations = {k: v for k, v in cfg.terminations.items() if k == "time_out"}   # keep playing after a stumble
  cfg.episode_length_s = int(1e9)
  if "motion" in cfg.commands and isinstance(cfg.commands["motion"], MotionCommandCfg):
    if not motion_file:
      raise SystemExit("tracking task: pass --motion-file <npz>")
    cfg.commands["motion"].motion_file = str(Path(motion_file).expanduser())
    cfg.commands["motion"].sampling_mode = "start"
  if cmd is not None and "twist" in cfg.commands:
    tw = cfg.commands["twist"]
    tw.ranges.lin_vel_x = (cmd[0], cmd[0]); tw.ranges.lin_vel_y = (cmd[1], cmd[1]); tw.ranges.ang_vel_z = (cmd[2], cmd[2])
    tw.heading_command = False; tw.ranges.heading = None; tw.rel_standing_envs = 0.0
    # fixed command for every env: switch off the special command modes (heading, world-frame, forward-only,
    # in-place turn, small-command practice) that would otherwise overwrite it
    for k in ("rel_heading_envs", "rel_world_envs", "rel_forward_envs", "rel_turn_envs", "rel_small_envs",
              "rel_yaw_only_envs"):
      if hasattr(tw, k):
        setattr(tw, k, 0.0)
  cfg.events.pop("push_robot", None)
  if render:
    cfg.viewer.width, cfg.viewer.height = width, height
  return ManagerBasedRlEnv(cfg=cfg, device=device, render_mode="rgb_array" if render else None)


class OnnxPolicy:
  """An exported ONNX policy (with the time step input for tracking policies)."""
  def __init__(self, path: str):
    """Load the ONNX file on CPU."""
    import onnxruntime as ort
    self.sess = ort.InferenceSession(str(Path(path).expanduser()), providers=["CPUExecutionProvider"])
    self.inputs = [i.name for i in self.sess.get_inputs()]

  def __call__(self, obs: np.ndarray, time_step: int | None) -> np.ndarray:
    feed = {"obs": obs.astype(np.float32)}
    if "time_step" in self.inputs:
      feed["time_step"] = np.array([[time_step or 0]], dtype=np.float32)
    return self.sess.run(["actions"], feed)[0]


def main():
  """Play an exported policy in its training environment, in the viewer or to a video."""
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--task", required=True)
  ap.add_argument("--policy", required=True, help="exported .onnx (or 'zero' to just show the scene)")
  ap.add_argument("--motion-file", default=None)
  ap.add_argument("--cmd", default=None, help="velocity tasks: vx,vy,wz")
  ap.add_argument("--seconds", type=float, default=20.0)
  ap.add_argument("--video", default=None, help="write an mp4 (headless)")
  ap.add_argument("--viewer", action="store_true", help="live MuJoCo viewer (Mac)")
  ap.add_argument("--width", type=int, default=960)
  ap.add_argument("--height", type=int, default=540)
  ap.add_argument("--device", default=None)
  a = ap.parse_args()
  device = a.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  cmd = tuple(float(x) for x in a.cmd.split(",")) if a.cmd else None

  register_tasks()
  env = build_env(a.task, a.motion_file, cmd, device, render=bool(a.video), width=a.width, height=a.height)
  policy = None if a.policy == "zero" else OnnxPolicy(a.policy)
  obs, _ = env.reset()
  n_act = env.action_manager.total_action_dim
  steps = int(a.seconds / env.step_dt)
  motion = env.command_manager.get_term("motion") if "motion" in env.command_manager.active_terms else None

  viewer = None
  if a.viewer:
    import mujoco
    import mujoco.viewer
    mj_model = env.sim.mj_model
    mj_data = mujoco.MjData(mj_model)
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

  frames = []
  for i in range(steps):
    o = obs["actor"] if isinstance(obs, dict) else obs
    if policy is None:
      act = np.zeros((1, n_act), dtype=np.float32)
    else:
      ts = int(motion.time_steps[0].item()) if motion is not None else None
      act = policy(o.detach().cpu().numpy(), ts)
    obs, _, _, _, _ = env.step(torch.as_tensor(act, device=env.device))
    if a.video:
      frames.append(env.render())
    if viewer is not None:
      if not viewer.is_running():
        break
      mj_data.qpos[:] = env.sim.data.qpos[0].cpu().numpy(); mj_data.qvel[:] = env.sim.data.qvel[0].cpu().numpy()
      mujoco.mj_forward(mj_model, mj_data)
      viewer.sync(); time.sleep(env.step_dt)
  if a.video:
    import mediapy as media
    out = Path(a.video).expanduser(); out.parent.mkdir(parents=True, exist_ok=True)
    media.write_video(str(out), [f for f in frames if f is not None], fps=round(1 / env.step_dt))
    print(f"[showcase] wrote {out} ({len(frames)} frames)")
  env.close()


if __name__ == "__main__":
  main()
