"""Convert a retargeted motion CSV to the motion .npz that mjlab's tracking task trains on, saved locally.

Same maths and output fields as mjlab 1.6.0's scripts/csv_to_npz.py (it reuses mjlab's MotionLoader:
resample to 50 Hz, finite-difference velocities, then forward kinematics in the robot's own scene to
log every body's world pose and velocity). Differences: no Weights & Biases upload, output path is an
argument, and the robot is selectable (g1 or wisco), so the Wisco retargeted CSV uses the same pipeline.

CSV layout (per row, input fps): base x y z, base quat x y z w, then joint positions in JOINT_ORDER[robot].

  python training/tracking/motion_to_npz.py --robot g1 --input data/lafan1/g1/dance1_subject2.csv \
      --output ~/hsim_data/motions/g1/dance1_subject2.npz [--video out.mp4] [--device cuda:0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]

G1_JOINTS = [
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
  "left_ankle_pitch_joint", "left_ankle_roll_joint",
  "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
  "right_ankle_pitch_joint", "right_ankle_roll_joint",
  "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
  "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
  "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
  "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def robot_setup(robot: str):
  """(scene cfg, joint names in CSV order) for the robot."""
  if robot == "g1":
    from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
    return unitree_g1_flat_tracking_env_cfg().scene, G1_JOINTS
  if robot == "wisco":
    sys.path.insert(0, str(REPO / "training" / "tracking" / "wisco_v1"))
    sys.path.insert(0, str(REPO / "training" / "locomotion" / "wisco_v1"))
    from wisco_tracking.env_cfg import wisco_flat_tracking_env_cfg, WISCO_CSV_JOINTS
    return wisco_flat_tracking_env_cfg().scene, list(WISCO_CSV_JOINTS)
  raise ValueError(robot)


def convert(robot: str, input_file: str, output: str, input_fps: float = 30.0, output_fps: float = 50.0,
            device: str = "cpu", video: str | None = None) -> dict:
  """Retargeted CSV to a motion npz at `output_fps`, with forward kinematics in the robot's training model."""
  from mjlab.scene import Scene
  from mjlab.scripts.csv_to_npz import MotionLoader
  from mjlab.sim.sim import Simulation, SimulationCfg

  if device.startswith("cuda") and not torch.cuda.is_available():
    device = "cpu"
  scene_cfg, joint_names = robot_setup(robot)
  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps
  scene = Scene(scene_cfg, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  renderer = None
  if video:
    from mjlab.viewer.offscreen_renderer import OffscreenRenderer
    from mjlab.viewer.viewer_config import ViewerConfig
    renderer = OffscreenRenderer(model=sim.mj_model, scene=scene, cfg=ViewerConfig(
      height=480, width=640, origin_type=ViewerConfig.OriginType.ASSET_ROOT, entity_name="robot",
      distance=2.5, elevation=-5.0, azimuth=20))
    renderer.initialize()

  motion = MotionLoader(motion_file=input_file, input_fps=input_fps, output_fps=output_fps, device=sim.device)
  robot_ent = scene["robot"]
  idx = robot_ent.find_joints(joint_names, preserve_order=True)[0]
  keys = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
  log = {k: [] for k in keys}
  frames = []
  scene.reset()
  while True:
    (pos, rot, lin_vel, ang_vel, dof_pos, dof_vel), reset_flag = motion.get_next_state()
    root = robot_ent.data.default_root_state.clone()
    root[:, 0:3] = pos
    root[:, :2] += scene.env_origins[:, :2]
    root[:, 3:7] = rot
    root[:, 7:10] = lin_vel
    root[:, 10:] = ang_vel
    robot_ent.write_root_state_to_sim(root)
    jp = robot_ent.data.default_joint_pos.clone(); jv = robot_ent.data.default_joint_vel.clone()
    jp[:, idx] = dof_pos; jv[:, idx] = dof_vel
    robot_ent.write_joint_state_to_sim(jp, jv)
    sim.forward()
    scene.update(sim.mj_model.opt.timestep)
    if renderer is not None:
      renderer.update(sim.data); frames.append(renderer.render())
    d = robot_ent.data
    log["joint_pos"].append(d.joint_pos[0].cpu().numpy().copy())
    log["joint_vel"].append(d.joint_vel[0].cpu().numpy().copy())
    log["body_pos_w"].append(d.body_link_pos_w[0].cpu().numpy().copy())
    log["body_quat_w"].append(d.body_link_quat_w[0].cpu().numpy().copy())
    log["body_lin_vel_w"].append(d.body_link_lin_vel_w[0].cpu().numpy().copy())
    log["body_ang_vel_w"].append(d.body_link_ang_vel_w[0].cpu().numpy().copy())
    if reset_flag:
      break
  out = {"fps": [output_fps], **{k: np.stack(v, 0) for k, v in log.items()}}
  Path(output).expanduser().parent.mkdir(parents=True, exist_ok=True)
  np.savez(Path(output).expanduser(), **out)
  if video and frames:
    import mediapy as media
    media.write_video(video, frames, fps=output_fps)
  info = {"frames": int(out["joint_pos"].shape[0]), "seconds": out["joint_pos"].shape[0] / output_fps,
          "joints": int(out["joint_pos"].shape[1]), "bodies": int(out["body_pos_w"].shape[1]),
          "min_body_z": float(out["body_pos_w"][..., 2].min())}
  print(f"wrote {output}: {info}")
  return info


def main():
  """Command line for `convert`."""
  ap = argparse.ArgumentParser()
  ap.add_argument("--robot", choices=("g1", "wisco"), required=True)
  ap.add_argument("--input", required=True)
  ap.add_argument("--output", required=True)
  ap.add_argument("--input-fps", type=float, default=30.0)
  ap.add_argument("--output-fps", type=float, default=50.0)
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--video", default=None)
  a = ap.parse_args()
  convert(a.robot, a.input, a.output, a.input_fps, a.output_fps, a.device, a.video)


if __name__ == "__main__":
  main()
