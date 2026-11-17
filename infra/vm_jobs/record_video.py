"""Record a short headless video of a trained mjlab policy (EGL), then exit.

mjlab's `play --video` records and then opens an interactive viewer (never exits headless), so the
morning report uses this instead. Same env/runner setup as mjlab.scripts.play (trained mode).

  MUJOCO_GL=egl <venv>/bin/python record_video.py --task TASK --checkpoint model_X.pt --out DIR
      [--steps 500] [--import-path DIR --import-module MOD] [--motion-file motion.npz] [--device cuda:0]
Writes DIR/video.mp4.
"""
import argparse
import importlib
import shutil
import sys
from dataclasses import asdict
from pathlib import Path


def main():
    """Record a video of a checkpoint playing its task."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--import-path", default=None)
    ap.add_argument("--import-module", default=None)
    ap.add_argument("--motion-file", default=None)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    if a.import_path:
        sys.path.insert(0, a.import_path)
    if a.import_module:
        importlib.import_module(a.import_module)
    import torch  # noqa: F401
    import mjlab.tasks  # noqa: F401  registers mjlab's tasks
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.wrappers import VideoRecorder

    env_cfg = load_env_cfg(a.task, play=True)
    agent_cfg = load_rl_cfg(a.task)
    env_cfg.scene.num_envs = 1
    if "motion" in env_cfg.commands:
        if not a.motion_file:
            sys.exit("tracking task needs --motion-file")
        env_cfg.commands["motion"].motion_file = a.motion_file

    out = Path(a.out)
    tmp = out / "_frames"
    out.mkdir(parents=True, exist_ok=True)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=a.device, render_mode="rgb_array")
    env = VideoRecorder(env, video_folder=tmp, step_trigger=lambda step: step == 0, video_length=a.steps,
                        disable_logger=True)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(a.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=a.device)
    runner.load(a.checkpoint, load_cfg={"actor": True}, strict=True, map_location=a.device)
    policy = runner.get_inference_policy(device=a.device)
    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(a.steps + 2):
            obs, _, _, _ = env.step(policy(obs))
    env.close()
    vids = sorted(tmp.rglob("*.mp4"))
    if not vids:
        sys.exit("no video produced")
    shutil.move(str(vids[-1]), out / "video.mp4")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {out / 'video.mp4'}")


if __name__ == "__main__":
    main()
