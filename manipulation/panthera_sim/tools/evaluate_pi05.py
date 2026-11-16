#!/usr/bin/env python3
"""Headless pi0.5 rollouts, two-camera MP4s, and fixed-seed stacking metrics."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

FPS = 30
TASK = "stack the three colored cubes"


@dataclass
class StackScore:
    """Color-ordered, table-supported, released stack sustained for one second."""
    hold_steps: int = 30
    stable_steps: int = 0
    max_stable_steps: int = 0
    success: bool = False
    first_success_step: int | None = None
    grasped: bool = False
    lifted: bool = False
    two_stacked: bool = False
    grasped_blocks: set = field(default_factory=set)

    def update(self, positions, holding, step):
        p = np.asarray(positions)
        if p.shape != (3, 3) or not np.isfinite(p).all():
            raise ValueError("Invalid three-block state")
        flags = np.asarray(holding, dtype=bool)
        self.grasped |= bool(flags.any())
        self.grasped_blocks.update(np.flatnonzero(flags).tolist())
        self.lifted |= bool(np.any(p[:, 2] > .0725 + .04))
        # Array order is red, green, blue. Green must support red, then blue.
        ordered = p[[1, 0, 2]]
        xy = np.linalg.norm(np.diff(ordered[:, :2], axis=0), axis=1)
        gaps = np.diff(ordered[:, 2])
        supported = abs(ordered[0, 2] - .0725) < .01
        pair = (xy <= .012) & (gaps > .032) & (gaps < .060)
        self.two_stacked |= bool(supported and pair[0] and not flags[0])
        stable = bool(supported and pair.all() and not flags.any())
        self.stable_steps = self.stable_steps + 1 if stable else 0
        self.max_stable_steps = max(self.max_stable_steps, self.stable_steps)
        if self.stable_steps >= self.hold_steps:
            if not self.success:
                self.first_success_step = step
            self.success = True
        return stable

    def record(self):
        return {"success": self.success, "grasped": self.grasped, "lifted": self.lifted,
                "two_stacked": self.two_stacked, "grasped_blocks": sorted(self.grasped_blocks),
                "max_stable_seconds": self.max_stable_steps / FPS,
                "time_to_success_s": None if self.first_success_step is None else (self.first_success_step + 1) / FPS}


def wilson_interval(successes, total, z=1.959963984540054):
    if total <= 0:
        raise ValueError("At least one episode is required")
    p = successes / total
    scale = 1 + z*z / total
    center = (p + z*z / (2*total)) / scale
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / scale
    return [max(0., center-half), min(1., center+half)]


def reset_scene(sim, seed, distribution):
    import mujoco
    from sim.panthera_env import mat_to_quat
    rng = np.random.default_rng(seed)
    sim.reset(randomize=True, rng=rng)
    if distribution == "matched":
        # Same start region/orientation and RNG draws as the scripted collector.
        # These fresh seeds are not conditioned on the planner later succeeding.
        rng.uniform(.10, .16); rng.uniform(.06, .08)
        rotation = sim.data.xmat[sim.object_bodies[0]].reshape(3, 3)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        yaw = (yaw + np.pi/4) % (np.pi/2) - np.pi/4
        pitch = np.deg2rad(55)
        c, s = np.cos(yaw), np.sin(yaw)
        rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
        c, s = np.cos(pitch), np.sin(pitch)
        quat = mat_to_quat(rz @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]))
        lo, hi, iterations, rot_tol = [.30, -.16, .18], [.46, .16, .28], 150, .01
    else:
        pitch = np.deg2rad(30)
        c, s = np.cos(pitch), np.sin(pitch)
        quat = mat_to_quat(np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]))
        lo, hi, iterations, rot_tol = [.30, -.16, .10], [.48, .16, .40], 100, .001
    for _ in range(100):
        target = rng.uniform(lo, hi)
        q, pe, re = sim.ik(target, quat, q_init=sim.q, max_joint_step=None, iters=iterations)
        if pe < .001 and re < rot_tol:
            break
    else:
        raise RuntimeError("Could not initialize a reachable arm pose; episode counts as failure")
    sim.data.qpos[sim.arm_qadr] = q
    sim.data.qvel[:] = 0
    sim.data.qpos[sim.finger_qadr] = [.04, -.04]
    sim.set_arm_ctrl(q, immediate=True)
    sim.set_gripper(1.)
    mujoco.mj_forward(sim.model, sim.data)


def run_episode(sim, renderers, cameras, select_action, *, seed, distribution, seconds,
                output, save_video=True, save_trace=False):
    import cv2
    import imageio.v2 as imageio
    from teleop.dataset_contract import PhysicsClock
    score = StackScore()
    started = time.monotonic()
    path = output / f"{distribution}_seed_{seed}"
    writer = None
    trace = {k: [] for k in ("qpos", "qvel", "ctrl", "eq_active", "grasp_flags", "sim_time", "action")}
    record = {"seed": seed, "distribution": distribution, "error": None, "video": None}
    clipped = 0
    queries = []
    steps = 0
    try:
        reset_scene(sim, seed, distribution)
        clock = PhysicsClock(FPS, sim.dt)
        if save_video:
            writer = imageio.get_writer(str(path.with_suffix(".mp4")), fps=FPS, codec="libx264",
                                       quality=7, macro_block_size=16, pixelformat="yuv420p")
            record["video"] = str(path.with_suffix(".mp4").resolve())
        for step in range(math.ceil(seconds * FPS)):
            images = []
            for renderer, camera in zip(renderers, cameras):
                renderer.update_scene(sim.data, camera)
                images.append(renderer.render().copy())
            state = np.r_[sim.q, float(sim.data.ctrl[sim.grip_act])].astype(np.float32)
            before = time.monotonic()
            action = np.asarray(select_action(images, state), dtype=np.float64).reshape(-1)
            queries.append(time.monotonic() - before)
            if action.shape != (7,) or not np.isfinite(action).all():
                raise RuntimeError(f"Invalid policy action: {action}")
            bounded = np.r_[np.clip(action[:6], sim.arm_range[:, 0], sim.arm_range[:, 1]),
                            np.clip(action[6], 0., .04)]
            clipped += int(not np.allclose(action, bounded, atol=1e-6))
            sim.set_arm_ctrl(bounded[:6])
            # Stored gripper is metres per finger; simulator expects [0,1].
            sim.set_gripper(float(bounded[6] / .04))
            sim.step(clock.next_steps())
            positions, _ = sim.object_poses()
            holding = sim.grasp_flags()
            score.update(positions, holding, step)
            steps = step + 1
            if save_trace:
                for key, value in {"qpos": sim.data.qpos, "qvel": sim.data.qvel, "ctrl": sim.data.ctrl,
                                   "eq_active": sim.data.eq_active, "grasp_flags": holding, "sim_time": sim.data.time, "action": action}.items():
                    trace[key].append(np.asarray(value).copy())
            if writer:
                frame = np.concatenate(images, axis=1)
                cv2.rectangle(frame, (0, 0), (512, 35), (0, 0, 0), -1)
                label = f"seed {seed} | {(step+1)/FPS:.1f}s | stable {score.stable_steps/FPS:.1f}s"
                cv2.putText(frame, label, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, .37, (255, 255, 255), 1)
                label = "SUCCESS" if score.success else "shoulder                              wrist"
                cv2.putText(frame, label, (6, 29), cv2.FONT_HERSHEY_SIMPLEX, .4,
                            (80, 255, 80) if score.success else (255, 255, 255), 1)
                writer.append_data(frame)
            if score.success:
                break
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        score.success = False
    finally:
        if writer:
            writer.close()
    if save_trace and steps:
        np.savez_compressed(path.with_suffix(".npz"), **{k: np.asarray(v) for k, v in trace.items()})
    record.update(score.record(), steps=steps, elapsed_s=time.monotonic()-started,
                  clipped_action_fraction=clipped / max(1, steps),
                  mean_control_inference_s=float(np.mean(queries)) if queries else None)
    path.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def summarize(records):
    total = len(records)
    successes = sum(r["success"] for r in records)
    return {"episodes": total, "successes": successes, "success_rate": successes / total,
            "success_95pct_wilson_interval": wilson_interval(successes, total),
            "rates": {k: sum(r[k] for r in records) / total for k in ("grasped", "lifted", "two_stacked")},
            "error_episodes": sum(r["error"] is not None for r in records)}


def wandb_metrics(report):
    metrics = {}
    for distribution, result in report["by_distribution"].items():
        for key in ("episodes", "successes", "success_rate", "error_episodes"):
            metrics[f"{distribution}/{key}"] = result[key]
        lo, hi = result["success_95pct_wilson_interval"]
        metrics[f"{distribution}/success_ci_low"] = lo
        metrics[f"{distribution}/success_ci_high"] = hi
        metrics.update({f"{distribution}/{key}_rate": value for key, value in result["rates"].items()})
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per selected distribution")
    parser.add_argument("--dynamics", choices=("contact-v2", "weld-v1"), default="contact-v2")
    parser.add_argument("--seconds", type=float, default=60.)
    parser.add_argument("--seed", type=int, default=260925000)
    parser.add_argument("--distribution", choices=["matched", "broad", "both"], default="both")
    parser.add_argument("--video-episodes", type=int, default=6, help="Number per distribution; -1 saves every rollout")
    parser.add_argument("--save-traces", action="store_true")
    parser.add_argument("--action-steps", type=int, default=10)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="panthera-pi05-ik3")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline"], default="online")
    parser.add_argument("--wandb-log-videos", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1 or args.seconds < 1 or args.video_episodes < -1:
        parser.error("Episodes must be positive, seconds >= 1, video episodes >= -1")
    if args.output.exists():
        parser.error("Use a fresh evaluation output directory to preserve earlier results")
    import mujoco
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.factory import make_pre_post_processors
    from tools.train_pi05_full import strict_from_pretrained, check_full_model
    from sim.panthera_env import PantheraSim
    from teleop.render_vla_dataset import shoulder_camera, wrist_camera
    checkpoint = args.checkpoint.resolve()
    contract = json.loads((checkpoint / "deployment.json").read_text())
    if contract["fps"] != FPS or contract["action_representation"] != "absolute":
        raise ValueError("Checkpoint control contract does not match this simulator")
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = "cuda"
    config.n_action_steps = args.action_steps
    if not 1 <= args.action_steps <= config.chunk_size:
        raise ValueError("action-steps must be between 1 and chunk_size")
    policy = strict_from_pretrained(PI05Policy, checkpoint, config=config).eval()
    check_full_model(policy)
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cuda"}})
    sim = PantheraSim(dynamics=args.dynamics)
    if sim.object_names != ["cube_red", "cube_green", "cube_blue"]:
        raise ValueError("Wrong simulator object order")
    renderers = [mujoco.Renderer(sim.model, height=256, width=256) for _ in range(2)]
    cameras = [shoulder_camera(sim.model), wrist_camera(sim.model)]

    @torch.inference_mode()
    def select_action(images, state):
        batch = {"observation.state": torch.from_numpy(state).unsqueeze(0), "task": [TASK]}
        for camera, pixels in zip(("shoulder", "wrist"), images):
            # Training observations are decoded from quality-92 JPEG. Match the
            # same rendering/encoding path before normalization at deployment.
            import cv2
            ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR),
                                      [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                raise RuntimeError("JPEG encoding failed")
            rgb = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            batch[f"observation.images.{camera}"] = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255
        return postprocessor(policy.select_action(preprocessor(batch))).detach().cpu().numpy()

    args.output.mkdir(parents=True)
    records = []
    distributions = ["matched", "broad"] if args.distribution == "both" else [args.distribution]
    settings = {**vars(args), "checkpoint": str(checkpoint), "output": str(args.output.resolve()),
                "fps": FPS, "success_hold_seconds": 1.0, "color_order": "green, red, blue",
                "max_horizontal_error_m": .012, "headless_simulation": True,
                "simulation_dynamics": sim.dynamics,
                "training_simulation_dynamics": contract.get("simulation_dynamics", "weld-v1"),
                "matched_starts": "collector start region, fresh seeds, no planner-success filtering"}
    wandb_run = None
    try:
        if args.wandb:
            import wandb
            training_config = json.loads((checkpoint / "train_config.json").read_text())
            training_run_id = training_config.get("wandb", {}).get("run_id")
            wandb_run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                mode=args.wandb_mode, name=args.output.name, job_type="evaluation",
                group=training_run_id or training_config.get("job_name"), dir=str(args.output),
                config={**settings, "training_run_id": training_run_id}, save_code=False)
            print("W&B evaluation run:", wandb_run.url, flush=True)
        for distribution in distributions:
            for i in range(args.episodes):
                seed = args.seed + i
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                policy.reset()
                record = run_episode(sim, renderers, cameras, select_action, seed=seed,
                    distribution=distribution, seconds=args.seconds, output=args.output,
                    save_video=args.video_episodes == -1 or i < args.video_episodes,
                    save_trace=args.save_traces)
                records.append(record)
                report = {"settings": settings, "by_distribution": {
                    d: summarize([r for r in records if r["distribution"] == d])
                    for d in distributions if any(r["distribution"] == d for r in records)}, "records": records}
                (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps({"episode": i+1, "distribution": distribution, **record}), flush=True)
                print(json.dumps(report["by_distribution"]), flush=True)
                if wandb_run is not None:
                    metrics = wandb_metrics(report)
                    wandb_run.summary.update(metrics)
                    if args.wandb_log_videos and record["video"]:
                        video = Path(record["video"])
                        if video.is_file() and video.stat().st_size > 1024:
                            metrics[f"videos/{distribution}/seed_{seed}"] = wandb.Video(str(video), format="mp4")
                    wandb_run.log(metrics, step=len(records))
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        for renderer in renderers:
            renderer.close()


if __name__ == "__main__":
    main()
