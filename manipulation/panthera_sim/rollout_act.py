#!/usr/bin/env python3
"""Roll out a trained LeRobot ACT checkpoint in the Panthera simulation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from tools.act_scene import rollout_metrics as task_stack_metrics, reset_fixed_arm
from teleop.dataset_contract import DEFAULT_CHECKPOINT, DEFAULT_DATASET, PhysicsClock, resolve_control_hz


REPO_ROOT = Path(__file__).resolve().parent


def reexec_in_act_venv() -> None:
    python = REPO_ROOT / ".venv-act" / "bin" / "python"
    if not python.is_file() or Path(sys.prefix).resolve() == python.parents[1].resolve():
        return
    if os.environ.get("ROBOT_ARM_ACT_REEXEC") == "1":
        return
    environment = os.environ.copy()
    environment["ROBOT_ARM_ACT_REEXEC"] = "1"
    os.execve(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]], environment)


def stack_metrics(positions: np.ndarray) -> dict:
    metrics = task_stack_metrics(positions)
    return {
        "object_positions": positions.tolist(),
        "horizontal_spread_m": metrics["horizontal_spread_m"],
        "vertical_gaps_m": metrics["vertical_gaps_m"],
        "max_object_height_m": metrics["max_height_m"],
        "two_stacked": metrics["two_stack"],
        "stacked": metrics["task_stack"],
    }


def main() -> None:
    reexec_in_act_venv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="control steps before stopping; 0 (default) runs until q",
    )
    parser.add_argument("--dynamics", choices=("contact-v2", "weld-v1"), default="contact-v2",
                        help="weld-v1 explicitly reproduces historical assisted physics")
    parser.add_argument("--hz", type=float, help="must match checkpoint FPS; read automatically by default")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--action-steps", type=int, help="execute this many actions per query; disables averaging unless explicitly requested")
    parser.add_argument("--max-joint-step", type=float, help="optional extra clipping for diagnostics; disabled by default")
    parser.add_argument(
        "--temporal-ensemble",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="blend overlapping ACT chunks (default: on for all checkpoints)",
    )
    parser.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        default=0.01,
        help="exponential weight coefficient for temporal ensembling (default: 0.01)",
    )
    parser.add_argument("--video", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--trace", type=Path, help="save physical states for failure inspection and corrective demonstrations")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--mujoco-gl", default="egl", choices=("egl", "glfw", "osmesa"))
    args = parser.parse_args()
    if args.steps < 0 or (args.hz is not None and args.hz <= 0) or (args.max_joint_step is not None and args.max_joint_step <= 0):
        parser.error("--steps must be nonnegative; --hz and --max-joint-step must be positive")
    if args.temporal_ensemble_coeff < 0:
        parser.error("--temporal-ensemble-coeff must be nonnegative")
    if args.no_display and args.steps == 0:
        parser.error("--no-display requires a positive --steps value")

    os.environ.setdefault("MUJOCO_GL", args.mujoco_gl)
    import cv2
    import mujoco
    import torch

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.utils import prepare_observation_for_inference

    sys.path.insert(0, str(REPO_ROOT / "sim"))
    sys.path.insert(0, str(REPO_ROOT / "teleop"))
    from keyboard import DEFAULT_ARM_START_RANGE, randomize_arm_start
    from panthera_env import PantheraSim
    from render_vla_dataset import shoulder_camera, wrist_camera

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint.expanduser().resolve()
    is_rl_checkpoint = (checkpoint / "rl_state.pt").is_file()
    inference_path = checkpoint / "inference.json"
    saved_inference = json.loads(inference_path.read_text()) if inference_path.is_file() else {}
    action_steps = args.action_steps if args.action_steps is not None else saved_inference.get("action_steps")
    temporal_ensemble = args.temporal_ensemble
    if temporal_ensemble is None:
        temporal_ensemble = False if args.action_steps is not None else saved_inference.get("temporal_ensemble", True)
    policy_config = PreTrainedConfig.from_pretrained(checkpoint)
    if action_steps is not None:
        if not 1 <= action_steps <= policy_config.chunk_size:
            parser.error("action-steps must be between 1 and the checkpoint chunk size")
        policy_config.n_action_steps = action_steps
    if temporal_ensemble:
        # Query a new chunk every step and blend its overlapping predictions.
        # Without this, ACT executes n_action_steps open-loop and can jump when
        # it replaces the exhausted action queue with an independently
        # predicted chunk.
        policy_config.n_action_steps = 1
        policy_config.temporal_ensemble_coeff = args.temporal_ensemble_coeff
    elif is_rl_checkpoint and action_steps is None:
        # Explicit compatibility/debug mode for older first-query RL runs.
        policy_config.n_action_steps = 1
        policy_config.temporal_ensemble_coeff = None
    if not temporal_ensemble:
        policy_config.temporal_ensemble_coeff = None
    policy = ACTPolicy.from_pretrained(checkpoint, config=policy_config).to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(checkpoint)
    )
    metadata = LeRobotDatasetMetadata("local/panthera_stack", root=args.dataset)

    args.hz = resolve_control_hz(checkpoint, args.hz, metadata.fps)
    contract_path = checkpoint / "deployment.json"
    contract = json.loads(contract_path.read_text()) if contract_path.is_file() else {}
    representation = contract.get("action_representation", "absolute")
    if representation not in ("absolute", "relative"):
        raise ValueError(f"Unsupported checkpoint action representation: {representation}")
    from tools.act_pickup import RelativeChunkExecutor, SustainedPickup
    relative_executor = RelativeChunkExecutor(policy, postprocessor) if representation == "relative" else None
    if args.max_joint_step is None and contract_path.is_file():
        args.max_joint_step = json.loads(contract_path.read_text()).get("max_joint_step")
    environment = contract.get("environment", {})
    sim = PantheraSim(REPO_ROOT / environment["scene"], dynamics=args.dynamics) if environment else PantheraSim(dynamics=args.dynamics)
    if environment and sim.object_names != environment["objects"]:
        raise ValueError("Checkpoint object order does not match the scene")
    physics_clock = PhysicsClock(args.hz, sim.dt)

    def reset_scene(seed: int) -> np.ndarray:
        nonlocal physics_clock
        physics_clock = PhysicsClock(args.hz, sim.dt)
        rng = np.random.default_rng(seed)
        sim.reset(randomize=True, rng=rng)
        if environment.get("arm_start") == "fixed":
            reset_fixed_arm(sim, environment)
        else:
            randomize_arm_start(sim, DEFAULT_ARM_START_RANGE, rng=rng)
        policy.reset()
        if relative_executor is not None:
            relative_executor.reset()
        positions, _ = sim.object_poses()
        return positions

    initial_positions = reset_scene(args.seed)
    renderers = (
        mujoco.Renderer(sim.model, height=256, width=256),
        mujoco.Renderer(sim.model, height=256, width=256),
    )
    cameras = (shoulder_camera(sim.model), wrist_camera(sim.model))
    writer = None
    if args.video:
        args.video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(args.video), cv2.VideoWriter_fourcc(*"mp4v"), args.hz, (512, 256)
        )
        if not writer.isOpened():
            raise SystemExit(f"could not open video output {args.video}")

    milestones = {"lifted": False, "grasped": False, "two_stacked": False, "three_stacked": False, "success": False, "sustained_pickup": False, "max_height_m": 0.0}
    pickup = SustainedPickup(args.hz, count=len(initial_positions))
    pickup_seconds = None
    trace = {key: [] for key in ("qpos", "qvel", "ctrl", "eq_active", "eq_data", "sim_time", "action", "grasp_flags")}
    stable_steps = 0
    started = time.monotonic()
    step = 0
    total_steps = 0
    reset_count = 0
    current_seed = args.seed
    limit = f"{args.steps} control steps" if args.steps else "until you quit"
    inference_mode = (
        f"temporal ensemble {args.temporal_ensemble_coeff:g}"
        if temporal_ensemble
        else f"{policy.config.n_action_steps}-step action queue"
    )
    print(
        f"Rolling out ACT {limit} at {args.hz:g} Hz "
        f"(seed {args.seed}; {inference_mode}; r resets, q quits)",
        flush=True,
    )
    try:
        while not args.steps or total_steps < args.steps:
            tick = time.monotonic()
            images = []
            for renderer, camera in zip(renderers, cameras):
                renderer.update_scene(sim.data, camera)
                images.append(renderer.render().copy())
            state = np.concatenate(
                [sim.q, [float(sim.data.ctrl[sim.grip_act])]]
            ).astype(np.float32)
            observation = prepare_observation_for_inference(
                {
                    "observation.state": state,
                    "observation.images.shoulder": images[0],
                    "observation.images.wrist": images[1],
                },
                device,
                task=environment.get("task", "stack the three colored cubes"),
                robot_type="panthera_ht_sim",
            )
            observation = preprocessor(observation)
            with torch.inference_mode():
                action = (relative_executor.select_action(observation, state) if relative_executor is not None
                          else postprocessor(policy.select_action(observation)))
            action = np.asarray(action.detach().cpu(), dtype=np.float64).reshape(-1)
            if action.shape != (7,) or not np.isfinite(action).all():
                raise RuntimeError(f"invalid ACT action: {action}")
            q_target = np.clip(action[:6], sim.arm_range[:, 0], sim.arm_range[:, 1])
            if args.max_joint_step is not None:
                q_target = np.clip(q_target, sim.q - args.max_joint_step, sim.q + args.max_joint_step)
            sim.set_arm_ctrl(q_target)
            sim.set_gripper(float(np.clip(action[6] / 0.04, 0.0, 1.0)))
            sim.step(physics_clock.next_steps())
            positions, _ = sim.object_poses()
            metrics = task_stack_metrics(positions)
            milestones["lifted"] |= bool(metrics["lifted_cubes"])
            milestones["grasped"] |= sim.grasped
            milestones["two_stacked"] |= bool(metrics["two_stack"])
            milestones["three_stacked"] |= bool(metrics["three_stack"])
            holding = sim.grasped
            flags = sim.grasp_flags()
            milestones["sustained_pickup"] |= pickup.update(positions[:, 2], flags, step)
            if pickup.success and pickup_seconds is None:
                pickup_seconds = (pickup.first_success_step + 1) / args.hz
            if args.trace:
                values = {"qpos": sim.data.qpos, "qvel": sim.data.qvel, "ctrl": sim.data.ctrl,
                          "eq_active": sim.data.eq_active, "eq_data": sim.model.eq_data,
                          "sim_time": sim.data.time, "action": action, "grasp_flags": flags}
                for name, value in values.items():
                    trace[name].append(np.asarray(value).copy())
            stable_steps = stable_steps + 1 if metrics["task_stack"] and not holding else 0
            milestones["success"] |= stable_steps >= max(1, round(environment.get("success_hold_seconds", .5) * args.hz))
            milestones["max_height_m"] = max(milestones["max_height_m"], metrics["max_height_m"])

            frame = cv2.cvtColor(np.concatenate(images, axis=1), cv2.COLOR_RGB2BGR)
            cv2.rectangle(frame, (0, 0), (512, 43), (0, 0, 0), -1)
            cv2.putText(
                frame,
                f"ACT step {step:04d}  grip {action[6]:.3f}m",
                (8, 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "[r] random reset  [q] quit",
                (8, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (90, 240, 90),
                1,
                cv2.LINE_AA,
            )
            if writer is not None:
                writer.write(frame)
            key = -1
            if not args.no_display:
                cv2.imshow("Panthera ACT rollout - shoulder | wrist", frame)
                key = cv2.waitKey(1) & 0xFF
            total_steps += 1
            if key == ord("q"):
                break
            if key == ord("r"):
                reset_count += 1
                current_seed = args.seed + reset_count
                initial_positions = reset_scene(current_seed)
                step = 0
                stable_steps = 0
                pickup = SustainedPickup(args.hz, count=len(initial_positions))
                print(f"Simulation reset (seed {current_seed})", flush=True)
                continue
            if not args.no_realtime:
                remaining = 1.0 / args.hz - (time.monotonic() - tick)
                if remaining > 0:
                    time.sleep(remaining)
            if step % 10 == 0:
                print(
                    f"step={step:03d} q={sim.q.round(2)} grip={action[6]:.3f}",
                    flush=True,
                )
            step += 1
    finally:
        if writer is not None:
            writer.release()
        for renderer in renderers:
            renderer.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    final_positions, _ = sim.object_poses()
    report = {
        "simulation_dynamics": sim.dynamics,
        "training_simulation_dynamics": contract.get("simulation_dynamics", "weld-v1"),
        "environment": environment,
        "action_representation": representation,
        "pickup_seconds": pickup_seconds,
        "inference": {"temporal_ensemble": bool(temporal_ensemble), "action_steps": policy_config.n_action_steps},
        "fps": args.hz,
        "milestones": milestones,
        "seed": args.seed,
        "final_seed": current_seed,
        "resets": reset_count,
        "steps": total_steps,
        "elapsed_seconds": time.monotonic() - started,
        "initial": stack_metrics(initial_positions),
        "final": stack_metrics(final_positions),
    }
    print(json.dumps(report, indent=2), flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    if args.trace:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.trace, **{key: np.asarray(value) for key, value in trace.items()})


if __name__ == "__main__":
    main()
