#!/usr/bin/env python3
"""PPO fine-tuning for the ACT three-block stacking policy.

The pretrained ACT visual representation and transformer are frozen.  During
rollout, their first decoder feature is cached and PPO trains the shared ACT
action head, a Gaussian exploration scale, and a privileged state critic.  An
RL checkpoint is still a normal LeRobot ACT checkpoint; ``rl_state.pt`` adds
the optimizer/critic state needed to resume training.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import shutil
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np


from teleop.dataset_contract import PhysicsClock, resolve_control_hz, DEFAULT_DATASET, DEFAULT_CHECKPOINT

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "act_rl" / "panthera_stack_v3"
DEFAULT_DEMOS = REPO_ROOT / "data"
CRITIC_DIM = 47
IMAGE_SHAPE = (256, 256, 3)


def reexec_in_act_venv() -> None:
    python = REPO_ROOT / ".venv-act" / "bin" / "python"
    if not python.is_file() or Path(sys.prefix).resolve() == python.parents[1].resolve():
        return
    if os.environ.get("ROBOT_ARM_ACT_RL_REEXEC") == "1":
        return
    environment = os.environ.copy()
    environment["ROBOT_ARM_ACT_RL_REEXEC"] = "1"
    os.execve(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]], environment)


@dataclass
class EpisodeStats:
    episode_return: float = 0.0
    length: int = 0
    grasp_events: int = 0
    max_height_m: float = 0.0
    max_stage: int = 0
    two_stack: bool = False
    success: bool = False
    demo_start: bool = False


def load_demo_snapshots(root: Path, per_episode: int) -> dict[str, np.ndarray]:
    """Load diverse post-first-stack states from successful demonstrations."""
    from sim.stack_task import stack_metrics
    from sim.panthera_env import PantheraSim
    from teleop.render_vla_dataset import recording_times, reconstruct_fingers
    reconstruction_sim = PantheraSim()

    snapshots: dict[str, list[np.ndarray]] = {
        key: [] for key in ("q", "dq", "ctrl", "obj_pos", "obj_quat", "finger_q")
    }
    episode_paths = sorted(root.glob("episode_*/data.npz"))
    for path in episode_paths:
        with np.load(path) as data:
            eligible = []
            for index, positions in enumerate(data["obj_pos"]):
                metrics = stack_metrics(positions)
                # Start after the first pair is complete, but before the third
                # cube has been lifted.  This avoids reconstructing an active
                # grasp constraint while still covering the whole final-block
                # approach from many arm poses and cube orders.
                if (
                    metrics["two_stack"]
                    and not metrics["three_stack"]
                    and metrics["lifted_cubes"] == 1
                    and data["ctrl"][index, 6] > 0.02
                ):
                    eligible.append(index)
            if not eligible:
                continue
            times, _ = recording_times(data, reconstruction_sim.dt)
            finger_q, _ = reconstruct_fingers(data, times, reconstruction_sim)
            chosen = np.unique(
                np.rint(np.linspace(0, len(eligible) - 1, per_episode)).astype(int)
            )
            for chosen_index in chosen:
                frame = eligible[int(chosen_index)]
                for key in snapshots:
                    snapshots[key].append(np.asarray(finger_q[frame] if key == "finger_q" else data[key][frame]).copy())
    if not snapshots["q"]:
        raise ValueError(f"no two-stack demonstration snapshots found under {root}")
    return {key: np.stack(values) for key, values in snapshots.items()}


class PantheraStackEnv:
    """Small, purpose-built environment around the existing PantheraSim."""

    def __init__(
        self,
        seed: int,
        episode_steps: int,
        hz: float,
        max_joint_step: float | None,
        reward_gamma: float,
        action_delta_coef: float,
        action_acceleration_coef: float,
        demo_snapshots: dict[str, np.ndarray] | None,
        demo_reset_probability: float,
    ):
        import mujoco

        sys.path.insert(0, str(REPO_ROOT / "sim"))
        sys.path.insert(0, str(REPO_ROOT / "teleop"))
        from keyboard import DEFAULT_ARM_START_RANGE, randomize_arm_start
        from panthera_env import GRIPPER_OPEN, PantheraSim
        from render_vla_dataset import shoulder_camera, wrist_camera
        from stack_task import StackReward

        self.randomize_arm_start = randomize_arm_start
        self.arm_start_range = DEFAULT_ARM_START_RANGE
        self.gripper_open = GRIPPER_OPEN
        self.sim = PantheraSim()
        self.mujoco = mujoco
        self.renderers = [
            mujoco.Renderer(self.sim.model, height=256, width=256),
            mujoco.Renderer(self.sim.model, height=256, width=256),
        ]
        self.cameras = [shoulder_camera(self.sim.model), wrist_camera(self.sim.model)]
        self.hz = hz
        self.physics_clock = PhysicsClock(hz, self.sim.dt)
        self.episode_steps = episode_steps
        self.max_joint_step = max_joint_step
        self.rewarder = StackReward(
            gamma=reward_gamma,
            action_delta_coef=action_delta_coef,
            action_acceleration_coef=action_acceleration_coef,
        )
        self.rng = np.random.default_rng(seed)
        self.demo_snapshots = demo_snapshots
        self.demo_reset_probability = demo_reset_probability
        self.current_reset_was_demo = False
        self.step_count = 0
        self.previous_action = np.zeros(7, dtype=np.float32)
        self.previous_action_delta = np.zeros(7, dtype=np.float32)
        self.was_grasped = False
        self.stats = EpisodeStats()
        self.reset()

    def _grasped(self) -> bool:
        return self.sim.grasped

    def reset(self, seed: int | None = None, demo_probability: float | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if demo_probability is not None:
            self.demo_reset_probability = demo_probability
        use_demo = bool(
            self.demo_snapshots is not None
            and self.demo_reset_probability > 0
            and self.rng.random() < self.demo_reset_probability
        )
        if use_demo:
            index = int(self.rng.integers(len(self.demo_snapshots["q"])))
            self.sim.reset(randomize=False)
            self.sim.data.qpos[self.sim.arm_qadr] = self.demo_snapshots["q"][index]
            self.sim.data.qvel[self.sim.arm_dofadr] = self.demo_snapshots["dq"][index]
            self.sim.data.ctrl[:7] = self.demo_snapshots["ctrl"][index]
            self.sim.data.qpos[self.sim.finger_qadr] = self.demo_snapshots["finger_q"][index]
            self.sim.set_object_poses(
                self.demo_snapshots["obj_pos"][index],
                self.demo_snapshots["obj_quat"][index],
            )
            for address in self.sim.object_dofadr:
                self.sim.data.qvel[address:address + 6] = 0.0
            self.sim.sync_control_state()
            self.mujoco.mj_forward(self.sim.model, self.sim.data)
        else:
            self.sim.reset(randomize=True, rng=self.rng)
            self.randomize_arm_start(self.sim, self.arm_start_range, rng=self.rng)
        self.current_reset_was_demo = use_demo
        self.physics_clock = PhysicsClock(self.hz, self.sim.dt)
        self.step_count = 0
        self.previous_action = np.concatenate(
            [self.sim.q, [float(self.sim.data.ctrl[self.sim.grip_act])]]
        ).astype(np.float32)
        self.previous_action_delta = np.zeros(7, dtype=np.float32)
        self.was_grasped = self._grasped()
        positions, _ = self.sim.object_poses()
        self.rewarder.reset(positions, self.sim.ee_pos(), self.was_grasped)
        self.stats = EpisodeStats(
            max_height_m=float(positions[:, 2].max()), demo_start=use_demo
        )

    def observation(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        images = []
        for renderer, camera in zip(self.renderers, self.cameras):
            renderer.update_scene(self.sim.data, camera)
            images.append(renderer.render().copy())
        state = np.concatenate(
            [self.sim.q, [float(self.sim.data.ctrl[self.sim.grip_act])]]
        ).astype(np.float32)
        return state, images[0], images[1]

    def critic_state(self) -> np.ndarray:
        positions, _ = self.sim.object_poses()
        ee = self.sim.ee_pos()
        lo, hi = self.sim.arm_range[:, 0], self.sim.arm_range[:, 1]
        q_scaled = 2.0 * (self.sim.q - lo) / (hi - lo) - 1.0
        dq_scaled = np.clip(self.sim.dq / 2.0, -2.0, 2.0)
        grip = np.array([self.sim.data.ctrl[self.sim.grip_act] / self.gripper_open])

        def xyz_scale(x):
            return (x - np.array([0.4, 0.0, 0.16])) / np.array([0.25, 0.30, 0.25])

        object_scaled = xyz_scale(positions).reshape(-1)
        ee_object = ((ee[None, :] - positions) / np.array([0.35, 0.35, 0.35])).reshape(-1)
        pairs = np.concatenate(
            [(positions[j] - positions[i]) / 0.15 for i, j in ((0, 1), (0, 2), (1, 2))]
        )
        grasp_flags = self.sim.grasp_flags().astype(float)
        return np.concatenate([
            q_scaled,
            dq_scaled,
            grip,
            xyz_scale(ee),
            object_scaled,
            ee_object,
            pairs,
            grasp_flags,
            [self.step_count / self.episode_steps],
        ]).astype(np.float32)

    def step(self, physical_action: np.ndarray) -> tuple[float, bool, dict, EpisodeStats | None]:
        action = np.asarray(physical_action, dtype=np.float64)
        q_target = np.clip(action[:6], self.sim.arm_range[:, 0], self.sim.arm_range[:, 1])
        if self.max_joint_step is not None:
            q_target = np.clip(q_target, self.sim.q - self.max_joint_step, self.sim.q + self.max_joint_step)
        executed = np.concatenate([q_target, [np.clip(action[6], 0.0, self.gripper_open)]])
        self.sim.set_arm_ctrl(q_target)
        self.sim.set_gripper(executed[6] / self.gripper_open)
        self.sim.step(self.physics_clock.next_steps())
        self.step_count += 1

        grasped = self._grasped()
        positions, _ = self.sim.object_poses()
        # Normalize each command change by its full per-tick range so arm and
        # gripper smoothness have comparable units.  Penalize acceleration as
        # well as velocity: alternating targets are what look like jitter.
        action_scale = np.array(
            [self.max_joint_step or 0.15] * 6 + [self.gripper_open], dtype=np.float64
        )
        action_delta = (executed - self.previous_action) / action_scale
        reward, success, info = self.rewarder.step(
            positions,
            self.sim.ee_pos(),
            grasped,
            action_delta=action_delta,
            action_acceleration=action_delta - self.previous_action_delta,
        )
        self.previous_action = executed.astype(np.float32)
        self.previous_action_delta = action_delta.astype(np.float32)
        self.stats.episode_return += reward
        self.stats.length += 1
        self.stats.grasp_events += int(grasped and not self.was_grasped)
        self.stats.max_height_m = max(self.stats.max_height_m, float(info["max_height_m"]))
        stage = 3 if info["three_stack"] else 2 if info["two_stack"] else 1 if grasped else 0
        self.stats.max_stage = max(self.stats.max_stage, stage)
        self.stats.two_stack |= bool(info["two_stack"])
        self.stats.success |= success
        self.was_grasped = grasped

        done = success or self.step_count >= self.episode_steps
        finished = self.stats if done else None
        return reward, done, info, finished

    def close(self) -> None:
        for renderer in self.renderers:
            renderer.close()


def _shared_array(spec: dict, handles: dict, key: str) -> np.ndarray:
    """Attach to one shared-memory array inside an environment worker."""
    item = spec[key]
    handle = shared_memory.SharedMemory(name=item["name"])
    handles[key] = handle
    return np.ndarray(tuple(item["shape"]), dtype=np.dtype(item["dtype"]), buffer=handle.buf)


def _env_worker_main(connection, spec: dict, indices: list[int], seeds: list[int], config: dict) -> None:
    """Own and operate a group of MuJoCo environments in a spawned process."""
    handles: dict[str, shared_memory.SharedMemory] = {}
    envs: list[PantheraStackEnv] = []
    try:
        arrays = {key: _shared_array(spec, handles, key) for key in spec}
        envs = [
            PantheraStackEnv(
                seed,
                config["episode_steps"],
                config["hz"],
                config["max_joint_step"],
                config["gamma"],
                config["action_delta_coef"],
                config["action_acceleration_coef"],
                config["demo_snapshots"],
                config["demo_reset_probability"],
            )
            for seed in seeds
        ]
        connection.send(("ready", None))
        while True:
            command = connection.recv()
            if command == "observe":
                for index, env in zip(indices, envs):
                    state, shoulder, wrist = env.observation()
                    arrays["state"][index] = state
                    arrays["shoulder"][index] = shoulder
                    arrays["wrist"][index] = wrist
                    arrays["critic"][index] = env.critic_state()
                connection.send(("ok", None))
            elif command == "step":
                completed = []
                for index, env in zip(indices, envs):
                    reward, done, _info, finished = env.step(arrays["action"][index])
                    arrays["reward"][index] = reward
                    arrays["done"][index] = done
                    if finished is not None:
                        completed.append((index, vars(finished).copy()))
                        env.reset()
                    # Keep the post-step/bootstrap state available without a
                    # separate render-and-transfer round trip.
                    arrays["critic"][index] = env.critic_state()
                connection.send(("ok", completed))
            elif command == "reset" or (
                isinstance(command, tuple) and command[0] == "reset"
            ):
                seeds_for_reset = command[1] if isinstance(command, tuple) else None
                demo_probability = command[2] if isinstance(command, tuple) else None
                for index, env in zip(indices, envs):
                    env.reset(
                        None if seeds_for_reset is None else seeds_for_reset[index],
                        demo_probability,
                    )
                    arrays["critic"][index] = env.critic_state()
                connection.send(("ok", None))
            elif command == "close":
                connection.send(("ok", None))
                break
            else:
                raise ValueError(f"unknown environment worker command {command!r}")
    except EOFError:
        pass
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        for env in envs:
            env.close()
        for handle in handles.values():
            handle.close()
        connection.close()


class LocalEnvBatch:
    """Single-process implementation used by ``--env-workers 1``."""

    def __init__(self, seeds: list[int], episode_steps: int, hz: float, max_joint_step: float | None,
                 gamma: float, action_delta_coef: float, action_acceleration_coef: float,
                 demo_snapshots: dict[str, np.ndarray] | None,
                 demo_reset_probability: float):
        self.envs = [
            PantheraStackEnv(
                seed, episode_steps, hz, max_joint_step, gamma,
                action_delta_coef, action_acceleration_coef,
                demo_snapshots, demo_reset_probability,
            )
            for seed in seeds
        ]
        self._critic = np.stack([env.critic_state() for env in self.envs])

    def observe(self):
        observations = [env.observation() for env in self.envs]
        state, shoulder, wrist = zip(*observations)
        self._critic = np.stack([env.critic_state() for env in self.envs])
        return (np.stack(state), np.stack(shoulder), np.stack(wrist)), self._critic

    def step(self, actions: np.ndarray):
        rewards = np.empty(len(self.envs), dtype=np.float32)
        dones = np.empty(len(self.envs), dtype=np.float32)
        completed: list[dict | None] = [None] * len(self.envs)
        for index, env in enumerate(self.envs):
            reward, done, _info, finished = env.step(actions[index])
            rewards[index] = reward
            dones[index] = done
            if finished is not None:
                completed[index] = vars(finished).copy()
                env.reset()
        self._critic = np.stack([env.critic_state() for env in self.envs])
        return rewards, dones, completed

    def critic_states(self) -> np.ndarray:
        return self._critic

    def reset_all(self, seeds: list[int] | None = None,
                  demo_probability: float | None = None) -> None:
        for index, env in enumerate(self.envs):
            env.reset(None if seeds is None else seeds[index], demo_probability)
        self._critic = np.stack([env.critic_state() for env in self.envs])

    def close(self) -> None:
        for env in self.envs:
            env.close()


class ParallelEnvBatch:
    """Spawned MuJoCo workers exchanging bulk data through shared memory."""

    _LAYOUT = {
        "state": ((7,), np.float32),
        "shoulder": (IMAGE_SHAPE, np.uint8),
        "wrist": (IMAGE_SHAPE, np.uint8),
        "critic": ((CRITIC_DIM,), np.float32),
        "action": ((7,), np.float32),
        "reward": ((), np.float32),
        "done": ((), np.float32),
    }

    def __init__(
        self,
        seeds: list[int],
        workers: int,
        episode_steps: int,
        hz: float,
        max_joint_step: float | None,
        gamma: float,
        action_delta_coef: float,
        action_acceleration_coef: float,
        demo_snapshots: dict[str, np.ndarray] | None,
        demo_reset_probability: float,
    ):
        self.num_envs = len(seeds)
        self.handles: dict[str, shared_memory.SharedMemory] = {}
        self.arrays: dict[str, np.ndarray] = {}
        self.spec: dict[str, dict] = {}
        self.processes: list[mp.Process] = []
        self.connections = []
        context = mp.get_context("spawn")
        try:
            for key, (tail_shape, dtype) in self._LAYOUT.items():
                shape = (self.num_envs, *tail_shape)
                size = int(np.prod(shape)) * np.dtype(dtype).itemsize
                handle = shared_memory.SharedMemory(create=True, size=size)
                array = np.ndarray(shape, dtype=dtype, buffer=handle.buf)
                array.fill(0)
                self.handles[key] = handle
                self.arrays[key] = array
                self.spec[key] = {
                    "name": handle.name,
                    "shape": shape,
                    "dtype": np.dtype(dtype).str,
                }

            groups = [group.tolist() for group in np.array_split(np.arange(self.num_envs), workers)]
            config = {
                "episode_steps": episode_steps,
                "hz": hz,
                "max_joint_step": max_joint_step,
                "gamma": gamma,
                "action_delta_coef": action_delta_coef,
                "action_acceleration_coef": action_acceleration_coef,
                "demo_snapshots": demo_snapshots,
                "demo_reset_probability": demo_reset_probability,
            }
            for indices in groups:
                parent, child = context.Pipe()
                process = context.Process(
                    target=_env_worker_main,
                    args=(child, self.spec, indices, [seeds[i] for i in indices], config),
                    daemon=True,
                )
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            self._receive_all("startup")
            self.reset_all()
        except BaseException:
            self.close()
            raise

    def _receive_all(self, operation: str):
        payloads = []
        for connection, process in zip(self.connections, self.processes):
            try:
                status, payload = connection.recv()
            except EOFError as error:
                raise RuntimeError(
                    f"environment worker {process.pid} exited during {operation} "
                    f"with code {process.exitcode}"
                ) from error
            if status not in ("ok", "ready"):
                raise RuntimeError(f"environment worker failed during {operation}:\n{payload}")
            payloads.append(payload)
        return payloads

    def _command(self, command):
        for connection in self.connections:
            connection.send(command)
        return self._receive_all(command)

    def observe(self):
        self._command("observe")
        observations = (
            self.arrays["state"],
            self.arrays["shoulder"],
            self.arrays["wrist"],
        )
        return observations, self.arrays["critic"]

    def step(self, actions: np.ndarray):
        self.arrays["action"][:] = actions
        payloads = self._command("step")
        completed: list[dict | None] = [None] * self.num_envs
        for payload in payloads:
            for index, stats in payload:
                completed[index] = stats
        return self.arrays["reward"].copy(), self.arrays["done"].copy(), completed

    def critic_states(self) -> np.ndarray:
        return self.arrays["critic"]

    def reset_all(self, seeds: list[int] | None = None,
                  demo_probability: float | None = None) -> None:
        if seeds is None and demo_probability is None:
            self._command("reset")
        else:
            self._command(("reset", seeds, demo_probability))

    def close(self) -> None:
        for connection, process in zip(self.connections, self.processes):
            if process.is_alive():
                try:
                    connection.send("close")
                except (BrokenPipeError, EOFError):
                    pass
        for connection, process in zip(self.connections, self.processes):
            if process.is_alive():
                try:
                    connection.recv()
                except (BrokenPipeError, EOFError):
                    pass
            connection.close()
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        self.connections.clear()
        self.processes.clear()
        for handle in self.handles.values():
            handle.close()
            try:
                handle.unlink()
            except FileNotFoundError:
                pass
        self.handles.clear()
        self.arrays.clear()


class ObservationNormalizer:
    """Vectorized equivalent of the LeRobot preprocessing used by rollout_act."""

    def __init__(self, stats: dict, device):
        import torch

        self.device = device
        self.values = {}
        for key in (
            "observation.state",
            "observation.images.shoulder",
            "observation.images.wrist",
            "action",
        ):
            self.values[key] = {
                name: torch.as_tensor(stats[key][name], dtype=torch.float32, device=device)
                for name in ("mean", "std")
            }

    def actor_batch(self, observations):
        import torch

        if (
            isinstance(observations, tuple)
            and len(observations) == 3
            and observations[0].ndim == 2
        ):
            states, shoulder, wrist = observations
        else:
            states, shoulder, wrist = zip(*observations)
            states, shoulder, wrist = np.stack(states), np.stack(shoulder), np.stack(wrist)
        state = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        state_stats = self.values["observation.state"]
        state = (state - state_stats["mean"]) / (state_stats["std"] + 1e-8)

        images = []
        for key, source in (
            ("observation.images.shoulder", shoulder),
            ("observation.images.wrist", wrist),
        ):
            tensor = torch.as_tensor(source, device=self.device)
            tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
            image_stats = self.values[key]
            tensor = (tensor - image_stats["mean"]) / (image_stats["std"] + 1e-8)
            images.append(tensor)
        return state, images

    def unnormalize_action(self, action):
        stats = self.values["action"]
        return action * stats["std"] + stats["mean"]


class BatchedFeatureEnsembler:
    """ACT temporal ensembling in decoder-feature space.

    ACT's action head is linear, so averaging the decoder features and then
    applying the head is exactly equivalent to averaging predicted actions.
    Keeping the effective feature lets PPO recompute the action likelihood
    after an update while still training the controller that is deployed.
    """

    def __init__(self, num_envs: int, chunk_size: int, feature_dim: int,
                 coefficient: float, device):
        import torch

        self.num_envs = num_envs
        self.chunk_size = chunk_size
        self.feature_dim = feature_dim
        self.device = device
        self.weights = torch.exp(
            -coefficient * torch.arange(chunk_size, device=device, dtype=torch.float32)
        )
        self.reset()

    def reset(self, dones=None) -> None:
        import torch

        if not hasattr(self, "pending") or dones is None:
            self.pending = torch.zeros(
                self.num_envs, self.chunk_size, self.feature_dim, device=self.device
            )
            self.counts = torch.zeros(
                self.num_envs, self.chunk_size, 1, dtype=torch.long, device=self.device
            )
            return
        mask = torch.as_tensor(dones, device=self.device, dtype=torch.bool)
        self.pending[mask] = 0
        self.counts[mask] = 0

    def update(self, feature_chunks):
        """Add ``(env, chunk, feature)`` predictions and consume this tick."""
        import torch

        counts = self.counts.clamp(max=self.chunk_size - 1)
        new_weight = self.weights[counts]
        old_weight = torch.where(
            self.counts > 0,
            torch.cumsum(self.weights, dim=0)[counts.clamp(min=1) - 1],
            torch.zeros_like(new_weight),
        )
        total_weight = old_weight + new_weight
        self.pending = (
            self.pending * old_weight + feature_chunks * new_weight
        ) / total_weight
        self.counts = (self.counts + 1).clamp(max=self.chunk_size)
        feature = self.pending[:, 0].clone()
        self.pending[:, :-1] = self.pending[:, 1:].clone()
        self.counts[:, :-1] = self.counts[:, 1:].clone()
        self.pending[:, -1] = 0
        self.counts[:, -1] = 0
        return feature


def explained_variance(target, prediction) -> float:
    import torch

    variance = torch.var(target)
    if variance < 1e-8:
        return math.nan
    return float(1.0 - torch.var(target - prediction) / variance)


def save_checkpoint(
    destination: Path,
    policy,
    preprocessor,
    postprocessor,
    critic,
    log_std,
    actor_optimizer,
    critic_optimizer,
    anchor_weight,
    anchor_bias,
    update: int,
    env_steps: int,
    args,
) -> None:
    temp = destination.with_name(f".{destination.name}.tmp")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    policy.save_pretrained(temp)
    (temp / "deployment.json").write_text(json.dumps(args.deployment, indent=2) + "\n")
    preprocessor.save_pretrained(temp)
    postprocessor.save_pretrained(temp)
    import torch

    torch.save(
        {
            "version": 1,
            "update": update,
            "env_steps": env_steps,
            "critic": critic.state_dict(),
            "log_std": log_std.detach().cpu(),
            "actor_optimizer": actor_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "anchor_weight": anchor_weight.detach().cpu(),
            "anchor_bias": anchor_bias.detach().cpu(),
            "args": vars(args),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        temp / "rl_state.pt",
    )
    (temp / "rl_config.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")
    if destination.exists():
        shutil.rmtree(destination)
    temp.rename(destination)


def main() -> None:
    reexec_in_act_venv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--demo-root", type=Path, default=DEFAULT_DEMOS)
    parser.add_argument("--demo-reset-probability", type=float, default=0.5)
    parser.add_argument("--demo-snapshots-per-episode", type=int, default=4)
    parser.add_argument("--resume", type=Path, help="RL checkpoint directory to resume")
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument(
        "--env-workers",
        type=int,
        default=4,
        help="MuJoCo worker processes; 1 keeps the legacy single-process loop",
    )
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--episode-steps", type=int, default=0, help="0: 15 seconds at checkpoint FPS")
    parser.add_argument("--hz", type=float, help="defaults to checkpoint FPS")
    parser.add_argument("--max-joint-step", type=float, help="optional extra action clipping")
    parser.add_argument("--ppo-epochs", type=int, default=6)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument(
        "--anchor-coef", type=float, default=5.0,
        help="penalty on normalized-action drift from the imitation policy",
    )
    parser.add_argument("--actor-lr", type=float, default=3e-6)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--init-action-std", type=float, default=0.04)
    parser.add_argument("--final-action-std", type=float, default=0.02)
    parser.add_argument("--exploration-anneal-updates", type=int, default=500)
    parser.add_argument("--target-kl", type=float, default=0.005)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--temporal-ensemble-coeff", type=float, default=0.01)
    parser.add_argument("--action-delta-coef", type=float, default=0.01)
    parser.add_argument("--action-acceleration-coef", type=float, default=0.02)
    parser.add_argument("--checkpoint-freq", type=int, default=10)
    parser.add_argument("--keep-checkpoints", type=int, default=5)
    parser.add_argument(
        "--eval-freq", type=int, default=25,
        help="run deterministic evaluation every N updates; 0 disables it",
    )
    parser.add_argument("--eval-episodes", type=int, default=24)
    parser.add_argument("--eval-demo-episodes", type=int, default=24)
    parser.add_argument(
        "--rollback-on-eval-regression",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="restore the best evaluated actor and decay its LR after a regression",
    )
    parser.add_argument("--rollback-lr-factor", type=float, default=0.5)
    parser.add_argument("--min-actor-lr", type=float, default=2.5e-7)
    parser.add_argument("--eval-seed", type=int, default=20261001)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--mujoco-gl", choices=("egl", "osmesa"), default="egl")
    args = parser.parse_args()

    if min(
        args.updates,
        args.num_envs,
        args.env_workers,
        args.rollout_steps,
    ) <= 0:
        parser.error(
            "updates, num-envs, env-workers, rollout-steps, and episode-steps must be positive"
        )
    if (
        args.minibatch_size <= 0
        or args.init_action_std <= 0
        or args.final_action_std <= 0
        or args.final_action_std > args.init_action_std
        or args.exploration_anneal_updates <= 0
        or args.checkpoint_freq <= 0
        or args.keep_checkpoints <= 0
        or args.eval_freq < 0
        or args.eval_episodes <= 0
        or args.eval_demo_episodes <= 0
        or args.demo_snapshots_per_episode <= 0
        or not 0.0 < args.rollback_lr_factor <= 1.0
        or args.min_actor_lr <= 0
        or args.min_actor_lr > args.actor_lr
    ):
        parser.error(
            "minibatch-size, action stds, exploration-anneal-updates, checkpoint-freq, "
            "keep-checkpoints, eval episode counts, and demo-snapshots-per-episode must "
            "be positive; final-action-std cannot exceed init-action-std; eval-freq "
            "must be nonnegative; rollback-lr-factor must be in (0, 1], and "
            "min-actor-lr cannot exceed actor-lr"
        )
    if min(
        args.temporal_ensemble_coeff,
        args.action_delta_coef,
        args.action_acceleration_coef,
        args.anchor_coef,
        args.entropy_coef,
    ) < 0:
        parser.error("ensemble, reward, anchor, and entropy coefficients must be nonnegative")
    if not 0.0 <= args.demo_reset_probability <= 1.0:
        parser.error("--demo-reset-probability must be between 0 and 1")
    os.environ.setdefault("MUJOCO_GL", args.mujoco_gl)

    import torch
    from torch import nn
    from torch.distributions import Normal

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = (args.resume or args.checkpoint).expanduser().resolve()
    source_contract = source / "deployment.json"
    if source_contract.is_file() and json.loads(source_contract.read_text()).get("action_representation", "absolute") != "absolute":
        parser.error("This RL trainer requires absolute-action checkpoints; relative pickup models are supported by rollout_act.py")
    dataset = args.dataset.expanduser().resolve()
    demo_root = args.demo_root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    demo_snapshots = load_demo_snapshots(demo_root, args.demo_snapshots_per_episode)
    print(
        f"loaded {len(demo_snapshots['q'])} two-stack snapshots from {demo_root}",
        flush=True,
    )

    config = PreTrainedConfig.from_pretrained(source)
    config.device = str(device)
    config.use_amp = device.type == "cuda"
    # Save the same receding-horizon temporal ensemble in every RL checkpoint
    # that is used during PPO collection and evaluation.
    config.n_action_steps = 1
    config.temporal_ensemble_coeff = args.temporal_ensemble_coeff
    policy = ACTPolicy.from_pretrained(source, config=config).to(device)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    for parameter in policy.model.action_head.parameters():
        parameter.requires_grad_(True)

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(source)
    )
    metadata = LeRobotDatasetMetadata("local/panthera_stack", root=dataset)
    args.hz = resolve_control_hz(source, args.hz, metadata.fps)
    args.episode_steps = args.episode_steps or round(15 * args.hz)
    if args.episode_steps < 0 or (args.max_joint_step is not None and args.max_joint_step <= 0):
        parser.error("invalid episode duration or action clipping")
    contract_path = source / "deployment.json"
    args.deployment = json.loads(contract_path.read_text()) if contract_path.is_file() else {"version": 1, "fps": args.hz}
    if args.max_joint_step is None:
        args.max_joint_step = args.deployment.get("max_joint_step")
    from sim.dynamics import CONTACT_DYNAMICS, recorded_dynamics
    if args.resume and recorded_dynamics(args.deployment) != CONTACT_DYNAMICS:
        parser.error("Cannot resume RL optimizer state across physics versions; use --checkpoint for a fresh run")
    args.deployment.setdefault("training_simulation_dynamics", recorded_dynamics(args.deployment))
    args.deployment["simulation_dynamics"] = CONTACT_DYNAMICS
    args.deployment["max_joint_step"] = args.max_joint_step
    normalizer = ObservationNormalizer(metadata.stats, device)

    critic = nn.Sequential(
        nn.Linear(CRITIC_DIM, 256), nn.Tanh(),
        nn.Linear(256, 256), nn.Tanh(),
        nn.Linear(256, 1),
    ).to(device)
    # Exploration follows a deterministic schedule.  Learning the variance
    # while also rewarding entropy kept the old run near its very noisy
    # initialization for all 700k steps.
    log_std = nn.Parameter(
        torch.full((7,), math.log(args.init_action_std), device=device),
        requires_grad=False,
    )
    actor_optimizer = torch.optim.Adam(
        policy.model.action_head.parameters(), lr=args.actor_lr, eps=1e-5
    )
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr, eps=1e-5)
    anchor_weight = policy.model.action_head.weight.detach().clone()
    anchor_bias = policy.model.action_head.bias.detach().clone()

    start_update = 0
    env_steps = 0
    if args.resume:
        state = torch.load(source / "rl_state.pt", map_location=device, weights_only=False)
        critic.load_state_dict(state["critic"])
        log_std.data.copy_(state["log_std"].to(device))
        try:
            actor_optimizer.load_state_dict(state["actor_optimizer"])
        except ValueError as error:
            raise SystemExit(
                "this checkpoint predates conservative RL v2 and cannot safely resume; "
                "start a fresh run from the imitation checkpoint"
            ) from error
        critic_optimizer.load_state_dict(state["critic_optimizer"])
        anchor_weight = state.get("anchor_weight", anchor_weight.cpu()).to(device)
        anchor_bias = state.get("anchor_bias", anchor_bias.cpu()).to(device)
        start_update = int(state["update"])
        env_steps = int(state["env_steps"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.type == "cuda" and state.get("torch_cuda_rng") is not None:
            # ``map_location=device`` also moves these saved byte tensors to
            # CUDA, but PyTorch's RNG restore API requires CPU ByteTensors.
            torch.cuda.set_rng_state_all([rng_state.cpu() for rng_state in state["torch_cuda_rng"]])
        print(f"resumed {source} at update={start_update} env_steps={env_steps}", flush=True)

    captured: list[torch.Tensor] = []

    def capture_decoder_feature(_module, inputs):
        captured.append(inputs[0].detach())

    hook = policy.model.action_head.register_forward_pre_hook(capture_decoder_feature)
    env_workers = min(args.env_workers, args.num_envs)
    env_seeds = [args.seed + 1009 * i + 104729 * start_update for i in range(args.num_envs)]
    if env_workers == 1:
        env_batch = LocalEnvBatch(
            env_seeds, args.episode_steps, args.hz, args.max_joint_step,
            args.gamma, args.action_delta_coef, args.action_acceleration_coef,
            demo_snapshots, args.demo_reset_probability,
        )
    else:
        env_batch = ParallelEnvBatch(
            env_seeds,
            env_workers,
            args.episode_steps,
            args.hz,
            args.max_joint_step,
            args.gamma,
            args.action_delta_coef,
            args.action_acceleration_coef,
            demo_snapshots,
            args.demo_reset_probability,
        )
    recent_episodes: deque[EpisodeStats] = deque(maxlen=100)
    metrics_path = args.output / "metrics.jsonl"
    started = time.monotonic()
    initial_env_steps = env_steps
    feature_dim = policy.config.dim_model
    ensembler = BatchedFeatureEnsembler(
        args.num_envs,
        policy.config.chunk_size,
        feature_dim,
        args.temporal_ensemble_coeff,
        device,
    )

    def actor_features(observations) -> torch.Tensor:
        state_tensor, image_tensors = normalizer.actor_batch(observations)
        captured.clear()
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            policy.model({OBS_STATE: state_tensor, OBS_IMAGES: image_tensors})
        if len(captured) != 1:
            raise RuntimeError(f"expected one ACT action-head call, captured {len(captured)}")
        return captured[0].float()

    def deterministic_evaluation(
        episodes: int, update: int, demo_probability: float
    ) -> dict[str, float]:
        """Evaluate random or curriculum starts on a repeatable reset sequence."""
        completed: list[EpisodeStats] = []
        seed_offset = 1_000_003 if demo_probability == 1.0 else 0
        eval_seeds = [
            args.eval_seed + seed_offset + 1009 * i for i in range(args.num_envs)
        ]
        env_batch.reset_all(eval_seeds, demo_probability)
        ensembler.reset()
        while len(completed) < episodes:
            observations, _critic_states = env_batch.observe()
            feature_chunks = actor_features(observations)
            feature = ensembler.update(feature_chunks)
            with torch.no_grad():
                normalized_action = policy.model.action_head(feature)
                physical_action = normalizer.unnormalize_action(normalized_action).cpu().numpy()
            _rewards, dones, finished_episodes = env_batch.step(physical_action)
            ensembler.reset(dones)
            for finished in finished_episodes:
                if finished is not None and len(completed) < episodes:
                    completed.append(EpisodeStats(**finished))
        # Evaluations must not make the subsequent training reset identical.
        training_seeds = [
            args.seed + 10_000_019 + 104729 * update + 1009 * i
            for i in range(args.num_envs)
        ]
        env_batch.reset_all(training_seeds, args.demo_reset_probability)
        ensembler.reset()
        prefix = "eval_demo" if demo_probability == 1.0 else "eval"
        return {
            f"{prefix}_return": float(np.mean([e.episode_return for e in completed])),
            f"{prefix}_grasp_rate": float(np.mean([e.grasp_events > 0 for e in completed])),
            f"{prefix}_two_stack_rate": float(np.mean([e.two_stack for e in completed])),
            f"{prefix}_success_rate": float(np.mean([e.success for e in completed])),
            f"{prefix}_max_height_m": float(np.mean([e.max_height_m for e in completed])),
        }

    def evaluation_score(values: dict[str, float]) -> tuple[float, ...]:
        """Real task milestones in priority order (never shaped return)."""
        return (
            values["eval_success_rate"],
            values["eval_demo_success_rate"],
            values["eval_two_stack_rate"],
            values["eval_grasp_rate"],
            values["eval_max_height_m"],
        )

    def scheduled_action_std(update: int) -> float:
        fraction = min(max((update - 1) / args.exploration_anneal_updates, 0.0), 1.0)
        return args.init_action_std + fraction * (
            args.final_action_std - args.init_action_std
        )

    best_path = args.output / "best.json"
    if best_path.is_file():
        best_record = json.loads(best_path.read_text())
        best_score = tuple(best_record["score"])
    else:
        baseline = deterministic_evaluation(args.eval_episodes, start_update, 0.0)
        baseline.update(
            deterministic_evaluation(args.eval_demo_episodes, start_update, 1.0)
        )
        best_score = evaluation_score(baseline)
        best_record = {
            "checkpoint": str(source),
            "update": start_update,
            "score": list(best_score),
            "metrics": baseline,
            "baseline": True,
        }
        best_path.write_text(json.dumps(best_record, indent=2) + "\n")
        with metrics_path.open("a") as stream:
            stream.write(json.dumps({
                "event": "baseline_evaluation",
                "update": start_update,
                "env_steps": env_steps,
                **baseline,
            }) + "\n")
        print(
            f"baseline eval={args.eval_episodes} return={baseline['eval_return']:.2f} "
            f"grasp={baseline['eval_grasp_rate']:.1%} "
            f"two={baseline['eval_two_stack_rate']:.1%} "
            f"success={baseline['eval_success_rate']:.1%}; "
            f"demo-start success={baseline['eval_demo_success_rate']:.1%}",
            flush=True,
        )

    def snapshot_actor() -> dict[str, torch.Tensor]:
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in policy.model.action_head.state_dict().items()
        }

    # Keep a known-good actor in memory.  Deterministic evaluation uses fixed
    # seeds, so a lower lexicographic milestone score is a real regression,
    # not evaluation sampling noise.  New output directories begin with the
    # source checkpoint as their guarded baseline.
    best_actor_state = snapshot_actor()

    print(
        f"PPO ACT fine-tuning on {device}: {args.num_envs} envs x "
        f"{args.rollout_steps} steps/update across {env_workers} worker(s); "
        f"action-head parameters="
        f"{sum(p.numel() for p in policy.model.action_head.parameters()):,}",
        flush=True,
    )
    try:
        for update in range(start_update + 1, args.updates + 1):
            with torch.no_grad():
                log_std.fill_(math.log(scheduled_action_std(update)))
            features_buf, critic_buf, actions_buf = [], [], []
            logp_buf, rewards_buf, dones_buf, values_buf = [], [], [], []
            rollout_reward = 0.0
            completed_episodes = 0

            for _ in range(args.rollout_steps):
                observations, critic_states = env_batch.observe()
                feature_chunks = actor_features(observations)
                feature = ensembler.update(feature_chunks)
                with torch.no_grad():
                    mean = policy.model.action_head(feature)
                    distribution = Normal(mean, log_std.exp().expand_as(mean))
                    normalized_action = distribution.sample()
                    log_prob = distribution.log_prob(normalized_action).sum(-1)
                    critic_state = torch.as_tensor(
                        critic_states, dtype=torch.float32, device=device
                    )
                    value = critic(critic_state).squeeze(-1)
                    physical_action = normalizer.unnormalize_action(normalized_action).cpu().numpy()

                step_rewards, step_dones, finished_episodes = env_batch.step(physical_action)
                ensembler.reset(step_dones)
                rollout_reward += float(step_rewards.sum())
                for finished in finished_episodes:
                    if finished is not None:
                        recent_episodes.append(EpisodeStats(**finished))
                        completed_episodes += 1

                features_buf.append(feature.cpu().half())
                critic_buf.append(critic_state.cpu())
                actions_buf.append(normalized_action.cpu())
                logp_buf.append(log_prob.cpu())
                rewards_buf.append(torch.from_numpy(step_rewards))
                dones_buf.append(torch.from_numpy(step_dones))
                values_buf.append(value.cpu())

            with torch.no_grad():
                final_states = torch.as_tensor(
                    env_batch.critic_states(), dtype=torch.float32, device=device
                )
                next_value = critic(final_states).squeeze(-1).cpu()

            rewards = torch.stack(rewards_buf)
            dones = torch.stack(dones_buf)
            values = torch.stack(values_buf)
            advantages = torch.zeros_like(rewards)
            last_gae = torch.zeros(args.num_envs)
            for t in reversed(range(args.rollout_steps)):
                alive = 1.0 - dones[t]
                following_value = next_value if t == args.rollout_steps - 1 else values[t + 1]
                delta = rewards[t] + args.gamma * following_value * alive - values[t]
                last_gae = delta + args.gamma * args.gae_lambda * alive * last_gae
                advantages[t] = last_gae
            returns = advantages + values

            flat_feature = torch.stack(features_buf).reshape(-1, feature_dim)
            flat_critic = torch.stack(critic_buf).reshape(-1, CRITIC_DIM)
            flat_action = torch.stack(actions_buf).reshape(-1, 7)
            flat_old_logp = torch.stack(logp_buf).reshape(-1)
            flat_advantage = advantages.reshape(-1)
            flat_return = returns.reshape(-1)
            flat_old_value = values.reshape(-1)
            flat_advantage = (flat_advantage - flat_advantage.mean()) / (
                flat_advantage.std() + 1e-8
            )

            batch_size = len(flat_advantage)
            indices = np.arange(batch_size)
            losses = {
                "policy": [], "value": [], "entropy": [], "anchor": [],
                "residual_rms": [], "kl": [], "clipfrac": [],
            }
            stop_early = False
            for _epoch in range(args.ppo_epochs):
                np.random.shuffle(indices)
                for start in range(0, batch_size, args.minibatch_size):
                    mb = torch.as_tensor(
                        indices[start:start + args.minibatch_size], device=device
                    )
                    mb_feature = flat_feature[mb.cpu()].to(device=device, dtype=torch.float32)
                    mb_action = flat_action[mb.cpu()].to(device)
                    mb_old_logp = flat_old_logp[mb.cpu()].to(device)
                    mb_advantage = flat_advantage[mb.cpu()].to(device)
                    mb_return = flat_return[mb.cpu()].to(device)
                    mb_old_value = flat_old_value[mb.cpu()].to(device)
                    mb_critic = flat_critic[mb.cpu()].to(device)

                    mean = policy.model.action_head(mb_feature)
                    with torch.no_grad():
                        base_mean = torch.nn.functional.linear(
                            mb_feature, anchor_weight, anchor_bias
                        )
                    distribution = Normal(mean, log_std.exp().expand_as(mean))
                    new_logp = distribution.log_prob(mb_action).sum(-1)
                    entropy = distribution.entropy().sum(-1).mean()
                    log_ratio = new_logp - mb_old_logp
                    ratio = log_ratio.exp()
                    policy_loss = -torch.min(
                        ratio * mb_advantage,
                        torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef) * mb_advantage,
                    ).mean()
                    residual = mean - base_mean
                    anchor_loss = residual.square().mean()
                    actor_loss = policy_loss - args.entropy_coef * entropy + args.anchor_coef * anchor_loss
                    actor_optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        policy.model.action_head.parameters(), args.max_grad_norm
                    )
                    actor_optimizer.step()

                    new_value = critic(mb_critic).squeeze(-1)
                    value_clipped = mb_old_value + torch.clamp(
                        new_value - mb_old_value, -args.clip_coef, args.clip_coef
                    )
                    value_loss = 0.5 * torch.max(
                        (new_value - mb_return).square(),
                        (value_clipped - mb_return).square(),
                    ).mean()
                    critic_optimizer.zero_grad(set_to_none=True)
                    (args.value_coef * value_loss).backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
                    critic_optimizer.step()

                    with torch.no_grad():
                        approx_kl = ((ratio - 1.0) - log_ratio).mean()
                        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()
                    losses["policy"].append(float(policy_loss.detach()))
                    losses["value"].append(float(value_loss.detach()))
                    losses["entropy"].append(float(entropy.detach()))
                    losses["anchor"].append(float(anchor_loss.detach()))
                    losses["residual_rms"].append(
                        float(residual.detach().square().mean().sqrt())
                    )
                    losses["kl"].append(float(approx_kl))
                    losses["clipfrac"].append(float(clipfrac))
                if args.target_kl > 0 and np.mean(losses["kl"][-math.ceil(batch_size / args.minibatch_size):]) > args.target_kl:
                    stop_early = True
                    break

            env_steps += batch_size
            elapsed = time.monotonic() - started
            episode_count = len(recent_episodes)
            random_episodes = [e for e in recent_episodes if not e.demo_start]
            demo_episodes = [e for e in recent_episodes if e.demo_start]
            summary = {
                "update": update,
                "env_steps": env_steps,
                "num_envs": args.num_envs,
                "env_workers": env_workers,
                "steps_per_second": (env_steps - initial_env_steps) / max(elapsed, 1e-6),
                "rollout_reward_per_step": rollout_reward / batch_size,
                "episodes_this_update": completed_episodes,
                "recent_episodes": episode_count,
                "recent_return": float(np.mean([e.episode_return for e in recent_episodes])) if episode_count else math.nan,
                "recent_grasp_rate": float(np.mean([e.grasp_events > 0 for e in random_episodes])) if random_episodes else math.nan,
                "recent_two_stack_rate": float(np.mean([e.two_stack for e in random_episodes])) if random_episodes else math.nan,
                "recent_success_rate": float(np.mean([e.success for e in random_episodes])) if random_episodes else math.nan,
                "recent_max_height_m": float(np.mean([e.max_height_m for e in random_episodes])) if random_episodes else math.nan,
                "recent_random_episodes": len(random_episodes),
                "recent_demo_episodes": len(demo_episodes),
                "recent_demo_grasp_rate": float(np.mean([e.grasp_events > 0 for e in demo_episodes])) if demo_episodes else math.nan,
                "recent_demo_success_rate": float(np.mean([e.success for e in demo_episodes])) if demo_episodes else math.nan,
                "policy_loss": float(np.mean(losses["policy"])),
                "value_loss": float(np.mean(losses["value"])),
                "entropy": float(np.mean(losses["entropy"])),
                "anchor_loss": float(np.mean(losses["anchor"])),
                "residual_rms": float(np.mean(losses["residual_rms"])),
                "approx_kl": float(np.mean(losses["kl"])),
                "clip_fraction": float(np.mean(losses["clipfrac"])),
                "explained_variance": explained_variance(flat_return, flat_old_value),
                "action_std": log_std.exp().detach().cpu().tolist(),
                "early_stop_kl": stop_early,
                "actor_lr": float(actor_optimizer.param_groups[0]["lr"]),
            }
            is_best = False
            rolled_back = False
            if args.eval_freq and update % args.eval_freq == 0:
                evaluation = deterministic_evaluation(args.eval_episodes, update, 0.0)
                evaluation.update(
                    deterministic_evaluation(args.eval_demo_episodes, update, 1.0)
                )
                summary.update(evaluation)
                score = evaluation_score(evaluation)
                is_best = score > best_score
                summary["beats_baseline_or_best"] = is_best
                if is_best:
                    best_score = score
                    best_actor_state = snapshot_actor()
                elif score < best_score and args.rollback_on_eval_regression:
                    policy.model.action_head.load_state_dict(best_actor_state)
                    # Adam momentum accumulated during the bad excursion points
                    # back toward it.  Discard it and make the next search more
                    # conservative instead of immediately repeating the drift.
                    actor_optimizer.state.clear()
                    for group in actor_optimizer.param_groups:
                        group["lr"] = max(
                            args.min_actor_lr,
                            float(group["lr"]) * args.rollback_lr_factor,
                        )
                    rolled_back = True
                    summary["actor_rolled_back"] = True
                    summary["actor_lr_after_rollback"] = float(
                        actor_optimizer.param_groups[0]["lr"]
                    )
            with metrics_path.open("a") as stream:
                stream.write(json.dumps(summary) + "\n")
            print(
                f"update={update:04d} steps={env_steps:07d} fps={summary['steps_per_second']:.1f} "
                f"reward/step={summary['rollout_reward_per_step']:+.3f} "
                f"return100={summary['recent_return']:.2f} "
                f"grasp100={summary['recent_grasp_rate']:.1%} "
                f"two100={summary['recent_two_stack_rate']:.1%} "
                f"success100={summary['recent_success_rate']:.1%} "
                f"demo_success100={summary['recent_demo_success_rate']:.1%} "
                f"height100={summary['recent_max_height_m']:.3f}m "
                f"pi={summary['policy_loss']:+.3f} v={summary['value_loss']:.3f} "
                f"kl={summary['approx_kl']:.4f} clip={summary['clip_fraction']:.2f} "
                f"std={np.mean(summary['action_std']):.3f} "
                f"residual={summary['residual_rms']:.3f}",
                flush=True,
            )
            if "eval_success_rate" in summary:
                print(
                    f"  eval={args.eval_episodes} return={summary['eval_return']:.2f} "
                    f"grasp={summary['eval_grasp_rate']:.1%} "
                    f"two={summary['eval_two_stack_rate']:.1%} "
                    f"success={summary['eval_success_rate']:.1%} "
                    f"height={summary['eval_max_height_m']:.3f}m; "
                    f"demo-start success={summary['eval_demo_success_rate']:.1%} "
                    f"grasp={summary['eval_demo_grasp_rate']:.1%}",
                    flush=True,
                )
                if rolled_back:
                    print(
                        "  actor rollback=best-evaluated "
                        f"lr={actor_optimizer.param_groups[0]['lr']:.2e}",
                        flush=True,
                    )

            if (
                update % args.checkpoint_freq == 0
                or update == args.updates
                or "eval_success_rate" in summary
            ):
                checkpoint_dir = args.output / f"checkpoint_{update:06d}"
                save_checkpoint(
                    checkpoint_dir, policy, preprocessor, postprocessor, critic, log_std,
                    actor_optimizer, critic_optimizer, anchor_weight, anchor_bias,
                    update, env_steps, args,
                )
                (args.output / "latest.json").write_text(json.dumps({
                    "checkpoint": str(checkpoint_dir), "update": update, "env_steps": env_steps
                }, indent=2) + "\n")
                if is_best:
                    best_record = {
                        "checkpoint": str(checkpoint_dir),
                        "update": update,
                        "score": list(best_score),
                        "metrics": evaluation,
                        "baseline": False,
                    }
                    best_path.write_text(json.dumps(best_record, indent=2) + "\n")
                    print(f"best_checkpoint={checkpoint_dir}", flush=True)
                checkpoints = sorted(args.output.glob("checkpoint_*"))
                protected = Path(best_record["checkpoint"]).resolve()
                removable = [
                    path for path in checkpoints[:-args.keep_checkpoints]
                    if path.resolve() != protected
                ]
                for old in removable:
                    shutil.rmtree(old)
                print(f"checkpoint={checkpoint_dir}", flush=True)
    finally:
        hook.remove()
        env_batch.close()


if __name__ == "__main__":
    main()
