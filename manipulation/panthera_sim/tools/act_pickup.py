"""First-pickup data windows, action decoding, and physical validation for ACT."""
from __future__ import annotations

import argparse
from collections import deque
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from teleop.dataset_contract import DEFAULT_DATASET, DEFAULT_RENDERED, file_hash, validate_rendered
from tools.lerobot_image_cache import _dataset_signature


class SustainedPickup:
    """Require one particular cube to be held above the table continuously."""
    def __init__(self, hz, count=3):
        self.required = max(1, round(.5 * hz))
        self.ticks = np.zeros(count, dtype=int)
        self.success = False
        self.first_success_step = None

    def update(self, heights, grasp_flags, step):
        self.ticks = np.where((np.asarray(heights) > .0975) & np.asarray(grasp_flags), self.ticks + 1, 0)
        if np.any(self.ticks >= self.required):
            self.success = True
            if self.first_success_step is None:
                self.first_success_step = step
        return self.success


def scalar_data(dataset):
    import pyarrow.parquet as pq
    columns = ("index", "episode_index", "frame_index", "observation.state", "action")
    tables = [pq.read_table(path, columns=list(columns)) for path in sorted((dataset / "data").glob("**/*.parquet"))]
    result = {key: np.concatenate([np.asarray(t[key].to_pylist()) for t in tables]) for key in columns}
    if not np.array_equal(result["index"], np.arange(len(result["index"]))):
        raise ValueError("Expected ordered contiguous dataset indices")
    return result


def build_manifest(output, dataset=DEFAULT_DATASET, rendered=DEFAULT_RENDERED):
    render = validate_rendered(rendered)
    scalars = scalar_data(dataset)
    episodes = {}
    for index, name in enumerate(render["episodes"]):
        data = np.load(rendered / name / "trajectory.npz")
        # Legacy demonstrations do not contain latch flags. Closed-command plus
        # sustained elevation is a crop heuristic, not the rollout success metric.
        held = (data["obj_pos"][:, :, 2] > .0975) & (data["ctrl"][:, 6:7] < .028)
        ticks = np.zeros(3, dtype=int)
        endpoint = None
        for frame, active in enumerate(held):
            ticks = np.where(active, ticks + 1, 0)
            if np.any(ticks >= 15):
                endpoint = frame
                break
        if endpoint is None or endpoint < 1:
            raise ValueError(f"No first sustained demonstration lift found: {name}")
        ids = np.flatnonzero(scalars["episode_index"] == index)
        if len(ids) != len(data["q"]) - 1 or endpoint > len(ids):
            raise ValueError(f"Rendered/LeRobot episode mismatch: {name}")
        # Observation at endpoint-1 has ctrl[endpoint] as its final target.
        episodes[str(index)] = {"source": name, "start": int(ids[0]),
                                "end_exclusive": int(ids[0] + endpoint), "frames": endpoint,
                                "demonstrated_pickup_seconds": endpoint / 30}
    manifest = {"version": 1, "fps": 30, "dataset": str(dataset.resolve()),
                "dataset_signature": _dataset_signature(dataset),
                "render_manifest_sha256": file_hash(rendered / "manifest.json"),
                "crop_rule": "first same cube elevated >25mm and command <28mm for 15 frames; endpoint is final target",
                "episodes": episodes, "total_frames": sum(r["frames"] for r in episodes.values())}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_manifest(path, dataset):
    manifest = json.loads(path.read_text())
    if manifest["fps"] != 30 or manifest["dataset_signature"] != _dataset_signature(dataset):
        raise ValueError("Pickup manifest does not match this 30 Hz dataset")
    provenance = json.loads((dataset / "meta/provenance.json").read_text())
    if manifest["render_manifest_sha256"] != provenance["render_manifest_sha256"]:
        raise ValueError("Pickup manifest has different source rendering")
    return manifest


def crop_action_chunk(action, padding, state, remaining, representation):
    """Mask the crop boundary before transforming absolute arm targets."""
    action = action.clone()
    padding = padding.clone()
    valid = min(len(action), remaining)
    if valid < 1:
        raise ValueError("Observation has no valid pickup target")
    action[valid:] = action[valid-1].clone()
    padding[valid:] = True
    if representation == "relative":
        action[:, :6] -= state[:6]
    elif representation != "absolute":
        raise ValueError(f"Unknown action representation: {representation}")
    return action, padding


class PickupDataset(torch.utils.data.Dataset):
    def __init__(self, base, manifest, representation):
        self.base, self.representation = base, representation
        self.ends = {int(key): value["end_exclusive"] for key, value in manifest["episodes"].items()}
        scalars = base.hf_dataset.select_columns(["index", "episode_index"]).with_format(None).to_dict()
        indices = np.asarray(scalars["index"])
        episodes = np.asarray(scalars["episode_index"])
        self.indices = np.flatnonzero([int(i) < self.ends[int(ep)] for i, ep in zip(indices, episodes)]).tolist()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        item = dict(self.base[self.indices[index]])
        remaining = self.ends[int(item["episode_index"])] - int(item["index"])
        item["action"], item["action_is_pad"] = crop_action_chunk(
            item["action"], item["action_is_pad"], item["observation.state"], remaining, self.representation)
        return item


def pickup_stats(dataset, manifest, training_episodes, chunk_size, representation, original_stats):
    """Fit action scaling on valid training chunk targets only, never held-out ones."""
    data = scalar_data(dataset)
    chunks = []
    for ep in training_episodes:
        bounds = manifest["episodes"][str(ep)]
        ids = np.arange(bounds["start"], bounds["end_exclusive"])
        for horizon in range(chunk_size):
            current = ids[ids+horizon < bounds["end_exclusive"]]
            values = data["action"][current+horizon].copy()
            if representation == "relative":
                values[:, :6] -= data["observation.state"][current, :6]
            chunks.append(values)
    values = np.concatenate(chunks)
    stats = copy.deepcopy(original_stats)
    stats["action"] = {"mean": values.mean(0).astype(np.float32),
                       "std": np.maximum(values.std(0), 1e-6).astype(np.float32),
                       "min": values.min(0).astype(np.float32), "max": values.max(0).astype(np.float32)}
    return stats


class RelativeChunkExecutor:
    """Decode to physical absolute targets before queueing or temporal averaging."""
    def __init__(self, policy, postprocessor):
        self.policy, self.postprocessor = policy, postprocessor
        self.queue = deque()
        self.ensembler = None
        if policy.config.temporal_ensemble_coeff is not None:
            from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
            self.ensembler = ACTTemporalEnsembler(policy.config.temporal_ensemble_coeff, policy.config.chunk_size)

    def reset(self):
        self.queue.clear()
        if self.ensembler is not None:
            self.ensembler.reset()

    def select_action(self, observation, physical_state):
        if self.ensembler is not None or not self.queue:
            chunk = self.postprocessor(self.policy.predict_action_chunk(observation)).clone()
            anchor = torch.as_tensor(physical_state[:6], device=chunk.device, dtype=chunk.dtype)
            chunk[..., :6] += anchor
            if self.ensembler is not None:
                return self.ensembler.update(chunk)
            self.queue.extend(chunk[:, :self.policy.config.n_action_steps].transpose(0, 1))
        return self.queue.popleft()


class PhysicalErrors:
    def __init__(self, representation):
        import mujoco
        from sim.panthera_env import PantheraSim
        self.mujoco, self.sim = mujoco, PantheraSim()
        self.data = mujoco.MjData(self.sim.model)
        self.representation = representation
        self.positions, self.joints, self.grippers = [], [], []

    def add(self, prediction, target, state):
        prediction, target, state = [x.detach().cpu().numpy().copy() for x in (prediction, target, state)]
        if self.representation == "relative":
            prediction[:, :6] += state[:, :6]
            target[:, :6] += state[:, :6]
        self.joints.extend(np.abs(prediction[:, :6]-target[:, :6]).mean(1)*180/np.pi)
        self.grippers.extend(np.abs(prediction[:, 6]-target[:, 6])*1000)
        for predicted, demonstrated in zip(prediction, target):
            positions = []
            for q in (predicted, demonstrated):
                self.data.qpos[self.sim.arm_qadr] = q[:6]
                self.mujoco.mj_kinematics(self.sim.model, self.data)
                positions.append(self.data.site_xpos[self.sim.ee_site].copy())
            self.positions.append(np.linalg.norm(positions[0]-positions[1])*1000)

    def summary(self):
        return {"validation_first_target_mm": float(np.mean(self.positions)),
                "validation_first_target_p95_mm": float(np.percentile(self.positions, 95)),
                "validation_first_joint_mae_deg": float(np.mean(self.joints)),
                "validation_first_gripper_mae_mm": float(np.mean(self.grippers))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.output)
    print(json.dumps({"episodes": len(manifest["episodes"]), "frames": manifest["total_frames"]}))
