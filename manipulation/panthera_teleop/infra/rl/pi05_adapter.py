"""Strict, lazy pi0.5 sampling for EXPO; importing this module loads no model.

The sampling unit is one exact observation and independent model noise, never a
demonstration action substituted for a policy proposal. Saved processors retain
the checkpoint's quantiles. This module does not contact robots or start jobs.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import subprocess
import threading

LEROBOT_COMMIT = "e624f3f7f8411ec3a02635d06e79373341e5ef35"
CAMERAS = ("overhead", "wrist_port2", "wrist_port3")
IMAGE_KEYS = tuple(f"observation.images.{name}" for name in CAMERAS)
ACTION_NAMES = tuple(f"{side}.{name}" for side in ("left", "right")
                     for name in (*[f"joint{i}" for i in range(1, 7)], "gripper_rad"))
TASK = "put the gear in the carrier and put a pin in it"
HORIZON = 50
ACTION_DIM = 14


def _digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_signature(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


@dataclass(frozen=True)
class CandidateBatch:
    normalized: object
    physical: object
    checkpoint_id: str
    processor_id: str
    observation_id: str
    seed: int


class PI05Adapter:
    """Resolve a local checkpoint once; load weights only on explicit use.

    ``sample_candidates`` returns normalized and physical tensors [N,50,14].
    ``infer_candidates`` accepts the existing dashboard's JPEG transport payload.
    Checkpoint and loaded parameter mutations invalidate this adapter/cache key.
    """

    def __init__(self, checkpoint, device="cuda"):
        self.checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
        if not self.checkpoint.is_dir():
            raise ValueError("Expected a local pretrained checkpoint directory")
        if (self.checkpoint / "BASE_TRAINING_INCOMPLETE").exists():
            raise ValueError("Cannot sample an incompletely saved base-training checkpoint")
        required = ("config.json", "model.safetensors", "deployment.json",
                    "policy_preprocessor.json", "policy_postprocessor.json")
        for name in required:
            if not (self.checkpoint / name).is_file():
                raise FileNotFoundError(f"Incomplete checkpoint: {name}")
        self.device = str(device)
        self._files = tuple(sorted(path for path in self.checkpoint.rglob("*")
                                   if path.is_file() and path.suffix in (".json", ".safetensors")))
        if any(path.is_symlink() for path in self._files):
            raise ValueError("Checkpoint contents must be fixed local files")
        self._signatures = {path: _file_signature(path) for path in self._files}
        manifest = {str(path.relative_to(self.checkpoint)): _digest_file(path) for path in self._files}
        self._assert_unchanged()
        self.checkpoint_id = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        processor_manifest = {key: value for key, value in manifest.items()
                              if "processor" in key or "normalizer" in key}
        self.processor_id = hashlib.sha256(json.dumps(processor_manifest, sort_keys=True).encode()).hexdigest()
        self.deployment = json.loads((self.checkpoint / "deployment.json").read_text())
        if (self.deployment.get("fps") != 20 or self.deployment.get("task") != TASK
                or self.deployment.get("action_representation") != "absolute"):
            raise ValueError("Checkpoint deployment must match Panthera absolute 20Hz task")
        self.policy = self.preprocessor = self.postprocessor = None
        self._parameter_versions = None
        self._lock = threading.RLock()

    def _assert_unchanged(self):
        files = tuple(sorted(path for path in self.checkpoint.rglob("*")
                             if path.is_file() and path.suffix in (".json", ".safetensors")))
        if files != self._files or any(_file_signature(path) != self._signatures[path] for path in files):
            raise RuntimeError("Checkpoint changed; resolve a new immutable adapter")
        if getattr(self, "policy", None) is not None and self._parameter_versions is not None:
            current = tuple((id(p), p._version) for p in self.policy.parameters())
            if current != self._parameter_versions:
                raise RuntimeError("Policy parameters changed; save/reload before caching proposals")

    @staticmethod
    def _verify_runtime_revision():
        import lerobot
        for directory in Path(lerobot.__file__).resolve().parents:
            if (directory / ".git").exists():
                revision = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"],
                                          check=True, capture_output=True, text=True).stdout.strip()
                dirty = subprocess.run(["git", "-C", str(directory), "status", "--porcelain", "--",
                                        "src/lerobot"], check=True, capture_output=True, text=True).stdout
                if revision != LEROBOT_COMMIT or dirty.strip():
                    raise RuntimeError("EXPO adapter requires the clean pinned LeRobot checkout")
                return
        raise RuntimeError("Cannot verify LeRobot revision; install the pinned editable checkout")

    @staticmethod
    def _validate_config(config):
        if (config.type != "pi05" or config.chunk_size != HORIZON or config.use_relative_actions
                or tuple(config.action_feature_names or ()) != ACTION_NAMES
                or set(config.image_features) != set(IMAGE_KEYS)
                or tuple(config.input_features["observation.state"].shape) != (ACTION_DIM,)
                or tuple(config.output_features["action"].shape) != (ACTION_DIM,)):
            raise ValueError("Expected pi05, absolute 14D actions/state, three Panthera cameras, horizon 50")
        for feature in ("STATE", "ACTION"):
            mode = config.normalization_mapping[feature]
            if getattr(mode, "value", mode) != "QUANTILES":
                raise ValueError("Checkpoint state/action normalization must use saved quantiles")
        if (getattr(config, "use_visual_memory", False) or getattr(config, "use_proprioceptive_memory", False)
                or getattr(config, "rtc_training_max_delay", 0)
                or getattr(getattr(config, "rtc_config", None), "enabled", False)):
            raise ValueError("This EXPO adapter requires a stateless, non-RTC pi05 checkpoint")

    def load(self):
        """Explicit heavyweight operation; strict weight errors propagate."""
        with self._lock:
            self._assert_unchanged()
            if self.policy is not None:
                return self
            self._verify_runtime_revision()
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.policies.pi05.modeling_pi05 import PI05Policy
            from lerobot.policies.factory import make_pre_post_processors
            from safetensors.torch import load_file
            config = PreTrainedConfig.from_pretrained(self.checkpoint, local_files_only=True)
            self._validate_config(config)
            config.device = self.device
            config.compile_model = False
            config.gradient_checkpointing = False
            policy = PI05Policy(config)
            state = load_file(str(self.checkpoint / "model.safetensors"))
            state = policy._fix_pytorch_state_dict_keys(state, config)
            state = {key if key.startswith("model.") else f"model.{key}": value
                     for key, value in state.items()}
            state = policy._prepare_pretrained_state_dict(state)
            aliases = {}
            for name, parameter in policy.named_parameters(remove_duplicate=False):
                aliases.setdefault(id(parameter), []).append(name)
            for names in aliases.values():
                saved = next((state[name] for name in names if name in state), None)
                if saved is not None:
                    for name in names:
                        state.setdefault(name, saved)
            policy.load_state_dict(state, strict=True)
            del state
            pre, post = make_pre_post_processors(
                config, pretrained_path=str(self.checkpoint),
                preprocessor_overrides={"device_processor": {"device": self.device}})
            self.policy = policy.to(self.device).eval()
            self.preprocessor, self.postprocessor = pre, post
            self._parameter_versions = tuple((id(p), p._version) for p in self.policy.parameters())
            self._assert_unchanged()
            return self

    @property
    def normalization(self):
        """Quantiles actually loaded by the saved processors, never new stats."""
        self.load()
        result = {"checkpoint_id": self.checkpoint_id, "processor_id": self.processor_id}
        for feature, label in (("action", "action"), ("observation.state", "state")):
            matches = []
            for step in self.preprocessor.steps:
                stats = getattr(step, "_tensor_stats", {}).get(feature)
                if stats is not None and "q01" in stats and "q99" in stats:
                    matches.append((stats, getattr(step, "eps", 1e-8)))
            if len(matches) != 1:
                raise ValueError(f"Saved processor must contain exactly one quantile map for {feature}")
            stats, epsilon = matches[0]
            lower, upper = stats["q01"].detach().cpu().reshape(-1), stats["q99"].detach().cpu().reshape(-1)
            import torch
            if (lower.numel() != ACTION_DIM or upper.numel() != ACTION_DIM
                    or not torch.isfinite(lower).all() or not torch.isfinite(upper).all()
                    or (upper < lower).any()):
                raise ValueError(f"Invalid saved 14D quantiles for {feature}")
            result[label] = {"q01": lower.tolist(), "q99": upper.tolist(),
                             "mode": "QUANTILES", "eps": float(epsilon)}
        return result

    def denormalize_actions(self, normalized):
        """Apply the checkpoint's actual saved postprocessor to edited chunks."""
        import torch
        self.load()
        self._assert_unchanged()
        actions = torch.as_tensor(normalized).float()
        if actions.shape[-1] != ACTION_DIM or not torch.isfinite(actions).all():
            raise ValueError("Expected finite normalized 14D actions")
        with torch.inference_mode():
            return self.postprocessor(actions.clone()).detach().cpu().clone()

    def _observation(self, observation):
        import torch
        if set(key for key in observation if key.startswith("observation.images.")) != set(IMAGE_KEYS):
            raise ValueError("All three exact named camera observations are required")
        if observation.get("task", TASK) != TASK:
            raise ValueError("Task differs from immutable checkpoint deployment")
        state = torch.as_tensor(observation["observation.state"]).detach().cpu().float().clone()
        if state.shape == (1, ACTION_DIM):
            state = state[0]
        if state.shape != (ACTION_DIM,) or not torch.isfinite(state).all():
            raise ValueError("Expected one finite physical 14D state")
        batch = {"observation.state": state, "task": TASK}
        digest = hashlib.sha256(TASK.encode())
        digest.update(state.numpy().tobytes())
        for key in IMAGE_KEYS:
            image = torch.as_tensor(observation[key]).detach().cpu().clone()
            if image.shape == (1, 3, 256, 256):
                image = image[0]
            if image.shape != (3, 256, 256):
                raise ValueError(f"Expected cropped RGB CHW 256x256 image: {key}")
            if image.dtype != torch.uint8 and not image.is_floating_point():
                raise ValueError("Images must be uint8 RGB or RGB floating point arrays")
            image = image.float() / 255 if image.dtype == torch.uint8 else image.float()
            if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise ValueError("Images must be uint8 RGB or finite RGB floats in [0,1]")
            batch[key] = image
            digest.update(key.encode())
            digest.update(image.numpy().tobytes())
        return batch, digest.hexdigest()

    def sample_candidates(self, observation, num_candidates=8, seed=0):
        """Sample N genuine normalized flow-policy chunks at the same observation."""
        import torch
        if type(num_candidates) is not int or not 1 <= num_candidates <= 64:
            raise ValueError("num_candidates must be an integer between 1 and 64")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0,2**63)")
        raw, observation_id = self._observation(observation)
        with self._lock:
            self.load()
            with torch.inference_mode():
                self.policy.reset()
                prepared = self.preprocessor(raw)
                repeated = {}
                for key, value in prepared.items():
                    if isinstance(value, torch.Tensor):
                        if value.shape[0] != 1:
                            raise ValueError("Saved processor must retain the one-observation batch")
                        repeated[key] = value.repeat(num_candidates, *([1] * (value.ndim - 1)))
                    elif isinstance(value, list):
                        repeated[key] = value * num_candidates
                    else:
                        repeated[key] = value
                generator = torch.Generator(device=self.device).manual_seed(seed)
                noise = torch.randn((num_candidates, HORIZON, self.policy.config.max_action_dim),
                                    device=self.device, generator=generator)
                normalized = self.policy.predict_action_chunk(repeated, noise=noise)
                if normalized.shape != (num_candidates, HORIZON, ACTION_DIM) or not torch.isfinite(normalized).all():
                    raise RuntimeError("Invalid genuine pi05 candidate output")
                physical = self.postprocessor(normalized.clone())
                if physical.shape != normalized.shape or not torch.isfinite(physical).all():
                    raise RuntimeError("Saved postprocessor produced invalid actions")
                self._assert_unchanged()
                return CandidateBatch(normalized.detach().cpu().clone(), physical.detach().cpu().clone(),
                                      self.checkpoint_id, self.processor_id, observation_id, seed)

    def infer_candidates(self, payload, n=8, seed=0):
        """Decode the existing inference payload without changing the observation."""
        import numpy as np
        from PIL import Image
        if set(payload["images"]) != set(CAMERAS):
            raise ValueError("Expected all three Panthera camera JPEGs")
        observation = {"observation.state": payload["state"], "task": payload.get("task", TASK)}
        for name in CAMERAS:
            with Image.open(io.BytesIO(base64.b64decode(payload["images"][name], validate=True))) as image:
                if image.mode != "RGB" or image.size != (256, 256):
                    raise ValueError("Expected cropped RGB 256x256 JPEG")
                observation[f"observation.images.{name}"] = np.asarray(image).copy().transpose(2, 0, 1)
        return self.sample_candidates(observation, num_candidates=n, seed=seed)
