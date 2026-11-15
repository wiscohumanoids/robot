"""EXPO-FT residual actor and REDQ learner, without a VLA runtime dependency.

The caller supplies genuine current-base proposals in the same frozen
checkpoint normalization as executed actions. The base VLA is maintained by
the separate adapter; this learner never samples substitute base actions.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

import torch
from torch import Tensor, nn

from .models import QEnsemble, SquashedGaussianEditor


@dataclass(frozen=True)
class EXPOConfig:
    obs_dim: int = 2318
    action_dim: int = 14
    chunk_size: int = 1
    hidden_dim: int = 256
    num_critics: int = 10
    target_subset_size: int = 2
    residual_bound: float = .05
    utd_ratio: int = 20
    critic_lr: float = 3e-4
    editor_lr: float = 3e-4
    temperature_lr: float = 3e-4
    tau: float = .005
    initial_temperature: float = .1
    target_entropy: float | None = None
    entropy_scale: float = 1.
    learn_temperature: bool = True
    grad_clip_norm: float = 10.
    seed: int = 0
    action_lower: tuple[float | None, ...] | None = None
    action_upper: tuple[float | None, ...] | None = None

    def __post_init__(self):
        for field in ("obs_dim", "action_dim", "chunk_size", "hidden_dim", "num_critics",
                      "target_subset_size", "utd_ratio"):
            value = getattr(self, field)
            if type(value) is not int or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if self.target_subset_size > self.num_critics:
            raise ValueError("target_subset_size cannot exceed num_critics")
        for field in ("residual_bound", "initial_temperature", "grad_clip_norm"):
            if not math.isfinite(getattr(self, field)) or getattr(self, field) <= 0:
                raise ValueError(f"{field} must be positive and finite")
        for field in ("critic_lr", "editor_lr", "temperature_lr", "entropy_scale"):
            if not math.isfinite(getattr(self, field)) or getattr(self, field) < 0:
                raise ValueError(f"{field} must be nonnegative and finite")
        if not math.isfinite(self.tau) or not 0 <= self.tau <= 1:
            raise ValueError("tau must lie in [0,1]")
        if self.target_entropy is not None and not math.isfinite(self.target_entropy):
            raise ValueError("target_entropy must be finite")
        if type(self.learn_temperature) is not bool or type(self.seed) is not int:
            raise ValueError("learn_temperature must be boolean and seed must be integer")
        for name in ("action_lower", "action_upper"):
            bounds = getattr(self, name)
            if bounds is not None:
                if len(bounds) != self.action_dim or any(v is not None and not math.isfinite(v) for v in bounds):
                    raise ValueError(f"{name} must have action_dim finite entries or None for unrestricted dimensions")
                object.__setattr__(self, name, tuple(None if v is None else float(v) for v in bounds))
        if self.action_lower is not None and self.action_upper is not None:
            if any(lo is not None and hi is not None and lo > hi
                   for lo, hi in zip(self.action_lower, self.action_upper)):
                raise ValueError("action_lower must not exceed action_upper")


def chunk_bellman_target(reward_return: Tensor, bootstrap_discount: Tensor,
                         next_q: Tensor) -> Tensor:
    """Chunk return plus already-discounted continuation; no entropy backup.

    bootstrap_discount includes gamma**C and the terminal mask. Truncation
    handling and reward discounting are the replay contract's responsibility.
    """
    if reward_return.ndim != 1 or reward_return.shape != bootstrap_discount.shape or next_q.shape != reward_return.shape:
        raise ValueError("Bellman inputs must have matching one-dimensional shapes")
    return reward_return + bootstrap_discount * next_q


@torch.no_grad()
def polyak_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    """target <- (1-tau) target + tau source, including floating buffers."""
    if not math.isfinite(tau) or not 0 <= tau <= 1:
        raise ValueError("tau must lie in [0,1]")
    source_state, target_state = source.state_dict(), target.state_dict()
    if source_state.keys() != target_state.keys():
        raise ValueError("Source and target structures differ")
    for name, value in target_state.items():
        update = source_state[name]
        if value.shape != update.shape:
            raise ValueError(f"Source and target shapes differ for {name}")
        if value.is_floating_point():
            value.lerp_(update, tau)
        else:
            value.copy_(update)


def select_candidate_actions(candidates: Tensor, scores: Tensor) -> dict[str, Tensor]:
    """Select one candidate independently for every row, keeping its index."""
    if candidates.ndim != 4 or scores.shape != candidates.shape[:2] or candidates.shape[1] < 1:
        raise ValueError("Expected candidates[B,K,C,A] and scores[B,K]")
    if not torch.isfinite(candidates).all() or not torch.isfinite(scores).all():
        raise ValueError("Candidates and scores must be finite")
    selected = scores.argmax(dim=1)
    actions = candidates[torch.arange(len(candidates), device=candidates.device), selected]
    return dict(actions=actions, selected=selected, scores=scores)


class EXPOLearner:
    """Bounded editor, ten-Q REDQ, and on-the-fly base/edit selection.

    update performs ``utd_ratio`` critic updates on the supplied replay batch,
    with fresh edits and target subsamples, followed by one editor/temperature
    update. A caller may supply fresh replay minibatches by using utd_ratio=1.
    Current proposals must come from the adapter's current base identity.
    """
    CHECKPOINT_VERSION = 1

    def __init__(self, config: EXPOConfig, device: str | torch.device = "cpu"):
        self.config = config
        self.device = torch.device(device)
        flat_dim = config.chunk_size * config.action_dim
        # Isolate initialization from the calling process's random state.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.seed)
            self.critic = QEnsemble(config.obs_dim, flat_dim, config.hidden_dim,
                                    config.num_critics).to(self.device)
            self.editor = SquashedGaussianEditor(config.obs_dim, flat_dim,
                                                 config.hidden_dim, config.residual_bound).to(self.device)
        self.target_critic = deepcopy(self.critic).requires_grad_(False).eval()
        self.log_temperature = nn.Parameter(torch.tensor(math.log(config.initial_temperature),
                                                         device=self.device))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.editor_optimizer = torch.optim.Adam(self.editor.parameters(), lr=config.editor_lr)
        self.temperature_optimizer = torch.optim.Adam([self.log_temperature], lr=config.temperature_lr)
        # The upstream adjusted target accounts for the scaled edit density.
        self.target_entropy = (config.target_entropy if config.target_entropy is not None
                               else -flat_dim / 2 + flat_dim * math.log(config.residual_bound))
        self.rng = torch.Generator(device=self.device).manual_seed(config.seed)
        self.updates = self.critic_updates = self.editor_updates = 0
        self.checkpoint_metadata: dict[str, Any] = {}

    def _project_actions(self, actions: Tensor,
                         candidate_transform: Callable[[Tensor], Tensor] | None = None) -> Tensor:
        """Project normalized actions consistently for behavior, TD and editor.

        Config bounds are checkpoint-normalized physical limits and serialized
        with the learner. None leaves that action coordinate unrestricted.
        Hooks must use differentiable tensor operations for actor training.
        """
        cfg = self.config
        if cfg.action_lower is not None:
            lower = actions.new_tensor([v if v is not None else -math.inf for v in cfg.action_lower])
            actions = torch.maximum(actions, lower)
        if cfg.action_upper is not None:
            upper = actions.new_tensor([v if v is not None else math.inf for v in cfg.action_upper])
            actions = torch.minimum(actions, upper)
        if candidate_transform is not None:
            transformed = candidate_transform(actions)
            if (not isinstance(transformed, Tensor) or transformed.shape != actions.shape or
                    transformed.device != actions.device or not transformed.is_floating_point() or
                    not torch.isfinite(transformed).all()):
                raise ValueError("candidate_transform must preserve candidate shape/device and finite floating values")
            if actions.requires_grad and not transformed.requires_grad:
                raise ValueError("candidate_transform must preserve action gradients during editor training")
            actions = transformed.to(dtype=actions.dtype)
        return actions

    @property
    def temperature(self) -> Tensor:
        return self.log_temperature.exp()

    def _tensor(self, value: Any) -> Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=self.device).detach()

    def _check_obs_candidates(self, obs: Tensor, base_actions: Tensor) -> None:
        cfg = self.config
        if obs.ndim != 2 or obs.shape[1] != cfg.obs_dim or len(obs) < 1:
            raise ValueError(f"Expected observation shape [B,{cfg.obs_dim}]")
        if (base_actions.ndim != 4 or base_actions.shape[0] != len(obs) or
                base_actions.shape[1] < 1 or base_actions.shape[2:] != (cfg.chunk_size, cfg.action_dim)):
            raise ValueError(f"Expected base_actions[B,N,{cfg.chunk_size},{cfg.action_dim}] with N>=1")
        if not torch.isfinite(obs).all() or not torch.isfinite(base_actions).all():
            raise ValueError("Observations and base actions must be finite")

    def _subset(self, deterministic: bool = False) -> Tensor:
        cfg = self.config
        if deterministic:
            return torch.arange(cfg.target_subset_size, device=self.device)
        return torch.randperm(cfg.num_critics, device=self.device,
                              generator=self.rng)[:cfg.target_subset_size]

    @torch.no_grad()
    def _select(self, obs: Tensor, base_actions: Tensor,
                deterministic: bool = False,
                candidate_transform: Callable[[Tensor], Tensor] | None = None) -> dict[str, Tensor]:
        self._check_obs_candidates(obs, base_actions)
        b, n, c, a = base_actions.shape
        repeated_obs = obs.repeat_interleave(n, dim=0)
        edits, _ = self.editor.sample(repeated_obs, base_actions.reshape(b * n, c * a),
                                     deterministic=deterministic, generator=self.rng)
        edited = base_actions + edits.reshape_as(base_actions)
        candidates = self._project_actions(torch.cat((base_actions, edited), dim=1), candidate_transform)
        all_obs = obs.repeat_interleave(2 * n, dim=0)
        q = self.target_critic(all_obs, candidates.reshape(b * 2 * n, c * a),
                               indices=self._subset(deterministic))
        scores = q.min(dim=0).values.reshape(b, 2 * n)
        selection = select_candidate_actions(candidates, scores)
        selection["is_edited"] = selection["selected"] >= n
        return selection

    def select_actions(self, obs: Any, base_actions: Any,
                       deterministic: bool = False,
                       candidate_transform: Callable[[Tensor], Tensor] | None = None) -> dict[str, Tensor]:
        """Rank target-Q scores over raw then edited candidates, including N=1.

        Only edit magnitude is bounded; normalized actions outside [-1,1] are
        retained because checkpoint quantiles are not robot joint limits.
        An optional safety projection is applied to every candidate before
        scoring, so Q compares the actions that the controller can execute.
        """
        return self._select(self._tensor(obs), self._tensor(base_actions), deterministic,
                            candidate_transform)

    def _batch(self, batch: Mapping[str, Any]) -> dict[str, Tensor]:
        names = ("obs", "next_obs", "actions", "base_actions", "next_base_actions",
                 "reward_return", "bootstrap_discount")
        missing = set(names) - set(batch)
        if missing:
            raise ValueError(f"Missing replay fields: {sorted(missing)}")
        tensors = {key: self._tensor(batch[key]) for key in names}
        self._check_obs_candidates(tensors["obs"], tensors["base_actions"])
        self._check_obs_candidates(tensors["next_obs"], tensors["next_base_actions"])
        count = len(tensors["obs"])
        if len(tensors["next_obs"]) != count:
            raise ValueError("Current and next observation batch sizes differ")
        if tensors["actions"].shape != (count, self.config.chunk_size, self.config.action_dim):
            raise ValueError("Executed action shape does not match fixed chunk contract")
        for key in ("reward_return", "bootstrap_discount"):
            if tensors[key].shape != (count,):
                raise ValueError(f"{key} must have shape [B]")
        if any(not torch.isfinite(tensors[key]).all() for key in names):
            raise ValueError("Replay tensors must be finite")
        discount = tensors["bootstrap_discount"]
        if ((discount < 0) | (discount > 1)).any():
            raise ValueError("bootstrap_discount must lie in [0,1]")
        for key in ("transition_valid", "editor_train_valid"):
            value = torch.as_tensor(batch.get(key, torch.ones(count, dtype=torch.bool)), device=self.device)
            if value.shape != (count,) or value.dtype != torch.bool:
                raise ValueError(f"{key} must be a boolean vector[B]")
            tensors[key] = value.detach().clone()
        tensors["editor_train_valid"] &= tensors["transition_valid"]
        return tensors

    def update(self, batch: Mapping[str, Any],
               candidate_transform: Callable[[Tensor], Tensor] | None = None) -> dict[str, float]:
        data = self._batch(batch)
        valid = data["transition_valid"]
        editor_valid = data["editor_train_valid"]
        self.critic.train()
        self.editor.train()
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.editor_optimizer.zero_grad(set_to_none=True)
        self.temperature_optimizer.zero_grad(set_to_none=True)
        metrics: dict[str, float] = {"valid_transitions": float(valid.sum()),
                                     "valid_editor_transitions": float(editor_valid.sum())}
        if not valid.any():
            metrics.update(skipped_update=1., updates=float(self.updates),
                           critic_updates=float(self.critic_updates), editor_updates=float(self.editor_updates))
            return metrics
        d = {key: value[valid] for key, value in data.items()}
        actions = d["actions"].flatten(1)
        for _ in range(self.config.utd_ratio):
            with torch.no_grad():
                choice = self._select(d["next_obs"], d["next_base_actions"],
                                      candidate_transform=candidate_transform)
                # Independent REDQ subsample for the continuation value, as in
                # upstream: candidate ranking and Bellman evaluation differ.
                next_q = self.target_critic(d["next_obs"], choice["actions"].flatten(1),
                                             indices=self._subset()).min(dim=0).values
                target = chunk_bellman_target(d["reward_return"], d["bootstrap_discount"], next_q)
            qs = self.critic(d["obs"], actions)
            loss = (qs - target.unsqueeze(0)).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite critic loss")
            self.critic_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.grad_clip_norm,
                                                  error_if_nonfinite=True)
            self.critic_optimizer.step()
            polyak_update(self.critic, self.target_critic, self.config.tau)
            self.critic_updates += 1
            metrics.update(critic_loss=float(loss.detach()), critic_grad_norm=float(norm),
                           q_mean=float(qs.detach().mean()), target_q_mean=float(target.mean()),
                           selected_edit_fraction=float(choice["is_edited"].float().mean()))
        # Clear critic gradients before actor work. Critic parameters are frozen
        # while derivatives still propagate from Q through the edited action.
        self.critic_optimizer.zero_grad(set_to_none=True)
        if editor_valid.any():
            obs = data["obs"][editor_valid]
            replay_actions = data["actions"][editor_valid].flatten(1)
            self.critic.requires_grad_(False)
            try:
                edits, log_prob = self.editor.sample(obs, replay_actions, generator=self.rng)
                edited_actions = (replay_actions + edits).reshape(-1, 1, self.config.chunk_size,
                                                                  self.config.action_dim)
                projected = self._project_actions(edited_actions, candidate_transform).flatten(1)
                q_edit = self.critic(obs, projected).mean(dim=0)
                actor_loss = (self.config.entropy_scale * self.temperature.detach() * log_prob - q_edit).mean()
                if not torch.isfinite(actor_loss):
                    raise FloatingPointError("Nonfinite editor loss")
                self.editor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.editor.parameters(), self.config.grad_clip_norm,
                                                      error_if_nonfinite=True)
                self.editor_optimizer.step()
                self.editor_updates += 1
                entropy = -log_prob.detach().mean()
                metrics.update(editor_loss=float(actor_loss.detach()), editor_grad_norm=float(norm),
                               entropy=float(entropy), edit_q=float(q_edit.detach().mean()),
                               mean_absolute_edit=float(edits.detach().abs().mean()))
                if self.config.learn_temperature:
                    temperature_loss = self.temperature * (entropy - self.target_entropy)
                    if not torch.isfinite(temperature_loss):
                        raise FloatingPointError("Nonfinite temperature loss")
                    self.temperature_optimizer.zero_grad(set_to_none=True)
                    temperature_loss.backward()
                    if not torch.isfinite(self.log_temperature.grad).all():
                        raise FloatingPointError("Nonfinite temperature gradient")
                    self.temperature_optimizer.step()
                    metrics["temperature_loss"] = float(temperature_loss.detach())
            finally:
                self.critic.requires_grad_(True)
        self.updates += 1
        metrics.update(temperature=float(self.temperature.detach()), updates=float(self.updates),
                       critic_updates=float(self.critic_updates), editor_updates=float(self.editor_updates),
                       skipped_update=0.)
        return metrics

    def save(self, path: str | Path) -> Path:
        """Atomically save learner networks, targets, optimizers and local RNG."""
        path = Path(path)
        # Restrict metadata to portable, finite JSON data; never pickle caller
        # objects alongside the tensor/primitive learner state.
        if not isinstance(self.checkpoint_metadata, dict):
            raise ValueError('checkpoint_metadata must be a JSON mapping')
        metadata = json.loads(json.dumps(self.checkpoint_metadata, allow_nan=False))
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(version=self.CHECKPOINT_VERSION, config=asdict(self.config),
                       checkpoint_metadata=metadata,
                       critic=self.critic.state_dict(), target_critic=self.target_critic.state_dict(),
                       editor=self.editor.state_dict(), log_temperature=self.log_temperature.detach(),
                       critic_optimizer=self.critic_optimizer.state_dict(),
                       editor_optimizer=self.editor_optimizer.state_dict(),
                       temperature_optimizer=self.temperature_optimizer.state_dict(),
                       rng_state=self.rng.get_state(), updates=self.updates,
                       critic_updates=self.critic_updates, editor_updates=self.editor_updates)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                              suffix=".tmp", delete=False) as output:
                temporary = Path(output.name)
                torch.save(payload, output)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return path

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "EXPOLearner":
        # Only primitive values and tensors are serialized; weights_only keeps
        # a checkpoint from executing arbitrary pickle objects during loading.
        payload = torch.load(path, map_location=device, weights_only=True)
        if payload.get("version") != cls.CHECKPOINT_VERSION:
            raise ValueError("Unsupported EXPO checkpoint version")
        learner = cls(EXPOConfig(**payload["config"]), device)
        metadata = payload.get('checkpoint_metadata', {})
        if not isinstance(metadata, dict):
            raise ValueError('checkpoint_metadata must be a JSON mapping')
        learner.checkpoint_metadata = json.loads(json.dumps(metadata, allow_nan=False))
        for name in ("critic", "target_critic", "editor"):
            getattr(learner, name).load_state_dict(payload[name], strict=True)
        with torch.no_grad():
            learner.log_temperature.copy_(payload["log_temperature"])
        for name in ("critic_optimizer", "editor_optimizer", "temperature_optimizer"):
            getattr(learner, name).load_state_dict(payload[name])
        learner.rng.set_state(payload["rng_state"].cpu())
        learner.updates = int(payload["updates"])
        learner.critic_updates = int(payload["critic_updates"])
        learner.editor_updates = int(payload["editor_updates"])
        return learner
