"""PPO + AMP runner for the Wisco velocity task.

The actor/critic are the normal velocity policy. The discriminator is an extra
training-only network over reference and policy transitions. At inference the
discriminator is not used; the exported actor is the same ONNX policy as the
velocity task.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
from rsl_rl.utils import check_nan

from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .discriminator import AMPDiscriminator
from .features import FeatureBuilder, FeatureSpec, FeatureNormalizer, reference_pairs
from .motion import load_clips


class AMPOnPolicyRunner(VelocityOnPolicyRunner):
  """Velocity PPO plus an AMP discriminator and style reward."""

  def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
    """Build the normal PPO runner, then the AMP state."""
    super().__init__(env, train_cfg, log_dir, device)

    amp_cfg = dict(self.cfg.get("amp") or {})
    if not amp_cfg.get("motion_files"):
      raise ValueError("AMP needs at least one motion file: --agent.amp.motion-files <clip.npz>")

    self._amp_cfg = amp_cfg
    self._spec = FeatureSpec.from_env(self.env.unwrapped)
    self._builder = FeatureBuilder(self._spec)

    clips = load_clips(tuple(amp_cfg["motion_files"]))
    raw_reference = []
    for clip in clips:
      raw_reference.append(self._builder.from_motion(clip))
    reference = np.concatenate(raw_reference, axis=0)
    if amp_cfg.get("normalize_features", True):
      self._normalizer = FeatureNormalizer.fit(reference)
    else:
      self._normalizer = FeatureNormalizer(
        np.zeros(self._spec.dim, dtype=np.float32), np.ones(self._spec.dim, dtype=np.float32))

    self._reference_pairs = reference_pairs(clips, self._builder, self._normalizer).to(self.device)
    self.discriminator = AMPDiscriminator(
      input_dim=self._spec.dim * 2,
      hidden_dims=tuple(amp_cfg.get("hidden_dims", (1024, 512))),
      activation=str(amp_cfg.get("activation", "elu")),
    ).to(self.device)
    self.disc_optimizer = torch.optim.Adam(
      self.discriminator.parameters(),
      lr=float(amp_cfg.get("learning_rate", 1e-4)),
      weight_decay=float(amp_cfg.get("weight_decay", 1e-4)),
    )
    self._style_weight = float(amp_cfg.get("style_weight", 0.1))
    self._style_warmup = int(amp_cfg.get("style_warmup_iterations", 100))
    self._style_ramp = max(1, int(amp_cfg.get("style_ramp_iterations", 500)))
    self._expert_batch_size = int(amp_cfg.get("expert_batch_size", 4096))
    self._policy_batch_size = int(amp_cfg.get("policy_batch_size", 4096))
    self._gradient_penalty = float(amp_cfg.get("gradient_penalty", 10.0))
    self._style_coef = 0.0
    self._last_amp_metrics: dict[str, float] = {}

  def _features(self) -> torch.Tensor:
    """Normalized AMP features from the live environment."""
    return self._normalizer.normalize_torch(self._builder.from_env(self.env.unwrapped))

  @staticmethod
  def _pairs(prev: torch.Tensor, curr: torch.Tensor) -> torch.Tensor:
    """Concatenate consecutive feature vectors into discriminator inputs."""
    return torch.cat((prev, curr), dim=-1)

  def _style_weight_at(self, iteration: int) -> float:
    """Zero before warmup, then linear ramp to the configured weight."""
    if self._style_weight <= 0.0 or iteration < self._style_warmup:
      return 0.0
    return self._style_weight * min(1.0, (iteration - self._style_warmup + 1) / self._style_ramp)

  def _update_discriminator(self, policy_pairs: list[torch.Tensor]) -> dict[str, float]:
    """One discriminator update on reference and policy transitions."""
    if not policy_pairs:
      return {}
    policy = torch.cat(policy_pairs, dim=0)
    if len(policy) > self._policy_batch_size:
      idx = torch.randperm(len(policy), device=policy.device)[:self._policy_batch_size]
      policy = policy[idx]
    n = min(self._expert_batch_size, len(self._reference_pairs))
    idx = torch.randint(0, len(self._reference_pairs), (n,), device=self._reference_pairs.device)
    expert = self._reference_pairs[idx]

    self.discriminator.train()
    loss, metrics = self.discriminator.loss(
      expert, policy, gradient_penalty=self._gradient_penalty)
    self.disc_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    self.disc_optimizer.step()
    metrics["amp/style_coef"] = self._style_coef
    self._last_amp_metrics = metrics
    return metrics

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
    """PPO rollout/update loop with an AMP discriminator update each iteration."""
    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=int(self.env.max_episode_length))

    obs = self.env.get_observations().to(self.device)
    self.alg.train_mode()
    if self.is_distributed:
      print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
      self.alg.broadcast_parameters()

    self.logger.init_logging_writer()
    amp_prev = self._features()

    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations
    for it in range(start_it, total_it):
      start = time.time()
      self._style_coef = self._style_weight_at(it)
      policy_pairs: list[torch.Tensor] = []
      style_sum = 0.0
      style_count = 0

      with torch.inference_mode():
        for _ in range(self.cfg["num_steps_per_env"]):
          actions = self.alg.act(obs)
          obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
          if self.cfg.get("check_for_nan", True):
            check_nan(obs, rewards, dones)
          obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)

          amp_curr = self._features()
          pair = self._pairs(amp_prev, amp_curr)
          with torch.no_grad():
            style = self.discriminator.style_reward(pair) * (1.0 - dones.float())
          if self._style_coef > 0.0:
            rewards = rewards + self._style_coef * style
            keep = ~dones.bool()
            if keep.any():
              policy_pairs.append(pair[keep].detach())
              style_sum += float(style[keep].mean())
              style_count += 1
          amp_prev = amp_curr

          self.alg.process_env_step(obs, rewards, dones, extras)
          intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"].get("rnd_cfg") else None
          self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

        stop = time.time()
        collect_time = stop - start
        start = stop
        self.alg.compute_returns(obs)

      loss_dict = self.alg.update()
      amp_metrics = self._update_discriminator(policy_pairs)
      loss_dict.update(amp_metrics)
      if style_count:
        loss_dict["amp/style_reward"] = style_sum / style_count
      loss_dict["amp/style_coef"] = self._style_coef

      stop = time.time()
      learn_time = stop - start
      self.current_learning_iteration = it
      self.logger.log(
        it=it,
        start_it=start_it,
        total_it=total_it,
        collect_time=collect_time,
        learn_time=learn_time,
        loss_dict=loss_dict,
        learning_rate=self.alg.learning_rate,
        action_std=self.alg.get_policy().output_std,
        rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"].get("rnd_cfg") else None,
      )

      if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
        self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

    if self.logger.writer is not None:
      self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
      self.logger.stop_logging_writer()

  def save(self, path: str, infos: dict | None = None) -> None:
    """Save the normal checkpoint plus discriminator and normalizer state."""
    super().save(path, infos)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint["amp"] = {
      "discriminator": {k: v.cpu() for k, v in self.discriminator.state_dict().items()},
      "normalizer": self._normalizer.state_dict(),
      "style_coef": self._style_coef,
    }
    torch.save(checkpoint, path)

  def load(self, path: str, load_cfg: dict | None = None, strict: bool = True,
           map_location: str | None = None) -> dict:
    """Restore normal PPO state and the AMP discriminator/normalizer."""
    raw = torch.load(path, map_location=map_location or self.device, weights_only=False)
    infos = super().load(path, load_cfg, strict, map_location)
    amp = raw.get("amp")
    if amp:
      self.discriminator.load_state_dict(amp["discriminator"])
      self._normalizer.load_state_dict(amp["normalizer"])
      self._style_coef = float(amp.get("style_coef", self._style_coef))
    else:
      print(f"[WARN] {path} has no AMP state; starting a fresh discriminator")
    return infos
