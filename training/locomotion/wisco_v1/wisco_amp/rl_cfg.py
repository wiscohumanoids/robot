"""Runner configuration and AMP hyperparameters."""
from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@dataclass
class AmpCfg:
  """Reference motions and discriminator settings."""
  motion_files: tuple[str, ...] = field(default_factory=tuple)
  """Retargeted .npz clips in training-model joint/body order."""
  hidden_dims: tuple[int, ...] = (1024, 512)
  activation: str = "elu"
  learning_rate: float = 1.0e-4
  weight_decay: float = 1.0e-4
  gradient_penalty: float = 10.0
  expert_batch_size: int = 4096
  policy_batch_size: int = 4096
  style_weight: float = 0.1
  style_warmup_iterations: int = 100
  style_ramp_iterations: int = 500
  normalize_features: bool = True
  """Normalize features with reference-dataset mean/std."""


@dataclass
class WiscoAmpRunnerCfg(RslRlOnPolicyRunnerCfg):
  """PPO settings plus AMP fields; PPO values match the velocity task."""
  amp: AmpCfg = field(default_factory=AmpCfg)

  experiment_name: str = "wisco_v1_amp"
  save_interval: int = 50
  max_iterations: int = 30_000

  actor: RslRlModelCfg = field(default_factory=lambda: RslRlModelCfg(
    hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True,
    distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
  ))
  critic: RslRlModelCfg = field(default_factory=lambda: RslRlModelCfg(
    hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True,
  ))
  algorithm: RslRlPpoAlgorithmCfg = field(default_factory=lambda: RslRlPpoAlgorithmCfg(
    value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.01,
    num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive",
    gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0,
  ))


def wisco_amp_runner_cfg() -> WiscoAmpRunnerCfg:
  """Fresh AMP runner config (mjlab mutates configs)."""
  return WiscoAmpRunnerCfg()
