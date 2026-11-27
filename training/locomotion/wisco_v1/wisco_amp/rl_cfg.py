"""amp-rsl-rl configuration for the Wisco velocity task."""
from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@dataclass
class AmpDatasetCfg:
  """Expert motion data and sampling."""
  amp_data_path: str = ""
  """Folder containing the .npy files produced by prepare_motion.py."""
  datasets: dict[str, float] = field(default_factory=lambda: {"walk": 1.0})
  """Dataset name (without .npy) -> sampling weight."""
  slow_down_factor: float = 1.0
  velocity_representation: str = "body_fixed"
  amp_joint_names: tuple[str, ...] | None = None
  """None lets AMPLoader use the dataset's joints_list."""


@dataclass
class AmpDiscriminatorCfg:
  """AMP discriminator settings."""
  hidden_dims: tuple[int, ...] = (1024, 512)
  reward_scale: float = 2.0
  loss_type: str = "BCEWithLogits"
  empirical_normalization: bool = True
  num_frames: int = 2
  frame_stride: int = 1


@dataclass
class WiscoAmpRunnerCfg(RslRlOnPolicyRunnerCfg):
  """PPO + AMP runner configuration."""
  discriminator: AmpDiscriminatorCfg = field(default_factory=AmpDiscriminatorCfg)
  dataset: AmpDatasetCfg = field(default_factory=AmpDatasetCfg)
  style_weight: float = 0.2
  obs_groups: dict[str, tuple[str, ...]] = field(default_factory=lambda: {
    "actor": ("actor",), "critic": ("critic",), "amp": ("amp",),
  })

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
    class_name="AMP_PPO",
    value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.01,
    num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive",
    gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0,
  ))


def wisco_amp_runner_cfg() -> WiscoAmpRunnerCfg:
  """Fresh runner config (mjlab mutates configs)."""
  return WiscoAmpRunnerCfg()
