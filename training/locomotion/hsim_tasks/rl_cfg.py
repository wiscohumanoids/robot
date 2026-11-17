"""PPO settings: identical to mjlab 1.6.0's G1 velocity runner; experiment name g1_velocity_arms_free,
max_iterations 20000 (an upper bound: stop the run once it is good)."""
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg


def g1_velocity_arms_free_ppo_runner_cfg():
  """mjlab's G1 PPO settings with this experiment's name and length."""
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = "g1_velocity_arms_free"
  cfg.max_iterations = 20_000
  cfg.save_interval = 50
  return cfg
