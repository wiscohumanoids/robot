"""PPO settings: identical to mjlab 1.6.0's G1 velocity runner; experiment name g1_velocity_v2,
max_iterations 20000 (the coordinator stops the run when it is good)."""
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg


def g1_velocity_v2_ppo_runner_cfg():
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = "g1_velocity_v2"
  cfg.max_iterations = 20_000
  cfg.save_interval = 50
  return cfg
