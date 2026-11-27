"""Wisco V1 velocity (walking) task, arms-free recipe.

The task is assembled from small, purpose-specific modules:

  recipe.py     tuning constants
  sensors.py    feet contact and non-arm self-collision sensors
  actions.py    policy joints and viewer body
  commands.py   velocity command sampling and command curriculum stages
  events.py     domain randomization, scripted arms, payloads and pushes
  rewards.py    reward terms and metrics
  curriculum.py performance-gated difficulty
  play.py       evaluation overrides
"""
from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from ...robot.robot_cfg import get_wisco_robot_cfg
from .actions import configure_actions
from .commands import configure_commands
from .curriculum import add_curriculum
from .events import add_events
from .play import configure_play
from .rewards import add_rewards
from .sensors import add_sensors


def wisco_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """The Wisco V1 flat velocity task; `play` fixes the curriculum for evaluation."""
  cfg = make_velocity_env_cfg()

  # --- from unitree_g1_rough_env_cfg -------------------------------------------------------
  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 70

  cfg.scene.entities = {"robot": get_wisco_robot_cfg()}                       # WISCO

  self_col_sensors = add_sensors(cfg)
  configure_actions(cfg)
  twist_cmd = configure_commands(cfg)
  add_events(cfg)
  add_rewards(cfg, self_col_sensors)
  add_curriculum(cfg)

  # --- from unitree_g1_flat_env_cfg --------------------------------------------------------
  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.scene.sensors = tuple(s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan")
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    configure_play(cfg, twist_cmd)

  return cfg
