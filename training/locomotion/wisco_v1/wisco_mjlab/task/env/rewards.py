"""Rewards and metrics for the Wisco velocity task."""
from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from ...robot import linkage_penalty
from ...robot import robot_cfg as RC
from ...robot.robot_cfg import P
from ...terms import arms_free_terms
from .recipe import (
  ANKLE_MOTOR_OVERLOAD_WEIGHT,
  ANKLE_THERMAL_WEIGHT,
  POSE_STD_RUNNING,
  POSE_STD_WALKING,
  TRACK_ANG_STD,
  TRACK_LIN_STD,
)


def add_rewards(cfg: ManagerBasedRlEnvCfg, self_col_sensors: tuple[str, ...]) -> None:
  """Tracking, posture, contact and ankle terms; also the matching metrics."""
  # ARMS-FREE: velocity tracking starts at the G1 defaults and tightens with L2.
  cfg.rewards["track_linear_velocity"].params["std"] = TRACK_LIN_STD[0]
  cfg.rewards["track_angular_velocity"].params["std"] = TRACK_ANG_STD[0]
  cfg.rewards["track_angular_velocity"].func = arms_free_terms.track_yaw_rate   # yaw rate only
  # Fixed-std copies for the curriculum gate (weight 1e-3: ~no effect on training).
  cfg.rewards["gate_track_lin"] = RewardTermCfg(
    func=mdp.track_linear_velocity, weight=1e-3,
    params={**cfg.rewards["track_linear_velocity"].params, "std": TRACK_LIN_STD[0]})
  cfg.rewards["gate_track_ang"] = RewardTermCfg(
    func=arms_free_terms.track_yaw_rate, weight=1e-3,
    params={**cfg.rewards["track_angular_velocity"].params, "std": TRACK_ANG_STD[0]})

  # ARMS-FREE: posture and joint-limit terms on the policy's joints only.
  cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=RC.LOCOMOTION_ACTION_NAMES)
  cfg.rewards["dof_pos_limits"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=RC.LOCOMOTION_ACTION_NAMES)
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = POSE_STD_WALKING
  cfg.rewards["pose"].params["std_running"] = POSE_STD_RUNNING

  cfg.rewards["upright"].params["asset_cfg"].body_names = (P.TORSO_BODY,)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (P.TORSO_BODY,)

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = P.FOOT_SITES

  cfg.rewards["body_ang_vel"].weight = -0.05
  # ARMS-FREE: only the policy-controlled bodies, not the scripted arms/payload. Weight -0.005 because
  # Wisco is heavier and taller than the G1. The whole-body value is logged as a metric for comparison.
  cfg.rewards["angular_momentum"] = RewardTermCfg(
    func=arms_free_terms.policy_body_angular_momentum_sq,
    weight=-0.005,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=(".*",), preserve_order=True)},
  )
  cfg.rewards["angular_momentum"].params["asset_cfg"] = SceneEntityCfg(
    "robot", body_names=tuple(f"^{b}$" for b in RC.POLICY_BODIES), preserve_order=True)
  cfg.metrics["angmom_wholebody_sq"] = MetricsTermCfg(func=arms_free_terms.wholebody_angular_momentum_sq)

  # Feet air time (0.05-0.5 s swings) whenever command > 0.1, so slow commands get short steps.
  cfg.rewards["air_time"].weight = 0.25
  cfg.rewards["air_time"].params["command_threshold"] = 0.1

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=arms_free_terms.self_collision_cost_multi,
    weight=-1.0,
    params={"sensor_names": self_col_sensors, "force_threshold": 10.0},
  )

  # WISCO: ankle linkage penalty + metrics to watch peak/RMS/rated ankle motor torque.
  cfg.rewards["ankle_motor_overload"] = RewardTermCfg(
    func=linkage_penalty.ankle_motor_overload,
    weight=ANKLE_MOTOR_OVERLOAD_WEIGHT,
    params={"motor_limit": P.X4_36_PEAK_NM},
  )
  cfg.metrics["ankle_motor_peak_nm"] = MetricsTermCfg(func=linkage_penalty.ankle_motor_peak, reduce="max")
  # Sustained torque above rated (thermal). With all 4 motors at 50 % over rated the term is -1.0/s,
  # half the max linear-tracking reward (2.0/s).
  cfg.rewards["ankle_motor_thermal"] = RewardTermCfg(
    func=linkage_penalty.ankle_motor_thermal, weight=ANKLE_THERMAL_WEIGHT,
    params={"rated_nm": P.X4_36_RATED_NM, "tau_s": 2.0})
  cfg.metrics["ankle_motor_frac_above_rated"] = MetricsTermCfg(
    func=linkage_penalty.ankle_motor_frac_above_rated, params={"rated_nm": P.X4_36_RATED_NM})
  cfg.metrics["ankle_motor_rms_nm"] = MetricsTermCfg(func=linkage_penalty.ankle_motor_rms)
