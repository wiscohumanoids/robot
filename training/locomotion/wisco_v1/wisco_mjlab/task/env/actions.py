"""Actions and viewer body for the Wisco velocity task."""
from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg

from ...robot import robot_cfg as RC
from ...robot.robot_cfg import P


def configure_actions(cfg: ManagerBasedRlEnvCfg) -> None:
  """Drive only the policy's joints (legs + waist) and show the torso in the viewer."""
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = RC.LOCOMOTION_ACTION_NAMES                   # ARMS-FREE: legs + waist only (15)
  joint_pos_action.preserve_order = True
  joint_pos_action.scale = RC.LOCOMOTION_ACTION_SCALE                            # WISCO

  cfg.viewer.body_name = P.TORSO_BODY                                            # WISCO: "torso"
