"""Velocity runner that also records which joints the policy drives.

mjlab's ONNX metadata lists all robot joints (joint_names, gains, default pose) but its
action_scale only covers the action term's joints. For a legs+waist policy the deploy side
needs to know which 15 joints those are, so we add:
  action_joint_names   the action term's joint names, in action order
  held_joint_names     the remaining actuated joints (held by another controller on the robot)
"""
from mjlab.rl.exporter_utils import attach_metadata_to_onnx
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


class SubsetVelocityRunner(VelocityOnPolicyRunner):
  """Velocity runner that writes which joints the policy drives into the ONNX metadata."""
  def save(self, path: str, infos=None):
    """Save as usual, then add the driven joint names to the exported ONNX."""
    super().save(path, infos)
    try:
      _, _, onnx_path = self._get_export_paths(path)
      env = self.env.unwrapped
      term = env.action_manager.get_term("joint_pos")
      acted = list(term.target_names)
      robot = env.scene["robot"]
      held = [n for n in robot.joint_names if n not in acted]
      attach_metadata_to_onnx(str(onnx_path), {"action_joint_names": acted, "held_joint_names": held})
    except Exception as e:
      print(f"[WARN] subset metadata not attached (training continues): {e}")
