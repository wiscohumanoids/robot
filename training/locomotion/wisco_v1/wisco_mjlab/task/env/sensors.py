"""Contact and self-collision sensors for the Wisco velocity task."""
from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, RingPatternCfg, TerrainHeightSensorCfg

from ...robot.robot_cfg import P


def add_sensors(cfg: ManagerBasedRlEnvCfg) -> tuple[str, ...]:
  """Add feet-contact and non-arm self-collision sensors; return the self-collision sensor names."""
  site_names = P.FOOT_SITES

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(ObjRef(type="site", name=s, entity="robot") for s in site_names)
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",                # same body names as the G1
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  # ARMS-FREE: self-collision counts only contacts where neither body is an arm link. The arms are moved by the
  # scripted controls-stack stand-in, so the policy can't avoid arm contacts. A contact sensor takes one secondary
  # element, so the non-arm pairs are covered by one sensor per group: left leg vs the whole right leg, and both
  # legs vs the pelvis and each torso link. Arm links appear in none of them.
  left_leg = r"^left_(hip_pitch|hip_roll|hip_yaw|knee|ankle_pitch|ankle_roll)_link$"
  legs = r"^(left|right)_(hip_pitch|hip_roll|hip_yaw|knee|ankle_pitch|ankle_roll)_link$"

  def _self_col(name, primary, secondary_mode, secondary):
    return ContactSensorCfg(
      name=name, primary=ContactMatch(mode="body", pattern=primary, entity="robot"),
      secondary=ContactMatch(mode=secondary_mode, pattern=secondary, entity="robot"),
      fields=("found", "force"), reduce="none", num_slots=1, history_length=4)

  self_col_sensors = (
    _self_col("selfcol_leg_leg", left_leg, "subtree", "right_hip_pitch_link"),
    _self_col("selfcol_legs_pelvis", legs, "body", "pelvis"),
    _self_col("selfcol_legs_torso_yaw", legs, "body", "torso_yaw_link"),
    _self_col("selfcol_legs_torso_roll", legs, "body", "torso_roll_link"),
    _self_col("selfcol_legs_torso", legs, "body", "torso"),
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground_cfg,) + self_col_sensors
  return tuple(s.name for s in self_col_sensors)
