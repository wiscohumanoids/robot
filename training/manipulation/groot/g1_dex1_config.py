"""GR00T modality config for our sim G1 + Dex1-1 grippers, standing at a counter (NEW_EMBODIMENT).

Same API in GR00T N1.6 (n1d6 branch) and N1.7 (main). Passed to launch_finetune.py with
--modality-config-path; after fine-tuning it is saved inside the checkpoint.

Why NEW_EMBODIMENT and not the pretrained REAL_G1 tag: REAL_G1 is real G1 hardware with Dex3 hands,
relative wrist-EEF + joint actions, and whole-body commands (base_height_command, navigate_command).
Our data is sim G1 with Dex1 parallel grippers, standing still, commanded in joint space. Matching
REAL_G1's keys would mean faking hand joints and nav commands, so we register our own layout and
follow REAL_G1's style where it fits: arm joints as RELATIVE actions, grippers ABSOLUTE.

Layout (must match meta/modality.json written by raw_to_lerobot.py):
  state  : left_arm 7, right_arm 7, left_gripper 1, right_gripper 1, waist 3   (19)
  action : left_arm 7, right_arm 7, left_gripper 1, right_gripper 1            (16)
  video  : ego_view (head camera)
  grippers: 0 = closed, 1 = open
"""
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

# 16 steps at 20 Hz = 0.8 s of actions per prediction; the robot executes the first 8 (EXECUTION_HORIZON).
ACTION_HORIZON = 16

g1_dex1_config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["ego_view"]),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["left_arm", "right_arm", "left_gripper", "right_gripper", "waist"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=["left_arm", "right_arm", "left_gripper", "right_gripper"],
        action_configs=[
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
        ],
    ),
    "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"]),
}

register_modality_config(g1_dex1_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
