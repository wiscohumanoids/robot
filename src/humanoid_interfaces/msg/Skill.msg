# One step of a task plan. Element of SkillSequence.skills.
#
# `name` is the skill vocabulary shared by the planner and the behavior tree.
# Vocabulary today (see INTERFACE_CONTRACT.md "Skill vocabulary"):
#   "navigate_to"  -> BT calls the nav2_msgs/NavigateToPose action with target_pose
#   "pick", "place", "handover", ... -> BT calls the ExecuteManipulation action
#                     with task=name, object_id=object_id, target_pose=target_pose
# Adding a skill name is an interface change: update INTERFACE_CONTRACT.md and
# the behavior tree in the same PR.

string name
string object_id                          # "" if the skill does not act on an object
geometry_msgs/PoseStamped target_pose     # frame_id normally "map"; unused if not needed
