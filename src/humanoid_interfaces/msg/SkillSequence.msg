# A task plan: an ordered list of skills to execute one after another.
#
# Published by: task_planner (stub today, LLM-based planner later), on /skill_sequence
# Consumed by:  behavior_tree (stub sequential executor today, BehaviorTree.CPP later)
#
# A new message replaces any plan the behavior tree is not yet executing; the BT
# ignores a plan that arrives while it is busy (see behavior_tree_stub/README.md).

std_msgs/Header header
string intent               # the /user_intent text this plan was made for
Skill[] skills
