# A task plan: an ordered list of skills to execute one after another.
#
# Published by: task_planner (stub today, LLM-based planner later), on /skill_sequence
# Consumed by:  behavior_tree
#
# The behavior tree queues plans (FIFO) and runs them one after another; a plan
# that arrives while it is busy waits its turn (see behavior_tree/README.md).

std_msgs/Header header
string intent               # the /user_intent text this plan was made for
string plan_id              # unique per plan; a resend carries the same id and the
                            # behavior tree ignores ids it has already seen ("" = no dedupe)
Skill[] skills
